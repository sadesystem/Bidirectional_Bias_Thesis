from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from transformers import AutoModelForMaskedLM, AutoTokenizer

METHODS = ("original", "within_word_l2r")
PRONOUN_SET = {"he", "him", "his", "she", "her", "hers"}
PRONOUN_RE = re.compile(r"\b(?:he|him|his|she|her|hers)\b", flags=re.IGNORECASE)


@dataclass
class SpanInfo:
    text: str
    needle: str
    tokens: List[str]
    ids: List[int]
    offsets: List[Tuple[int, int]]
    char_span: Optional[Tuple[int, int]]
    tok_span: Optional[Tuple[int, int]]


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("SimpleGenderBiasAnalysis")


def setup_output_dirs(output_dir: str, dataset_name: str, model_name: str):
    model_short = model_name.replace("/", "-").replace("-uncased", "").lower()
    dataset_short = str(dataset_name).lower()
    root = Path(output_dir) / dataset_short / model_short
    root.mkdir(parents=True, exist_ok=True)
    return {"root": root}


def resolve_torch_device(device: Optional[str] = None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return resolved


def _load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def _load_masked_lm_with_validation(model_name: str):
    model, loading_info = AutoModelForMaskedLM.from_pretrained(
        model_name,
        output_loading_info=True,
    )
    missing_keys = loading_info.get("missing_keys", []) or []
    mlm_head_markers = (
        "cls.predictions.",
        "lm_head.",
        "mlm_head.",
        "masked_lm",
        "vocab_projector",
        "pred_layer",
        "decoder.",
        "generator_lm_head",
    )
    missing_mlm_head = [k for k in missing_keys if any(marker in k for marker in mlm_head_markers)]
    if missing_mlm_head:
        preview = ", ".join(missing_mlm_head[:6])
        if len(missing_mlm_head) > 6:
            preview += ", ..."
        raise RuntimeError(
            f"Model '{model_name}' is not valid for masked-LM PLL analysis: "
            f"MLM-head parameters were missing at load time ({preview})."
        )
    return model


def get_model_lower_case(tokenizer: AutoTokenizer) -> bool:
    return getattr(tokenizer, "do_lower_case", False) is True


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def to_model_casing(text: str, lower: bool) -> str:
    return text.lower() if lower else text


def token_to_word(token: str) -> str:
    surface = token.replace("▁", " ").lstrip().lstrip("Ġ")
    if surface.startswith("##"):
        surface = surface[2:]
    return surface.lower()


def _token_surface_for_alignment(token: str) -> str:
    surface = token.replace("▁", " ").lstrip()
    surface = surface.lstrip("Ġ")
    if surface.startswith("##"):
        surface = surface[2:]
    return surface


def _tokenize_with_offset_mapping(tokenizer: AutoTokenizer, text: str):
    try:
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True, return_offsets_mapping=True)
        return enc, enc["offset_mapping"][0].tolist()
    except NotImplementedError:
        pass

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    offsets = []
    pos = 0
    for tok in tokens:
        if tok in getattr(tokenizer, "all_special_tokens", []) or tok in ("[CLS]", "[SEP]", "<s>", "</s>", "<pad>", "[PAD]"):
            offsets.append((0, 0))
            continue
        surface = _token_surface_for_alignment(tok)
        if not surface:
            offsets.append((0, 0))
            continue
        idx = text.find(surface, pos)
        if idx == -1:
            idx = text.find(surface)
        if idx == -1:
            offsets.append((0, 0))
            continue
        start, end = idx, idx + len(surface)
        offsets.append((start, end))
        pos = end
    return enc, offsets


def find_span_in_text(text: str, needle: str, tokenizer: AutoTokenizer, lower: bool) -> SpanInfo:
    target_text = text.lower() if lower else text
    target_needle = needle.lower() if lower else needle
    enc, offsets = _tokenize_with_offset_mapping(tokenizer, target_text)
    ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    start = target_text.find(target_needle)
    if start == -1:
        return SpanInfo(text, needle, tokens, ids, offsets, None, None)
    end = start + len(target_needle)
    idxs = [i for i, (a, b) in enumerate(offsets) if a != b and not (b <= start or a >= end)]
    tok_span = (min(idxs), max(idxs)) if idxs else None
    return SpanInfo(text, needle, tokens, ids, offsets, (start, end), tok_span)


def _find_token_span_for_char_span(
    text: str,
    tokenizer: AutoTokenizer,
    lower: bool,
    char_span: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    target_text = text.lower() if lower else text
    _, offsets = _tokenize_with_offset_mapping(tokenizer, target_text)
    idxs = [i for i, (a, b) in enumerate(offsets) if a != b and not (b <= char_span[0] or a >= char_span[1])]
    if not idxs:
        return None
    return (min(idxs), max(idxs))


def _word_ids_from_offsets(text: str, offsets: List[Tuple[int, int]]) -> List[Optional[int]]:
    word_spans = [match.span() for match in re.finditer(r"\S+", text)]
    word_ids: List[Optional[int]] = []
    for start, end in offsets:
        if start == end:
            word_ids.append(None)
            continue
        word_id = None
        for idx, (w_start, w_end) in enumerate(word_spans):
            if start >= w_start and end <= w_end:
                word_id = idx
                break
        word_ids.append(word_id)
    return word_ids


def find_changed_pronoun_spans(
    sent_male: str,
    sent_female: str,
    tokenizer: AutoTokenizer,
    lower: bool,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[str], Optional[str], str]:
    male_matches = list(PRONOUN_RE.finditer(sent_male))
    female_matches = list(PRONOUN_RE.finditer(sent_female))
    if not male_matches or not female_matches:
        return None, None, None, None, "missing pronoun"
    if len(male_matches) != len(female_matches):
        return None, None, None, None, "pronoun count mismatch"
    for male_match, female_match in zip(male_matches, female_matches):
        male_word = male_match.group(0).lower()
        female_word = female_match.group(0).lower()
        if male_word == female_word:
            continue
        male_span = _find_token_span_for_char_span(sent_male, tokenizer, lower, male_match.span())
        female_span = _find_token_span_for_char_span(sent_female, tokenizer, lower, female_match.span())
        if male_span is None or female_span is None:
            return None, None, male_word, female_word, "pronoun alignment failed"
        return male_span, female_span, male_word, female_word, "ok"
    return None, None, None, None, "no differing pronoun"


def _mask_positions_for_mode(
    pos: int,
    span: Tuple[int, int],
    word_ids: List[Optional[int]],
    which_masking: str,
) -> List[int]:
    if which_masking == "original":
        return [pos]
    if which_masking == "within_word_l2r":
        current_word = word_ids[pos] if pos < len(word_ids) else None
        if current_word is None:
            return [pos]
        positions = [j for j in range(pos, span[1] + 1) if word_ids[j] == current_word]
        return positions if pos in positions else [pos] + positions
    raise NotImplementedError(f"Unsupported masking mode for simple runner: {which_masking}")


def compute_pll_for_span(
    ids: torch.Tensor,
    span: Tuple[int, int],
    model,
    mask_id: Optional[int],
    word_ids: Optional[List[Optional[int]]] = None,
    which_masking: str = "original",
) -> float:
    if mask_id is None:
        raise RuntimeError("This tokenizer does not define a mask token.")
    x = ids.clone()
    total = 0.0
    if word_ids is None:
        word_ids = [None] * ids.shape[1]
    with torch.no_grad():
        for pos in range(span[0], span[1] + 1):
            mask_positions = _mask_positions_for_mode(pos, span, word_ids, which_masking)
            originals = {j: x[0, j].item() for j in mask_positions}
            for j in mask_positions:
                x[0, j] = mask_id
            logits = model(x).logits[0, pos]
            log_probs = torch.log_softmax(logits, dim=-1)
            total += log_probs[originals[pos]].item()
            for j, original in originals.items():
                x[0, j] = original
    return float(total)


def process_single_example(
    model,
    tokenizer,
    sent_male: str,
    sent_female: str,
    context: str,
    lower: bool,
    device: Optional[torch.device] = None,
) -> Dict:
    if device is None:
        device = next(model.parameters()).device
    sm = to_model_casing(sent_male, lower)
    sw = to_model_casing(sent_female, lower)
    cx = to_model_casing(context, lower)
    result = {
        "Context": context,
        "Sentence_m": sent_male,
        "Sentence_w": sent_female,
    }

    span_m = find_span_in_text(sm, cx, tokenizer, lower)
    span_w = find_span_in_text(sw, cx, tokenizer, lower)
    if span_m.tok_span and span_w.tok_span:
        ids_m = torch.tensor([span_m.ids], dtype=torch.long, device=device)
        ids_w = torch.tensor([span_w.ids], dtype=torch.long, device=device)
        word_ids_m = _word_ids_from_offsets(sm, span_m.offsets)
        word_ids_w = _word_ids_from_offsets(sw, span_w.offsets)
        for method_key in METHODS:
            suffix = "" if method_key == "original" else f"_{method_key}"
            pll_m = compute_pll_for_span(ids_m, span_m.tok_span, model, tokenizer.mask_token_id, word_ids_m, method_key)
            pll_w = compute_pll_for_span(ids_w, span_w.tok_span, model, tokenizer.mask_token_id, word_ids_w, method_key)
            delta = pll_m - pll_w
            p_male = sigmoid(delta)
            result.update(
                {
                    f"PLL_m_context{suffix}": pll_m,
                    f"PLL_w_context{suffix}": pll_w,
                    f"Context_Delta{suffix}": delta,
                    f"Context_P_male{suffix}": p_male,
                    f"Context_P_female{suffix}": 1.0 - p_male,
                    f"Context_Label{suffix}": "male" if p_male > 0.5 else ("female" if p_male < 0.5 else "tie"),
                }
            )
    else:
        for method_key in METHODS:
            suffix = "" if method_key == "original" else f"_{method_key}"
            result[f"Context_Delta{suffix}"] = np.nan

    enc_m2, offsets_m2 = _tokenize_with_offset_mapping(tokenizer, sm)
    enc_w2, offsets_w2 = _tokenize_with_offset_mapping(tokenizer, sw)
    ids_m2 = enc_m2["input_ids"].to(device)
    ids_w2 = enc_w2["input_ids"].to(device)
    word_ids_m2 = _word_ids_from_offsets(sm, offsets_m2)
    word_ids_w2 = _word_ids_from_offsets(sw, offsets_w2)
    pm, pw, pron_m_txt, pron_w_txt, pron_reason = find_changed_pronoun_spans(sm, sw, tokenizer, lower)
    if pm and pw:
        for method_key in METHODS:
            suffix = "" if method_key == "original" else f"_{method_key}"
            pll_pm = compute_pll_for_span(ids_m2, pm, model, tokenizer.mask_token_id, word_ids_m2, method_key)
            pll_pw = compute_pll_for_span(ids_w2, pw, model, tokenizer.mask_token_id, word_ids_w2, method_key)
            delta_pron = pll_pm - pll_pw
            p_pron = sigmoid(delta_pron)
            result.update(
                {
                    f"PLL_m_pronoun{suffix}": pll_pm,
                    f"PLL_w_pronoun{suffix}": pll_pw,
                    f"Pronoun_Delta{suffix}": delta_pron,
                    f"Pronoun_P_male{suffix}": p_pron,
                    f"Pronoun_P_female{suffix}": 1.0 - p_pron,
                    f"Pronoun_Label{suffix}": "male" if p_pron > 0.5 else ("female" if p_pron < 0.5 else "tie"),
                }
            )
        result["PronounToken_m"] = pron_m_txt
        result["PronounToken_w"] = pron_w_txt
        result["SpanOK_pronoun"] = True
        result["Pronoun_Skip_Reason"] = ""
    else:
        for method_key in METHODS:
            suffix = "" if method_key == "original" else f"_{method_key}"
            result[f"Pronoun_Delta{suffix}"] = np.nan
        result["PronounToken_m"] = pron_m_txt
        result["PronounToken_w"] = pron_w_txt
        result["SpanOK_pronoun"] = False
        result["Pronoun_Skip_Reason"] = pron_reason
    return result


def generate_method_results_csv(results: List[Dict], method_key: str) -> pd.DataFrame:
    df = pd.DataFrame(results)
    suffix = "" if method_key == "original" else f"_{method_key}"
    common_cols = ["Context", "Sentence_m", "Sentence_w", "idx", "PronounToken_m", "PronounToken_w", "SpanOK_pronoun", "Pronoun_Skip_Reason"]
    if "HB" in df.columns:
        common_cols.append("HB")
    method_cols = [
        f"PLL_m_context{suffix}",
        f"PLL_w_context{suffix}",
        f"Context_Delta{suffix}",
        f"Context_P_male{suffix}",
        f"Context_P_female{suffix}",
        f"Context_Label{suffix}",
        f"PLL_m_pronoun{suffix}",
        f"PLL_w_pronoun{suffix}",
        f"Pronoun_Delta{suffix}",
        f"Pronoun_P_male{suffix}",
        f"Pronoun_P_female{suffix}",
        f"Pronoun_Label{suffix}",
    ]
    cols = [c for c in common_cols + method_cols if c in df.columns]
    return df[cols].copy()


def generate_method_aggregates(results: List[Dict], method_key: str, min_count: int = 1) -> pd.DataFrame:
    df = pd.DataFrame(results)
    delta_col = "Context_Delta" if method_key == "original" else f"Context_Delta_{method_key}"
    if delta_col not in df.columns or "Context" not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby("Context")[delta_col].agg(["mean", "std", "count"]).reset_index()
    grouped.columns = ["Context", "mean_delta", "std_delta", "count"]
    grouped = grouped[grouped["count"] >= min_count]
    grouped["mean_abs_delta"] = grouped["mean_delta"].abs()
    return grouped.sort_values(["mean_delta", "mean_abs_delta"], ascending=[False, False])


def _interpret_cohens_d(value: float) -> str:
    abs_d = abs(value)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def _get_method_stats(deltas: np.ndarray) -> Dict:
    if deltas.size == 0:
        return {}
    m_count = int((deltas > 0).sum())
    f_count = int((deltas < 0).sum())
    t_count = int((deltas == 0).sum())
    total = m_count + f_count + t_count
    return {
        "n": int(deltas.size),
        "mean": float(np.mean(deltas)),
        "std": float(np.std(deltas)),
        "male_count": m_count,
        "female_count": f_count,
        "tie_count": t_count,
        "male_pct": 100.0 * m_count / total if total > 0 else 0.0,
        "female_pct": 100.0 * f_count / total if total > 0 else 0.0,
        "tie_pct": 100.0 * t_count / total if total > 0 else 0.0,
    }


def _fdr_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    n = len(p_values)
    if n == 0:
        return [], []
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values, dtype=float)[sorted_idx]
    thresholds = np.arange(1, n + 1) / n * alpha
    rejected_sorted = sorted_p <= thresholds
    if rejected_sorted.any():
        max_idx = int(np.where(rejected_sorted)[0].max())
        rejected_sorted = np.arange(n) <= max_idx
    adjusted_sorted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running = min(running, sorted_p[i] * n / rank)
        adjusted_sorted[i] = running
    rejected = np.zeros(n, dtype=bool)
    adjusted = np.empty(n, dtype=float)
    rejected[sorted_idx] = rejected_sorted
    adjusted[sorted_idx] = np.clip(adjusted_sorted, 0.0, 1.0)
    return rejected.tolist(), adjusted.tolist()


def compute_method_statistical_significance(
    results: List[Dict],
    method_key: str,
    min_count: int = 3,
    alpha: float = 0.05,
) -> Dict:
    df = pd.DataFrame(results)
    delta_col = "Context_Delta" if method_key == "original" else f"Context_Delta_{method_key}"
    stats_report = {
        "method": method_key,
        "overall": {},
        "per_context": {},
        "summary": {},
    }
    if delta_col not in df.columns:
        stats_report["overall"]["error"] = f"Column {delta_col} not found"
        return stats_report
    ctx_deltas = df[delta_col].dropna().to_numpy()
    if len(ctx_deltas) < 2:
        stats_report["overall"]["error"] = "Insufficient data for statistical tests"
        return stats_report
    t_stat, t_pval = stats.ttest_1samp(ctx_deltas, 0)
    mean_delta = np.mean(ctx_deltas)
    std_delta = np.std(ctx_deltas, ddof=1)
    cohens_d = mean_delta / std_delta if std_delta > 0 else 0.0
    stats_report["overall"]["t_test"] = {
        "statistic": float(t_stat),
        "p_value": float(t_pval),
        "significant": t_pval < alpha,
    }
    stats_report["overall"]["effect_size"] = {
        "cohens_d": float(cohens_d),
        "interpretation": _interpret_cohens_d(cohens_d),
    }
    stats_report["overall"]["descriptive"] = {
        "n": len(ctx_deltas),
        "mean": float(mean_delta),
        "std": float(std_delta),
        "median": float(np.median(ctx_deltas)),
        "min": float(np.min(ctx_deltas)),
        "max": float(np.max(ctx_deltas)),
    }
    use = df[["Context", delta_col]].dropna()
    context_stats = {}
    for ctx, group in use.groupby("Context"):
        deltas = group[delta_col].to_numpy()
        if len(deltas) < min_count:
            continue
        ctx_mean = np.mean(deltas)
        ctx_std = np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0
        if len(deltas) >= 2:
            ctx_t, ctx_p = stats.ttest_1samp(deltas, 0)
            ctx_d = ctx_mean / ctx_std if ctx_std > 0 else 0.0
            context_stats[ctx] = {
                "n": len(deltas),
                "mean_delta": float(ctx_mean),
                "std_delta": float(ctx_std),
                "t_statistic": float(ctx_t),
                "p_value": float(ctx_p),
                "cohens_d": float(ctx_d),
                "significant_005": ctx_p < alpha,
                "direction": "male" if ctx_mean > 0 else "female",
            }
    stats_report["per_context"] = context_stats
    if context_stats:
        p_values = [v["p_value"] for v in context_stats.values()]
        contexts = list(context_stats.keys())
        rejected, adjusted_pvals = _fdr_correction(p_values, alpha)
        for idx, ctx in enumerate(contexts):
            context_stats[ctx]["p_value_fdr"] = float(adjusted_pvals[idx])
            context_stats[ctx]["significant_fdr"] = bool(rejected[idx])
        stats_report["summary"]["multiple_comparison"] = {
            "n_contexts_tested": len(p_values),
            "uncorrected_significant": int(sum(1 for p in p_values if p < alpha)),
            "fdr_significant": int(sum(rejected)),
        }
    return stats_report


def save_method_significance_report(stats_report: Dict, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx_df = pd.DataFrame(
        [
            {
                "Context": ctx,
                "n": data.get("n", 0),
                "mean_delta": data.get("mean_delta", 0),
                "std_delta": data.get("std_delta", 0),
                "t_statistic": data.get("t_statistic", 0),
                "p_value": data.get("p_value", 1),
                "p_value_fdr": data.get("p_value_fdr", 1),
                "cohens_d": data.get("cohens_d", 0),
                "significant_fdr": data.get("significant_fdr", False),
                "direction": data.get("direction", ""),
            }
            for ctx, data in stats_report.get("per_context", {}).items()
        ]
    )
    if not ctx_df.empty:
        ctx_df = ctx_df.sort_values(["significant_fdr", "p_value_fdr", "p_value"], ascending=[False, True, True])
    ctx_df.to_csv(output_dir / "context_significance.csv", index=False)
    lines = [
        "=" * 70,
        f"STATISTICAL SIGNIFICANCE - {stats_report.get('method', 'unknown')}",
        "=" * 70,
        "",
    ]
    overall = stats_report.get("overall", {})
    if "error" in overall:
        lines.append(f"ERROR: {overall['error']}")
    else:
        desc = overall.get("descriptive", {})
        t_test = overall.get("t_test", {})
        effect = overall.get("effect_size", {})
        lines.extend(
            [
                "OVERALL BIAS ANALYSIS",
                "-" * 40,
                f"  N samples        : {desc.get('n', 'N/A')}",
                f"  Mean Delta       : {desc.get('mean', 0):.6f}",
                f"  Std Delta        : {desc.get('std', 0):.6f}",
                f"  t-statistic      : {t_test.get('statistic', 0):.4f}",
                f"  p-value          : {t_test.get('p_value', 1):.2e}",
                f"  Significant      : {'YES' if t_test.get('significant') else 'NO'} (α=0.05)",
                f"  Cohen's d        : {effect.get('cohens_d', 0):.4f}",
                f"  Interpretation   : {effect.get('interpretation', 'N/A')}",
                "",
            ]
        )
        mc = stats_report.get("summary", {}).get("multiple_comparison", {})
        if mc:
            lines.extend(
                [
                    "PER-CONTEXT MULTIPLE TESTING",
                    "-" * 40,
                    f"  Contexts tested          : {mc.get('n_contexts_tested', 0)}",
                    f"  Uncorrected significant  : {mc.get('uncorrected_significant', 0)}",
                    f"  FDR significant          : {mc.get('fdr_significant', 0)}",
                    "",
                ]
            )
    (output_dir / "statistical_significance.txt").write_text("\n".join(lines))


def generate_method_summary(results: List[Dict], model_name: str, dataset_name: str, method_key: str) -> List[str]:
    df = pd.DataFrame(results)
    delta_col = "Context_Delta" if method_key == "original" else f"Context_Delta_{method_key}"
    pron_col = "Pronoun_Delta" if method_key == "original" else f"Pronoun_Delta_{method_key}"
    ctx = df[delta_col].dropna().to_numpy() if delta_col in df.columns else np.array([])
    pron = df[pron_col].dropna().to_numpy() if pron_col in df.columns else np.array([])
    ctx_stats = _get_method_stats(ctx)
    pron_stats = _get_method_stats(pron)
    lines = [
        "=" * 70,
        f"SIMPLE METHOD SUMMARY - {method_key}",
        "=" * 70,
        f"Model:   {model_name}",
        f"Dataset: {dataset_name}",
        "",
        "CONTEXT SUMMARY",
        "-" * 40,
        f"N:               {ctx_stats.get('n', 0)}",
        f"Mean delta:      {ctx_stats.get('mean', float('nan')):+.6f}" if ctx_stats else "Mean delta:      N/A",
        f"Std:             {ctx_stats.get('std', float('nan')):.6f}" if ctx_stats else "Std:             N/A",
        f"Male pref %:     {ctx_stats.get('male_pct', 0):.2f}" if ctx_stats else "Male pref %:     N/A",
        f"Female pref %:   {ctx_stats.get('female_pct', 0):.2f}" if ctx_stats else "Female pref %:   N/A",
        f"Tie %:           {ctx_stats.get('tie_pct', 0):.2f}" if ctx_stats else "Tie %:           N/A",
        "",
        "PRONOUN SUMMARY",
        "-" * 40,
        f"N:               {pron_stats.get('n', 0)}",
        f"Mean delta:      {pron_stats.get('mean', float('nan')):+.6f}" if pron_stats else "Mean delta:      N/A",
        f"Std:             {pron_stats.get('std', float('nan')):.6f}" if pron_stats else "Std:             N/A",
        f"Male pref %:     {pron_stats.get('male_pct', 0):.2f}" if pron_stats else "Male pref %:     N/A",
        f"Female pref %:   {pron_stats.get('female_pct', 0):.2f}" if pron_stats else "Female pref %:   N/A",
        f"Tie %:           {pron_stats.get('tie_pct', 0):.2f}" if pron_stats else "Tie %:           N/A",
    ]

    stats_report = compute_method_statistical_significance(results, method_key, min_count=3, alpha=0.05)
    overall = stats_report.get("overall", {})
    if "t_test" in overall:
        lines.extend(
            [
                "",
                "CONTEXT SIGNIFICANCE",
                "-" * 40,
                f"t-statistic:     {overall['t_test'].get('statistic', 0):+.4f}",
                f"p-value:         {overall['t_test'].get('p_value', 1):.2e}",
                f"Significant:     {'YES' if overall['t_test'].get('significant') else 'NO'}",
                f"Cohen's d:       {overall.get('effect_size', {}).get('cohens_d', 0):+.4f}",
            ]
        )
    mc = stats_report.get("summary", {}).get("multiple_comparison", {})
    if mc:
        lines.extend(
            [
                "",
                "PER-CONTEXT SIGNIFICANCE SUMMARY",
                "-" * 40,
                f"Contexts tested:      {mc.get('n_contexts_tested', 0)}",
                f"Uncorrected sig.:     {mc.get('uncorrected_significant', 0)}",
                f"FDR significant:      {mc.get('fdr_significant', 0)}",
                "Per-context details:  context_significance.csv",
            ]
        )
    return lines


def _escape_latex(text: str) -> str:
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def generate_latex_table_single_method(
    results: List[Dict],
    model_name: str,
    dataset_name: str,
    method_key: str,
) -> str:
    df = pd.DataFrame(results)
    suffix = "" if method_key == "original" else f"_{method_key}"
    ctx_col = f"Context_Delta{suffix}"
    pron_col = f"Pronoun_Delta{suffix}"

    ctx = df[ctx_col].dropna().to_numpy() if ctx_col in df.columns else np.array([])
    pron = df[pron_col].dropna().to_numpy() if pron_col in df.columns else np.array([])
    ctx_stats = _get_method_stats(ctx)
    pron_stats = _get_method_stats(pron)

    corr = float("nan")
    agreement = float("nan")
    if ctx_col in df.columns and pron_col in df.columns:
        valid_both = df[[ctx_col, pron_col]].dropna()
        if len(valid_both) > 1:
            try:
                corr, _ = stats.pearsonr(valid_both[ctx_col], valid_both[pron_col])
            except Exception:
                corr = float("nan")
            agreement = float(((valid_both[ctx_col] > 0) == (valid_both[pron_col] > 0)).mean()) * 100.0

    model_escaped = _escape_latex(model_name)
    dataset_escaped = _escape_latex(dataset_name)
    method_escaped = _escape_latex(method_key)
    ctx_m = f"{ctx_stats.get('male_pct', 0):.1f}" if ctx_stats else "N/A"
    ctx_f = f"{ctx_stats.get('female_pct', 0):.1f}" if ctx_stats else "N/A"
    ctx_mean = f"{ctx_stats.get('mean', 0):.3f}" if ctx_stats else "N/A"
    ctx_std = f"{ctx_stats.get('std', 0):.3f}" if ctx_stats else "N/A"
    pron_m = f"{pron_stats.get('male_pct', 0):.1f}" if pron_stats else "N/A"
    pron_f = f"{pron_stats.get('female_pct', 0):.1f}" if pron_stats else "N/A"
    pron_mean = f"{pron_stats.get('mean', 0):.3f}" if pron_stats else "N/A"
    pron_std = f"{pron_stats.get('std', 0):.3f}" if pron_stats else "N/A"
    corr_str = f"{corr:.4f}" if not np.isnan(corr) else "N/A"
    agreement_str = f"{agreement:.1f}" if not np.isnan(agreement) else "N/A"

    return f"""\\begin{{table*}}[t]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccccccc}}
\\toprule
Model &
\\multicolumn{{4}}{{c}}{{Context Masking}} &
\\multicolumn{{4}}{{c}}{{Pronoun Masking}} &
\\multicolumn{{2}}{{c}}{{Bidir}} \\\\
\\cmidrule(lr){{2-5}}\\cmidrule(lr){{6-9}}\\cmidrule(lr){{10-11}}
 & M (\\%) & F (\\%) & mean & sd &
   M (\\%) & F (\\%) & mean & sd &
   Corr ($r$) & Agreement (\\%) \\\\
\\midrule
{method_escaped} &
{ctx_m} & {ctx_f} & {ctx_mean} & {ctx_std} &
{pron_m} & {pron_f} & {pron_mean} & {pron_std} &
{corr_str} & {agreement_str} \\\\
\\bottomrule
\\end{{tabular}}
}}
\\caption{{Simple-score summary for {method_escaped} on \\textsc{{{model_escaped}}} / \\textsc{{{dataset_escaped}}}. Agreement is the percentage of examples where context and pronoun scores prefer the same direction.}}
\\label{{tab:simple-{method_key}-{dataset_name.lower().replace(' ', '-')}}}
\\end{{table*}}
"""


def generate_comparison_summary(results: List[Dict], model_name: str, dataset_name: str) -> List[str]:
    df = pd.DataFrame(results)
    lines = [
        "=" * 70,
        "COMPARISON SUMMARY",
        "=" * 70,
        f"Model:   {model_name}",
        f"Dataset: {dataset_name}",
        "",
        f"{'Method':<18} {'Ctx Mean':>12} {'Ctx SD':>10} {'Ctx M%':>8} {'Pr Mean':>12} {'Pr SD':>10} {'Pr M%':>8}",
        "-" * 84,
    ]
    for method_key in METHODS:
        ctx_col = "Context_Delta" if method_key == "original" else f"Context_Delta_{method_key}"
        pron_col = "Pronoun_Delta" if method_key == "original" else f"Pronoun_Delta_{method_key}"
        ctx = df[ctx_col].dropna().to_numpy() if ctx_col in df.columns else np.array([])
        pron = df[pron_col].dropna().to_numpy() if pron_col in df.columns else np.array([])
        ctx_stats = _get_method_stats(ctx)
        pron_stats = _get_method_stats(pron)
        ctx_mean = ctx_stats.get("mean", float("nan"))
        ctx_std = ctx_stats.get("std", float("nan"))
        ctx_male = ctx_stats.get("male_pct", float("nan"))
        pron_mean = pron_stats.get("mean", float("nan"))
        pron_std = pron_stats.get("std", float("nan"))
        pron_male = pron_stats.get("male_pct", float("nan"))
        lines.append(
            f"{method_key:<18} {ctx_mean:>+12.6f} {ctx_std:>10.6f} {ctx_male:>8.2f} "
            f"{pron_mean:>+12.6f} {pron_std:>10.6f} {pron_male:>8.2f}"
        )

    lines.extend(["", "CONTEXT SIGNIFICANCE", "-" * 40])
    for method_key in METHODS:
        stats_report = compute_method_statistical_significance(results, method_key, min_count=3, alpha=0.05)
        overall = stats_report.get("overall", {})
        mc = stats_report.get("summary", {}).get("multiple_comparison", {})
        if "t_test" in overall:
            lines.append(
                f"{method_key:<18} p={overall['t_test'].get('p_value', 1):.2e} "
                f"d={overall.get('effect_size', {}).get('cohens_d', 0):+.4f}"
            )
            if mc:
                lines.append(
                    f"{'':<18} contexts={mc.get('n_contexts_tested', 0)} "
                    f"fdr_sig={mc.get('fdr_significant', 0)}"
                )
    return lines


def generate_latex_table_comparison(results: List[Dict], model_name: str, dataset_name: str) -> str:
    df = pd.DataFrame(results)
    rows = []
    for method_key in METHODS:
        ctx_col = "Context_Delta" if method_key == "original" else f"Context_Delta_{method_key}"
        pron_col = "Pronoun_Delta" if method_key == "original" else f"Pronoun_Delta_{method_key}"
        ctx = df[ctx_col].dropna().to_numpy() if ctx_col in df.columns else np.array([])
        pron = df[pron_col].dropna().to_numpy() if pron_col in df.columns else np.array([])
        ctx_stats = _get_method_stats(ctx)
        pron_stats = _get_method_stats(pron)
        corr = float("nan")
        agreement = float("nan")
        if ctx_col in df.columns and pron_col in df.columns:
            valid_both = df[[ctx_col, pron_col]].dropna()
            if len(valid_both) > 1:
                try:
                    corr, _ = stats.pearsonr(valid_both[ctx_col], valid_both[pron_col])
                except Exception:
                    corr = float("nan")
                agreement = float(((valid_both[ctx_col] > 0) == (valid_both[pron_col] > 0)).mean()) * 100.0

        ctx_m = f"{ctx_stats.get('male_pct', 0):.1f}" if ctx_stats else "N/A"
        ctx_f = f"{ctx_stats.get('female_pct', 0):.1f}" if ctx_stats else "N/A"
        ctx_mean = f"{ctx_stats.get('mean', 0):.3f}" if ctx_stats else "N/A"
        ctx_std = f"{ctx_stats.get('std', 0):.3f}" if ctx_stats else "N/A"
        pron_m = f"{pron_stats.get('male_pct', 0):.1f}" if pron_stats else "N/A"
        pron_f = f"{pron_stats.get('female_pct', 0):.1f}" if pron_stats else "N/A"
        pron_mean = f"{pron_stats.get('mean', 0):.3f}" if pron_stats else "N/A"
        pron_std = f"{pron_stats.get('std', 0):.3f}" if pron_stats else "N/A"
        corr_str = f"{corr:.4f}" if not np.isnan(corr) else "N/A"
        agreement_str = f"{agreement:.1f}" if not np.isnan(agreement) else "N/A"
        rows.append(
            f"{_escape_latex(method_key)} & {ctx_m} & {ctx_f} & {ctx_mean} & {ctx_std} & "
            f"{pron_m} & {pron_f} & {pron_mean} & {pron_std} & {corr_str} & {agreement_str} \\\\"
        )

    model_escaped = _escape_latex(model_name)
    dataset_escaped = _escape_latex(dataset_name)
    rows_text = "\n".join(rows)
    return f"""\\begin{{table*}}[t]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccccccc}}
\\toprule
Method &
\\multicolumn{{4}}{{c}}{{Context Masking}} &
\\multicolumn{{4}}{{c}}{{Pronoun Masking}} &
\\multicolumn{{2}}{{c}}{{Bidir}} \\\\
\\cmidrule(lr){{2-5}}\\cmidrule(lr){{6-9}}\\cmidrule(lr){{10-11}}
 & M (\\%) & F (\\%) & mean & sd &
   M (\\%) & F (\\%) & mean & sd &
   Corr ($r$) & Agreement (\\%) \\\\
\\midrule
{rows_text}
\\bottomrule
\\end{{tabular}}
}}
\\caption{{Simple-score method comparison for \\textsc{{{model_escaped}}} on \\textsc{{{dataset_escaped}}}. Agreement is the percentage of examples where context and pronoun scores prefer the same direction.}}
\\label{{tab:simple-comparison-{dataset_name.lower().replace(' ', '-')}}}
\\end{{table*}}
"""


def generate_organized_output(
    results: List[Dict],
    model_name: str,
    dataset_name: str,
    output_dir: Path,
    min_count: int = 1,
    alpha: float = 0.05,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method_dirs = {}
    comparison_rows = []
    for method_key in METHODS:
        method_dir = output_dir / method_key
        method_dir.mkdir(parents=True, exist_ok=True)
        generate_method_results_csv(results, method_key).to_csv(method_dir / "results.csv", index=False)
        agg_df = generate_method_aggregates(results, method_key, min_count)
        if not agg_df.empty:
            agg_df.to_csv(method_dir / "context_aggregates.csv", index=False)
        summary_lines = generate_method_summary(results, model_name, dataset_name, method_key)
        (method_dir / "summary.txt").write_text("\n".join(summary_lines))
        (method_dir / "table.tex").write_text(
            generate_latex_table_single_method(results, model_name, dataset_name, method_key)
        )
        stats_report = compute_method_statistical_significance(results, method_key, min_count, alpha)
        save_method_significance_report(stats_report, method_dir)
        comparison_rows.append(
            {
                "method": method_key,
                "context_mean": stats_report.get("overall", {}).get("descriptive", {}).get("mean"),
                "context_n": stats_report.get("overall", {}).get("descriptive", {}).get("n"),
                "context_p_value": stats_report.get("overall", {}).get("t_test", {}).get("p_value"),
                "context_cohens_d": stats_report.get("overall", {}).get("effect_size", {}).get("cohens_d"),
            }
        )
        method_dirs[method_key] = method_dir
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "summary.txt").write_text("\n".join(generate_comparison_summary(results, model_name, dataset_name)))
    (comparison_dir / "table.tex").write_text(
        generate_latex_table_comparison(results, model_name, dataset_name)
    )
    pd.DataFrame(comparison_rows).to_csv(comparison_dir / "all_methods_summary.csv", index=False)
    method_dirs["comparison"] = comparison_dir
    return method_dirs
