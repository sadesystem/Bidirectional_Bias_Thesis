#!/usr/bin/env python
"""
Simple masked-LM bias evaluation.

context/pronoun scoring methods:
- original
- within_word_l2r
"""

import argparse
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent

from bert_pil_utils_simple import (
    _load_masked_lm_with_validation,
    _load_tokenizer,
    generate_organized_output,
    get_model_lower_case,
    process_single_example,
    resolve_torch_device,
    setup_logging,
    setup_output_dirs,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run simple BERT gender-bias evaluation with only original and within_word_l2r"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to CSV dataset with sent_m, sent_w, context columns",
    )
    parser.add_argument(
        "--model",
        default="bert-base-uncased",
        help="HuggingFace masked-LM checkpoint",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "results"),
        help="Output directory root",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=3,
        help="Minimum examples per context for aggregation",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    log = setup_logging(args.debug)
    log.info("Loading dataset: %s", args.dataset)
    df = pd.read_csv(args.dataset)
    required_cols = {"sent_m", "sent_w", "context"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {', '.join(sorted(missing_cols))}")

    dataset_name = Path(args.dataset).stem
    dirs = setup_output_dirs(args.output, dataset_name, args.model)
    log.info("Output directory: %s", dirs["root"])

    tokenizer = _load_tokenizer(args.model)
    model = _load_masked_lm_with_validation(args.model)
    device = resolve_torch_device(args.device)
    model.to(device)
    model.eval()
    lower = get_model_lower_case(tokenizer)
    log.info("Using device: %s", device)

    results = []
    total = len(df)
    progress = tqdm(total=total, desc="Examples", unit="ex") if tqdm is not None else None
    for idx, row in df.iterrows():
        result = process_single_example(
            model=model,
            tokenizer=tokenizer,
            sent_male=str(row["sent_m"]),
            sent_female=str(row["sent_w"]),
            context=str(row["context"]),
            lower=lower,
            device=device,
        )
        if "HB" in row and pd.notna(row["HB"]):
            result["HB"] = str(row["HB"])
        result["idx"] = idx
        results.append(result)
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()

    saved = generate_organized_output(
        results=results,
        model_name=args.model,
        dataset_name=dataset_name,
        output_dir=dirs["root"],
        min_count=args.min_count,
        alpha=args.alpha,
    )
    log.info("Saved simple method outputs under %s", dirs["root"])
    for key, value in saved.items():
        log.info("  %s -> %s", key, value)


if __name__ == "__main__":
    main()
