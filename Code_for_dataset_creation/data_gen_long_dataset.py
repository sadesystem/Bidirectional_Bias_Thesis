#!/usr/bin/env python3
"""
Lengthen sentences in Dataset_long_unmasked.csv using GPT-5.

This script reads the sent_less column, sends batches of 10 sentences to GPT-5
to be lengthened to 11-14 words, validates the results with a separate API call,
and writes the accepted sentences back as new columns (sent_less_long, sent_more_long).
"""

import json
import re
import pandas as pd
from openai import BadRequestError, OpenAI

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

API_KEY = (
    "YOUR_OPENAI_KEY"
)

MODEL = "gpt-5"
DATASET_PATH = "Dataset_long_unmasked.csv"
BATCH_SIZE = 10          # process 10 sentences at a time
TARGET_MIN_WORDS = 11    # minimum word count (inclusive)
TARGET_MAX_WORDS = 14    # maximum word count (inclusive)
MAX_RETRIES = 3          # max regeneration attempts for a failed sentence
TEMPERATURE = 0.7

# Characters that are forbidden in the output (except commas are allowed)
FORBIDDEN_CHARS = re.compile(r"['\u2019\u2018`\u00b4\-]")

# ──────────────────────────────────────────────────────────────────────
# System / user prompts for GENERATION
# ──────────────────────────────────────────────────────────────────────

GENERATION_SYSTEM_PROMPT = (
    "You are a sentence editor. You will receive sentences that end with the "
    "word 'him'. Your job is to make each sentence longer so that it contains "
    "between 11 and 14 words (inclusive). Follow these rules strictly:\n"
    "1. Keep the original sentence structure and word order intact. Do NOT "
    "rearrange any existing words.\n"
    "2. Only ADD meaningful words or short phrases that provide additional "
    "context. Every original word must remain in the sentence, unchanged.\n"
    "3. The sentence must still end with 'him.' as the last word.\n"
    "4. The first two words of the original sentence MUST" 
    "remain together as the first two words of the lengthened sentence. Do NOT "
    "insert any words before or between them.\n"
    "5. Do NOT use special characters such as hyphens (-), apostrophes ('), "
    "backticks (`), or acute accents. Commas are allowed.\n"
    "6. The output must be a single sentence. Do NOT split it into multiple "
    "sentences or add extra periods, exclamation marks, or question marks "
    "in the middle.\n"
    "7. Use neutral, professional language.\n"
    "8. Return a JSON object with a single key 'sentences' whose value is a "
    "list of strings, one per input sentence, in the same order.\n"
)

# ──────────────────────────────────────────────────────────────────────
# System / user prompts for VALIDATION
# ──────────────────────────────────────────────────────────────────────

VALIDATION_SYSTEM_PROMPT = (
    "You are a strict sentence validator. You will receive pairs of "
    "(original_sentence, lengthened_sentence). For each pair, check ALL of the "
    "following rules and return true only if every rule passes:\n"
    "1. The lengthened sentence has between 11 and 14 words (inclusive).\n"
    "2. Every word from the original sentence appears in the lengthened "
    "sentence in the same relative order.\n"
    "3. The lengthened sentence ends with 'him.' as the final word.\n"
    "4. The first two words of the original sentence remain the first two "
    "words of the lengthened sentence (nothing inserted before or between them).\n"
    "5. There are no special characters such as hyphens (-), apostrophes ('), "
    "backticks (`), or acute accents.\n"
    "6. There is only one sentence (no extra periods, exclamation marks, or "
    "question marks in the middle).\n"
    "7. The added words are meaningful and make grammatical sense.\n"
    "Return a JSON object with a single key 'results' whose value is a list "
    "of booleans, one per pair, in the same order.\n"
)

# ──────────────────────────────────────────────────────────────────────
# JSON schemas for structured output
# ──────────────────────────────────────────────────────────────────────

GENERATION_SCHEMA = {
    "type": "json_schema",
    "name": "lengthened_sentences",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "sentences": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["sentences"],
    },
}

VALIDATION_SCHEMA = {
    "type": "json_schema",
    "name": "validation_results",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "items": {"type": "boolean"},
            }
        },
        "required": ["results"],
    },
}


# ──────────────────────────────────────────────────────────────────────
# Helper: local (rule-based) validation
# ──────────────────────────────────────────────────────────────────────

def local_validate(original: str, lengthened: str) -> bool:
    """
    Run fast, deterministic checks on a lengthened sentence.
    Returns True only if ALL checks pass.
    """
    # Strip whitespace
    lengthened = lengthened.strip()

    # 1. Word count must be 11-14
    words = lengthened.split()
    if not (TARGET_MIN_WORDS <= len(words) <= TARGET_MAX_WORDS):
        return False

    # 2. Must end with "him." (last word is "him" with period)
    if not lengthened.endswith("him."):
        return False

    # 3. First two words must match the original (e.g. "The mover")
    orig_words_raw = original.strip().split()
    if len(words) < 2 or len(orig_words_raw) < 2:
        return False
    if words[0].lower() != orig_words_raw[0].lower() or words[1].lower() != orig_words_raw[1].lower():
        return False

    # 4. No forbidden special characters
    if FORBIDDEN_CHARS.search(lengthened):
        return False

    # 5. No mid-sentence terminators (only one sentence)
    #    Remove trailing period, then check no periods/!/? remain
    inner = lengthened[:-1]  # strip the final period
    if re.search(r"[.!?]", inner):
        return False

    # 6. All original words appear in order
    orig_words = original.strip().rstrip(".").split()
    leng_words = lengthened.rstrip(".").split()
    oi = 0
    for w in leng_words:
        if oi < len(orig_words) and w.lower() == orig_words[oi].lower():
            oi += 1
    if oi != len(orig_words):
        return False

    return True


# ──────────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────────

def call_generate(client: OpenAI, sentences: list[str]) -> list[str]:
    """
    Send a batch of sentences to GPT-5 for lengthening.
    Returns a list of lengthened sentences.
    """
    # Build user prompt listing the sentences
    numbered = "\n".join(
        f"{i+1}. {s}" for i, s in enumerate(sentences)
    )
    user_prompt = (
        f"Lengthen each of the following {len(sentences)} sentences to "
        f"11-14 words. Return them in the same order.\n\n{numbered}"
    )

    request_kwargs = {
        "model": MODEL,
        "instructions": GENERATION_SYSTEM_PROMPT,
        "input": user_prompt,
        "text": {"format": GENERATION_SCHEMA},
        "temperature": TEMPERATURE,
    }

    try:
        resp = client.responses.create(**request_kwargs)
    except BadRequestError as exc:
        if "temperature" in str(exc).lower():
            # Model does not support explicit temperature; retry without it
            print("  Model rejected temperature parameter; retrying without it.")
            request_kwargs.pop("temperature", None)
            resp = client.responses.create(**request_kwargs)
        else:
            raise

    data = json.loads(resp.output_text)
    return data["sentences"]


def call_validate(
    client: OpenAI,
    originals: list[str],
    lengthened: list[str],
) -> list[bool]:
    """
    Ask GPT-5 to validate each (original, lengthened) pair.
    Returns a list of booleans (True = valid).
    """
    pairs = "\n".join(
        f"{i+1}. Original: {o}\n   Lengthened: {l}"
        for i, (o, l) in enumerate(zip(originals, lengthened))
    )
    user_prompt = (
        f"Validate each of the following {len(originals)} pairs.\n\n{pairs}"
    )

    request_kwargs = {
        "model": MODEL,
        "instructions": VALIDATION_SYSTEM_PROMPT,
        "input": user_prompt,
        "text": {"format": VALIDATION_SCHEMA},
        "temperature": 0,  # deterministic validation
    }

    try:
        resp = client.responses.create(**request_kwargs)
    except BadRequestError as exc:
        if "temperature" in str(exc).lower():
            # Model does not support explicit temperature; retry without it
            print("  Model rejected temperature parameter; retrying without it.")
            request_kwargs.pop("temperature", None)
            resp = client.responses.create(**request_kwargs)
        else:
            raise

    data = json.loads(resp.output_text)
    return data["results"]


# ──────────────────────────────────────────────────────────────────────
# Main processing loop
# ──────────────────────────────────────────────────────────────────────

def process_dataset():
    """
    Main entry point.
    1. Reads the CSV.
    2. Processes sent_less in batches of 10.
    3. Validates each batch (local + GPT-5 validation).
    4. Retries failures up to MAX_RETRIES times.
    5. Derives sent_more_long by replacing trailing 'him.' with 'her.'.
    6. Writes the new columns back to the CSV.
    """

    # ── Load dataset ─────────────────────────────────────────────────
    df = pd.read_csv(DATASET_PATH)
    total_rows = len(df)
    print(f"Loaded {total_rows} rows from {DATASET_PATH}")

    # Initialise new columns with empty strings so partial progress is visible
    if "sent_less_long" not in df.columns:
        df["sent_less_long"] = ""
    if "sent_more_long" not in df.columns:
        df["sent_more_long"] = ""

    # ── OpenAI client ────────────────────────────────────────────────
    client = OpenAI(api_key=API_KEY)

    # ── Identify rows that still need processing ─────────────────────
    # (rows where sent_less_long is empty)
    pending_mask = df["sent_less_long"].astype(str).isin(["", "nan"])
    pending_indices = df.index[pending_mask].tolist()
    print(f"Rows to process: {len(pending_indices)}")

    # ── Process in batches of BATCH_SIZE ─────────────────────────────
    batch_num = 0
    i = 0
    while i < len(pending_indices):
        batch_indices = pending_indices[i : i + BATCH_SIZE]
        batch_sentences = df.loc[batch_indices, "sent_less"].tolist()
        batch_num += 1

        print(f"\n── Batch {batch_num} ({len(batch_indices)} sentences, "
              f"rows {batch_indices[0]}-{batch_indices[-1]}) ──")

        # Track which sentences within this batch still need work
        # Key: position in batch_indices, Value: original sentence
        remaining = {pos: sent for pos, sent in enumerate(batch_sentences)}
        results = [""] * len(batch_indices)  # final accepted sentences

        attempt = 0
        while remaining and attempt < MAX_RETRIES:
            attempt += 1
            retry_positions = sorted(remaining.keys())
            retry_sentences = [remaining[p] for p in retry_positions]

            print(f"  Attempt {attempt}: generating {len(retry_sentences)} sentence(s)...")

            # ── Step 1: Generate lengthened sentences ────────────────
            try:
                generated = call_generate(client, retry_sentences)
            except Exception as e:
                print(f"  ERROR in generation API call: {e}")
                continue  # try again on next attempt

            # Safety: if API returns fewer sentences than expected, pad
            while len(generated) < len(retry_sentences):
                generated.append("")

            # ── Step 2: Local (rule-based) validation ────────────────
            local_pass = [
                local_validate(orig, gen)
                for orig, gen in zip(retry_sentences, generated)
            ]
            print(f"  Local validation: {sum(local_pass)}/{len(local_pass)} passed")

            # Collect locally-passed sentences for GPT validation
            gpt_check_positions = [
                p for p, ok in zip(retry_positions, local_pass) if ok
            ]
            gpt_check_originals = [
                remaining[p] for p in gpt_check_positions
            ]
            gpt_check_generated = [
                generated[retry_positions.index(p)]
                for p in gpt_check_positions
            ]

            # ── Step 3: GPT-5 validation (separate call) ────────────
            gpt_pass = []
            if gpt_check_positions:
                try:
                    gpt_pass = call_validate(
                        client, gpt_check_originals, gpt_check_generated
                    )
                except Exception as e:
                    print(f"  ERROR in validation API call: {e}")
                    # Treat all as failed if validation call fails
                    gpt_pass = [False] * len(gpt_check_positions)

                # Pad if the model returns fewer results
                while len(gpt_pass) < len(gpt_check_positions):
                    gpt_pass.append(False)

                print(f"  GPT validation: {sum(gpt_pass)}/{len(gpt_pass)} passed")

            # ── Step 4: Accept valid, keep invalid for retry ─────────
            accepted_count = 0
            # Mark locally-failed sentences as still remaining (already there)
            # Mark GPT-failed sentences as still remaining
            gpt_idx = 0
            for p, lok in zip(retry_positions, local_pass):
                if lok:
                    # This went through GPT validation
                    if gpt_pass[gpt_idx]:
                        # Accepted
                        results[p] = generated[retry_positions.index(p)]
                        del remaining[p]
                        accepted_count += 1
                    gpt_idx += 1
                # If local failed, sentence stays in remaining

            print(f"  Accepted: {accepted_count}, "
                  f"remaining for retry: {len(remaining)}")

        # ── Handle sentences that failed all retries ─────────────────
        if remaining:
            print(f"  WARNING: {len(remaining)} sentence(s) could not be "
                  f"lengthened after {MAX_RETRIES} attempts. "
                  f"Leaving original columns unchanged for those rows.")

        # ── Write accepted results into the dataframe ────────────────
        for pos, sent_long in enumerate(results):
            if sent_long:
                row_idx = batch_indices[pos]
                df.at[row_idx, "sent_less_long"] = sent_long
                # Derive sent_more_long by replacing final "him." with "her."
                sent_more_long = re.sub(r"\bhim\.$", "her.", sent_long)
                df.at[row_idx, "sent_more_long"] = sent_more_long

        # ── Save after every batch so progress is not lost ───────────
        df.to_csv(DATASET_PATH, index=False)
        print(f"  Saved progress to {DATASET_PATH}")

        i += BATCH_SIZE

    # ── Final summary ────────────────────────────────────────────────
    filled = (df["sent_less_long"].astype(str) != "") & (
        df["sent_less_long"].astype(str) != "nan"
    )
    print(f"\n{'='*60}")
    print(f"Done. {filled.sum()}/{total_rows} rows have lengthened sentences.")
    print(f"Output saved to {DATASET_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_dataset()

