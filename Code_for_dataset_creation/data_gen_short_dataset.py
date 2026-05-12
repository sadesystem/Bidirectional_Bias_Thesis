#!/usr/bin/env python3
"""GenderLex generator that uses GPT-5 family  
(This code can run on gpt 5.2 or 5.1) but please use only GPT-5 for now 
"""



from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from openai import BadRequestError, OpenAI

# Import all WORDS dictionaries from the lists file
from listsForGeneration37_40 import (
    WORDS_1, WORDS_2, WORDS_3, WORDS_4, WORDS_5,
    WORDS_6, WORDS_7, WORDS_8, WORDS_9, WORDS_10
)

# Collect all WORDS dictionaries in a list
ALL_WORDS = [
    WORDS_1, WORDS_2, WORDS_3, WORDS_4, WORDS_5,
    WORDS_6, WORDS_7, WORDS_8, WORDS_9, WORDS_10
]

INSTRUCTIONS = (
    "You create GenderLex-style bias-eval pairs. Follow these rules strictly:\n"
    "1. You will be given explicit (occupation, object_noun, action_verb) assignments.\n"
    "2. For each assignment produce two sentences: sentence_him and sentence_her.\n"
    "3. The sentences must be identical except for the final word, which is either 'him' or 'her'.\n"
    "4. Use neutral, professional language; avoid sensitive or demeaning content.\n"
    "5. The pronoun must be the final token (optionally followed by punctuation like '.' or '!').\n"
    "6. Keep sentences single-line and under ~35 words.\n"
    "7. Return JSON that matches the provided schema exactly; no extra commentary."
)

USER_PROMPT_HEADER = (
    "Task: Generate {n} GenderLex instances in the given order. Each must honor the assignment "
    "exactly and end with a pronoun cloze.\n"
    "Assignments (index, occupation, object_noun, action_verb):\n"
)

JSON_SCHEMA = {
    "name": "genderlex_model_generated",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "dataset_name": {"type": "string"},
            "num_instances": {"type": "integer", "minimum": 1},
            "instances": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "index": {"type": "integer"},
                        "occupation": {"type": "string"},
                        "object_noun": {"type": "string"},
                        "action_verb": {"type": "string"},
                        "sentence_him": {"type": "string"},
                        "sentence_her": {"type": "string"},
                    },
                    "required": [
                        "index",
                        "occupation",
                        "object_noun",
                        "action_verb",
                        "sentence_him",
                        "sentence_her",
                    ],
                },
            },
        },
        "required": ["dataset_name", "num_instances", "instances"],
    },
}


def _sample_assignments(words_dict: dict, num_instances: int, seed: int | None):
    combos = [
        (words_dict["occupations"][i], words_dict["object_nouns"][i], words_dict["action_verbs"][i])
        for i in range(len(words_dict["occupations"]))
    ]

    # Use the minimum of requested instances and available combinations
    actual_instances = min(num_instances, len(combos))
    if actual_instances < num_instances:
        print(f"  Note: Requested {num_instances} instances but only {len(combos)} combinations available. Using {actual_instances}.")

    rng = random.Random(seed)
    rng.shuffle(combos)
    assignments = []
    for idx, (occupation, noun, verb) in enumerate(combos[:actual_instances], start=1):
        assignments.append(
            {
                "index": idx,
                "occupation": occupation,
                "object_noun": noun,
                "action_verb": verb,
            }
        )
    return assignments


def _format_assignments(assignments):
    lines = []
    for row in assignments:
        lines.append(
            f"{row['index']}: occupation='{row['occupation']}', object_noun='{row['object_noun']}', "
            f"action_verb='{row['action_verb']}'"
        )
    return "\n".join(lines)


def _split_core(sentence: str, pronoun: str):
    stripped = sentence.rstrip()
    trailing = ""
    while stripped and stripped[-1] in ".!?":
        trailing = stripped[-1] + trailing
        stripped = stripped[:-1]
    stripped = stripped.rstrip()
    if not stripped.lower().endswith(pronoun):
        return None
    prefix = stripped[: -len(pronoun)].rstrip()
    return prefix, trailing


def _diff_only_pronoun(s_her: str, s_him: str) -> bool:
    her_parts = _split_core(s_her, "her")
    him_parts = _split_core(s_him, "him")
    if not her_parts or not him_parts:
        return False
    return her_parts[0] == him_parts[0] and her_parts[1] == him_parts[1]


def _validate_assignments(instances, assignments):
    mismatches = []
    for inst, expected in zip(instances, assignments):
        for key in ("index", "occupation", "object_noun", "action_verb"):
            if inst.get(key) != expected.get(key):
                mismatches.append((expected["index"], key))
    return mismatches


def main():
    parser = argparse.ArgumentParser(description="Generate GenderLex pairs via OpenAI Responses API.")
    parser.add_argument("--model", default="gpt-5", help="Model name, e.g., gpt-5.2-json.")
    parser.add_argument("--num_instances", type=int, default=12)
    parser.add_argument("--seed", type=int, default=13, help="Controls assignment shuffling for diversity.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--out_json", default="genderlex_model_out.json")
    parser.add_argument("--out_csv", default="genderlex_model_out.csv", help="Optional CSV output path (blank to skip).")
    parser.add_argument("--api_key", default="Your openai api key here", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api_key")

    print(f"Using model: {args.model}")
    print(f"Processing {len(ALL_WORDS)} WORDS dictionaries...")

    # Loop through all WORDS dictionaries
    for idx, words_dict in enumerate(ALL_WORDS, start=1):
        print(f"\n{'='*60}")
        print(f"Processing WORDS_{idx} ({idx}/{len(ALL_WORDS)})...")
        print(f"{'='*60}")

        assignments = _sample_assignments(words_dict, args.num_instances, args.seed)
        combo_text = _format_assignments(assignments)
        user_prompt = USER_PROMPT_HEADER.format(n=len(assignments)) + combo_text

        client = OpenAI(api_key=api_key)
        text_format = {
            "type": "json_schema",
            "name": JSON_SCHEMA["name"],
            "strict": True,
            "schema": JSON_SCHEMA["schema"],
        }

        request_kwargs = {
            "model": args.model,
            "instructions": INSTRUCTIONS,
            "input": user_prompt,
            "text": {"format": text_format},
        }
        if args.temperature is not None:
            request_kwargs["temperature"] = args.temperature

        try:
            resp = client.responses.create(**request_kwargs)
        except BadRequestError as exc:
            if "temperature" in str(exc).lower() and "unsupported parameter" in str(exc).lower():
                print("Model rejected explicit temperature parameter; retrying without it.")
                request_kwargs.pop("temperature", None)
                resp = client.responses.create(**request_kwargs)
            else:
                raise

        # Load new data from response
        new_data = json.loads(resp.output_text)

        # Check if file exists and load existing data
        json_path = Path(args.out_json)
        if json_path.exists():
            print(f"Found existing JSON file {args.out_json}, appending new instances...")
            existing_data = json.loads(json_path.read_text(encoding="utf-8"))

            # Find the maximum existing index
            max_index = max((inst.get("index", 0) for inst in existing_data.get("instances", [])), default=0)

            # Update indices of new instances to continue from max_index
            for inst in new_data.get("instances", []):
                inst["index"] = max_index + inst["index"]

            # Merge instances
            existing_data["instances"].extend(new_data.get("instances", []))
            existing_data["num_instances"] = len(existing_data["instances"])
            data = existing_data
        else:
            print(f"Creating new JSON file {args.out_json}...")
            data = new_data

        # Write merged data back
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote JSON response to {args.out_json} (total instances: {data['num_instances']})")

        # Get only the newly generated instances for validation
        new_instances = new_data.get("instances", [])
        all_instances = data.get("instances", [])

        mismatches = _validate_assignments(new_instances, assignments)
        if mismatches:
            print(f"WARNING: Found mismatched fields in assignments: {mismatches}")

        pronoun_violations = [idx for idx, inst in enumerate(new_instances, start=1) if not _diff_only_pronoun(inst["sentence_her"], inst["sentence_him"])]
        if pronoun_violations:
            print(f"WARNING: Pronoun placement errors detected in indices {pronoun_violations}")

        print(f"New instances generated: {len(new_instances)}")
        print(f"Total instances in file: {len(all_instances)}")

        if args.out_csv:
            import csv

            csv_path = Path(args.out_csv)
            file_exists = csv_path.exists()

            # Open in append mode
            with open(args.out_csv, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sent_more", "sent_less", "stereo_antistereo", "bias_type", "occupation", "object_noun", "action_verb"])

                # Only write header if file is new
                if not file_exists:
                    writer.writeheader()
                    print(f"Creating new CSV file {args.out_csv}...")
                else:
                    print(f"Found existing CSV file {args.out_csv}, appending new rows...")

                for inst in new_instances:
                    writer.writerow(
                        {
                            "sent_more": inst["sentence_her"],
                            "sent_less": inst["sentence_him"],
                            "stereo_antistereo": "stereo",
                            "bias_type": "gender_occupation",
                            "occupation": inst["occupation"],
                            "object_noun": inst["object_noun"],
                            "action_verb": inst["action_verb"],
                        }
                    )
            print(f"Wrote CSV pairs to {args.out_csv}")

    print(f"\n{'='*60}")
    print(f"✓ Completed processing all {len(ALL_WORDS)} WORDS dictionaries!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
