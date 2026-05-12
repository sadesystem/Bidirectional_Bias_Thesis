#!/usr/bin/env bash

set -u

MODELS=(
  "bert-large-uncased"
  "roberta-large"
  "facebook/bart-large"
  "answerdotai/ModernBERT-large"
  "distilbert/distilbert-base-uncased"
  "distilbert/distilroberta-base"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/dataset_04_12_2026}"

mapfile -t DATASETS < <(find "$DATASET_ROOT" -maxdepth 2 -type f -name '*_cleaned.csv' | sort)

run_count=0
success_count=0
fail_count=0
skip_count=0

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "[error] no datasets found under: $DATASET_ROOT"
  exit 1
fi

for dataset in "${DATASETS[@]}"; do
  dataset_path="$dataset"
  dataset_label="${dataset_path#$SCRIPT_DIR/}"

  if [[ ! -f "$dataset_path" ]]; then
    echo "[skip] dataset not found: $dataset_label"
    skip_count=$((skip_count + 1))
    continue
  fi

  for model in "${MODELS[@]}"; do
    run_count=$((run_count + 1))
    echo "[run ] dataset=$dataset_label model=$model"

    if "$PYTHON_BIN" "$SCRIPT_DIR/main_simple.py" \
      --dataset "$dataset_path" \
      --model "$model" \
      --output "$OUTPUT_DIR"; then
      echo "[ ok ] dataset=$dataset_label model=$model"
      success_count=$((success_count + 1))
    else
      echo "[fail] dataset=$dataset_label model=$model"
      fail_count=$((fail_count + 1))
    fi
  done
done

echo
echo "Finished."
echo "Runs attempted: $run_count"
echo "Succeeded:      $success_count"
echo "Failed:         $fail_count"
echo "Datasets skipped: $skip_count"
