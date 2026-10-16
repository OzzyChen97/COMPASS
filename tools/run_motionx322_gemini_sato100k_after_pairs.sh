#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/workspace/Wilson/T2M-GPT
PYTHON=/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python
DATA_ROOT="$ROOT/dataset/Motion-X-322"
PAIR_RAW="$DATA_ROOT/preserving_pairs_gemini31_human_recipe.jsonl"
PAIR_FILTERED="$DATA_ROOT/preserving_pairs_gemini31_human_recipe_filtered.jsonl"
PAIR_LOG="$DATA_ROOT/build_train_pairs_gemini31_human_recipe.log"
PIPELINE_LOG="$ROOT/ablation/motionx322_sato/gemini31_human_recipe_100k_pipeline.log"
mkdir -p "$(dirname "$PIPELINE_LOG")"

log_stage() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" >> "$PIPELINE_LOG"
}

log_stage "waiting for Motion-X Gemini pair generation"
while tmux -L motionxgemini has-session -t generate 2>/dev/null; do
  sleep 60
done

cd "$ROOT"

log_stage "starting HumanML3D-recipe LLM judge filtering with gemini-3.1-flash-lite"
"$PYTHON" tools/build_motionx322_pairs_human_recipe.py \
  --dataset-root "$DATA_ROOT" \
  --output-raw "$PAIR_RAW" \
  --output-filtered "$PAIR_FILTERED" \
  --model gemini-3.1-flash-lite \
  --base-url https://di2lab.zeabur.app/v1 \
  --api-key-env DI2LAB_API_KEY \
  --batch-size 1 --workers 16 --seed 42 \
  --filter-batch-size 10 --filter-workers 4 --filter-max-retries 10 \
  --filter-only \
  >> "$PAIR_LOG" 2>&1

log_stage "validating filtered Motion-X training pairs"
"$PYTHON" tools/validate_motionx322_dataset.py \
  --dataset-root "$DATA_ROOT" --train-pairs "$PAIR_FILTERED" \
  --train-raw "$PAIR_RAW" \
  --train-decisions "$PAIR_FILTERED.decisions.jsonl" \
  >> "$PIPELINE_LOG" 2>&1

log_stage "Motion-X Gemini training-pair dataset build and validation finished; training not started"
