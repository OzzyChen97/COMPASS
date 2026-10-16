#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PAIR_MODEL="${SATO_PAIR_MODEL:-gemini-3.1-flash-lite}"
PAIR_BASE_URL="${OPENAI_BASE_URL:-https://di2lab.zeabur.app/v1}"
PAIR_API_KEY_ENV="${PAIR_API_KEY_ENV:-DI2LAB_API_KEY}"
PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-5}"
PAIR_WORKERS="${PAIR_WORKERS:-16}"
PAIR_SEED="${PAIR_SEED:-123}"

if [[ -z "${!PAIR_API_KEY_ENV:-}" ]]; then
  echo "Missing API key in environment variable ${PAIR_API_KEY_ENV}" >&2
  exit 1
fi

generate_dataset() {
  local dataset_root="$1"
  local output_prefix="$2"
  local combined_output="${output_prefix}.json"

  "${PYTHON_BIN}" tools/build_sato_pairs.py \
    --dataset-root "${dataset_root}" \
    --split test \
    --protocol test \
    --output "${combined_output}" \
    --model "${PAIR_MODEL}" \
    --base-url "${PAIR_BASE_URL}" \
    --api-key-env "${PAIR_API_KEY_ENV}" \
    --preserve-prompt-style humanml3d \
    --batch-size "${PAIR_BATCH_SIZE}" \
    --workers "${PAIR_WORKERS}" \
    --seed "${PAIR_SEED}"

  "${PYTHON_BIN}" tools/finalize_sato_test_levels.py \
    --progress "${combined_output}.progress.jsonl" \
    --output-prefix "${output_prefix}" \
    --seed "${PAIR_SEED}"
}

generate_dataset \
  dataset/KIT-ML \
  dataset/KIT-ML/test_perb_gemini31_flash_lite_human_recipe

generate_dataset \
  dataset/Motion-X-322 \
  dataset/Motion-X-322/test_perb_gemini31_flash_lite_human_recipe
