#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU="${1:?usage: $0 GPU START_SEED NUM_REPEATS [OUT_DIR]}"
START_SEED="${2:?missing start seed}"
NUM_REPEATS="${3:?missing repeat count}"
OUT_DIR="${4:-MotionGPT/experiments/robustness/humanml3d_repaired_sato_pp_p123/seed_${START_SEED}}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
ASSET_ROOT="${ASSET_ROOT:-MotionGPT/experiments/robustness/humanml3d_caption_locked_sato_mix_repair_all_p123_4631_20260810}"
SAMPLE_INDEX="${SAMPLE_INDEX:-${ASSET_ROOT}/sample_index_seed123.json}"
if [[ -z "${PAIR_FILE:-}" ]]; then
  pair_candidates=("${ASSET_ROOT}"/pairs*.json)
  if [[ "${#pair_candidates[@]}" -ne 1 || ! -f "${pair_candidates[0]}" ]]; then
    echo "expected exactly one pairs*.json under ${ASSET_ROOT}" >&2
    exit 1
  fi
  PAIR_FILE="${pair_candidates[0]}"
fi

for path in "${PAIR_FILE}" "${SAMPLE_INDEX}"; do
  [[ -f "${path}" ]] || { echo "missing required asset: ${path}" >&2; exit 1; }
done

mkdir -p "${OUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" \
  ablation/repeat20_perturb_fast_batch.py \
  --pair-data-path "${PAIR_FILE}" \
  --expected-sample-index "${SAMPLE_INDEX}" \
  --out-dir "${OUT_DIR}" \
  --report-md "${OUT_DIR}/README.md" \
  --start-seed "${START_SEED}" \
  --num-repeats "${NUM_REPEATS}" \
  --batch-size 32 \
  --caption-locked \
  --caption-locked-length-policy repair-all \
  --fast-batch \
  --sampling-pair-rng-mode sequential \
  --model sato_pp_allloss_l1_iter130k \
  --cfg-scale 1.0 \
  --cfg-uncond zero \
  --sample-mode categorical \
  --sample-temperature 1.0 \
  --top-k 0 \
  --top-p 1.0 \
  --wandb-mode disabled
