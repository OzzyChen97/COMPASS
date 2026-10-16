#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
SATO_DIR="ablation/motionx322_sato/max_token75_humanml3d_recipe_150k_bs132_seed123"
BEST_CKPT="${MOTIONX322_SATO_BEST_CKPT:-${SATO_DIR}/sato_pp_best_fid.pth}"
LAST_CKPT="${MOTIONX322_SATO_LAST_CKPT:-${SATO_DIR}/sato_pp_last.pth}"
PAIR_PREFIX="${MOTIONX322_PAIR_PREFIX:-dataset/Motion-X-322/test_perb_gemini31}"
P2P3_PAIR="${MOTIONX322_P2P3_PAIR:-dataset/Motion-X-322/test_perb_gemini31_p2p3.json}"
MIXED_PAIR="${MOTIONX322_MIXED_PAIR:-${PAIR_PREFIX}_mixed_full.json}"
OUT_ROOT="${MOTIONX322_LEVEL_MIX_EVAL_ROOT:-ablation/motionx322_sato/best10k_last110k_level_mix_repeat20}"
SPEC_PATH="${OUT_ROOT}/best10k_last110k_spec.json"

for path in "$BEST_CKPT" "$LAST_CKPT" "$MIXED_PAIR"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

mkdir -p "$OUT_ROOT"
date -Is > "${OUT_ROOT}/started_at.txt"

"$PYTHON" tools/build_sato_level_mix.py \
  --p2 "${PAIR_PREFIX}_p2.json" \
  --p3 "${PAIR_PREFIX}_p3.json" \
  --output "$P2P3_PAIR" \
  --seed 123

"$PYTHON" tools/write_sato_compare_spec.py \
  --model "motionx322_sato_best_fid_10k=${BEST_CKPT}" \
  --model "motionx322_sato_last_110k=${LAST_CKPT}" \
  --output "$SPEC_PATH"

levels=(p2p3_balanced p123_human_ratio)
pair_files=("$P2P3_PAIR" "$MIXED_PAIR")

for index in 0 1; do
  level="${levels[$index]}"
  pair_file="${pair_files[$index]}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" \
    ablation/repeat20_perturb_fast_batch.py \
    --spec-json "$SPEC_PATH" \
    --pair-data-path "$pair_file" \
    --out-dir "${OUT_ROOT}/${level}" \
    --report-md "${OUT_ROOT}/${level}.md" \
    --start-seed 123 \
    --num-repeats 20 \
    --batch-size 32 \
    --caption-locked \
    --fast-batch \
    --cfg-scale 1.3 \
    --sample-mode categorical \
    --wandb-mode disabled \
    > "${OUT_ROOT}/${level}_driver.log" 2>&1
done

date -Is > "${OUT_ROOT}/complete.txt"
