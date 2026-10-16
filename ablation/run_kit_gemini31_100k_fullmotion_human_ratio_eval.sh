#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
REPEATS="${KIT_EVAL_REPEATS:-20}"
CFG_SCALE="${KIT_EVAL_CFG_SCALE:-1.3}"

PAIR_FILE="dataset/KIT-ML/test_perb_gemini31_fullmotion_mixed_full.json"
SPEC="ablation/kit_t2mgpt_sato/gemini31_fullpairs_100k_levels_spec.json"
OUT_ROOT="${KIT_EVAL_OUT_ROOT:-ablation/kit_t2mgpt_sato/gemini31_100k_fullmotion_human_ratio_cfg1p3_repeat20}"
REPORT_STEM="ablation/kit_t2mgpt_sato/KIT_GEMINI31_100K_FULLMOTION_HUMAN_RATIO_REPEAT20"

for path in "$PAIR_FILE" "$SPEC"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

if [[ -e "${OUT_ROOT}/complete.txt" ]]; then
  echo "Refusing to overwrite completed evaluation: $OUT_ROOT" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
printf '%s\n' "$$" > "${OUT_ROOT}/pipeline.pid"
date -Is > "${OUT_ROOT}/started_at.txt"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" \
  ablation/repeat20_perturb_fast_batch.py \
  --spec-json "$SPEC" \
  --pair-data-path "$PAIR_FILE" \
  --out-dir "$OUT_ROOT" \
  --report-md "${OUT_ROOT}.md" \
  --start-seed 123 \
  --num-repeats "$REPEATS" \
  --batch-size 32 \
  --caption-locked \
  --fast-batch \
  --cfg-scale "$CFG_SCALE" \
  --cfg-uncond zero \
  --sample-mode categorical \
  --wandb-mode disabled \
  > "${OUT_ROOT}_driver.log" 2>&1

"$PYTHON" tools/summarize_kit_ablation_levels.py \
  --input "MIXED=${OUT_ROOT}/summary.json" \
  --output-json "${REPORT_STEM}.json" \
  --output-csv "${REPORT_STEM}.csv" \
  --output-md "${REPORT_STEM}.md" \
  --title "KIT-ML 100k SATO++ vs T2M-GPT Baseline: Full-Motion Human-Ratio P1/P2/P3"

date -Is > "${OUT_ROOT}/complete.txt"
echo "[$(date -Is)] KIT 100k full-motion Human-ratio repeat${REPEATS} evaluation complete"
