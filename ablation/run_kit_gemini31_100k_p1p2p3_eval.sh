#!/usr/bin/env bash
# KIT-ML P1/P2/P3 level evaluation: T2M-GPT baseline vs SATO++ (gemini31 full-pair 100k).
# Waits for the 100k training run to write sato_pp_final.pth, then runs repeat20
# caption-locked fast-batch evaluation at CFG 1.3 on each pure perturbation level.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
REPEATS="${KIT_EVAL_REPEATS:-20}"
CFG_SCALE="${KIT_EVAL_CFG_SCALE:-1.3}"

TRAIN_DIR="ablation/kit_t2mgpt_sato/gemini31_fullpairs_human_recipe_100k_seed123"
FINAL_CKPT="${TRAIN_DIR}/sato_pp_final.pth"
SPEC="ablation/kit_t2mgpt_sato/gemini31_fullpairs_100k_levels_spec.json"
OUT_ROOT="${KIT_EVAL_OUT_ROOT:-ablation/kit_t2mgpt_sato/gemini31_100k_levels_cfg1p3_repeat20}"
REPORT_STEM="ablation/kit_t2mgpt_sato/KIT_GEMINI31_100K_P1P2P3_REPEAT20"

LEVELS=(p1 p2 p3)
PAIRS=(
  dataset/KIT-ML/test_perb_gemini31_p1.json
  dataset/KIT-ML/test_perb_gemini31_p2.json
  dataset/KIT-ML/test_perb_gemini31_p3.json
)

for path in "$SPEC" "${PAIRS[@]}"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

mkdir -p "$OUT_ROOT"
printf '%s\n' "$$" > "${OUT_ROOT}/pipeline.pid"
date -Is > "${OUT_ROOT}/started_at.txt"

# ---- Wait for training to finish ----------------------------------------
WAIT_TIMEOUT_SEC="${KIT_EVAL_WAIT_TIMEOUT_SEC:-43200}"
waited=0
while [[ ! -f "$FINAL_CKPT" ]]; do
  if (( waited >= WAIT_TIMEOUT_SEC )); then
    echo "[$(date -Is)] Timed out after ${WAIT_TIMEOUT_SEC}s waiting for $FINAL_CKPT" >&2
    exit 1
  fi
  if (( waited % 600 == 0 )); then
    echo "[$(date -Is)] Waiting for $FINAL_CKPT (${waited}s elapsed)"
    tail -n 1 "${TRAIN_DIR}/pipeline.log" 2>/dev/null || true
  fi
  sleep 60
  waited=$((waited + 60))
done

# Let the checkpoint finish being written and the trainer release the GPU.
echo "[$(date -Is)] Found $FINAL_CKPT, settling before evaluation"
sleep 120
while pgrep -f "train_sato_pp_ddp.py.*gemini31_fullpairs_human_recipe_100k_seed123" > /dev/null; do
  echo "[$(date -Is)] Training process still alive, waiting for GPU release"
  sleep 60
done

echo "[$(date -Is)] Starting P1/P2/P3 repeat${REPEATS} evaluation at CFG ${CFG_SCALE}"

# ---- Evaluate: one call per level, both models inside; single GPU so serial ----
for index in "${!LEVELS[@]}"; do
  level="${LEVELS[$index]}"
  pair="${PAIRS[$index]}"
  output="${OUT_ROOT}/${level}"
  echo "[$(date -Is)] eval level=${level} pair=${pair}"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" \
    ablation/repeat20_perturb_fast_batch.py \
    --spec-json "$SPEC" \
    --pair-data-path "$pair" \
    --out-dir "$output" \
    --report-md "${output}.md" \
    --start-seed 123 \
    --num-repeats "$REPEATS" \
    --batch-size 32 \
    --caption-locked \
    --fast-batch \
    --cfg-scale "$CFG_SCALE" \
    --cfg-uncond zero \
    --sample-mode categorical \
    --wandb-mode disabled \
    > "${output}_driver.log" 2>&1
done

# ---- Summarize ----------------------------------------------------------
inputs=()
for level in "${LEVELS[@]}"; do
  inputs+=(--input "${level^^}=${OUT_ROOT}/${level}/summary.json")
done

"$PYTHON" tools/summarize_kit_ablation_levels.py \
  "${inputs[@]}" \
  --output-json "${REPORT_STEM}.json" \
  --output-csv "${REPORT_STEM}.csv" \
  --output-md "${REPORT_STEM}.md"

date -Is > "${OUT_ROOT}/complete.txt"
echo "[$(date -Is)] KIT gemini31 100k P1/P2/P3 repeat${REPEATS} evaluation complete"
