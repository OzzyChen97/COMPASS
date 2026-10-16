#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
REPEATS="${KIT_P2P3_REPEATS:-20}"
CFG_SCALE="${KIT_P2P3_CFG_SCALE:-1.3}"
OUT_ROOT="${KIT_P2P3_OUT_ROOT:-ablation/gemini31_test_perb_latest_cfg1p3_repeat20_20260725}"
PAIR_POOL="dataset/KIT-ML/test_perb.json"
PAIR_MIX="dataset/KIT-ML/test_perb_gemini31_p2p3_balanced.json"
SPEC="${OUT_ROOT}/kit_latest_baseline_spec.json"

for path in "$PAIR_POOL" "$SPEC"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

"$PYTHON" tools/build_mixed_perturbation_pool.py \
  --input "$PAIR_POOL" \
  --output "$PAIR_MIX" \
  --weights 0 0.5 0.5 \
  --seed 123

run_eval() {
  local model="$1" output="$2"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" \
    ablation/repeat20_perturb_fast_batch.py \
    --spec-json "$SPEC" \
    --model "$model" \
    --pair-data-path "$PAIR_MIX" \
    --out-dir "$output" \
    --report-md "${output}.md" \
    --start-seed 123 \
    --num-repeats "$REPEATS" \
    --batch-size 32 \
    --caption-locked \
    --fast-batch \
    --cfg-scale "$CFG_SCALE" \
    --sample-mode categorical \
    --wandb-mode disabled \
    > "${output}_driver.log" 2>&1
}

run_eval kit_sato_latest_85k "${OUT_ROOT}/kit_p2p3_latest_85k" &
pid_sato="$!"
run_eval kit_t2mgpt_baseline_best "${OUT_ROOT}/kit_p2p3_baseline" &
pid_baseline="$!"

failed=0
for pid in "$pid_sato" "$pid_baseline"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

date -Is > "${OUT_ROOT}/kit_p2p3_complete.txt"
