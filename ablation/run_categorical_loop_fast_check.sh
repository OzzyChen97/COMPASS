#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
SATO_ROOT="ablation/kit_t2mgpt_sato"
PAIR_FILE="dataset/KIT-ML/test_perb_gemini31_mixed_full.json"
SPEC15="${SATO_ROOT}/eval_corrected_mixed_15k_spec.json"
SPEC80="${SATO_ROOT}/eval_corrected_mixed_80k_spec.json"

run_eval() {
  local gpu="$1" spec="$2" model="$3" mode="$4" out_dir="$5"
  fast_args=()
  if [[ "$mode" == "fast" ]]; then
    fast_args=(--fast-batch)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" ablation/repeat20_perturb_fast_batch.py \
    --spec-json "$spec" --model "$model" --pair-data-path "$PAIR_FILE" \
    --out-dir "$out_dir" --report-md "${out_dir}.md" \
    --start-seed 123 --num-repeats 1 --batch-size 32 \
    "${fast_args[@]}" > "${out_dir}_driver.log" 2>&1
}

run_eval 0 "$SPEC15" kit_t2mgpt_baseline loop "${SATO_ROOT}/parity_cat_baseline_loop" &
pid0="$!"
run_eval 1 "$SPEC15" kit_t2mgpt_sato_pp_full_gradient loop "${SATO_ROOT}/parity_cat_sato15k_loop" &
pid1="$!"
run_eval 2 "$SPEC80" kit_t2mgpt_sato_pp_full_gradient loop "${SATO_ROOT}/parity_cat_sato80k_loop" &
pid2="$!"
(
  run_eval 3 "$SPEC15" kit_t2mgpt_baseline fast "${SATO_ROOT}/parity_cat_baseline_fast"
  run_eval 3 "$SPEC15" kit_t2mgpt_sato_pp_full_gradient fast "${SATO_ROOT}/parity_cat_sato15k_fast"
  run_eval 3 "$SPEC80" kit_t2mgpt_sato_pp_full_gradient fast "${SATO_ROOT}/parity_cat_sato80k_fast"
) &
pid3="$!"

failed=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

"$PYTHON" tools/compare_categorical_loop_fast.py \
  --pair "baseline=${SATO_ROOT}/parity_cat_baseline_loop/summary.json=${SATO_ROOT}/parity_cat_baseline_fast/summary.json" \
  --pair "sato15k=${SATO_ROOT}/parity_cat_sato15k_loop/summary.json=${SATO_ROOT}/parity_cat_sato15k_fast/summary.json" \
  --pair "sato80k=${SATO_ROOT}/parity_cat_sato80k_loop/summary.json=${SATO_ROOT}/parity_cat_sato80k_fast/summary.json" \
  --output-json "${SATO_ROOT}/CATEGORICAL_FAST_LOOP_CHECK.json" \
  --output-md "${SATO_ROOT}/CATEGORICAL_FAST_LOOP_CHECK.md"

date -Is > "${SATO_ROOT}/categorical_fast_loop_check_complete.txt"
