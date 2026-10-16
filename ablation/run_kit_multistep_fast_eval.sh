#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
SEED="${SEED:-123}"
REPEATS="${KIT_EVAL_REPEATS:-20}"
SATO_ROOT="ablation/kit_t2mgpt_sato"
EXP_NAME="humanml3d_recipe_35k_lrmilestones25k30k32p5k_seed${SEED}"
EXP_DIR="${SATO_ROOT}/${EXP_NAME}"
PAIR_TEST="dataset/KIT-ML/test_perb_gemini31_mixed_full.json"
FINAL_SPEC="${EXP_DIR}/eval_final_spec.json"
BEST_SPEC="${EXP_DIR}/eval_best_spec.json"
FAST_ROOT="${SATO_ROOT}/${EXP_NAME}_humanml_ref_fast_repeat${REPEATS}"

for path in "$PAIR_TEST" "$FINAL_SPEC" "$BEST_SPEC"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

printf '%s\n' "$$" > "${EXP_DIR}/fast_eval_pipeline.pid"
date -Is > "${EXP_DIR}/fast_eval_started_at.txt"

run_eval() {
  local gpu="$1" spec="$2" model="$3" output="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" ablation/repeat20_perturb_fast_batch.py \
    --spec-json "$spec" --model "$model" --pair-data-path "$PAIR_TEST" \
    --out-dir "$output" --report-md "${output}.md" \
    --start-seed 123 --num-repeats "$REPEATS" --batch-size 32 \
    --humanml-fast-protocol > "${output}_driver.log" 2>&1
}

run_eval 0 "$FINAL_SPEC" kit_t2mgpt_baseline "${FAST_ROOT}_baseline" &
pid_baseline="$!"
run_eval 1 "$FINAL_SPEC" kit_sato_humanml3d_recipe_multistep_final35k "${FAST_ROOT}_final35k" &
pid_final="$!"
run_eval 2 "$BEST_SPEC" kit_sato_humanml3d_recipe_multistep_best "${FAST_ROOT}_best" &
pid_best="$!"

failed=0
for pid in "$pid_baseline" "$pid_final" "$pid_best"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

REPORT_STEM="${SATO_ROOT}/KIT_HUMANML3D_RECIPE_35K_LR25K_30K_32P5K_HUMANML_REF_FAST"
"$PYTHON" tools/summarize_human_recipe_eval.py \
  --baseline "${FAST_ROOT}_baseline/summary.json" \
  --candidate "final35k=${FAST_ROOT}_final35k/summary.json" \
  --candidate "best=${FAST_ROOT}_best/summary.json" \
  --output-json "${REPORT_STEM}.json" \
  --output-md "${REPORT_STEM}.md" \
  --protocol-description \
  "HumanML3D-reference fast categorical Repeat-N evaluation: standard shuffled/drop-last loader, batch size 32, 704 samples per repeat."

date -Is > "${EXP_DIR}/fast_eval_complete.txt"
echo "[$(date -Is)] KIT HumanML-reference fast Repeat${REPEATS} complete"
