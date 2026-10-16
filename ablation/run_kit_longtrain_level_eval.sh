#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
SPEC="ablation/kit_t2mgpt_sato/longtrain_60k_85k_eval_spec.json"
OUT_ROOT="ablation/kit_t2mgpt_sato/longtrain_levels_repeat20_humanml_fast"
REPORT_STEM="ablation/kit_t2mgpt_sato/KIT_LONGTRAIN_35K_TO_85K_LEVELS_REPEAT20"
REPEATS="${KIT_EVAL_REPEATS:-20}"

MODELS=(
  kit_sato_longtrain_60k
  kit_sato_longtrain_70k
  kit_sato_longtrain_80k
  kit_sato_longtrain_85k
)
LABELS=(60k 70k 80k 85k)
GPUS=(0 1 2 3)
LEVELS=(p1p2p3 p2p3 p3)
PAIRS=(
  dataset/KIT-ML/test_perb_gemini31_mixed_full.json
  dataset/KIT-ML/test_perb_gemini31_p2p3.json
  dataset/KIT-ML/test_perb_gemini31_p3.json
)

mkdir -p "$OUT_ROOT"
printf '%s\n' "$$" > "${OUT_ROOT}/pipeline.pid"
date -Is > "${OUT_ROOT}/started_at.txt"

run_checkpoint() {
  local gpu="$1" model="$2" label="$3"
  local index level pair output
  mkdir -p "${OUT_ROOT}/${label}"
  for index in "${!LEVELS[@]}"; do
    level="${LEVELS[$index]}"
    pair="${PAIRS[$index]}"
    output="${OUT_ROOT}/${label}/${level}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" ablation/repeat20_perturb_fast_batch.py \
      --spec-json "$SPEC" \
      --model "$model" \
      --pair-data-path "$pair" \
      --out-dir "$output" \
      --report-md "${output}.md" \
      --start-seed 123 \
      --num-repeats "$REPEATS" \
      --batch-size 32 \
      --humanml-fast-protocol > "${output}_driver.log" 2>&1
  done
}

pids=()
for index in "${!MODELS[@]}"; do
  run_checkpoint "${GPUS[$index]}" "${MODELS[$index]}" "${LABELS[$index]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

sources=(
  --source "P1P2P3:baseline=ablation/kit_t2mgpt_sato/humanml3d_recipe_35k_lrmilestones25k30k32p5k_seed123_humanml_ref_fast_repeat20_baseline/summary.json"
  --source "P1P2P3:35k=ablation/kit_t2mgpt_sato/humanml3d_recipe_35k_lrmilestones25k30k32p5k_seed123_humanml_ref_fast_repeat20_final35k/summary.json"
  --source "P2P3:baseline=ablation/kit_t2mgpt_sato/eval_p2p3_baseline/summary.json"
  --source "P2P3:35k=ablation/kit_t2mgpt_sato/eval_p2p3_sato35k/summary.json"
  --source "P3:baseline=ablation/kit_t2mgpt_sato/eval_p3only_baseline/summary.json"
  --source "P3:35k=ablation/kit_t2mgpt_sato/eval_p3only_sato35k/summary.json"
)
for label in "${LABELS[@]}"; do
  for level in "${LEVELS[@]}"; do
    case "$level" in
      p1p2p3) display_level="P1P2P3" ;;
      p2p3) display_level="P2P3" ;;
      p3) display_level="P3" ;;
    esac
    sources+=(--source "${display_level}:${label}=${OUT_ROOT}/${label}/${level}/summary.json")
  done
done

"$PYTHON" tools/summarize_kit_longtrain_levels.py \
  "${sources[@]}" \
  --output-json "${REPORT_STEM}.json" \
  --output-csv "${REPORT_STEM}.csv" \
  --output-md "${REPORT_STEM}.md"

date -Is > "${OUT_ROOT}/complete.txt"
echo "[$(date -Is)] KIT long-training level Repeat${REPEATS} evaluation complete"
