#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
SPEC="ablation/kit_t2mgpt_sato/longtrain_60k_85k_eval_spec.json"
MODEL="kit_sato_longtrain_85k"
OUT_ROOT="ablation/kit_t2mgpt_sato/longtrain_85k_cfg1p30_levels_repeat20_humanml_fast"
REPORT_STEM="ablation/kit_t2mgpt_sato/KIT_85K_CFG1P00_VS_CFG1P30_LEVELS_REPEAT20"
REPEATS="${KIT_EVAL_REPEATS:-20}"

LEVELS=(p1p2p3 p2p3 p3)
GPUS=(0 1 2)
PAIRS=(
  dataset/KIT-ML/test_perb_gemini31_mixed_full.json
  dataset/KIT-ML/test_perb_gemini31_p2p3.json
  dataset/KIT-ML/test_perb_gemini31_p3.json
)

mkdir -p "$OUT_ROOT"
printf '%s\n' "$$" > "${OUT_ROOT}/pipeline.pid"
date -Is > "${OUT_ROOT}/started_at.txt"

run_level() {
  local gpu="$1" level="$2" pair="$3"
  local output="${OUT_ROOT}/${level}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" ablation/repeat20_perturb_fast_batch.py \
    --spec-json "$SPEC" \
    --model "$MODEL" \
    --pair-data-path "$pair" \
    --out-dir "$output" \
    --report-md "${output}.md" \
    --start-seed 123 \
    --num-repeats "$REPEATS" \
    --batch-size 32 \
    --humanml-fast-protocol \
    --cfg-scale 1.3 \
    --cfg-uncond zero > "${output}_driver.log" 2>&1
}

pids=()
for index in "${!LEVELS[@]}"; do
  run_level "${GPUS[$index]}" "${LEVELS[$index]}" "${PAIRS[$index]}" &
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

"$PYTHON" tools/compare_sato_checkpoints.py \
  --reference "P1P2P3=ablation/kit_t2mgpt_sato/longtrain_levels_repeat20_humanml_fast/85k/p1p2p3/summary.json" \
  --reference "P2P3=ablation/kit_t2mgpt_sato/longtrain_levels_repeat20_humanml_fast/85k/p2p3/summary.json" \
  --reference "P3=ablation/kit_t2mgpt_sato/longtrain_levels_repeat20_humanml_fast/85k/p3/summary.json" \
  --candidate "P1P2P3=${OUT_ROOT}/p1p2p3/summary.json" \
  --candidate "P2P3=${OUT_ROOT}/p2p3/summary.json" \
  --candidate "P3=${OUT_ROOT}/p3/summary.json" \
  --reference-name "CFG-1.0" \
  --candidate-name "CFG-1.3" \
  --output-json "${REPORT_STEM}.json" \
  --output-csv "${REPORT_STEM}.csv" \
  --output-md "${REPORT_STEM}.md"

date -Is > "${OUT_ROOT}/complete.txt"
echo "[$(date -Is)] KIT 85k CFG 1.3 level Repeat${REPEATS} evaluation complete"
