#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:---dry-run}"
GPU="${2:-}"
PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
PAIR="dataset/Motion-X-322/test_perb_gemini31_max75_mixed_full.json"
FROZEN_ROOT="ablation/motionx322_sato/gemini31_100k_full_test_human_protocol_cfg1p3_repeat20"
SPEC="$FROZEN_ROOT/baseline_vs_sato100k_spec.json"
FROZEN_SUMMARY="$FROZEN_ROOT/combined/summary.json"
MODEL="motionx322_gemini31_sato100k"
OUT="${MOTIONX322_SATO_PP_PARITY_OUT:-ablation/motionx322_sato/table3_sato_pp_seed123_parity_cfg1p3}"
REPORT="$OUT/report.md"
REFERENCE="$OUT/frozen_reference_manifest.json"
RESULT="$OUT/parity_result.json"

for path in "$PAIR" "$SPEC" "$FROZEN_SUMMARY"; do
  [[ -f "$path" ]] || { echo "missing required file: $path" >&2; exit 1; }
done
mkdir -p "$OUT"

"$PYTHON" tools/check_motionx322_sato_pp_seed123_parity.py \
  --pair-file "$PAIR" --spec-json "$SPEC" --frozen-summary "$FROZEN_SUMMARY" \
  --output "$REFERENCE"

command=(
  "$PYTHON" ablation/repeat20_perturb_fast_batch.py
  --spec-json "$SPEC" --model "$MODEL" --pair-data-path "$PAIR"
  --out-dir "$OUT/current" --report-md "$REPORT"
  --start-seed 123 --num-repeats 1 --batch-size 32
  --humanml-fast-protocol --cfg-scale 1.3 --cfg-uncond zero
  --sample-mode categorical --sample-temperature 1 --top-k 0 --top-p 1
  --wandb-mode disabled
)

if [[ "$MODE" == "--dry-run" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=<free-gpu>'
  printf ' %q' "${command[@]}"
  printf '\n'
  echo "Frozen reference written to $REFERENCE"
  exit 0
fi
if [[ "$MODE" != "--run" || -z "$GPU" ]]; then
  echo "usage: $0 --dry-run | --run GPU" >&2
  exit 2
fi

gpu_status="$(nvidia-smi --id="$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
IFS=',' read -r memory_used utilization <<< "$gpu_status"
memory_used="${memory_used// /}"
utilization="${utilization// /}"
if [[ ! "$memory_used" =~ ^[0-9]+$ || ! "$utilization" =~ ^[0-9]+$ ]] \
    || (( memory_used > 512 || utilization > 5 )); then
  echo "refusing to use GPU $GPU: memory=${memory_used:-unknown} MiB utilization=${utilization:-unknown}%" >&2
  exit 1
fi
if [[ -e "$OUT/complete.txt" ]]; then
  echo "refusing to overwrite completed parity output: $OUT" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU" "${command[@]}" > "$OUT/driver.log" 2>&1
"$PYTHON" tools/check_motionx322_sato_pp_seed123_parity.py \
  --pair-file "$PAIR" --spec-json "$SPEC" --frozen-summary "$FROZEN_SUMMARY" \
  --current-out "$OUT/current" --current-summary "$OUT/current/summary.json" \
  --atol "${MOTIONX322_SATO_PP_PARITY_ATOL:-1e-5}" --output "$RESULT"
date -Is > "$OUT/complete.txt"
echo "SATO++ Motion-X seed-123 parity passed: $RESULT"
