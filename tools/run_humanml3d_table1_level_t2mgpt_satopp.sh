#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 LEVEL MODEL GPU START_SEED NUM_REPEATS [RUN_ROOT]" >&2
  echo "  LEVEL: P1|P2|P3; MODEL: t2mgpt|satopp" >&2
  exit 2
fi

LEVEL="${1^^}"
MODEL_ALIAS="${2,,}"
GPU="$3"
START_SEED="$4"
NUM_REPEATS="$5"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${6:-${HUMANML_LEVEL_RUN_ROOT:-$ROOT/ablation/humanml3d_table1_levels/runs}}"
PYTHON_BIN="${T2MGPT_PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"

case "$LEVEL" in
  P1|P2|P3) ;;
  *) echo "unsupported level: $LEVEL" >&2; exit 2 ;;
esac
case "$MODEL_ALIAS" in
  t2mgpt) MODEL_NAME="baseline_t2mgpt_original" ;;
  satopp) MODEL_NAME="sato_pp_allloss_l1_iter130k" ;;
  *) echo "unsupported model: $MODEL_ALIAS" >&2; exit 2 ;;
esac
if ! [[ "$START_SEED" =~ ^[0-9]+$ && "$NUM_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "START_SEED and NUM_REPEATS must be positive integers" >&2
  exit 2
fi
END_SEED=$((START_SEED + NUM_REPEATS - 1))
if (( START_SEED < 123 || END_SEED > 142 )); then
  echo "formal seeds must stay within 123..142, got $START_SEED..$END_SEED" >&2
  exit 2
fi

LOWER="${LEVEL,,}"
PAIR="$ROOT/dataset/HumanML3D/test_perb_table1_level_${LOWER}.json"
INDEX="$ROOT/ablation/humanml3d_table1_levels/index_audit/${LOWER}_sample_index_seed123.json"
GROUP_ROOT="$RUN_ROOT/t2mgpt_satopp/$LOWER/$MODEL_NAME"
DIAGNOSTICS="$GROUP_ROOT/diagnostics"
REPORT="$GROUP_ROOT/report.md"
LOG="$GROUP_ROOT/driver_seed${START_SEED}_${END_SEED}.log"

for required in "$PYTHON_BIN" "$PAIR" "$INDEX"; do
  [[ -e "$required" ]] || { echo "missing required asset: $required" >&2; exit 1; }
done

wait_for_idle_gpu() {
  local max_memory="${HUMANML_LEVEL_GPU_MAX_MEMORY_MIB:-512}"
  local max_util="${HUMANML_LEVEL_GPU_MAX_UTILIZATION:-5}"
  local max_checks="${HUMANML_LEVEL_GPU_IDLE_MAX_CHECKS:-12}"
  local poll_seconds="${HUMANML_LEVEL_GPU_IDLE_POLL_SECONDS:-5}"
  local attempt gpu_status memory_used utilization

  [[ "${HUMANML_LEVEL_SKIP_GPU_GUARD:-0}" == "1" ]] && return 0
  if ! [[ "$max_memory" =~ ^[0-9]+$ && "$max_util" =~ ^[0-9]+$ \
      && "$max_checks" =~ ^[1-9][0-9]*$ && "$poll_seconds" =~ ^[0-9]+$ ]]; then
    echo "invalid GPU guard configuration" >&2
    return 2
  fi

  for ((attempt = 1; attempt <= max_checks; attempt++)); do
    if ! gpu_status="$(nvidia-smi --id="$GPU" \
        --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"; then
      echo "failed to query GPU $GPU" >&2
      return 1
    fi
    IFS=',' read -r memory_used utilization <<< "$gpu_status"
    memory_used="${memory_used// /}"
    utilization="${utilization// /}"
    if [[ "$memory_used" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] \
        && (( memory_used <= max_memory && utilization <= max_util )); then
      if (( attempt > 1 )); then
        echo "GPU $GPU became idle after $attempt checks"
      fi
      return 0
    fi
    if (( attempt == max_checks )); then
      echo "GPU $GPU stayed busy after $max_checks checks: " \
        "memory=${memory_used:-unknown} MiB utilization=${utilization:-unknown}%" >&2
      return 1
    fi
    echo "waiting for idle GPU $GPU ($attempt/$max_checks): " \
      "memory=${memory_used:-unknown} MiB utilization=${utilization:-unknown}%"
    sleep "$poll_seconds"
  done
}

wait_for_idle_gpu

mkdir -p "$GROUP_ROOT" "$DIAGNOSTICS"
cd "$ROOT"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="$GPU"

command=(
  "$PYTHON_BIN" ablation/repeat20_perturb_fast_batch.py
  --model "$MODEL_NAME"
  --pair-data-path "$PAIR"
  --expected-sample-index "$INDEX"
  --allowed-levels "$LEVEL"
  --out-dir "$GROUP_ROOT"
  --report-md "$REPORT"
  --start-seed "$START_SEED"
  --num-repeats "$NUM_REPEATS"
  --batch-size 32
  --num-workers 0
  --caption-locked
  --caption-locked-length-policy repair-all
  --fast-batch
  --sampling-pair-rng-mode sequential
  --cfg-scale 1.0
  --cfg-uncond zero
  --sample-mode categorical
  --sample-temperature 1.0
  --top-k 0
  --top-p 1.0
  --diagnostics-dir "$DIAGNOSTICS"
  --wandb-mode disabled
)

printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU" > "$GROUP_ROOT/command_seed${START_SEED}_${END_SEED}.txt"
printf ' %q' "${command[@]}" >> "$GROUP_ROOT/command_seed${START_SEED}_${END_SEED}.txt"
printf '\n' >> "$GROUP_ROOT/command_seed${START_SEED}_${END_SEED}.txt"

"${command[@]}" 2>&1 | tee "$LOG"

"$PYTHON_BIN" - "$GROUP_ROOT/$MODEL_NAME/runs.csv" "$START_SEED" "$END_SEED" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
start, end = map(int, sys.argv[2:])
with path.open(newline="", encoding="utf-8") as handle:
    seeds = {int(row["seed"]) for row in csv.DictReader(handle)}
missing = sorted(set(range(start, end + 1)) - seeds)
if missing:
    raise SystemExit(f"missing completed seeds in {path}: {missing}")
PY

date -Is > "$GROUP_ROOT/complete_seed${START_SEED}_${END_SEED}.txt"