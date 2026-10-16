#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 LEVEL GPU START_SEED END_SEED [RUN_ROOT]" >&2
  exit 2
fi

LEVEL="${1^^}"
GPU="$2"
START_SEED="$3"
END_SEED="$4"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${5:-${HUMANML_LEVEL_RUN_ROOT:-$ROOT/ablation/humanml3d_table1_levels/runs}}"
case "$LEVEL" in
  P1|P2|P3) ;;
  *) echo "unsupported level: $LEVEL" >&2; exit 2 ;;
esac
if ! [[ "$START_SEED" =~ ^[0-9]+$ && "$END_SEED" =~ ^[0-9]+$ ]] \
    || (( START_SEED < 123 || END_SEED > 142 || START_SEED > END_SEED )); then
  echo "seed range must be an ordered subset of 123..142" >&2
  exit 2
fi

LOWER="${LEVEL,,}"
SOURCE_ROOT="$ROOT/ablation/humanml3d_table1_levels/motiongpt_sources/$LOWER"
MANIFEST="$SOURCE_ROOT/manifest.json"
CONFIG_ROOT="$SOURCE_ROOT/configs"
PAIR="$ROOT/dataset/HumanML3D/test_perb_table1_level_${LOWER}.json"
INDEX="$ROOT/ablation/humanml3d_table1_levels/index_audit/${LOWER}_sample_index_seed123.json"
OUTPUT_ROOT="$RUN_ROOT/motiongpt/$LOWER"
LOG="$OUTPUT_ROOT/logs/worker_gpu${GPU}_seed${START_SEED}_${END_SEED}.log"

for required in "$MANIFEST" "$PAIR" "$INDEX"; do
  [[ -f "$required" ]] || { echo "missing required asset: $required" >&2; exit 1; }
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

mkdir -p "$OUTPUT_ROOT/logs"
export MOTIONGPT_HUMANML_MANIFEST="$MANIFEST"
export MOTIONGPT_HUMANML_CONFIG_ROOT="$CONFIG_ROOT"
export MOTIONGPT_HUMANML_CONFIG_PATTERN="motiongpt_humanml3d_seed%s.yaml"
export MOTIONGPT_HUMANML_EVAL_SCRIPT="$ROOT/MotionGPT/eval_loader_humanml3d_table1.py"
export PAIR_FILE="$PAIR"
export FIXED_SAMPLE_INDEX="$INDEX"
export OUTPUT_ROOT
export LOADER_MODE=caption-locked
export CAPTION_LENGTH_POLICY=repair-all
export ALLOWED_LEVELS="$LEVEL"
export PAIR_RNG_MODE=sequential
export LENGTH_MODE=generated
export SAMPLING_KWARGS_MODE=explicit
export PROMPT_MODE=official-default
export NUM_WORKERS=0

bash "$ROOT/MotionGPT/scripts/run_humanml3d_loader_eval_shard.sh" \
  "$GPU" "$START_SEED" "$END_SEED" 2>&1 | tee "$LOG"

for seed in $(seq "$START_SEED" "$END_SEED"); do
  for required in \
    "$OUTPUT_ROOT/seed${seed}.json" \
    "$OUTPUT_ROOT/seed${seed}_generation_audit.json" \
    "$OUTPUT_ROOT/seed${seed}_diagnostics.json" \
    "$OUTPUT_ROOT/sample_indices/seed${seed}.json" \
    "$OUTPUT_ROOT/retrieval_embeddings/seed${seed}.npz"; do
    [[ -s "$required" ]] || { echo "missing completed artifact: $required" >&2; exit 1; }
  done
done

date -Is > "$OUTPUT_ROOT/complete_gpu${GPU}_seed${START_SEED}_${END_SEED}.txt"