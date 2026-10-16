#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/workspace/Wilson/T2M-GPT"
MGPT_ROOT="$ROOT/MotionGPT"
RUN_ROOT="$MGPT_ROOT/experiments/motiongpt_continue_last_to_100k_noearly_20260820"
PYTHON_BIN="/mnt/workspace/Wilson/miniconda3/envs/mgpt/bin/python"

MX_CFG="$RUN_ROOT/configs/motionx322_stage3_continue_100k.yaml"
MX_EXP="$MGPT_ROOT/experiments/mgpt/Instruct_MOTIONX322_OFFICIAL_FRESH_R2_SEED123_20260817"
MX_LOG="$RUN_ROOT/motionx322_continue_100k_restart_shorttmp.log"
STATE="$RUN_ROOT/state.tsv"

mkdir -p /tmp/mgptc100k

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/mnt/workspace/Wilson/miniconda3/envs/mgpt/lib/python3.11/site-packages/torch/lib:/mnt/workspace/Wilson/miniconda3/envs/mgpt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export TMPDIR="/tmp/mgptc100k"

record() {
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" >> "$STATE"
}

fail_if_bad_log() {
  local label="$1"
  local log="$2"
  if rg -qi 'Traceback|CUDA out of memory|ProcessExitedException|ChildFailedError|NCCL WARN|non-finite validation|Detected nan|Detected NaN|loss[^[:alnum:]]*[=:][^[:space:]]*(nan|inf)' "$log"; then
    record "FAIL:${label}:error_signature"
    exit 1
  fi
  if ! rg -q 'Training ends!' "$log"; then
    record "FAIL:${label}:missing_training_ends"
    exit 1
  fi
}

cd "$MGPT_ROOT"

record "restart:motionx322:gpu0-3:no_earlystop:short_tmpdir"
MASTER_PORT=29633 "$PYTHON_BIN" train.py \
  --cfg "$MX_CFG" \
  --cfg_assets configs/assets_motionx322.yaml \
  --device 0 1 2 3 \
  --nodebug \
  --resume "$MX_EXP" > "$MX_LOG" 2>&1
record "done:motionx322"
fail_if_bad_log "motionx322" "$MX_LOG"
record "complete"
