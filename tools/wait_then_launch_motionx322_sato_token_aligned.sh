#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${MOTIONX322_SATO_WAIT_SESSION:-motionx322_sato_token_aligned_wait}"
TRAIN_SESSION="${MOTIONX322_SATO_TRAIN_SESSION:-motionx322_sato_token_aligned_train}"
STATE_DIR="${ROOT}/output_motionx322_sato/motionx322_sato_token_aligned_earlystop_pgd_mild_v2_seed123"
WATCH_LOG="${STATE_DIR}/gpu_wait.log"
LAUNCH_LOG="${STATE_DIR}/launcher.log"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
POLL_SECONDS="${MOTIONX322_SATO_GPU_POLL_SECONDS:-60}"
MAX_USED_MIB="${MOTIONX322_SATO_GPU_MAX_USED_MIB:-1024}"
MAX_UTIL_PERCENT="${MOTIONX322_SATO_GPU_MAX_UTIL_PERCENT:-5}"
REQUIRED_CONSECUTIVE="${MOTIONX322_SATO_GPU_REQUIRED_CONSECUTIVE:-2}"

mkdir -p "${STATE_DIR}"
exec >>"${WATCH_LOG}" 2>&1
printf '%s watcher_started gpus=%s used_limit=%s util_limit=%s consecutive=%s\n' \
  "$(date -Is)" "$GPUS" "$MAX_USED_MIB" "$MAX_UTIL_PERCENT" "$REQUIRED_CONSECUTIVE"

IFS=',' read -r -a REQUESTED_GPUS <<< "$GPUS"
consecutive=0
while true; do
  mapfile -t rows < <(nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits)
  ready=1
  snapshot=()
  for requested in "${REQUESTED_GPUS[@]}"; do
    found=0
    for row in "${rows[@]}"; do
      IFS=',' read -r index used util <<< "$row"
      index="${index//[[:space:]]/}"
      used="${used//[[:space:]]/}"
      util="${util//[[:space:]]/}"
      if [[ "$index" == "$requested" ]]; then
        found=1
        snapshot+=("${index}:${used}MiB:${util}%")
        if (( used > MAX_USED_MIB || util > MAX_UTIL_PERCENT )); then
          ready=0
        fi
        break
      fi
    done
    if (( ! found )); then
      ready=0
      snapshot+=("${requested}:missing")
    fi
  done

  if (( ready )); then
    ((consecutive += 1))
  else
    consecutive=0
  fi
  printf '%s ready=%s consecutive=%s snapshot=%s\n' \
    "$(date -Is)" "$ready" "$consecutive" "${snapshot[*]}"

  if (( consecutive >= REQUIRED_CONSECUTIVE )); then
    break
  fi
  sleep "$POLL_SECONDS"
done

if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
  printf '%s train_session_already_exists session=%s\n' "$(date -Is)" "$TRAIN_SESSION"
  exit 1
fi
if pgrep -af 'motionx322_sato_token_aligned_earlystop_pgd_mild_v2_seed123' >/dev/null; then
  printf '%s matching_training_process_already_exists\n' "$(date -Is)"
  exit 1
fi

printf '%s launching_training session=%s\n' "$(date -Is)" "$TRAIN_SESSION"
tmux new-session -d -s "$TRAIN_SESSION" \
  "cd '$ROOT' && GPUS='$GPUS' NPROC='${#REQUESTED_GPUS[@]}' tools/run_motionx322_sato50k.sh earlystop-pgd >'$LAUNCH_LOG' 2>&1"
printf '%s training_session_started session=%s launcher_log=%s\n' \
  "$(date -Is)" "$TRAIN_SESSION" "$LAUNCH_LOG"