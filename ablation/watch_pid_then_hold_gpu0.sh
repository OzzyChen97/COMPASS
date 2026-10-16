#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 WATCH_PID" >&2
  exit 2
fi

WATCH_PID="$1"
POLL_SECONDS="${POLL_SECONDS:-5}"
HOLDER_SESSION="${HOLDER_SESSION:-motionx322-gpu0-holder}"
HOLDER_LOG="${HOLDER_LOG:-/mnt/workspace/Wilson/T2M-GPT/ablation/motionx322_sato/gemini31_100k_full_test_human_protocol_cfg1p3_repeat20/gpu0_holder.log}"
PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
HOLDER_SCRIPT="${HOLDER_SCRIPT:-/mnt/workspace/Wilson/gpu_1card.py}"

command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
if [[ -z "${HOLDER_COMMAND:-}" ]]; then
  [[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; exit 1; }
  [[ -f "$HOLDER_SCRIPT" ]] || { echo "Holder script is missing: $HOLDER_SCRIPT" >&2; exit 1; }
  HOLDER_COMMAND="exec '$PYTHON' '$HOLDER_SCRIPT'"
fi
mkdir -p "$(dirname "$HOLDER_LOG")"

printf '%s watching PID %s; holder session=%s\n' \
  "$(date --iso-8601=seconds)" "$WATCH_PID" "$HOLDER_SESSION"
while kill -0 "$WATCH_PID" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done
printf '%s PID %s exited; starting holder\n' \
  "$(date --iso-8601=seconds)" "$WATCH_PID"

if tmux has-session -t "$HOLDER_SESSION" 2>/dev/null; then
  printf '%s holder session already exists: %s\n' \
    "$(date --iso-8601=seconds)" "$HOLDER_SESSION"
  exit 0
fi

tmux new-session -d -s "$HOLDER_SESSION" -- \
  bash -lc "$HOLDER_COMMAND >> '$HOLDER_LOG' 2>&1"
printf '%s holder session started: %s; log=%s\n' \
  "$(date --iso-8601=seconds)" "$HOLDER_SESSION" "$HOLDER_LOG"
