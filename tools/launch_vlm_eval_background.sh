#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/Wilson/T2M-GPT}"
LOG_DIR="${LOG_DIR:-$ROOT/ablation/vlm_judge/logs}"
PID_FILE="${PID_FILE:-$LOG_DIR/vlm_eval_background.pid}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/vlm_eval_background.log}"
SESSION_NAME="${SESSION_NAME:-t2mgpt_vlm_eval}"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "already running: $old_pid"
    echo "log: $LOG_FILE"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

cd "$ROOT"
if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already running: $SESSION_NAME"
    echo "log: $LOG_FILE"
    exit 0
  fi
  if [[ "${APPEND_LOG:-0}" == "1" ]]; then
    tmux new-session -d -s "$SESSION_NAME" "cd '$ROOT' && bash tools/run_vlm_eval_background.sh >> '$LOG_FILE' 2>&1"
  else
    tmux new-session -d -s "$SESSION_NAME" "cd '$ROOT' && bash tools/run_vlm_eval_background.sh > '$LOG_FILE' 2>&1"
  fi
  tmux display-message -p -t "$SESSION_NAME" "#{pane_pid}" > "$PID_FILE"
elif [[ "${APPEND_LOG:-0}" == "1" ]]; then
  setsid bash tools/run_vlm_eval_background.sh >> "$LOG_FILE" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
else
  setsid bash tools/run_vlm_eval_background.sh > "$LOG_FILE" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
fi

echo "started: $(cat "$PID_FILE")"
echo "session: $SESSION_NAME"
echo "log: $LOG_FILE"
