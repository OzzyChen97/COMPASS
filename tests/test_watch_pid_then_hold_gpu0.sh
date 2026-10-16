#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WATCHER="$ROOT/ablation/watch_pid_then_hold_gpu0.sh"
SESSION="test-gpu0-holder-$$"
TMP_DIR="$(mktemp -d)"
MARKER="$TMP_DIR/started"
LOG="$TMP_DIR/holder.log"
WATCH_LOG="$TMP_DIR/watcher.log"

cleanup() {
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

sleep 2 &
watched_pid=$!
HOLDER_SESSION="$SESSION" \
HOLDER_LOG="$LOG" \
HOLDER_COMMAND="printf started > '$MARKER'; sleep 30" \
POLL_SECONDS=0.1 \
  bash "$WATCHER" "$watched_pid" > "$WATCH_LOG" 2>&1 &
watcher_pid=$!

sleep 0.5
[[ ! -e "$MARKER" ]] || { echo "holder started before watched process exited"; exit 1; }
wait "$watched_pid"
wait "$watcher_pid"
for _ in {1..30}; do
  [[ -e "$MARKER" ]] && break
  sleep 0.1
done
[[ -e "$MARKER" ]] || { echo "holder did not start after watched process exited"; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null || { echo "holder tmux session is missing"; exit 1; }

HOLDER_SESSION="$SESSION" \
HOLDER_LOG="$LOG" \
HOLDER_COMMAND="printf duplicate >> '$MARKER'" \
POLL_SECONDS=0.1 \
  bash "$WATCHER" "$watched_pid" >> "$WATCH_LOG" 2>&1

[[ "$(cat "$MARKER")" == "started" ]] || { echo "existing holder session was duplicated"; exit 1; }
[[ "$(tmux list-sessions -F '#{session_name}' | grep -Fxc "$SESSION")" -eq 1 ]] || {
  echo "expected exactly one holder session"
  exit 1
}

echo "PASS: watcher starts one holder only after watched PID exits"
