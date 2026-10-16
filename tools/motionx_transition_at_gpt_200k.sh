#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
LAUNCHER="ablation/launch_motionx_t2mgpt_sato.sh"
GPT_CKPT="${MOTIONX_GPT_200K_CKPT:-output_motionx/GPT_MOTIONX/net_iter_200000.pth}"
PIPELINE_PID_FILE="ablation/motionx_t2mgpt_sato/pipeline.pid"
QUALITY_META="dataset/Motion-X/test_perb_deterministic_p123.meta.json"
POLL_SECONDS="${MOTIONX_TRANSITION_POLL_SECONDS:-30}"

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

checkpoint_is_valid() {
  [[ -s "$GPT_CKPT" ]] || return 1
  "$PYTHON" -c '
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu")
assert int(payload.get("iter", -1)) == 200000
assert isinstance(payload.get("trans"), dict) and payload["trans"]
' "$GPT_CKPT" >/dev/null 2>&1
}

collect_descendants() {
  local parent="$1"
  local status pid ppid
  for status in /proc/[0-9]*/status; do
    [[ -r "$status" ]] || continue
    pid="${status#/proc/}"
    pid="${pid%/status}"
    ppid="$(awk '/^PPid:/{print $2}' "$status" 2>/dev/null || true)"
    if [[ "$ppid" == "$parent" ]]; then
      collect_descendants "$pid"
      printf '%s\n' "$pid"
    fi
  done
}

stop_pipeline_tree() {
  local root_pid="$1"
  local pid
  mapfile -t descendants < <(collect_descendants "$root_pid")
  if ((${#descendants[@]})); then
    kill -TERM "${descendants[@]}" 2>/dev/null || true
  fi
  kill -TERM "$root_pid" 2>/dev/null || true

  for _ in {1..30}; do
    alive=0
    for pid in "${descendants[@]}" "$root_pid"; do
      [[ -d "/proc/$pid" ]] && alive=1
    done
    ((alive == 0)) && return 0
    sleep 1
  done

  for pid in "${descendants[@]}" "$root_pid"; do
    [[ -d "/proc/$pid" ]] && kill -KILL "$pid" 2>/dev/null || true
  done
}

log "Waiting for validated GPT checkpoint: $GPT_CKPT"
while ! checkpoint_is_valid; do
  latest="$(tail -n 200 output_motionx/GPT_MOTIONX/run.log 2>/dev/null | sed -n 's/.*Train step \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
  log "GPT progress: ${latest:-unknown}/200000"
  sleep "$POLL_SECONDS"
done

log "Validated GPT checkpoint at 200000 steps"
pipeline_pid="$(cat "$PIPELINE_PID_FILE")"
if [[ -d "/proc/$pipeline_pid" ]]; then
  log "Stopping GPT pipeline tree rooted at PID $pipeline_pid"
  stop_pipeline_tree "$pipeline_pid"
fi

export MOTIONX_GPT_CKPT="$GPT_CKPT"
export TORCH_DISTRIBUTED_TIMEOUT_SECONDS="${TORCH_DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
log "Starting four-GPU SATO++ training from $GPT_CKPT"
bash "$LAUNCHER" train-sato

log "Checking perturbation quality before final evaluation"
"$PYTHON" tools/check_sato_pair_quality.py \
  --meta "$QUALITY_META" --max-fallback-rate 0.02

log "Starting final four-level evaluation"
bash "$LAUNCHER" eval
log "Motion-X 200k GPT -> SATO++ -> evaluation pipeline completed"
