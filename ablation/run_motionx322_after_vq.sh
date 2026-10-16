#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
LAUNCHER="${ROOT}/ablation/launch_motionx322_sato.sh"
VQ_CONTRACT="${ROOT}/ablation/motionx322_sato/contracts/vq.json"
POLL_SECONDS="${MOTIONX322_AFTER_VQ_POLL_SECONDS:-60}"

log() {
  echo "[$(date '+%F %T')] $*"
}

log "Waiting for Motion-X 322D VQ training to finish"
while pgrep -f 'train_vq_ddp.py .*--dataname motionx322' >/dev/null; do
  sleep "$POLL_SECONDS"
done

log "VQ worker exited; waiting for the immutable best-checkpoint contract"
for _ in $(seq 1 10); do
  [[ -s "$VQ_CONTRACT" ]] && break
  sleep 30
done
if [[ ! -s "$VQ_CONTRACT" ]]; then
  log "ERROR: VQ contract was not created; downstream training will not start"
  exit 1
fi

"$PYTHON" tools/stage_checkpoint.py verify --contract "$VQ_CONTRACT"
log "VQ best checkpoint verified; starting tokenize -> GPT -> SATO++"
bash "$LAUNCHER" after-vq
log "Motion-X 322D downstream training completed"
