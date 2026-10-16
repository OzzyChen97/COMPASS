#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-preflight}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
SEED="${SEED:-123}"
DATA_ROOT="${MOTIONX_DATA_ROOT:-dataset/Motion-X}"
CHECKPOINT_ROOT="${MOTIONX_EVALUATOR_CHECKPOINT_ROOT:-checkpoints}"
WANDB_PROJECT="${WANDB_PROJECT:-SATO++-MotionX-Ablation}"
WANDB_GROUP="${WANDB_GROUP:-MotionX-Evaluator-263D-seed${SEED}}"
WANDB_MODE="${WANDB_MODE:-online}"
HOLDER_PID_FILE="${MOTIONX_HOLDER_PID_FILE:-ablation/motionx_t2mgpt_sato/gpu3_medium_holder.pid}"

torchrun4() {
  CUDA_VISIBLE_DEVICES="$GPUS" \
    NCCL_DEBUG="${MOTIONX_NCCL_DEBUG:-WARN}" \
    TORCH_DISTRIBUTED_TIMEOUT_SECONDS="${TORCH_DISTRIBUTED_TIMEOUT_SECONDS:-7200}" \
    "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node "$NPROC" "$@"
}

release_known_holder() {
  [[ -f "$HOLDER_PID_FILE" ]] || return 0
  local pid cmdline
  pid="$(tr -d '[:space:]' < "$HOLDER_PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  [[ -r "/proc/$pid/cmdline" ]] || return 0
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  if [[ "$cmdline" != *"tools/hold_gpu.py"* ]]; then
    echo "Refusing to stop PID $pid because it is not the known GPU holder: $cmdline" >&2
    exit 1
  fi
  echo "Stopping known GPU holder PID $pid"
  kill -TERM "$pid"
  for _ in $(seq 1 60); do
    [[ -d "/proc/$pid" ]] || break
    sleep 1
  done
  if [[ -d "/proc/$pid" ]]; then
    echo "GPU holder PID $pid did not stop within 60 seconds" >&2
    exit 1
  fi
}

assert_gpus_free() {
  local processes
  processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$processes" ]]; then
    echo "GPU processes remain; refusing to kill unknown processes:" >&2
    echo "$processes" >&2
    exit 1
  fi
}

start_gpu3_holder() {
  local holder_log holder_pid
  holder_log="${HOLDER_PID_FILE%.pid}.log"
  mkdir -p "$(dirname "$HOLDER_PID_FILE")"
  CUDA_VISIBLE_DEVICES=3 nohup "$PYTHON" tools/hold_gpu.py \
    --device cuda:0 --memory-gb 48 --compute-size 4096 \
    --compute-seconds 0.5 --cycle-seconds 1.0 \
    > "$holder_log" 2>&1 &
  holder_pid="$!"
  echo "$holder_pid" > "$HOLDER_PID_FILE"
  sleep 3
  if [[ ! -d "/proc/$holder_pid" ]]; then
    echo "GPU3 holder failed to start; see $holder_log" >&2
    exit 1
  fi
  echo "GPU3 medium-utilization holder started as PID $holder_pid"
}

preflight() {
  "$PYTHON" -m compileall -q \
    train_motionx_evaluator_ddp.py dataset models utils
  "$PYTHON" tools/validate_sato_dataset.py \
    --dataset-root "$DATA_ROOT" --expected-dim 263 \
    --max-missing-fraction 0 --require-disjoint-train-val --full-scan
  "$PYTHON" - <<'PY'
from pathlib import Path
import numpy as np
import torch
import wandb

root = Path("dataset/Motion-X")
assert np.load(root / "Mean.npy").shape == (263,)
assert np.load(root / "Std.npy").shape == (263,)
for name, expected in (("train", 17250), ("val", 1399), ("test", 4839)):
    count = len([line for line in (root / f"{name}.txt").read_text().splitlines() if line])
    assert count == expected, (name, count, expected)
assert Path("glove/our_vab_data.npy").is_file()
assert torch.cuda.device_count() >= 4, torch.cuda.device_count()
print(f"torch={torch.__version__} wandb={wandb.__version__} cuda_devices={torch.cuda.device_count()}")
PY
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
}

smoke() {
  local smoke_root
  smoke_root="/tmp/motionx_evaluator_smoke_${USER:-root}_$$"
  echo "Smoke checkpoints: $smoke_root"
  torchrun4 train_motionx_evaluator_ddp.py \
    --stage decomp --data-root "$DATA_ROOT" --checkpoint-root "$smoke_root" \
    --seed "$SEED" --limit-ids 64 --max-epochs 2 --num-workers 2 \
    --log-every 10 --save-every 1 --wandb-mode disabled
  torchrun4 train_motionx_evaluator_ddp.py \
    --stage match --data-root "$DATA_ROOT" --checkpoint-root "$smoke_root" \
    --seed "$SEED" --limit-ids 64 --max-epochs 2 --min-epochs 2 \
    --patience 30 --num-workers 2 --log-every 2 --save-every 1 \
    --wandb-mode disabled
  "$PYTHON" - <<PY
import torch
root = "$smoke_root"
decomp = torch.load(f"{root}/motionx/Decomp_SP001_SM001_H512/model/latest.tar", map_location="cpu")
match = torch.load(f"{root}/motionx/text_mot_match/model/finest.tar", map_location="cpu")
assert decomp["ep"] == 2
assert match["epoch"] >= 1
assert {"movement_encoder", "text_encoder", "motion_encoder"} <= match.keys()
print("Four-GPU evaluator smoke test passed")
PY
}

train_decomp() {
  torchrun4 train_motionx_evaluator_ddp.py \
    --stage decomp --data-root "$DATA_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" \
    --seed "$SEED" --max-epochs 100 --min-epochs 60 --patience 10 \
    --decomp-global-batch 128 \
    --decomp-lr-milestones 60 80 --decomp-lr-gamma 0.5 \
    --lr 1e-4 --feat-bias 5 --window-size 40 --num-workers 4 \
    --log-every 50 --save-every 10 --resume \
    --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" \
    --wandb-name "MotionX-Decomp-263D-4GPU-seed${SEED}" \
    --wandb-mode "$WANDB_MODE" --wandb-watch-log gradients --wandb-watch-freq 200
}

train_match() {
  torchrun4 train_motionx_evaluator_ddp.py \
    --stage match --data-root "$DATA_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" \
    --seed "$SEED" --max-epochs 300 --min-epochs 120 --patience 30 \
    --match-global-batch 32 --negative-margin 10 --lr 1e-4 \
    --match-lr-milestones 40 80 --match-lr-gamma 0.5 \
    --feat-bias 5 --unit-length 4 --max-text-len 20 \
    --max-motion-length 196 --num-workers 4 --log-every 50 \
    --save-every 25 --resume \
    --wandb-project "$WANDB_PROJECT" --wandb-group "$WANDB_GROUP" \
    --wandb-name "MotionX-TextMotionMatch-263D-4GPU-seed${SEED}" \
    --wandb-mode "$WANDB_MODE" --wandb-watch-log gradients --wandb-watch-freq 200
}

validate() {
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PYTHON" train_motionx_evaluator_ddp.py \
    --stage validate --data-root "$DATA_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" \
    --seed "$SEED" --match-global-batch 32 --num-workers 4 \
    --wandb-mode disabled
}

case "$STAGE" in
  preflight)
    preflight
    ;;
  smoke)
    release_known_holder
    assert_gpus_free
    preflight
    smoke
    ;;
  train-decomp)
    release_known_holder
    assert_gpus_free
    train_decomp
    ;;
  train-match)
    release_known_holder
    assert_gpus_free
    train_match
    start_gpu3_holder
    ;;
  validate)
    validate
    ;;
  all)
    release_known_holder
    assert_gpus_free
    preflight
    smoke
    train_decomp
    train_match
    start_gpu3_holder
    validate
    ;;
  *)
    echo "Usage: $0 {preflight|smoke|train-decomp|train-match|validate|all}" >&2
    exit 2
    ;;
esac
