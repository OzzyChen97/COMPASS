#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
DATA_ROOT="${MOTIONX322_DATA_ROOT:-dataset/Motion-X-322}"
CHECKPOINT_ROOT="${MOTIONX322_CHECKPOINT_ROOT:-checkpoints}"
CONTRACT_ROOT="${MOTIONX322_CONTRACT_ROOT:-ablation/motionx322_sato/contracts}"
SEED="${SEED:-123}"
WANDB_PROJECT="${WANDB_PROJECT:-SATO++-MotionX322-Ablation}"
WANDB_MODE="${WANDB_MODE:-online}"

DECOMP_CHECKPOINT="${CHECKPOINT_ROOT}/motionx322/Decomp_SP001_SM001_H512/model/finest.tar"
MATCH_CHECKPOINT="${CHECKPOINT_ROOT}/motionx322/text_mot_match/model/finest.tar"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON" train_motionx_evaluator_ddp.py \
  --stage decomp --data-root "$DATA_ROOT" --dataset-key motionx322 \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --decomp-global-batch 128 --eval-batch-size 32 \
  --lr 1e-4 --max-epochs 300 --decomp-min-epochs 300 \
  --decomp-patience 10 --num-workers 4 --seed "$SEED" --resume \
  --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE" \
  --wandb-group "MotionX322-Evaluator-seed${SEED}" \
  --wandb-name "MotionX322-Decomp-322D-seed${SEED}"

"$PYTHON" tools/stage_checkpoint.py record \
  --stage evaluator_decomp --checkpoint "$DECOMP_CHECKPOINT" \
  --dataset-root "$DATA_ROOT" \
  --contract "${CONTRACT_ROOT}/evaluator_decomp.json"
"$PYTHON" tools/stage_checkpoint.py verify \
  --contract "${CONTRACT_ROOT}/evaluator_decomp.json"

"$PYTHON" train_motionx_evaluator_ddp.py \
  --stage match --data-root "$DATA_ROOT" --dataset-key motionx322 \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --match-global-batch 32 --eval-batch-size 32 \
  --lr 1e-4 --max-epochs 300 --min-epochs 120 --patience 30 \
  --negative-margin 10 --unit-length 4 --max-text-len 20 \
  --num-workers 4 --seed "$SEED" \
  --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE" \
  --wandb-group "MotionX322-Evaluator-seed${SEED}" \
  --wandb-name "MotionX322-TextMotionMatch-322D-seed${SEED}"

"$PYTHON" tools/stage_checkpoint.py record \
  --stage evaluator_match --checkpoint "$MATCH_CHECKPOINT" \
  --dataset-root "$DATA_ROOT" \
  --contract "${CONTRACT_ROOT}/evaluator_match.json" \
  --upstream "decomp=${DECOMP_CHECKPOINT}"

"$PYTHON" train_motionx_evaluator_ddp.py \
  --stage validate --data-root "$DATA_ROOT" --dataset-key motionx322 \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --eval-batch-size 32 --num-workers 4 --seed "$SEED"
