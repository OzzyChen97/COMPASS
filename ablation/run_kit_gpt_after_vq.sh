#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/workspace/Wilson/T2M-GPT"
VQ_SESSION="${KIT_VQ_SESSION:-kit-vq}"
VQ_LOG="${KIT_VQ_LOG:-${ROOT}/output_kit/VQVAE_KIT/run.log}"
FOLLOW_LOG="${KIT_FOLLOW_LOG:-${ROOT}/ablation/kit_t2mgpt_sato/gpt_followup.log}"
POLL_SECONDS="${KIT_POLL_SECONDS:-60}"
SEED="${SEED:-123}"
GPU="${GPU:-1}"

cd "$ROOT"

while tmux has-session -t "$VQ_SESSION" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

if ! grep -q "Train. Iter 300000" "$VQ_LOG" 2>/dev/null; then
  printf '%s VQ session exited before iteration 300000; GPT was not started.\n' \
    "$(date --iso-8601=seconds)" >> "$FOLLOW_LOG"
  exit 1
fi

printf '%s VQ completed; starting KIT tokenization and T2M-GPT on GPU %s.\n' \
  "$(date --iso-8601=seconds)" "$GPU" >> "$FOLLOW_LOG"

env GPU="$GPU" SEED="$SEED" WANDB_PROJECT="${WANDB_PROJECT:-SATO++-KIT-Ablation}" \
  bash ablation/launch_kit_t2mgpt_sato.sh train-gpt \
  >> "$FOLLOW_LOG" 2>&1

printf '%s KIT T2M-GPT baseline completed.\n' \
  "$(date --iso-8601=seconds)" >> "$FOLLOW_LOG"
