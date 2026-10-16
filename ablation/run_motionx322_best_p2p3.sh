#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
VQ_CKPT="${MOTIONX322_VQ_CKPT:-output_motionx322/VQVAE_MOTIONX322/net_best_fid.pth}"
GPT_CKPT="${MOTIONX322_GPT_CKPT:-output_motionx322/GPT_MOTIONX322_MAXTOK75_200K/net_best_fid.pth}"
SATO_CKPT="${MOTIONX322_SATO_CKPT:-ablation/motionx322_sato/max_token75_humanml3d_recipe_150k_bs132_seed123/sato_pp_best_fid.pth}"
PAIR_PREFIX="${MOTIONX322_PAIR_PREFIX:-dataset/Motion-X-322/test_perb_gemini31}"
OUT_ROOT="${MOTIONX322_P2P3_EVAL_ROOT:-ablation/motionx322_sato/best10k_p2p3_repeat20}"
SPEC_PATH="${OUT_ROOT}/motionx322_best10k_spec.json"
WANDB_PROJECT="${WANDB_PROJECT:-SATO++-MotionX-Ablation}"
WANDB_MODE="${WANDB_MODE:-online}"

for path in "$VQ_CKPT" "$GPT_CKPT" "$SATO_CKPT"; do
  [[ -f "$path" ]] || { echo "Missing checkpoint: $path" >&2; exit 1; }
done

mkdir -p "$OUT_ROOT"
date -Is > "${OUT_ROOT}/started_at.txt"

"$PYTHON" tools/write_sato_eval_spec.py \
  --dataname motionx322_75 \
  --vq-checkpoint "$VQ_CKPT" \
  --baseline-checkpoint "$GPT_CKPT" \
  --sato-checkpoint "$SATO_CKPT" \
  --baseline-name motionx322_t2mgpt_best \
  --sato-name motionx322_sato_best_fid_10k \
  --block-size 76 \
  --output "$SPEC_PATH"

for level in p2 p3; do
  pair_file="${PAIR_PREFIX}_${level}.json"
  [[ -f "$pair_file" ]] || { echo "Missing pair data: $pair_file" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" \
    ablation/repeat20_perturb_fast_batch.py \
    --spec-json "$SPEC_PATH" \
    --pair-data-path "$pair_file" \
    --out-dir "${OUT_ROOT}/${level}" \
    --report-md "${OUT_ROOT}/${level}.md" \
    --start-seed 123 \
    --num-repeats 20 \
    --batch-size 32 \
    --caption-locked \
    --fast-batch \
    --cfg-scale 1.3 \
    --sample-mode categorical \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-mode "$WANDB_MODE" \
    --wandb-group MotionX322-P2P3-Best10k-Repeat20 \
    --wandb-name "MotionX322-${level}-Best10k-CFG1p3-Repeat20" \
    --wandb-tags "motionx322,evaluation,best10k,repeat20,cfg1.3,${level}" \
    > "${OUT_ROOT}/${level}_driver.log" 2>&1
done

date -Is > "${OUT_ROOT}/complete.txt"
