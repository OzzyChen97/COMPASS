#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
SEED="${SEED:-123}"
GPUS="${GPUS:-0}"
NPROC="${KIT_SATO_NPROC_PER_NODE:-1}"
MICRO_BATCH="${KIT_SATO_BATCH_SIZE:-132}"
GRAD_ACCUM="${KIT_SATO_GRAD_ACCUM_STEPS:-1}"
GLOBAL_BATCH="$((MICRO_BATCH * NPROC * GRAD_ACCUM))"

PAIR_TRAIN="${KIT_PAIR_TRAIN:-dataset/KIT-ML/preserving_pairs_gemini31_full_filtered.jsonl}"
VQ_CKPT="${KIT_VQ_CKPT:-output_kit/VQVAE_KIT/net_best_fid.pth}"
GPT_CKPT="${KIT_GPT_CKPT:-output_kit/GPT_KIT/net_best_fid.pth}"
SATO_ROOT="${KIT_SATO_ROOT:-ablation/kit_t2mgpt_sato}"
EXP_NAME="${KIT_SATO_EXP_NAME:-gemini31_fullpairs_human_recipe_100k_seed${SEED}}"
EXP_DIR="${SATO_ROOT}/${EXP_NAME}"
WANDB_PROJECT="${WANDB_PROJECT:-SATO++-KIT-Ablation}"
WANDB_MODE="${WANDB_MODE:-online}"

for path in "$PAIR_TRAIN" "$VQ_CKPT" "$GPT_CKPT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

if [[ -e "${EXP_DIR}/sato_pp_last.pth" || -e "${EXP_DIR}/sato_pp_final.pth" ]]; then
  echo "Refusing to overwrite an existing training run: $EXP_DIR" >&2
  exit 1
fi

mkdir -p "$EXP_DIR"
printf '%s\n' "$$" > "${EXP_DIR}/pipeline.pid"
date -Is > "${EXP_DIR}/pipeline_started_at.txt"

echo "[$(date -Is)] Starting KIT Gemini full-pair SATO++ from GPT best-FID"
echo "experiment=$EXP_NAME global_batch=$GLOBAL_BATCH pair_data=$PAIR_TRAIN"

CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC" \
  train_sato_pp_ddp.py \
  --dataname kit \
  --resume-pth "$VQ_CKPT" \
  --resume-trans "$GPT_CKPT" \
  --pair-data-path "$PAIR_TRAIN" \
  --out-dir "$SATO_ROOT" \
  --exp-name "$EXP_NAME" \
  --vq-name VQVAE \
  --batch-size "$MICRO_BATCH" \
  --grad-accum-steps "$GRAD_ACCUM" \
  --nb-code 512 \
  --code-dim 512 \
  --output-emb-width 512 \
  --down-t 2 \
  --stride-t 2 \
  --width 512 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --quantizer ema_reset \
  --block-size 51 \
  --embed-dim-gpt 1024 \
  --clip-dim 512 \
  --num-layers 9 \
  --n-head-gpt 16 \
  --ff-rate 4 \
  --drop-out-rate 0.1 \
  --total-iter 100000 \
  --lr 2e-4 \
  --lr-scheduler 60000 \
  --gamma 0.05 \
  --optimizer adamw \
  --weight-decay 1e-6 \
  --pkeep 0.5 \
  --teacher-loss-type l1 \
  --lambda-teach 0.5 \
  --lambda-align 0.2 \
  --lambda-var 0.5 \
  --lambda-cov 0.04 \
  --lambda-diff 0.5 \
  --lambda-loc 0.1 \
  --preserve-ratio 0.5 \
  --attribution-K 16 \
  --tau 1.0 \
  --freeze-clip \
  --eval-iter 5000 \
  --save-iter 5000 \
  --milestone-save-iter 10000 \
  --print-iter 50 \
  --seed "$SEED" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-group "KIT-Gemini31-fullpairs-seed${SEED}" \
  --wandb-name "KIT-Gemini31-fullpairs-SATO++-100k-globalbs${GLOBAL_BATCH}-seed${SEED}" \
  --wandb-tags "kit,t2m-gpt,sato++,gemini-3.1-flash-lite,build-full-pairs,humanml3d-recipe,no-second-order,100k,globalbs${GLOBAL_BATCH},seed${SEED}" \
  --wandb-mode "$WANDB_MODE"

date -Is > "${EXP_DIR}/train_complete.txt"
echo "[$(date -Is)] KIT Gemini full-pair SATO++ training complete"
