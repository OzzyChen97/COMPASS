#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/workspace/Wilson/T2M-GPT
PYTHON=/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python
PAIR_FILTERED="$ROOT/dataset/Motion-X-322/preserving_pairs_gemini31_human_recipe_filtered.jsonl"
GEN_DATA_ROOT="$ROOT/dataset/Motion-X-322-MaxToken75"
GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-2}"
EXP_NAME=max_token75_gemini31_human_recipe_100k_lr60k_bs132_seed123
EXP_DIR="$ROOT/ablation/motionx322_sato/$EXP_NAME"
TRAIN_LOG="$EXP_DIR/manual_train.log"

cd "$ROOT"

test -s "$PAIR_FILTERED"
test -f output_motionx322/VQVAE_MOTIONX322/net_best_fid.pth
test -f output_motionx322/GPT_MOTIONX322_MAXTOK75_200K/net_best_fid.pth
test ! -e "$EXP_DIR/sato_pp_last.pth"

if [[ "$NPROC" != "2" ]]; then
  echo "This launcher requires two processes for GPUs 0 and 1; got NPROC=$NPROC" >&2
  exit 1
fi

"$PYTHON" tools/build_motionx322_token_cap.py \
  --source dataset/Motion-X-322 \
  --output "$GEN_DATA_ROOT" \
  --max-motion-tokens 75
"$PYTHON" -c "from dataset.config import dataset_max_token_length; assert dataset_max_token_length('motionx322_75', 4) == 76"

mapfile -t holder_pids < <(
  pgrep -f '^/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python /mnt/workspace/Wilson/gpu_1card.py$' || true
)
if (( ${#holder_pids[@]} > 0 )); then
  kill -TERM "${holder_pids[@]}"
  sleep 10
fi

mkdir -p "$EXP_DIR"

CUDA_VISIBLE_DEVICES="$GPUS" TORCH_DISTRIBUTED_TIMEOUT_SECONDS=7200 \
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" \
  train_sato_pp_ddp.py \
  --dataname motionx322_75 \
  --resume-pth output_motionx322/VQVAE_MOTIONX322/net_best_fid.pth \
  --resume-trans output_motionx322/GPT_MOTIONX322_MAXTOK75_200K/net_best_fid.pth \
  --pair-data-path "$PAIR_FILTERED" \
  --out-dir ablation/motionx322_sato --exp-name "$EXP_NAME" --vq-name VQVAE \
  --batch-size 66 --grad-accum-steps 1 --num-workers 4 \
  --nb-code 512 --code-dim 512 --output-emb-width 512 \
  --down-t 2 --stride-t 2 --width 512 --depth 3 \
  --dilation-growth-rate 3 --quantizer ema_reset \
  --block-size 76 --embed-dim-gpt 1024 --clip-dim 512 \
  --num-layers 9 --n-head-gpt 16 --ff-rate 4 --drop-out-rate 0.1 \
  --total-iter 100000 --lr 2e-4 --lr-scheduler 60000 --gamma 0.05 \
  --optimizer adamw --weight-decay 1e-6 --pkeep 0.5 \
  --teacher-loss-type l1 --lambda-teach 0.5 --lambda-align 0.2 \
  --lambda-var 0.5 --lambda-cov 0.04 --lambda-diff 0.5 \
  --lambda-loc 0.1 --preserve-ratio 0.5 --attribution-K 16 \
  --tau 1.0 --freeze-clip --eval-iter 5000 --save-iter 5000 \
  --milestone-save-iter 10000 --print-iter 50 --seed 123 \
  --wandb-project SATO++-MotionX-Ablation --wandb-mode online \
  --wandb-entity HumanoidGAM \
  --wandb-group MotionX322-SATO++-Gemini31-seed123 \
  --wandb-name MotionX322-Gemini31-SATO++-100k-lr60k-bs132-seed123 \
  --wandb-tags motionx322,smplx322,max-token75,sato++,100k,gemini31,humanml3d-recipe,lr60k,globalbs132,seed123 \
  >> "$TRAIN_LOG" 2>&1
