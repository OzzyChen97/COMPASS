#!/usr/bin/env bash
set -euo pipefail
cd /mnt/workspace/Wilson/T2M-GPT

# Baseline T2M-GPT + synonym augmentation, single GPU 7
export CUDA_VISIBLE_DEVICES=7

exec /mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python train_t2m_trans_aug.py \
  --dataname t2m \
  --batch-size 64 \
  --total-iter 200000 \
  --resume-pth pretrained/VQVAE/net_best_fid.pth \
  --resume-trans pretrained/VQTransformer_corruption05/net_best_fid.pth \
  --out-dir output_GPT_Final \
  --exp-name baseline_aug_syn \
  --vq-name VQVAE \
  --down-t 2 \
  --block-size 51 \
  --embed-dim-gpt 1024 \
  --num-layers 9 \
  --n-head-gpt 16 \
  --pkeep 0.5 \
  --print-iter 200 \
  --eval-iter 5000 \
  --lr 2e-4 \
  --lr-scheduler 60000 \
  --gamma 0.05 \
  --aug-prob 0.5 \
  --pair-data-path dataset/HumanML3D/preserving_pairs_full.jsonl
