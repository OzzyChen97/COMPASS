#!/usr/bin/env bash
set -euo pipefail
cd /mnt/workspace/Wilson/T2M-GPT

PYTHON=/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python
TORCHRUN="/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/torchrun"

mkdir -p output_GPT_Final/sato_v3a_strong_teach_low_sato_2gpu
mkdir -p output_GPT_Final/sato_v3b_med_teach_med_sato_2gpu
mkdir -p output_GPT_Final/sato_v3c_l1_teach_low_sato_2gpu
mkdir -p output_GPT_Final/02_ablation_v2/ablate_gen_teacher_100k
mkdir -p output_GPT_Final/02_ablation_v2/ablate_preserve_only_100k

# ============================================================
# Experiment v3a: Strong KL teacher + very low SATO++ losses
# GPUs 0-1, lambda_teach=2.0, all SATO++ lambdas at 1/4 original
# ============================================================
nohup bash -c "
cd /mnt/workspace/Wilson/T2M-GPT
export CUDA_VISIBLE_DEVICES=0,1
$TORCHRUN --nproc_per_node=2 --master_port=29500 train_sato_pp_ddp.py \
    --dataname t2m \
    --resume-pth pretrained/VQVAE/net_best_fid.pth \
    --resume-trans pretrained/VQTransformer_corruption05/net_best_fid.pth \
    --out-dir output_GPT_Final \
    --exp-name sato_v3a_strong_teach_low_sato_2gpu \
    --vq-name VQVAE --down-t 2 --block-size 51 \
    --embed-dim-gpt 1024 --num-layers 9 --n-head-gpt 16 \
    --pkeep 0.5 --print-iter 200 --eval-iter 10000 \
    --save-iter 5000 --milestone-save-iter 20000 --total-iter 200000 \
    --batch-size 32 \
    --lr 2e-4 --lr-scheduler 30000 --gamma 0.1 \
    --optimizer adamw --freeze-clip \
    --attribution-method grad_input --attribution-K 16 --tau 1.0 \
    --clip-dim-attr 512 --proj-type linear --proj-dim 512 \
    --preserve-ratio 0.5 \
    --pair-data-path dataset/HumanML3D/preserving_pairs_filtered.jsonl \
    --teacher-loss-type kl \
    --lambda-teach 2.0 \
    --lambda-align 0.05 \
    --lambda-var 0.1 \
    --lambda-cov 0.01 \
    --lambda-diff 0.1 \
    --lambda-loc 0.02 \
    --seed 123
" > output_GPT_Final/sato_v3a_strong_teach_low_sato_2gpu/nohup.log 2>&1 &
PID_A=$!
echo "v3a (GPU 0-1): PID=$PID_A"

# ============================================================
# Experiment v3b: Medium KL teacher + medium SATO++ losses
# GPUs 2-3, lambda_teach=1.0, SATO++ lambdas at 1/2 original
# ============================================================
nohup bash -c "
cd /mnt/workspace/Wilson/T2M-GPT
export CUDA_VISIBLE_DEVICES=2,3
$TORCHRUN --nproc_per_node=2 --master_port=29501 train_sato_pp_ddp.py \
    --dataname t2m \
    --resume-pth pretrained/VQVAE/net_best_fid.pth \
    --resume-trans pretrained/VQTransformer_corruption05/net_best_fid.pth \
    --out-dir output_GPT_Final \
    --exp-name sato_v3b_med_teach_med_sato_2gpu \
    --vq-name VQVAE --down-t 2 --block-size 51 \
    --embed-dim-gpt 1024 --num-layers 9 --n-head-gpt 16 \
    --pkeep 0.5 --print-iter 200 --eval-iter 10000 \
    --save-iter 5000 --milestone-save-iter 20000 --total-iter 200000 \
    --batch-size 32 \
    --lr 2e-4 --lr-scheduler 30000 --gamma 0.1 \
    --optimizer adamw --freeze-clip \
    --attribution-method grad_input --attribution-K 16 --tau 1.0 \
    --clip-dim-attr 512 --proj-type linear --proj-dim 512 \
    --preserve-ratio 0.5 \
    --pair-data-path dataset/HumanML3D/preserving_pairs_filtered.jsonl \
    --teacher-loss-type kl \
    --lambda-teach 1.0 \
    --lambda-align 0.1 \
    --lambda-var 0.2 \
    --lambda-cov 0.02 \
    --lambda-diff 0.2 \
    --lambda-loc 0.05 \
    --seed 123
" > output_GPT_Final/sato_v3b_med_teach_med_sato_2gpu/nohup.log 2>&1 &
PID_B=$!
echo "v3b (GPU 2-3): PID=$PID_B"

# ============================================================
# Experiment v3c: L1 teacher + low SATO++ losses
# GPUs 4-5, lambda_teach=1.0 L1, SATO++ lambdas at 1/4
# ============================================================
nohup bash -c "
cd /mnt/workspace/Wilson/T2M-GPT
export CUDA_VISIBLE_DEVICES=4,5
$TORCHRUN --nproc_per_node=2 --master_port=29502 train_sato_pp_ddp.py \
    --dataname t2m \
    --resume-pth pretrained/VQVAE/net_best_fid.pth \
    --resume-trans pretrained/VQTransformer_corruption05/net_best_fid.pth \
    --out-dir output_GPT_Final \
    --exp-name sato_v3c_l1_teach_low_sato_2gpu \
    --vq-name VQVAE --down-t 2 --block-size 51 \
    --embed-dim-gpt 1024 --num-layers 9 --n-head-gpt 16 \
    --pkeep 0.5 --print-iter 200 --eval-iter 10000 \
    --save-iter 5000 --milestone-save-iter 20000 --total-iter 200000 \
    --batch-size 32 \
    --lr 2e-4 --lr-scheduler 30000 --gamma 0.1 \
    --optimizer adamw --freeze-clip \
    --attribution-method grad_input --attribution-K 16 --tau 1.0 \
    --clip-dim-attr 512 --proj-type linear --proj-dim 512 \
    --preserve-ratio 0.5 \
    --pair-data-path dataset/HumanML3D/preserving_pairs_filtered.jsonl \
    --teacher-loss-type l1 \
    --lambda-teach 1.0 \
    --lambda-align 0.05 \
    --lambda-var 0.1 \
    --lambda-cov 0.01 \
    --lambda-diff 0.1 \
    --lambda-loc 0.02 \
    --seed 123
" > output_GPT_Final/sato_v3c_l1_teach_low_sato_2gpu/nohup.log 2>&1 &
PID_C=$!
echo "v3c (GPU 4-5): PID=$PID_C"

# ============================================================
# Ablation 1: gen + teacher only (no SATO++ losses) - GPU 6
# ============================================================
nohup bash -c "
cd /mnt/workspace/Wilson/T2M-GPT
export CUDA_VISIBLE_DEVICES=6
$PYTHON train_sato_pp.py \
    --dataname t2m \
    --resume-pth pretrained/VQVAE/net_best_fid.pth \
    --resume-trans pretrained/VQTransformer_corruption05/net_best_fid.pth \
    --out-dir output_GPT_Final/02_ablation_v2 \
    --exp-name ablate_gen_teacher_100k \
    --vq-name VQVAE --down-t 2 --block-size 51 \
    --embed-dim-gpt 1024 --num-layers 9 --n-head-gpt 16 \
    --pkeep 0.5 --print-iter 200 --eval-iter 10000 \
    --save-iter 5000 --milestone-save-iter 20000 --total-iter 100000 \
    --batch-size 64 \
    --lr 2e-4 --lr-scheduler 30000 --gamma 0.1 \
    --optimizer adamw --freeze-clip \
    --attribution-method grad_input --attribution-K 16 --tau 1.0 \
    --clip-dim-attr 512 --proj-type linear --proj-dim 512 \
    --preserve-ratio 0.5 \
    --pair-data-path dataset/HumanML3D/preserving_pairs_filtered.jsonl \
    --teacher-loss-type kl \
    --lambda-teach 1.0 \
    --lambda-align 0.0 \
    --lambda-var 0.0 \
    --lambda-cov 0.0 \
    --lambda-diff 0.0 \
    --lambda-loc 0.0 \
    --seed 123
" > output_GPT_Final/02_ablation_v2/ablate_gen_teacher_100k/nohup.log 2>&1 &
PID_D=$!
echo "ablate_gen+teacher (GPU 6): PID=$PID_D"

# ============================================================
# Ablation 2: gen + teacher + preserve losses only - GPU 7
# Tests SimSiam + VICReg without altering losses
# ============================================================
nohup bash -c "
cd /mnt/workspace/Wilson/T2M-GPT
export CUDA_VISIBLE_DEVICES=7
$PYTHON train_sato_pp.py \
    --dataname t2m \
    --resume-pth pretrained/VQVAE/net_best_fid.pth \
    --resume-trans pretrained/VQTransformer_corruption05/net_best_fid.pth \
    --out-dir output_GPT_Final/02_ablation_v2 \
    --exp-name ablate_preserve_only_100k \
    --vq-name VQVAE --down-t 2 --block-size 51 \
    --embed-dim-gpt 1024 --num-layers 9 --n-head-gpt 16 \
    --pkeep 0.5 --print-iter 200 --eval-iter 10000 \
    --save-iter 5000 --milestone-save-iter 20000 --total-iter 100000 \
    --batch-size 64 \
    --lr 2e-4 --lr-scheduler 30000 --gamma 0.1 \
    --optimizer adamw --freeze-clip \
    --attribution-method grad_input --attribution-K 16 --tau 1.0 \
    --clip-dim-attr 512 --proj-type linear --proj-dim 512 \
    --preserve-ratio 0.5 \
    --pair-data-path dataset/HumanML3D/preserving_pairs_filtered.jsonl \
    --teacher-loss-type kl \
    --lambda-teach 1.0 \
    --lambda-align 0.05 \
    --lambda-var 0.1 \
    --lambda-cov 0.01 \
    --lambda-diff 0.0 \
    --lambda-loc 0.0 \
    --seed 123
" > output_GPT_Final/02_ablation_v2/ablate_preserve_only_100k/nohup.log 2>&1 &
PID_E=$!
echo "ablate_preserve (GPU 7): PID=$PID_E"

echo ""
echo "============================================"
echo "All 5 experiments launched!"
echo "============================================"
echo "v3a (GPU 0-1): PID=$PID_A  | teach=2.0 KL, sato=low"
echo "v3b (GPU 2-3): PID=$PID_B  | teach=1.0 KL, sato=med"
echo "v3c (GPU 4-5): PID=$PID_C  | teach=1.0 L1, sato=low"
echo "ablate_gen (GPU 6): PID=$PID_D  | gen+teacher only"
echo "ablate_preserve (GPU 7): PID=$PID_E  | gen+teacher+preserve"
echo ""
echo "Monitor: tail -f output_GPT_Final/<exp_name>/run.log"
echo "============================================"
