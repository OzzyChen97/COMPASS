#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-preflight}"
SEED="${SEED:-123}"
GPU="${GPU:-0}"
GPUS="${GPUS:-0,1,2,3}"
NPROC_PER_NODE="${KIT_SATO_NPROC_PER_NODE:-4}"
PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
DATA_ROOT="${KIT_DATA_ROOT:-dataset/KIT-ML}"
VQ_OUT="${KIT_VQ_OUT:-output_kit/VQVAE_KIT}"
GPT_OUT="${KIT_GPT_OUT:-output_kit/GPT_KIT}"
SATO_ROOT="${KIT_SATO_ROOT:-ablation/kit_t2mgpt_sato}"
SATO_EXP_NAME="${KIT_SATO_EXP_NAME:-humanml3d_recipe_35k_seed${SEED}}"
SATO_OUT="${SATO_ROOT}/${SATO_EXP_NAME}"
PAIR_TRAIN="${KIT_PAIR_TRAIN:-${DATA_ROOT}/preserving_pairs_filtered.jsonl}"
PAIR_TEST="${KIT_PAIR_TEST:-${DATA_ROOT}/test_perb.json}"
PAIR_TEST_PREFIX="${KIT_PAIR_TEST_PREFIX:-${DATA_ROOT}/test_perb_gemini31}"
VQ_CKPT="${KIT_VQ_CKPT:-${VQ_OUT}/net_best_fid.pth}"
GPT_CKPT="${KIT_GPT_CKPT:-${GPT_OUT}/net_best_fid.pth}"
SATO_CKPT="${KIT_SATO_CKPT:-${SATO_OUT}/sato_pp_final.pth}"
SATO_RESUME="${KIT_SATO_RESUME:-}"
BATCH_SIZE="${KIT_SATO_BATCH_SIZE:-33}"
GLOBAL_BATCH_SIZE="$((BATCH_SIZE * NPROC_PER_NODE))"
SATO_TOTAL_ITER="${KIT_SATO_TOTAL_ITER:-35000}"
SATO_LR_SCHEDULER="${KIT_SATO_LR_SCHEDULER:-60000}"
SATO_GAMMA="${KIT_SATO_GAMMA:-0.05}"
read -r -a SATO_LR_SCHEDULER_ARGS <<< "$SATO_LR_SCHEDULER"
if [[ "${#SATO_LR_SCHEDULER_ARGS[@]}" -eq 0 ]]; then
  echo "KIT_SATO_LR_SCHEDULER must contain at least one milestone" >&2
  exit 2
fi
EVAL_NAME="${KIT_EVAL_NAME:-humanml3d_recipe_35k_seed${SEED}_repeat20}"
EVAL_REPEATS="${KIT_EVAL_REPEATS:-20}"
WANDB_PROJECT="${WANDB_PROJECT:-SATO++-KIT-Ablation}"
WANDB_GROUP="${WANDB_GROUP:-KIT-T2M-GPT-baseline-seed${SEED}}"
SATO_WANDB_GROUP="${SATO_WANDB_GROUP:-KIT-HumanML3D-recipe-seed${SEED}}"
SATO_WANDB_NAME="${SATO_WANDB_NAME:-KIT-HumanML3D-recipe-35k-globalbs${GLOBAL_BATCH_SIZE}-seed${SEED}}"
SATO_WANDB_TAGS="${SATO_WANDB_TAGS:-kit,t2m-gpt,sato++,humanml3d-recipe,no-second-order,35k,ddp,${NPROC_PER_NODE}gpu,globalbs${GLOBAL_BATCH_SIZE},seed${SEED}}"
WANDB_MODE="${WANDB_MODE:-online}"
PAIR_MODEL="${KIT_PAIR_MODEL:-gpt-5.4-mini}"

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "Missing required path: $1" >&2
    exit 1
  fi
}

case "$STAGE" in
  preflight)
    require_path "${DATA_ROOT}/train.txt"
    require_path "${DATA_ROOT}/val.txt"
    require_path "${DATA_ROOT}/test.txt"
    require_path "${DATA_ROOT}/new_joint_vecs"
    require_path "${DATA_ROOT}/texts"
    "$PYTHON" tools/validate_sato_dataset.py \
      --dataset-root "$DATA_ROOT" \
      --expected-dim 251
    ;;

  pairs)
    "$PYTHON" tools/build_sato_pairs.py \
      --dataset-root "$DATA_ROOT" \
      --split train \
      --output "$PAIR_TRAIN" \
      --protocol train \
      --model "$PAIR_MODEL" \
      --batch-size 16 \
      --workers 8 \
      --seed "$SEED"
    "$PYTHON" tools/build_sato_pairs.py \
      --dataset-root "$DATA_ROOT" \
      --split test \
      --output "$PAIR_TEST" \
      --protocol test \
      --model "$PAIR_MODEL" \
      --batch-size 16 \
      --workers 8 \
      --seed "$SEED"
    "$PYTHON" tools/validate_sato_dataset.py \
      --dataset-root "$DATA_ROOT" \
      --expected-dim 251 \
      --train-pairs "$PAIR_TRAIN" \
      --test-pairs "$PAIR_TEST"
    "$PYTHON" tools/finalize_test_perb_levels.py \
      --dataset-root "$DATA_ROOT" \
      --pool "$PAIR_TEST" \
      --output-prefix "$PAIR_TEST_PREFIX" \
      --seed "$SEED"
    ;;

  finalize-test-levels)
    require_path "$PAIR_TEST"
    "$PYTHON" tools/finalize_test_perb_levels.py \
      --dataset-root "$DATA_ROOT" \
      --pool "$PAIR_TEST" \
      --output-prefix "$PAIR_TEST_PREFIX" \
      --seed "$SEED"
    ;;

  train-vq)
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" train_vq.py \
      --batch-size 256 \
      --lr 2e-4 \
      --total-iter 300000 \
      --lr-scheduler 200000 \
      --nb-code 512 \
      --down-t 2 \
      --depth 3 \
      --dilation-growth-rate 3 \
      --out-dir output_kit \
      --dataname kit \
      --vq-act relu \
      --quantizer ema_reset \
      --loss-vel 0.5 \
      --recons-loss l1_smooth \
      --exp-name VQVAE_KIT \
      --print-iter 50 \
      --seed "$SEED" \
      --wandb-project "$WANDB_PROJECT" \
      --wandb-group "$WANDB_GROUP" \
      --wandb-name "KIT-VQVAE-baseline-seed${SEED}" \
      --wandb-tags "kit,t2m-gpt,baseline,vqvae,seed${SEED}" \
      --wandb-watch-log all \
      --wandb-watch-freq 200
    ;;

  tokenize)
    require_path "$VQ_CKPT"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" tools/prepare_motion_tokens.py \
      --dataname kit \
      --resume-pth "$VQ_CKPT" \
      --vq-name VQVAE \
      --device cuda:0
    ;;

  train-gpt)
    require_path "$VQ_CKPT"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" train_t2m_trans.py \
      --exp-name GPT_KIT \
      --batch-size 128 \
      --num-layers 9 \
      --embed-dim-gpt 1024 \
      --nb-code 512 \
      --n-head-gpt 16 \
      --block-size 51 \
      --ff-rate 4 \
      --drop-out-rate 0.1 \
      --resume-pth "$VQ_CKPT" \
      --vq-name VQVAE \
      --out-dir output_kit \
      --total-iter 300000 \
      --lr-scheduler 150000 \
      --lr 0.0001 \
      --dataname kit \
      --down-t 2 \
      --depth 3 \
      --quantizer ema_reset \
      --eval-iter 10000 \
      --pkeep 0.5 \
      --dilation-growth-rate 3 \
      --vq-act relu \
      --print-iter 50 \
      --seed "$SEED" \
      --wandb-project "$WANDB_PROJECT" \
      --wandb-group "$WANDB_GROUP" \
      --wandb-name "KIT-T2M-GPT-baseline-seed${SEED}" \
      --wandb-tags "kit,t2m-gpt,baseline,gpt,seed${SEED}" \
      --wandb-watch-log all \
      --wandb-watch-freq 200
    ;;

  train-sato)
    require_path "$VQ_CKPT"
    require_path "$GPT_CKPT"
    require_path "$PAIR_TRAIN"
    resume_args=()
    if [[ -n "$SATO_RESUME" ]]; then
      require_path "$SATO_RESUME"
      resume_args=(--resume-sato-pp "$SATO_RESUME")
    fi
    CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -m torch.distributed.run \
      --standalone \
      --nproc_per_node "$NPROC_PER_NODE" \
      train_sato_pp_ddp.py \
      --dataname kit \
      --resume-pth "$VQ_CKPT" \
      --resume-trans "$GPT_CKPT" \
      "${resume_args[@]}" \
      --pair-data-path "$PAIR_TRAIN" \
      --out-dir "$SATO_ROOT" \
      --exp-name "$SATO_EXP_NAME" \
      --vq-name VQVAE \
      --batch-size "$BATCH_SIZE" \
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
      --total-iter "$SATO_TOTAL_ITER" \
      --lr 2e-4 \
      --lr-scheduler "${SATO_LR_SCHEDULER_ARGS[@]}" \
      --gamma "$SATO_GAMMA" \
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
      --wandb-group "$SATO_WANDB_GROUP" \
      --wandb-name "$SATO_WANDB_NAME" \
      --wandb-tags "$SATO_WANDB_TAGS" \
      --wandb-mode "$WANDB_MODE"
    ;;

  eval)
    require_path "$VQ_CKPT"
    require_path "$GPT_CKPT"
    require_path "$SATO_CKPT"
    require_path "$PAIR_TEST"
    SPEC_PATH="${SATO_ROOT}/${EVAL_NAME}_spec.json"
    "$PYTHON" tools/write_sato_eval_spec.py \
      --dataname kit \
      --vq-checkpoint "$VQ_CKPT" \
      --baseline-checkpoint "$GPT_CKPT" \
      --sato-checkpoint "$SATO_CKPT" \
      --sato-name kit_sato_humanml3d_recipe_35k \
      --output "$SPEC_PATH"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" ablation/repeat20_perturb_fast_batch.py \
      --spec-json "$SPEC_PATH" \
      --pair-data-path "$PAIR_TEST" \
      --out-dir "${SATO_ROOT}/${EVAL_NAME}" \
      --report-md "${SATO_ROOT}/${EVAL_NAME}.md" \
      --start-seed 123 \
      --num-repeats "$EVAL_REPEATS" \
      --batch-size 32 \
      --caption-locked
    ;;

  *)
    echo "Usage: $0 {preflight|pairs|finalize-test-levels|train-vq|tokenize|train-gpt|train-sato|eval}" >&2
    exit 2
    ;;
esac
