#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-preflight}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
BYPY="${BYPY:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/bypy}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
SEED="${SEED:-123}"
DATA_ROOT="${MOTIONX322_DATA_ROOT:-dataset/Motion-X-322}"
GEN_DATA_ROOT="${MOTIONX322_GEN_DATA_ROOT:-dataset/Motion-X-322-MaxToken75}"
GEN_DATANAME="${MOTIONX322_GEN_DATANAME:-motionx322_75}"
MAX_MOTION_TOKENS="${MOTIONX322_MAX_MOTION_TOKENS:-75}"
ARCHIVE_ROOT="${MOTIONX322_ARCHIVE_ROOT:-${DATA_ROOT}/raw/archives}"
MOTION_ZIP="${MOTIONX322_MOTION_ZIP:-${ARCHIVE_ROOT}/motionx_smplx.zip}"
TEXT_ZIP="${MOTIONX322_TEXT_ZIP:-${ARCHIVE_ROOT}/motionx_seq_text.zip}"
SPLIT_ROOT="${MOTIONX322_SPLIT_ROOT:-dataset/Motion-X/raw/motionx-official/datasets}"

PAIR_MODEL="${SATO_PAIR_MODEL:-gpt-5.4-mini}"
PAIR_TRAIN="${MOTIONX322_PAIR_TRAIN:-${DATA_ROOT}/preserving_pairs_gpt54mini.jsonl}"
PAIR_TEST="${MOTIONX322_PAIR_TEST:-${DATA_ROOT}/test_perb.json}"
PAIR_PREFIX="${MOTIONX322_PAIR_PREFIX:-${DATA_ROOT}/test_perb_gemini31}"

OUTPUT_ROOT="${MOTIONX322_OUTPUT_ROOT:-output_motionx322}"
VQ_EXP="${MOTIONX322_VQ_EXP:-VQVAE_MOTIONX322}"
GPT_EXP="${MOTIONX322_GPT_EXP:-GPT_MOTIONX322_MAXTOK75_200K}"
VQ_OUT="${OUTPUT_ROOT}/${VQ_EXP}"
GPT_OUT="${OUTPUT_ROOT}/${GPT_EXP}"
VQ_CKPT="${MOTIONX322_VQ_CKPT:-${VQ_OUT}/net_best_fid.pth}"
GPT_CKPT="${MOTIONX322_GPT_CKPT:-${GPT_OUT}/net_best_fid.pth}"

SATO_ROOT="${MOTIONX322_SATO_ROOT:-ablation/motionx322_sato}"
SATO_EXP="${MOTIONX322_SATO_EXP:-max_token75_humanml3d_recipe_150k_bs132_seed${SEED}}"
SATO_OUT="${SATO_ROOT}/${SATO_EXP}"
SATO_CKPT="${MOTIONX322_SATO_CKPT:-${SATO_OUT}/sato_pp_best_fid.pth}"
SATO_RESUME="${MOTIONX322_SATO_RESUME:-}"
SATO_WANDB_RUN_ID="${MOTIONX322_SATO_WANDB_RUN_ID:-}"
CONTRACT_ROOT="${SATO_ROOT}/contracts"
TOKEN_CONTRACT="${CONTRACT_ROOT}/tokenize.json"
EVALUATOR_DECOMP="checkpoints/motionx322/Decomp_SP001_SM001_H512/model/finest.tar"
EVALUATOR_MATCH="checkpoints/motionx322/text_mot_match/model/finest.tar"
BLOCK_SIZE="${MOTIONX322_BLOCK_SIZE:-76}"
GPT_TOTAL_ITER="${MOTIONX322_GPT_TOTAL_ITER:-200000}"
GPT_EVAL_START_ITER="${MOTIONX322_GPT_EVAL_START_ITER:-100000}"
WANDB_PROJECT="${WANDB_PROJECT:-SATO++-MotionX322-Ablation}"
WANDB_MODE="${WANDB_MODE:-online}"
EVAL_ROOT="${MOTIONX322_EVAL_ROOT:-${SATO_ROOT}/repeat20_levels}"
SPEC_PATH="${SATO_ROOT}/motionx322_baseline_full_spec.json"

require_path() {
  [[ -e "$1" ]] || { echo "Missing required path: $1" >&2; exit 1; }
}

require_best_fid_checkpoint() {
  require_path "$1"
  [[ "$(basename "$1")" == "net_best_fid.pth" ]] || {
    echo "Refusing non-best checkpoint for downstream training: $1" >&2
    exit 1
  }
}

retry_command() {
  local attempt=1
  local max_attempts="${MOTIONX322_PAIR_MAX_ATTEMPTS:-20}"
  while ! "$@"; do
    if (( attempt >= max_attempts )); then
      echo "Command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    local delay=$((attempt * 30))
    (( delay > 300 )) && delay=300
    echo "Command failed on attempt ${attempt}; retrying in ${delay}s" >&2
    sleep "$delay"
    attempt=$((attempt + 1))
  done
}

torchrun4() {
  CUDA_VISIBLE_DEVICES="$GPUS" \
    TORCH_DISTRIBUTED_TIMEOUT_SECONDS="${TORCH_DISTRIBUTED_TIMEOUT_SECONDS:-7200}" \
    "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node "$NPROC" "$@"
}

record_checkpoint() {
  local stage="$1"
  local checkpoint="$2"
  shift 2
  "$PYTHON" tools/stage_checkpoint.py record \
    --stage "$stage" --checkpoint "$checkpoint" \
    --dataset-root "$DATA_ROOT" \
    --contract "${CONTRACT_ROOT}/${stage}.json" "$@"
}

record_generation_checkpoint() {
  local stage="$1"
  local checkpoint="$2"
  shift 2
  "$PYTHON" tools/stage_checkpoint.py record \
    --stage "$stage" --checkpoint "$checkpoint" \
    --dataset-root "$GEN_DATA_ROOT" \
    --contract "${CONTRACT_ROOT}/${stage}.json" "$@"
}

verify_contract() {
  "$PYTHON" tools/stage_checkpoint.py verify \
    --contract "${CONTRACT_ROOT}/$1.json"
}

case "$STAGE" in
  preflight)
    "$PYTHON" -m compileall -q dataset models options tools utils \
      train_motionx_evaluator_ddp.py train_vq_ddp.py \
      train_t2m_trans_ddp.py train_sato_pp_ddp.py
    "$PYTHON" tools/validate_motionx322_dataset.py \
      --dataset-root "$DATA_ROOT"
    "$0" generation-subset
    ;;

  download)
    mkdir -p "$ARCHIVE_ROOT"
    if [[ ! -f "$MOTION_ZIP" ]]; then
      "$BYPY" downfile \
        /Motion-X++/Motion-X/Motion-X/motionx_smplx.zip "$MOTION_ZIP"
    fi
    if [[ ! -f "$TEXT_ZIP" ]]; then
      "$BYPY" downfile \
        /Motion-X++/Motion-X/Motion-X/motionx_seq_text.zip "$TEXT_ZIP"
    fi
    ;;

  prepare)
    require_path "$MOTION_ZIP"
    require_path "$TEXT_ZIP"
    require_path "${SPLIT_ROOT}/train.txt"
    "$PYTHON" tools/prepare_motionx322_dataset.py \
      --motion-zip "$MOTION_ZIP" --text-zip "$TEXT_ZIP" \
      --split-root "$SPLIT_ROOT" --output "$DATA_ROOT" --overwrite
    "$PYTHON" tools/validate_motionx322_dataset.py \
      --dataset-root "$DATA_ROOT" --full-scan
    ;;

  pairs)
    require_path "${DATA_ROOT}/train.txt"
    retry_command "$PYTHON" tools/build_sato_pairs.py \
      --dataset-root "$DATA_ROOT" --split train --protocol train \
      --output "$PAIR_TRAIN" --model "$PAIR_MODEL" \
      --batch-size 16 --workers "${MOTIONX322_PAIR_WORKERS:-8}" \
      --max-retries 5 --completion-rounds 3 --seed "$SEED"
    retry_command "$PYTHON" tools/build_sato_pairs.py \
      --dataset-root "$DATA_ROOT" --split test --protocol test \
      --output "$PAIR_TEST" --model "$PAIR_MODEL" \
      --batch-size 16 --workers "${MOTIONX322_PAIR_WORKERS:-8}" \
      --max-retries 5 --completion-rounds 3 --seed "$SEED"
    "$PYTHON" tools/finalize_test_perb_levels.py \
      --dataset-root "$DATA_ROOT" \
      --pool "$PAIR_TEST" \
      --output-prefix "$PAIR_PREFIX" --seed "$SEED" \
      --mixed-ratios 0.46 0.27 0.27
    "$PYTHON" tools/validate_motionx322_dataset.py \
      --dataset-root "$DATA_ROOT" --train-pairs "$PAIR_TRAIN" \
      --test-prefix "$PAIR_PREFIX"
    ;;

  evaluator-decomp)
    torchrun4 train_motionx_evaluator_ddp.py \
      --stage decomp --data-root "$DATA_ROOT" --dataset-key motionx322 \
      --decomp-global-batch 128 --eval-batch-size 32 \
      --lr 1e-4 --max-epochs "${DECOMP_MAX_EPOCHS:-100}" --decomp-min-epochs "${DECOMP_MIN_EPOCHS:-60}" \
      --decomp-patience 10 --num-workers 4 --seed "$SEED" \
      --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE" \
      --wandb-group "MotionX322-Evaluator-seed${SEED}" \
      --wandb-name "MotionX322-Decomp-322D-seed${SEED}"
    require_path "$EVALUATOR_DECOMP"
    record_checkpoint evaluator_decomp "$EVALUATOR_DECOMP"
    ;;

  evaluator-match)
    verify_contract evaluator_decomp
    torchrun4 train_motionx_evaluator_ddp.py \
      --stage match --data-root "$DATA_ROOT" --dataset-key motionx322 \
      --match-global-batch 32 --eval-batch-size 32 \
      --lr 1e-4 --max-epochs 300 --min-epochs 120 --patience 30 \
      --num-workers 4 --seed "$SEED" \
      --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE" \
      --wandb-group "MotionX322-Evaluator-seed${SEED}" \
      --wandb-name "MotionX322-TextMotionMatch-322D-seed${SEED}"
    require_path "$EVALUATOR_MATCH"
    record_checkpoint evaluator_match "$EVALUATOR_MATCH" \
      --upstream "decomp=${EVALUATOR_DECOMP}"
    ;;

  evaluator-validate)
    verify_contract evaluator_match
    CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PYTHON" \
      train_motionx_evaluator_ddp.py --stage validate \
      --data-root "$DATA_ROOT" --dataset-key motionx322 \
      --eval-batch-size 32 --num-workers 4 --seed "$SEED"
    ;;

  train-vq)
    verify_contract evaluator_match
    require_path "checkpoints/motionx322/evaluator_validation.json"
    "$PYTHON" -c 'import json; from pathlib import Path; report=json.loads(Path("checkpoints/motionx322/evaluator_validation.json").read_text()); assert report.get("passed") is True, "Motion-X evaluator validation did not pass"'
    VQ_MICRO_BATCH="${MOTIONX322_VQ_MICRO_BATCH:-256}"
    VQ_GRAD_ACCUM="${MOTIONX322_VQ_GRAD_ACCUM_STEPS:-1}"
    VQ_GLOBAL_BATCH=$((VQ_MICRO_BATCH * VQ_GRAD_ACCUM * NPROC))
    if (( VQ_GLOBAL_BATCH != 256 )); then
      echo "Motion-X VQ global batch must match HumanML3D: ${VQ_MICRO_BATCH} * ${VQ_GRAD_ACCUM} * ${NPROC} = ${VQ_GLOBAL_BATCH}, expected 256" >&2
      exit 1
    fi
    torchrun4 train_vq_ddp.py \
      --dataname motionx322 --batch-size "$VQ_MICRO_BATCH" \
      --grad-accum-steps "$VQ_GRAD_ACCUM" \
      --num-workers 4 --lr 2e-4 --total-iter 300000 \
      --lr-scheduler 200000 --nb-code 512 --down-t 2 --depth 3 \
      --dilation-growth-rate 3 --out-dir "$OUTPUT_ROOT" \
      --exp-name "$VQ_EXP" --vq-act relu --quantizer ema_reset \
      --loss-vel 0.5 --recons-loss l1_smooth \
      --eval-iter 10000 --save-iter 10000 --print-iter 50 --seed "$SEED" \
      --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE" \
      --wandb-group "MotionX322-baseline-seed${SEED}" \
      --wandb-name "MotionX322-VQVAE-humanml-equivalent-seed${SEED}" \
      --wandb-tags "motionx322,smplx322,raw,30fps,full-sequence,vqvae,humanml-equivalent-ema,microbs${VQ_MICRO_BATCH},globalbs${VQ_GLOBAL_BATCH},seed${SEED}"
    require_path "$VQ_CKPT"
    record_checkpoint vq "$VQ_CKPT" \
      --upstream "evaluator=${EVALUATOR_MATCH}"
    ;;

  tokenize)
    verify_contract vq
    mkdir -p "${DATA_ROOT}/VQVAE/tokenize_logs"
    IFS=',' read -r -a gpu_array <<< "$GPUS"
    token_shards="${#gpu_array[@]}"
    if (( token_shards < 1 )); then
      echo "At least one GPU is required for tokenization" >&2
      exit 1
    fi
    pids=()
    for ((shard = 0; shard < token_shards; shard++)); do
      CUDA_VISIBLE_DEVICES="${gpu_array[$shard]}" "$PYTHON" \
        tools/prepare_motion_tokens.py --dataname motionx322 \
        --resume-pth "$VQ_CKPT" --vq-name VQVAE --device cuda:0 \
        --num-shards "$token_shards" --shard-index "$shard" \
        --allow-partial --overwrite \
        > "${DATA_ROOT}/VQVAE/tokenize_logs/shard${shard}.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    "$PYTHON" tools/validate_motionx322_dataset.py \
      --dataset-root "$DATA_ROOT" --vq-name VQVAE
    "$PYTHON" tools/tokenization_contract.py record \
      --dataset-root "$DATA_ROOT" --vq-name VQVAE \
      --checkpoint "$VQ_CKPT" --contract "$TOKEN_CONTRACT"
    ;;

  generation-subset)
    require_path "${DATA_ROOT}/manifest.jsonl"
    require_path "${DATA_ROOT}/VQVAE"
    "$PYTHON" tools/build_motionx322_token_cap.py \
      --source "$DATA_ROOT" --output "$GEN_DATA_ROOT" \
      --max-motion-tokens "$MAX_MOTION_TOKENS" --unit-length 4
    "$PYTHON" -c "from dataset.config import dataset_max_token_length; assert dataset_max_token_length('${GEN_DATANAME}', 4) == ${BLOCK_SIZE}"
    ;;

  train-gpt)
    verify_contract vq
    require_path "${GEN_DATA_ROOT}/VQVAE"
    "$PYTHON" tools/tokenization_contract.py verify \
      --dataset-root "$DATA_ROOT" --vq-name VQVAE \
      --checkpoint "$VQ_CKPT" --contract "$TOKEN_CONTRACT"
    GPT_MICRO_BATCH="${MOTIONX322_GPT_MICRO_BATCH:-32}"
    GPT_GRAD_ACCUM_DEFAULT=$((128 / (GPT_MICRO_BATCH * NPROC)))
    GPT_GRAD_ACCUM="${MOTIONX322_GPT_GRAD_ACCUM_STEPS:-$GPT_GRAD_ACCUM_DEFAULT}"
    GPT_GLOBAL_BATCH=$((GPT_MICRO_BATCH * GPT_GRAD_ACCUM * NPROC))
    if (( GPT_GLOBAL_BATCH != 128 )); then
      echo "Motion-X GPT global batch must match HumanML3D: ${GPT_MICRO_BATCH} * ${GPT_GRAD_ACCUM} * ${NPROC} = ${GPT_GLOBAL_BATCH}, expected 128" >&2
      exit 1
    fi
    torchrun4 train_t2m_trans_ddp.py \
      --dataname "$GEN_DATANAME" --exp-name "$GPT_EXP" --out-dir "$OUTPUT_ROOT" \
      --batch-size "$GPT_MICRO_BATCH" --grad-accum-steps "$GPT_GRAD_ACCUM" \
      --num-workers 4 \
      --num-layers 9 --embed-dim-gpt 1024 --nb-code 512 \
      --n-head-gpt 16 --block-size "$BLOCK_SIZE" --ff-rate 4 \
      --drop-out-rate 0.1 --resume-pth "$VQ_CKPT" --vq-name VQVAE \
      --total-iter "$GPT_TOTAL_ITER" --lr-scheduler 150000 --lr 1e-4 \
      --down-t 2 --depth 3 --quantizer ema_reset --pkeep 0.5 \
      --dilation-growth-rate 3 --vq-act relu \
      --skip-initial-eval \
      --eval-start-iter "$GPT_EVAL_START_ITER" --eval-iter 10000 \
      --save-iter 10000 --print-iter 50 --seed "$SEED" \
      --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE" \
      --wandb-group "MotionX322-baseline-seed${SEED}" \
      --wandb-name "MotionX322-T2M-GPT-200k-max-token75-seed${SEED}" \
      --wandb-tags "motionx322,smplx322,max-token75,gpt,200k,eval-from-100k,microbs${GPT_MICRO_BATCH},globalbs${GPT_GLOBAL_BATCH},seed${SEED}"
    require_path "$GPT_CKPT"
    record_generation_checkpoint gpt "$GPT_CKPT" \
      --upstream "vq=${VQ_CKPT}" \
      --upstream "tokens=${TOKEN_CONTRACT}" \
      --upstream "evaluator=${EVALUATOR_MATCH}"
    ;;

  train-sato)
    verify_contract gpt
    require_best_fid_checkpoint "$VQ_CKPT"
    require_best_fid_checkpoint "$GPT_CKPT"
    "$PYTHON" -c "from dataset.config import dataset_max_token_length; assert dataset_max_token_length('${GEN_DATANAME}', 4) == 76"
    require_path "$PAIR_TRAIN"
    SATO_MICRO_BATCH="${MOTIONX322_SATO_MICRO_BATCH:-132}"
    SATO_GRAD_ACCUM_DEFAULT=$((132 / (SATO_MICRO_BATCH * NPROC)))
    SATO_GRAD_ACCUM="${MOTIONX322_SATO_GRAD_ACCUM_STEPS:-$SATO_GRAD_ACCUM_DEFAULT}"
    SATO_GLOBAL_BATCH=$((SATO_MICRO_BATCH * SATO_GRAD_ACCUM * NPROC))
    if (( SATO_GLOBAL_BATCH != 132 )); then
      echo "Motion-X SATO++ global batch must match HumanML3D recipe: ${SATO_MICRO_BATCH} * ${SATO_GRAD_ACCUM} * ${NPROC} = ${SATO_GLOBAL_BATCH}, expected 132" >&2
      exit 1
    fi
    sato_resume_args=()
    if [[ -n "$SATO_RESUME" ]]; then
      require_path "$SATO_RESUME"
      sato_resume_args=(--resume-sato-pp "$SATO_RESUME")
    fi
    sato_wandb_args=(--wandb-entity "${WANDB_ENTITY:-HumanoidGAM}")
    if [[ -n "$SATO_WANDB_RUN_ID" ]]; then
      sato_wandb_args+=(--wandb-run-id "$SATO_WANDB_RUN_ID")
    fi
    torchrun4 train_sato_pp_ddp.py \
      --dataname "$GEN_DATANAME" --resume-pth "$VQ_CKPT" \
      --resume-trans "$GPT_CKPT" --pair-data-path "$PAIR_TRAIN" \
      "${sato_resume_args[@]}" \
      --out-dir "$SATO_ROOT" --exp-name "$SATO_EXP" --vq-name VQVAE \
      --batch-size "$SATO_MICRO_BATCH" --grad-accum-steps "$SATO_GRAD_ACCUM" \
      --num-workers 4 \
      --nb-code 512 --code-dim 512 --output-emb-width 512 \
      --down-t 2 --stride-t 2 --width 512 --depth 3 \
      --dilation-growth-rate 3 --quantizer ema_reset \
      --block-size "$BLOCK_SIZE" --embed-dim-gpt 1024 --clip-dim 512 \
      --num-layers 9 --n-head-gpt 16 --ff-rate 4 --drop-out-rate 0.1 \
      --total-iter 150000 --lr 2e-4 --lr-scheduler 60000 --gamma 0.05 \
      --optimizer adamw --weight-decay 1e-6 --pkeep 0.5 \
      --teacher-loss-type l1 --lambda-teach 0.5 --lambda-align 0.2 \
      --lambda-var 0.5 --lambda-cov 0.04 --lambda-diff 0.5 \
      --lambda-loc 0.1 --preserve-ratio 0.5 --attribution-K 16 \
      --tau 1.0 --freeze-clip --eval-iter 5000 --save-iter 5000 \
      --milestone-save-iter 10000 --print-iter 50 --seed "$SEED" \
      --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE" \
      "${sato_wandb_args[@]}" \
      --wandb-group "MotionX322-SATO++-seed${SEED}" \
      --wandb-name "MotionX322-max-token75-SATO++-150k-bs${SATO_MICRO_BATCH}-seed${SEED}" \
      --wandb-tags "motionx322,smplx322,max-token75,sato++,150k,humanml3d-recipe,microbs${SATO_MICRO_BATCH},globalbs${SATO_GLOBAL_BATCH},seed${SEED}"
    require_path "$SATO_CKPT"
    record_generation_checkpoint sato "$SATO_CKPT" \
      --upstream "vq=${VQ_CKPT}" --upstream "gpt=${GPT_CKPT}" \
      --upstream "evaluator=${EVALUATOR_MATCH}" \
      --upstream "pairs=${PAIR_TRAIN}"
    ;;

  eval)
    verify_contract sato
    "$PYTHON" tools/write_sato_eval_spec.py \
      --dataname "$GEN_DATANAME" --vq-checkpoint "$VQ_CKPT" \
      --baseline-checkpoint "$GPT_CKPT" --sato-checkpoint "$SATO_CKPT" \
      --baseline-name motionx322_t2mgpt_baseline \
      --sato-name motionx322_full_sato_pp --block-size "$BLOCK_SIZE" \
      --output "$SPEC_PATH"
    levels=(p1 p2 p3 mixed)
    pair_files=(
      "${PAIR_PREFIX}_p1.json" "${PAIR_PREFIX}_p2.json"
      "${PAIR_PREFIX}_p3.json" "${PAIR_PREFIX}_mixed_full.json"
    )
    IFS=',' read -r -a gpu_array <<< "$GPUS"
    pids=()
    mkdir -p "$EVAL_ROOT"
    for index in 0 1 2 3; do
      level="${levels[$index]}"
      pair_file="${pair_files[$index]}"
      require_path "$pair_file"
      CUDA_VISIBLE_DEVICES="${gpu_array[$index]}" "$PYTHON" \
        ablation/repeat20_perturb_fast_batch.py \
        --spec-json "$SPEC_PATH" --pair-data-path "$pair_file" \
        --out-dir "${EVAL_ROOT}/${level}" \
        --report-md "${EVAL_ROOT}/${level}.md" \
        --start-seed 123 --num-repeats 20 --batch-size 32 \
        --caption-locked --fast-batch --cfg-scale 1.3 \
        --sample-mode categorical --wandb-project "$WANDB_PROJECT" \
        --wandb-mode "$WANDB_MODE" \
        --wandb-group "MotionX322-Repeat20-seed${SEED}" \
        --wandb-name "MotionX322-${level}-CFG1p3-Repeat20" \
        --wandb-tags "motionx322,evaluation,repeat20,cfg1.3,${level}" \
        > "${EVAL_ROOT}/${level}_driver.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    ;;

  all)
    "$0" pairs
    "$0" evaluator-decomp
    "$0" evaluator-match
    "$0" evaluator-validate
    "$0" train-vq
    "$0" tokenize
    "$0" generation-subset
    "$0" train-gpt
    "$0" train-sato
    "$0" eval
    ;;

  after-vq)
    "$0" tokenize
    "$0" generation-subset
    "$0" train-gpt
    "$0" train-sato
    ;;

  after-tokenize)
    "$0" generation-subset
    "$0" train-gpt
    "$0" train-sato
    ;;

  *)
    echo "Unknown stage: $STAGE" >&2
    exit 2
    ;;
esac
