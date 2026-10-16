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
RAW_ROOT="${MOTIONX_RAW_ROOT:-${DATA_ROOT}/raw/bypy}"
JOINT_ROOT="${MOTIONX_JOINT_ROOT:-${DATA_ROOT}/raw/joint}"
SPLIT_REPO="${MOTIONX_SPLIT_REPO:-${DATA_ROOT}/raw/motionx-official}"
MOTIONX_GIT_URL="${MOTIONX_GIT_URL:-https://github.com/IDEA-Research/Motion-X.git}"
BYPY="${BYPY:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/bypy}"
SMPLX_MODEL_ROOT="${SMPLX_MODEL_ROOT:-/mnt/workspace/Wilson/ck/umrm/GENMO/inputs/checkpoints/body_models}"
OUTPUT_ROOT="${MOTIONX_OUTPUT_ROOT:-output_motionx}"
VQ_EXP="${MOTIONX_VQ_EXP:-VQVAE_MOTIONX}"
GPT_EXP="${MOTIONX_GPT_EXP:-GPT_MOTIONX}"
VQ_OUT="${OUTPUT_ROOT}/${VQ_EXP}"
GPT_OUT="${OUTPUT_ROOT}/${GPT_EXP}"
VQ_CKPT="${MOTIONX_VQ_CKPT:-${VQ_OUT}/net_best_fid.pth}"
GPT_CKPT="${MOTIONX_GPT_CKPT:-${GPT_OUT}/net_iter_200000.pth}"
SATO_ROOT="${MOTIONX_SATO_ROOT:-ablation/motionx_t2mgpt_sato}"
SATO_EXP="${MOTIONX_SATO_EXP:-humanml3d_recipe_150k_seed${SEED}}"
SATO_OUT="${SATO_ROOT}/${SATO_EXP}"
SATO_CKPT="${MOTIONX_SATO_CKPT:-${SATO_OUT}/sato_pp_final.pth}"
PAIR_TRAIN="${MOTIONX_PAIR_TRAIN:-${DATA_ROOT}/preserving_pairs_filtered.jsonl}"
PAIR_PREFIX="${MOTIONX_PAIR_PREFIX:-${DATA_ROOT}/test_perb_deterministic}"
PAIR_MODEL="${SATO_PAIR_MODEL:-gpt-5.4-mini}"
WANDB_PROJECT="${WANDB_PROJECT:-SATO++-MotionX-Ablation}"
WANDB_MODE="${WANDB_MODE:-online}"
EVAL_ROOT="${MOTIONX_EVAL_ROOT:-${SATO_ROOT}/repeat20_levels}"
SPEC_PATH="${SATO_ROOT}/motionx_baseline_full_spec.json"
HOLDER_ROOT="${SATO_ROOT}/gpu_holders"

require_path() {
  [[ -e "$1" ]] || { echo "Missing required path: $1" >&2; exit 1; }
}

retry_command() {
  local attempt=1
  local max_attempts="${MOTIONX_PAIR_MAX_ATTEMPTS:-100}"
  while ! "$@"; do
    if (( attempt >= max_attempts )); then
      echo "Command failed after ${attempt} attempts: $*" >&2
      return 1
    fi
    delay=$((attempt * 30))
    (( delay > 300 )) && delay=300
    echo "Command failed on attempt ${attempt}; retrying in ${delay}s: $*" >&2
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

case "$STAGE" in
  preflight)
    "$PYTHON" -m compileall -q dataset models options tools utils \
      train_vq_ddp.py train_t2m_trans_ddp.py train_sato_pp_ddp.py
    "$PYTHON" -c 'import torch, wandb, smplx, spacy; print(torch.__version__)'
    "$BYPY" info
    require_path checkpoints/t2m/text_mot_match/model/finest.tar
    require_path dataset/HumanML3D/new_joints/000021.npy
    require_path "${SMPLX_MODEL_ROOT}/smplx/SMPLX_NEUTRAL.npz"
    ;;

  download)
    mkdir -p "$(dirname "$SPLIT_REPO")"
    if [[ ! -d "${SPLIT_REPO}/.git" ]]; then
      git clone --depth 1 "$MOTIONX_GIT_URL" "$SPLIT_REPO"
    else
      git -C "$SPLIT_REPO" fetch --depth 1 origin main
      git -C "$SPLIT_REPO" checkout --detach FETCH_HEAD
    fi
    "$PYTHON" tools/download_motionx_bypy.py \
      --bypy "$BYPY" --remote-root Motion-X++ --output "$RAW_ROOT"
    ;;

  convert)
    require_path "${RAW_ROOT}/motion/motion_generation/smplx322"
    mkdir -p "$JOINT_ROOT"
    IFS=',' read -r -a gpu_array <<< "$GPUS"
    pids=()
    for shard in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES="${gpu_array[$shard]}" "$PYTHON" \
        tools/convert_motionx_smplx_to_joints.py \
        --input "${RAW_ROOT}/motion/motion_generation/smplx322" \
        --output "$JOINT_ROOT" --model-root "$SMPLX_MODEL_ROOT" \
        --device cuda:0 --num-shards 4 --shard-index "$shard" \
        > "${JOINT_ROOT}/conversion_shard${shard}.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    ;;

  prepare)
    require_path "${SPLIT_REPO}/datasets/train.txt"
    if ! "$PYTHON" -c 'import spacy; spacy.load("en_core_web_sm")' >/dev/null 2>&1; then
      "$PYTHON" -m spacy download en_core_web_sm
    fi
    "$PYTHON" tools/prepare_motionx_dataset.py \
      --raw-root "$RAW_ROOT" \
      --joint-root "$JOINT_ROOT" \
      --split-root "${SPLIT_REPO}/datasets" \
      --output "$DATA_ROOT" \
      --workers "${MOTIONX_PREP_WORKERS:-12}" \
      --source-fps 30 --target-fps 20
    "$PYTHON" tools/validate_sato_dataset.py \
      --dataset-root "$DATA_ROOT" --expected-dim 263 \
      --max-missing-fraction 0 --require-disjoint-train-val --full-scan
    ;;

  pairs)
    require_path "${DATA_ROOT}/train.txt"
    retry_command "$PYTHON" tools/build_sato_pairs.py \
      --dataset-root "$DATA_ROOT" --split train --protocol train \
      --output "$PAIR_TRAIN" --model "$PAIR_MODEL" \
      --batch-size 16 --workers "${MOTIONX_PAIR_WORKERS:-4}" \
      --max-retries 5 --seed "$SEED"
    retry_command "$PYTHON" tools/build_sato_pairs.py \
      --dataset-root "$DATA_ROOT" --split test --protocol test \
      --output "${PAIR_PREFIX}.json" --model "$PAIR_MODEL" \
      --batch-size 16 --workers "${MOTIONX_PAIR_WORKERS:-4}" \
      --max-retries 5 --seed "$SEED"
    "$PYTHON" tools/finalize_sato_test_levels.py \
      --progress "${PAIR_PREFIX}.json.progress.jsonl" \
      --output-prefix "$PAIR_PREFIX" --seed "$SEED" \
      --mixed-ratios 0.46 0.27 0.27
    if ! "$PYTHON" tools/check_sato_pair_quality.py \
      --meta "${PAIR_PREFIX}_p123.meta.json" --max-fallback-rate 0.02; then
      "$PYTHON" tools/repair_sato_test_fallbacks.py \
        --progress "${PAIR_PREFIX}.json.progress.jsonl" \
        --model "$PAIR_MODEL" --batch-size 16 \
        --workers "${MOTIONX_PAIR_WORKERS:-4}" \
        --max-retries 5 --max-fallback-rate 0.02
      "$PYTHON" tools/finalize_sato_test_levels.py \
        --progress "${PAIR_PREFIX}.json.progress.jsonl" \
        --output-prefix "$PAIR_PREFIX" --seed "$SEED" \
        --mixed-ratios 0.46 0.27 0.27
      "$PYTHON" tools/check_sato_pair_quality.py \
        --meta "${PAIR_PREFIX}_p123.meta.json" --max-fallback-rate 0.02
    fi
    "$PYTHON" tools/validate_sato_dataset.py \
      --dataset-root "$DATA_ROOT" --expected-dim 263 \
      --train-pairs "$PAIR_TRAIN" \
      --test-pairs "${PAIR_PREFIX}_mixed_human_ratio.json" \
      --require-disjoint-train-val
    ;;

  train-vq)
    torchrun4 train_vq_ddp.py \
      --dataname motionx --batch-size 64 --num-workers 8 \
      --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 \
      --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 \
      --out-dir "$OUTPUT_ROOT" --exp-name "$VQ_EXP" \
      --vq-act relu --quantizer ema_reset --loss-vel 0.5 \
      --recons-loss l1_smooth --eval-iter 10000 --save-iter 10000 \
      --print-iter 50 --seed "$SEED" \
      --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE" \
      --wandb-group "MotionX-baseline-seed${SEED}" \
      --wandb-name "MotionX-VQVAE-300k-4gpu-seed${SEED}" \
      --wandb-tags "motionx,body-only,263d,t2m-gpt,baseline,vqvae,4gpu,seed${SEED}" \
      --wandb-watch-log all --wandb-watch-freq 200
    ;;

  tokenize)
    require_path "$VQ_CKPT"
    pids=()
    mkdir -p "${DATA_ROOT}/VQVAE/tokenize_logs"
    IFS=',' read -r -a gpu_array <<< "$GPUS"
    for shard in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES="${gpu_array[$shard]}" "$PYTHON" tools/prepare_motion_tokens.py \
        --dataname motionx --resume-pth "$VQ_CKPT" --vq-name VQVAE \
        --device cuda:0 --num-shards 4 --shard-index "$shard" --allow-partial \
        > "${DATA_ROOT}/VQVAE/tokenize_logs/shard${shard}.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    "$PYTHON" tools/validate_sato_dataset.py \
      --dataset-root "$DATA_ROOT" --expected-dim 263 --vq-name VQVAE \
      --max-missing-fraction 0 --require-disjoint-train-val
    ;;

  train-gpt)
    require_path "$VQ_CKPT"
    require_path "${DATA_ROOT}/VQVAE"
    gpt_recovery_args=()
    if [[ "${MOTIONX_GPT_SKIP_INITIAL_EVAL:-0}" == "1" ]]; then
      gpt_recovery_args+=(--skip-initial-eval)
    fi
    torchrun4 train_t2m_trans_ddp.py \
      --dataname motionx --exp-name "$GPT_EXP" --out-dir "$OUTPUT_ROOT" \
      --batch-size 32 --num-workers 8 --num-layers 9 \
      --embed-dim-gpt 1024 --nb-code 512 --n-head-gpt 16 \
      --block-size 51 --ff-rate 4 --drop-out-rate 0.1 \
      --resume-pth "$VQ_CKPT" --vq-name VQVAE \
      --total-iter 200000 --lr-scheduler 150000 --lr 1e-4 \
      --down-t 2 --depth 3 --quantizer ema_reset --pkeep 0.5 \
      --dilation-growth-rate 3 --vq-act relu \
      --eval-iter 10000 --save-iter 10000 --print-iter 50 --seed "$SEED" \
      --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE" \
      --wandb-group "MotionX-baseline-seed${SEED}" \
      --wandb-name "MotionX-T2M-GPT-200k-4gpu-seed${SEED}" \
      --wandb-tags "motionx,body-only,263d,t2m-gpt,baseline,gpt,200k,4gpu,seed${SEED}" \
      --wandb-watch-log all --wandb-watch-freq 200 \
      "${gpt_recovery_args[@]}"
    ;;

  train-sato)
    require_path "$VQ_CKPT"
    require_path "$GPT_CKPT"
    require_path "$PAIR_TRAIN"
    torchrun4 train_sato_pp_ddp.py \
      --dataname motionx --resume-pth "$VQ_CKPT" --resume-trans "$GPT_CKPT" \
      --pair-data-path "$PAIR_TRAIN" --out-dir "$SATO_ROOT" --exp-name "$SATO_EXP" \
      --vq-name VQVAE --batch-size 33 --num-workers 4 \
      --nb-code 512 --code-dim 512 --output-emb-width 512 \
      --down-t 2 --stride-t 2 --width 512 --depth 3 \
      --dilation-growth-rate 3 --quantizer ema_reset \
      --block-size 51 --embed-dim-gpt 1024 --clip-dim 512 \
      --num-layers 9 --n-head-gpt 16 --ff-rate 4 --drop-out-rate 0.1 \
      --total-iter 150000 --lr 2e-4 --lr-scheduler 60000 --gamma 0.05 \
      --optimizer adamw --weight-decay 1e-6 --pkeep 0.5 \
      --teacher-loss-type l1 --lambda-teach 0.5 \
      --lambda-align 0.2 --lambda-var 0.5 --lambda-cov 0.04 \
      --lambda-diff 0.5 --lambda-loc 0.1 --preserve-ratio 0.5 \
      --attribution-K 16 --tau 1.0 --freeze-clip \
      --eval-iter 5000 --save-iter 5000 --milestone-save-iter 10000 \
      --print-iter 50 --seed "$SEED" \
      --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE" \
      --wandb-group "MotionX-SATO++-seed${SEED}" \
      --wandb-name "MotionX-full-SATO++-150k-4gpu-seed${SEED}" \
      --wandb-tags "motionx,body-only,263d,t2m-gpt,sato++,full,150k,humanml3d-recipe,no-second-order,4gpu,globalbs132,seed${SEED}"
    ;;

  eval)
    require_path "$VQ_CKPT"
    require_path "$GPT_CKPT"
    require_path "$SATO_CKPT"
    "$PYTHON" tools/write_sato_eval_spec.py \
      --dataname motionx --vq-checkpoint "$VQ_CKPT" \
      --baseline-checkpoint "$GPT_CKPT" --sato-checkpoint "$SATO_CKPT" \
      --baseline-name motionx_t2mgpt_baseline --sato-name motionx_full_sato_pp \
      --output "$SPEC_PATH"
    mkdir -p "$EVAL_ROOT"
    levels=(p1 p2 p3 mixed)
    pair_files=(
      "${PAIR_PREFIX}_p1.json"
      "${PAIR_PREFIX}_p2.json"
      "${PAIR_PREFIX}_p3.json"
      "${PAIR_PREFIX}_mixed_human_ratio.json"
    )
    IFS=',' read -r -a gpu_array <<< "$GPUS"
    pids=()
    for index in 0 1 2 3; do
      level="${levels[$index]}"
      pair_file="${pair_files[$index]}"
      require_path "$pair_file"
      CUDA_VISIBLE_DEVICES="${gpu_array[$index]}" "$PYTHON" \
        ablation/repeat20_perturb_fast_batch.py \
        --spec-json "$SPEC_PATH" --pair-data-path "$pair_file" \
        --out-dir "${EVAL_ROOT}/${level}" --report-md "${EVAL_ROOT}/${level}.md" \
        --start-seed 123 --num-repeats 20 --batch-size 32 \
        --humanml-fast-protocol --cfg-scale 1.3 --sample-mode categorical \
        --wandb-project "$WANDB_PROJECT" --wandb-mode "$WANDB_MODE" \
        --wandb-group "MotionX-Repeat20-seed${SEED}" \
        --wandb-name "MotionX-${level}-CFG1p3-Repeat20" \
        --wandb-tags "motionx,evaluation,repeat20,cfg1.3,${level},baseline-vs-full" \
        > "${EVAL_ROOT}/${level}_driver.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    "$PYTHON" tools/summarize_motionx_ablation.py \
      --source "P1=${EVAL_ROOT}/p1/summary.json" \
      --source "P2=${EVAL_ROOT}/p2/summary.json" \
      --source "P3=${EVAL_ROOT}/p3/summary.json" \
      --source "P1P2P3=${EVAL_ROOT}/mixed/summary.json" \
      --output-prefix "${SATO_ROOT}/MOTIONX_BASELINE_VS_FULL"
    ;;

  hold-gpus)
    mkdir -p "$HOLDER_ROOT"
    IFS=',' read -r -a gpu_array <<< "$GPUS"
    for index in "${!gpu_array[@]}"; do
      pid_file="${HOLDER_ROOT}/gpu${gpu_array[$index]}.pid"
      if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        continue
      fi
      nohup setsid env CUDA_VISIBLE_DEVICES="${gpu_array[$index]}" \
        "$PYTHON" tools/hold_gpu.py \
        --device cuda:0 --memory-gb "${MOTIONX_HOLDER_MEMORY_GB:-60}" \
        --compute-size 4096 --compute-seconds "${MOTIONX_HOLDER_COMPUTE_SECONDS:-0.25}" \
        --cycle-seconds 1.0 > "${HOLDER_ROOT}/gpu${gpu_array[$index]}.log" 2>&1 \
        < /dev/null &
      echo "$!" > "$pid_file"
    done
    ;;

  release-gpus)
    if [[ -d "$HOLDER_ROOT" ]]; then
      for pid_file in "$HOLDER_ROOT"/*.pid; do
        [[ -e "$pid_file" ]] || continue
        pid="$(cat "$pid_file")"
        if kill -0 "$pid" 2>/dev/null; then
          kill -TERM "$pid"
        fi
      done
      for attempt in {1..30}; do
        alive=0
        for pid_file in "$HOLDER_ROOT"/*.pid; do
          [[ -e "$pid_file" ]] || continue
          kill -0 "$(cat "$pid_file")" 2>/dev/null && alive=1
        done
        [[ "$alive" -eq 0 ]] && break
        sleep 1
      done
    fi
    ;;

  after-convert)
    "$0" hold-gpus
    "$0" prepare
    "$0" after-prepare
    ;;

  after-prepare)
    "$0" pairs > "${SATO_ROOT}/pairs.log" 2>&1 &
    pair_pid="$!"
    "$0" release-gpus
    "$0" train-vq
    "$0" tokenize
    "$0" train-gpt
    wait "$pair_pid"
    "$0" train-sato
    "$0" eval
    ;;

  after-vq)
    "$0" pairs > "${SATO_ROOT}/pairs.log" 2>&1 &
    pair_pid="$!"
    "$0" tokenize
    "$0" train-gpt
    wait "$pair_pid"
    "$0" train-sato
    "$0" eval
    ;;

  after-tokenize)
    "$0" train-gpt
    "$0" pairs
    "$0" train-sato
    "$0" eval
    ;;

  all)
    "$0" download
    "$0" convert
    "$0" after-convert
    ;;

  *)
    echo "Usage: $0 {preflight|download|convert|prepare|pairs|train-vq|tokenize|train-gpt|train-sato|eval|hold-gpus|release-gpus|after-tokenize|after-vq|after-prepare|after-convert|all}" >&2
    exit 2
    ;;
esac
