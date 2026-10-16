#!/usr/bin/env bash
set -euo pipefail

# Official SATO (not SATO++) Motion-X-322 adapter.  The default invocation
# launches the requested 4-GPU run; --dry-run prints the immutable command so
# CI and reviewers can audit it without starting a long training job.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SATO_DIR="${ROOT}/Stable-Text-to-Motion-Framework-official"
PYTHON_BIN="${PYTHON_BIN:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "${PYTHON_BIN}")/torchrun}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NPROC="${NPROC:-${#GPU_LIST[@]}}"
GLOBAL_BATCH_SIZE="${MOTIONX322_SATO_GLOBAL_BATCH_SIZE:-64}"
if (( GLOBAL_BATCH_SIZE % NPROC != 0 )); then
  echo "Global batch size ${GLOBAL_BATCH_SIZE} must be divisible by NPROC ${NPROC}" >&2
  exit 2
fi
BATCH_SIZE="${MOTIONX322_SATO_BATCH_SIZE:-$((GLOBAL_BATCH_SIZE / NPROC))}"
SEED="${SEED:-123}"

DATA_ROOT="${MOTIONX322_SATO_DATA_ROOT:-${ROOT}/dataset/Motion-X-322-MaxToken75}"
PAIR_JSONL="${MOTIONX322_SATO_PAIR_JSONL:-${ROOT}/dataset/Motion-X-322/token_aligned_lexical_pairs_sato_clip_train.jsonl}"
VQ_CKPT="${MOTIONX322_SATO_VQ_CKPT:-${ROOT}/output_motionx322/VQVAE_MOTIONX322/net_best_fid.pth}"
T2M_CKPT="${MOTIONX322_SATO_T2M_CKPT:-${ROOT}/output_motionx322/GPT_MOTIONX322_MAXTOK75_200K/net_best_fid.pth}"
CLIP_CKPT="${MOTIONX322_SATO_CLIP_CKPT:-${SATO_DIR}/pretrained/ViT-B-32.pt}"
OUT_DIR="${MOTIONX322_SATO_OUT_DIR:-${ROOT}/output_motionx322_sato}"
MODE="${1:-formal}"
if [[ "$MODE" == "smoke" ]]; then
  DEFAULT_TOTAL_ITER=2
  DEFAULT_PRINT_ITER=1
  DEFAULT_EXP_NAME="motionx322_sato_official_smoke_seed${SEED}"
  DEFAULT_DATASET_LIMIT=64
  DEFAULT_EARLY_STOPPING=0
  DEFAULT_PGD=0
elif [[ "$MODE" == "formal" || "$MODE" == "--dry-run" ]]; then
  DEFAULT_TOTAL_ITER=50000
  DEFAULT_PRINT_ITER=100
  DEFAULT_EXP_NAME="motionx322_sato_official_50k_seed${SEED}"
  DEFAULT_DATASET_LIMIT=0
  DEFAULT_EARLY_STOPPING=0
  DEFAULT_PGD=0
elif [[ "$MODE" == "earlystop" || "$MODE" == "earlystop-dry-run" ]]; then
  DEFAULT_TOTAL_ITER=100000
  DEFAULT_PRINT_ITER=100
  DEFAULT_EXP_NAME="motionx322_sato_official_earlystop_seed${SEED}"
  DEFAULT_DATASET_LIMIT=0
  DEFAULT_EARLY_STOPPING=1
  DEFAULT_PGD=0
elif [[ "$MODE" == "earlystop-pgd" || "$MODE" == "earlystop-pgd-dry-run" ]]; then
  DEFAULT_TOTAL_ITER=100000
  DEFAULT_PRINT_ITER=100
  DEFAULT_EXP_NAME="motionx322_sato_token_aligned_earlystop_pgd_mild_v2_seed${SEED}"
  DEFAULT_DATASET_LIMIT=0
  DEFAULT_EARLY_STOPPING=1
  DEFAULT_PGD=1
else
  echo "usage: $0 {smoke|formal|--dry-run|earlystop|earlystop-dry-run|earlystop-pgd|earlystop-pgd-dry-run}" >&2
  exit 2
fi
EXP_NAME="${MOTIONX322_SATO_EXP_NAME:-${DEFAULT_EXP_NAME}}"
TOTAL_ITER="${MOTIONX322_SATO_TOTAL_ITER:-${DEFAULT_TOTAL_ITER}}"
PRINT_ITER="${MOTIONX322_SATO_PRINT_ITER:-${DEFAULT_PRINT_ITER}}"
DATASET_LIMIT="${MOTIONX322_SATO_DATASET_LIMIT:-${DEFAULT_DATASET_LIMIT}}"
EARLY_STOPPING="${MOTIONX322_SATO_EARLY_STOPPING:-${DEFAULT_EARLY_STOPPING}}"
EARLY_STOPPING_PATIENCE="${MOTIONX322_SATO_EARLY_STOPPING_PATIENCE:-6}"
EARLY_STOPPING_MIN_VALIDATIONS="${MOTIONX322_SATO_EARLY_STOPPING_MIN_VALIDATIONS:-8}"
PGD="${MOTIONX322_SATO_PGD:-${DEFAULT_PGD}}"
PGD_RADIUS="${MOTIONX322_SATO_PGD_RADIUS:-0.01}"
PGD_STEPS="${MOTIONX322_SATO_PGD_STEPS:-1}"
PGD_STEP_SIZE="${MOTIONX322_SATO_PGD_STEP_SIZE:-0.005}"
PGD_PROBABILITY="${MOTIONX322_SATO_PGD_PROBABILITY:-0.25}"
PGD_NORM="${MOTIONX322_SATO_PGD_NORM:-l-infty}"
PGD_RANDOM_START="${MOTIONX322_SATO_PGD_RANDOM_START:-1}"
RESUME_TRAIN_STATE="${MOTIONX322_SATO_RESUME_TRAIN_STATE:-}"

if [[ "$MODE" == "--dry-run" || "$MODE" == "earlystop-dry-run" || "$MODE" == "earlystop-pgd-dry-run" ]]; then
  DRY_RUN=1
else
  DRY_RUN=0
fi

[[ -d "${DATA_ROOT}" ]] || { echo "Missing dataset root: ${DATA_ROOT}" >&2; exit 1; }
[[ -s "${PAIR_JSONL}" ]] || { echo "Missing preserving pair JSONL: ${PAIR_JSONL}" >&2; exit 1; }
[[ -f "${VQ_CKPT}" ]] || { echo "Missing VQ checkpoint: ${VQ_CKPT}" >&2; exit 1; }
[[ -f "${T2M_CKPT}" ]] || { echo "Missing T2M checkpoint: ${T2M_CKPT}" >&2; exit 1; }

COMMAND=("${TORCHRUN}" --standalone --nproc_per_node="${NPROC}" "${SATO_DIR}/train.py"
  --dataname motionx322_75
  --batch-size "${BATCH_SIZE}"
  --total-iter "${TOTAL_ITER}"
  --warm-up-iter 1000
  --lr 1e-4
  --lr-scheduler 60000
  --gamma 0.05
  --optimizer adamw
  --weight-decay 1e-6
  --print-iter "${PRINT_ITER}"
  --eval-iter "${MOTIONX322_SATO_EVAL_ITER:-5000}"
  --save-iter "${MOTIONX322_SATO_SAVE_ITER:-5000}"
  --seed "${SEED}"
  --vq-name VQVAE
  --skip-tokenize
  --pair-jsonl "${PAIR_JSONL}"
  --K 2
  --attention-loss-weight 0.05
  --resume-pth "${VQ_CKPT}"
  --resume-trans "${T2M_CKPT}"
  --clip_path "${CLIP_CKPT}"
  --block-size 76
  --num-workers "${MOTIONX322_SATO_NUM_WORKERS:-0}"
  --dataset-limit "${DATASET_LIMIT}"
  --out-dir "${OUT_DIR}"
  --exp-name "${EXP_NAME}")

if [[ -n "$RESUME_TRAIN_STATE" ]]; then
  [[ -f "$RESUME_TRAIN_STATE" ]] || { echo "Missing training-state checkpoint: $RESUME_TRAIN_STATE" >&2; exit 1; }
  COMMAND+=(--resume-train-state "$RESUME_TRAIN_STATE")
fi

if [[ "$EARLY_STOPPING" == "1" ]]; then
  COMMAND+=(
    --early-stopping
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
    --early-stopping-min-validations "${EARLY_STOPPING_MIN_VALIDATIONS}")
fi

if [[ "$PGD" == "1" ]]; then
  COMMAND+=(
    --pgd
    --pgd-radius "${PGD_RADIUS}"
    --pgd-steps "${PGD_STEPS}"
    --pgd-step-size "${PGD_STEP_SIZE}"
    --pgd-probability "${PGD_PROBABILITY}"
    --pgd-norm "${PGD_NORM}")
  if [[ "$PGD_RANDOM_START" == "1" ]]; then
    COMMAND+=(--pgd-random-start)
  fi
fi

printf 'CUDA_VISIBLE_DEVICES=%s ' "${GPUS}"
printf '%q ' "${COMMAND[@]}"
printf '\n'
if (( DRY_RUN )); then
  exit 0
fi

mkdir -p "${OUT_DIR}/${EXP_NAME}"
cd "${SATO_DIR}"
CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 \
  LD_LIBRARY_PATH="/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/lib/python3.10/site-packages/torch/lib:/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64" \
  "${COMMAND[@]}"
