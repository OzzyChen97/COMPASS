#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU START_SEED END_SEED" >&2
  exit 2
fi

GPU="$1"
START_SEED="$2"
END_SEED="$3"
ROOT=/mnt/workspace/Wilson/T2M-GPT
MGPT_ROOT="${ROOT}/MotionGPT"
PYTHON_BIN=/mnt/workspace/Wilson/miniconda3/envs/mgpt/bin/python
EVAL_SCRIPT="${MOTIONGPT_HUMANML_EVAL_SCRIPT:-${MGPT_ROOT}/eval_loader_robustness.py}"
MANIFEST="${MOTIONGPT_HUMANML_MANIFEST:-${MGPT_ROOT}/experiments/robustness/humanml3d_baseline_final/manifest.json}"
CONFIG_ROOT="${MOTIONGPT_HUMANML_CONFIG_ROOT:-${MGPT_ROOT}/experiments/robustness/humanml3d_baseline_final/configs}"
CONFIG_PATTERN="${MOTIONGPT_HUMANML_CONFIG_PATTERN:-motiongpt_baseline_seed%s.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MGPT_ROOT}/experiments/robustness/humanml3d_caption_locked_p123_4184_20seed_20260810}"
PAIR_RNG_MODE="${PAIR_RNG_MODE:-sequential}"
LENGTH_MODE="${LENGTH_MODE:-generated}"
SAMPLING_KWARGS_MODE="${SAMPLING_KWARGS_MODE:-explicit}"
PROMPT_MODE="${PROMPT_MODE:-official-default}"
LOADER_MODE="${LOADER_MODE:-caption-locked}"
CAPTION_LENGTH_POLICY="${CAPTION_LENGTH_POLICY:-strict}"
PAIR_FILE="${PAIR_FILE:-${ROOT}/dataset/HumanML3D/test_perb_deterministic.json}"
ALLOWED_LEVELS="${ALLOWED_LEVELS:-P1 P2 P3}"
FIXED_SAMPLE_INDEX="${FIXED_SAMPLE_INDEX:-}"
GENERATION_AUDIT_INPUT_ROOT="${GENERATION_AUDIT_INPUT_ROOT:-}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RETRIEVAL_EMBEDDING_ROOT="${RETRIEVAL_EMBEDDING_ROOT:-${OUTPUT_ROOT}/retrieval_embeddings}"

mkdir -p \
  "${OUTPUT_ROOT}/logs" \
  "${OUTPUT_ROOT}/sample_indices" \
  "${RETRIEVAL_EMBEDDING_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export LD_LIBRARY_PATH=/mnt/workspace/Wilson/miniconda3/envs/mgpt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export PYTHONNOUSERSITE=1
export TMPDIR=/tmp

for seed in $(seq "${START_SEED}" "${END_SEED}"); do
  printf -v config_name "${CONFIG_PATTERN}" "${seed}"
  config="${CONFIG_ROOT}/${config_name}"
  output="${OUTPUT_ROOT}/seed${seed}.json"
  log="${OUTPUT_ROOT}/logs/seed${seed}.log"
  sample_index="${OUTPUT_ROOT}/sample_indices/seed${seed}.json"
  retrieval_embedding="${RETRIEVAL_EMBEDDING_ROOT}/seed${seed}.npz"
  generation_audit="${OUTPUT_ROOT}/seed${seed}_generation_audit.json"
  diagnostics="${OUTPUT_ROOT}/seed${seed}_diagnostics.json"
  if [[ -n "${FIXED_SAMPLE_INDEX}" ]]; then
    if [[ ! -s "${sample_index}" ]]; then
      cp "${FIXED_SAMPLE_INDEX}" "${sample_index}"
    fi
  fi
  [[ -f "${config}" ]] || { echo "missing seed config: ${config}" >&2; exit 1; }
  if [[ -s "${output}" && -s "${retrieval_embedding}" \
      && -s "${sample_index}" && -s "${generation_audit}" \
      && -s "${diagnostics}" ]]; then
    echo "skip existing seed ${seed}: complete audited artifact set"
    continue
  fi
  echo "start seed ${seed} on physical GPU ${GPU}"
  start=$(date +%s)
  if [[ ! -s "${sample_index}" ]]; then
    "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
      --manifest "${MANIFEST}" \
      --config "${config}" \
      --seed "${seed}" \
      --batch-size 32 \
      --num-workers "${NUM_WORKERS}" \
      --loader-mode "${LOADER_MODE}" \
      --pair-file "${PAIR_FILE}" \
      --allowed-levels ${ALLOWED_LEVELS} \
      --caption-length-policy "${CAPTION_LENGTH_POLICY}" \
      --pair-rng-mode "${PAIR_RNG_MODE}" \
      --sampling-kwargs-mode "${SAMPLING_KWARGS_MODE}" \
      --prompt-mode "${PROMPT_MODE}" \
      --length-mode "${LENGTH_MODE}" \
      --index-only \
      --sample-index-output "${sample_index}" \
      --output "${OUTPUT_ROOT}/seed${seed}_index_audit.json"
  fi
  generation_audit_args=(
    --generation-audit-output "${generation_audit}"
    --generation-audit-batches 0
  )
  if [[ -n "${GENERATION_AUDIT_INPUT_ROOT}" ]]; then
    generation_audit_input="${GENERATION_AUDIT_INPUT_ROOT}/seed${seed}_generation_audit.json"
    [[ -s "${generation_audit_input}" ]] || {
      echo "missing generation replay audit: ${generation_audit_input}" >&2
      exit 1
    }
    generation_audit_args=(--generation-audit-input "${generation_audit_input}")
  fi
  "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --manifest "${MANIFEST}" \
    --config "${config}" \
    --seed "${seed}" \
    --device cuda:0 \
    --batch-size 32 \
    --num-workers "${NUM_WORKERS}" \
    --loader-mode "${LOADER_MODE}" \
    --pair-file "${PAIR_FILE}" \
    --allowed-levels ${ALLOWED_LEVELS} \
    --caption-length-policy "${CAPTION_LENGTH_POLICY}" \
    --pair-rng-mode "${PAIR_RNG_MODE}" \
    --sampling-kwargs-mode "${SAMPLING_KWARGS_MODE}" \
    --prompt-mode "${PROMPT_MODE}" \
    --generation-order original-first \
    --length-mode "${LENGTH_MODE}" \
    --expected-sample-index "${sample_index}" \
    "${generation_audit_args[@]}" \
    --retrieval-embedding-output "${retrieval_embedding}" \
    --diagnostics-output "${diagnostics}" \
    --output "${output}" 2>&1 | tee "${log}"
  end=$(date +%s)
  echo "done seed ${seed} wall_seconds=$((end-start))"
done
