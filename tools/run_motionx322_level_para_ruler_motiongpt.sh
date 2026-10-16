#!/usr/bin/env bash
# 新尺子（322 paraphrase-aware ruler）下的 Motion-X-322 P1/P2/P3 MotionGPT 家族评测。
# 与 tools/run_motionx322_level_motiongpt.sh 协议完全一致，
# 唯一差异: export T2MGPT_RULER_CHECKPOINT（eval_loader_robustness.build_root_evaluator 读取）。
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 MODEL_KEY LEVEL GPU START_SEED END_SEED [RUN_ROOT]" >&2
  echo "  MODEL_KEY: motiongpt|motiongpt_satopp; LEVEL: P1|P2|P3" >&2
  exit 2
fi

MODEL_KEY="${1,,}"
LEVEL="${2^^}"
GPU="$3"
START_SEED="$4"
END_SEED="$5"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${6:-${MOTIONX_LEVEL_RUN_ROOT:-$ROOT/ablation/motionx322_level_rprecision_para_ruler/runs}}"
ASSET_ROOT="${MOTIONX_LEVEL_ASSET_ROOT:-$ROOT/ablation/motionx322_level_rprecision}"
PYTHON_BIN="${MOTIONGPT_PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/mgpt/bin/python}"
RULER_CHECKPOINT="${MOTIONX_PARA_RULER_CHECKPOINT:-$ROOT/checkpoints/motionx322/text_mot_match_para_aware/model/finest.tar}"

case "$MODEL_KEY" in
  motiongpt|motiongpt_satopp) ;;
  *) echo "unsupported model key: $MODEL_KEY" >&2; exit 2 ;;
esac
case "$LEVEL" in
  P1|P2|P3) ;;
  *) echo "unsupported level: $LEVEL" >&2; exit 2 ;;
esac
if ! [[ "$START_SEED" =~ ^[0-9]+$ && "$END_SEED" =~ ^[0-9]+$ ]] \
    || (( START_SEED < 123 || END_SEED > 142 || START_SEED > END_SEED )); then
  echo "seed range must be an ordered subset of 123..142" >&2
  exit 2
fi

LOWER="${LEVEL,,}"
SOURCE_ROOT="$ASSET_ROOT/motiongpt_sources/$MODEL_KEY/$LOWER"
MANIFEST="$SOURCE_ROOT/manifest.json"
CONFIG_ROOT="$SOURCE_ROOT/configs"
PAIR="$ROOT/dataset/Motion-X-322/test_perb_gemini31_max75_${LOWER}.json"
INDEX="$ASSET_ROOT/index_audit/${LOWER}_sample_index_seed123.json"
OUTPUT_ROOT="$RUN_ROOT/motiongpt_family/$MODEL_KEY/$LOWER"
LOG_ROOT="$OUTPUT_ROOT/logs"

for required in "$PYTHON_BIN" "$MANIFEST" "$PAIR" "$INDEX" "$RULER_CHECKPOINT"; do
  [[ -e "$required" ]] || { echo "missing required asset: $required" >&2; exit 1; }
done

wait_for_available_gpu() {
  local min_free="${MOTIONX_LEVEL_GPU_MIN_FREE_MIB:-12288}"
  local max_checks="${MOTIONX_LEVEL_GPU_IDLE_MAX_CHECKS:-24}"
  local poll_seconds="${MOTIONX_LEVEL_GPU_IDLE_POLL_SECONDS:-10}"
  local attempt gpu_status free_memory
  [[ "${MOTIONX_LEVEL_SKIP_GPU_GUARD:-0}" == "1" ]] && return 0
  for ((attempt = 1; attempt <= max_checks; attempt++)); do
    gpu_status="$(nvidia-smi --id="$GPU" \
      --query-gpu=memory.free --format=csv,noheader,nounits)" || return 1
    free_memory="${gpu_status// /}"
    if [[ "$free_memory" =~ ^[0-9]+$ ]] && (( free_memory >= min_free )); then
      echo "GPU $GPU accepted: free=${free_memory} MiB required=${min_free} MiB"
      return 0
    fi
    if (( attempt == 1 || attempt % 20 == 0 )); then
      echo "GPU $GPU waiting for memory: free=${free_memory:-?} MiB required=${min_free} MiB" >&2
    fi
    if (( attempt == max_checks )); then
      echo "GPU $GPU lacked required free memory: free=${free_memory:-?} MiB required=${min_free} MiB" >&2
      return 1
    fi
    sleep "$poll_seconds"
  done
}

DRY_RUN="${MOTIONX_LEVEL_DRY_RUN:-0}"
if [[ "$DRY_RUN" != "1" ]]; then
  wait_for_available_gpu
  mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT/sample_indices" \
    "$OUTPUT_ROOT/generation_audits" "$OUTPUT_ROOT/retrieval_embeddings" \
    "$OUTPUT_ROOT/audits"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export PYTHONNOUSERSITE=1
  export LD_LIBRARY_PATH=/mnt/workspace/Wilson/miniconda3/envs/mgpt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
  export TMPDIR=/tmp
  export T2MGPT_RULER_CHECKPOINT="$RULER_CHECKPOINT"
fi

for seed in $(seq "$START_SEED" "$END_SEED"); do
  config="$CONFIG_ROOT/${MODEL_KEY}_motionx322_seed${seed}.yaml"
  output="$OUTPUT_ROOT/seed${seed}.json"
  sample_index="$OUTPUT_ROOT/sample_indices/seed${seed}.json"
  generation_audit="$OUTPUT_ROOT/generation_audits/seed${seed}.json"
  retrieval_embedding="$OUTPUT_ROOT/retrieval_embeddings/seed${seed}.npz"
  diagnostics="$OUTPUT_ROOT/audits/seed${seed}_diagnostics.json"
  log="$LOG_ROOT/seed${seed}.log"
  command_record="$OUTPUT_ROOT/command_seed${seed}.txt"
  [[ -f "$config" ]] || { echo "missing seed config: $config" >&2; exit 1; }

  if [[ "$DRY_RUN" != "1" ]]; then
    if [[ -s "$output" && -s "$sample_index" && -s "$generation_audit" \
        && -s "$retrieval_embedding" && -s "$diagnostics" \
        && -s "$command_record" && -s "$log" ]]; then
      echo "skip completed $MODEL_KEY/$LEVEL seed $seed"
      continue
    fi
    if [[ -e "$output" || -e "$generation_audit" || -e "$retrieval_embedding" ]]; then
      echo "incomplete existing artifact set for $MODEL_KEY/$LEVEL seed $seed; refusing overwrite" >&2
      exit 1
    fi
    if [[ -s "$sample_index" ]]; then
      cmp -s "$INDEX" "$sample_index" || {
        echo "existing sample index differs for $MODEL_KEY/$LEVEL seed $seed" >&2
        exit 1
      }
    else
      cp "$INDEX" "$sample_index"
    fi
  fi

  command=(
    "$PYTHON_BIN" MotionGPT/eval_loader_robustness.py
    --manifest "$MANIFEST"
    --config "$config"
    --dataset motionx322
    --expected-samples 7679
    --pair-file "$PAIR"
    --loader-mode caption-locked
    --allowed-levels "$LEVEL"
    --caption-length-policy repair-all
    --seed "$seed"
    --device cuda:0
    --batch-size 32
    --num-workers 0
    --spacy-model en_core_web_sm
    --sample-mode categorical
    --temperature 1
    --top-k 0
    --top-p 1
    --pair-rng-mode sequential
    --sampling-kwargs-mode explicit
    --prompt-mode official-default
    --generation-order original-first
    --length-mode generated
    --expected-sample-index "$sample_index"
    --generation-audit-output "$generation_audit"
    --generation-audit-batches 0
    --retrieval-embedding-output "$retrieval_embedding"
    --diagnostics-output "$diagnostics"
    --output "$output"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU"
    printf ' %q' "${command[@]}"
    printf '\n'
    continue
  fi
  printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU" > "$command_record"
  printf ' %q' "${command[@]}" >> "$command_record"
  printf '\n' >> "$command_record"
  (
    cd "$ROOT"
    "${command[@]}"
  ) 2>&1 | tee "$log"

  for artifact in "$output" "$sample_index" "$generation_audit" \
      "$retrieval_embedding" "$diagnostics" "$command_record" "$log"; do
    if [[ ! -s "$artifact" ]]; then
      echo "seed $seed lacks completed artifact: $artifact" >&2
      exit 1
    fi
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
date -Is > "$OUTPUT_ROOT/complete_gpu${GPU}_seed${START_SEED}_${END_SEED}.txt"