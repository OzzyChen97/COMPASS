#!/usr/bin/env bash
# 新尺子（322 paraphrase-aware ruler）下的 Motion-X-322 P1/P2/P3 T2M-GPT/SATO++ 评测。
# 与 tools/run_motionx322_level_t2mgpt_satopp.sh 协议完全一致，
# 唯一差异: 给 repeat20 driver 传 --evaluator-checkpoint（换检索尺子）。
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 LEVEL MODEL GPU START_SEED NUM_REPEATS [RUN_ROOT]" >&2
  echo "  LEVEL: P1|P2|P3; MODEL: t2mgpt|satopp" >&2
  exit 2
fi

LEVEL="${1^^}"
MODEL_ALIAS="${2,,}"
GPU="$3"
START_SEED="$4"
NUM_REPEATS="$5"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${6:-${MOTIONX_LEVEL_RUN_ROOT:-$ROOT/ablation/motionx322_level_rprecision_para_ruler/runs}}"
ASSET_ROOT="${MOTIONX_LEVEL_ASSET_ROOT:-$ROOT/ablation/motionx322_level_rprecision}"
PYTHON_BIN="${T2MGPT_PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
SPEC="$ROOT/ablation/motionx322_sato/gemini31_100k_full_test_human_protocol_cfg1p3_repeat20/baseline_vs_sato100k_spec.json"
RULER_CHECKPOINT="${MOTIONX_PARA_RULER_CHECKPOINT:-$ROOT/checkpoints/motionx322/text_mot_match_para_aware/model/finest.tar}"

case "$LEVEL" in
  P1|P2|P3) ;;
  *) echo "unsupported level: $LEVEL" >&2; exit 2 ;;
esac
case "$MODEL_ALIAS" in
  t2mgpt) MODEL_NAME="motionx322_t2mgpt_baseline_200k_best" ;;
  satopp) MODEL_NAME="motionx322_gemini31_sato100k" ;;
  *) echo "unsupported model: $MODEL_ALIAS" >&2; exit 2 ;;
esac
if ! [[ "$START_SEED" =~ ^[0-9]+$ && "$NUM_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "START_SEED and NUM_REPEATS must be positive integers" >&2
  exit 2
fi
END_SEED=$((START_SEED + NUM_REPEATS - 1))
if (( START_SEED < 123 || END_SEED > 142 )); then
  echo "formal seeds must stay within 123..142, got $START_SEED..$END_SEED" >&2
  exit 2
fi

LOWER="${LEVEL,,}"
PAIR="$ROOT/dataset/Motion-X-322/test_perb_gemini31_max75_${LOWER}.json"
INDEX="$ASSET_ROOT/index_audit/${LOWER}_sample_index_seed123.json"
GROUP_ROOT="$RUN_ROOT/t2m_family/$LOWER/$MODEL_ALIAS"
RETRIEVAL_EMBEDDINGS="$GROUP_ROOT/retrieval_embeddings"
GENERATION_AUDITS="$GROUP_ROOT/generation_audits"
REPORT="$GROUP_ROOT/report.md"
LOG="$GROUP_ROOT/driver_seed${START_SEED}_${END_SEED}.log"
COMMAND_RECORD="$GROUP_ROOT/command_seed${START_SEED}_${END_SEED}.txt"

for required in "$PYTHON_BIN" "$SPEC" "$PAIR" "$INDEX" "$RULER_CHECKPOINT"; do
  [[ -e "$required" ]] || { echo "missing required asset: $required" >&2; exit 1; }
done

wait_for_available_gpu() {
  local min_free="${MOTIONX_LEVEL_GPU_MIN_FREE_MIB:-8192}"
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

command=(
  "$PYTHON_BIN" ablation/repeat20_perturb_fast_batch.py
  --spec-json "$SPEC"
  --model "$MODEL_NAME"
  --pair-data-path "$PAIR"
  --expected-sample-index "$INDEX"
  --allowed-levels "$LEVEL"
  --evaluator-checkpoint "$RULER_CHECKPOINT"
  --out-dir "$GROUP_ROOT"
  --report-md "$REPORT"
  --start-seed "$START_SEED"
  --num-repeats "$NUM_REPEATS"
  --batch-size 32
  --num-workers 0
  --dataset-split test
  --spacy-model en_core_web_sm
  --caption-locked
  --caption-locked-length-policy repair-all
  --fast-batch
  --sampling-pair-rng-mode sequential
  --cfg-scale 1.3
  --cfg-uncond zero
  --sample-mode categorical
  --sample-temperature 1
  --top-k 0
  --top-p 1
  --retrieval-embedding-dir "$RETRIEVAL_EMBEDDINGS"
  --generation-audit-dir "$GENERATION_AUDITS"
  --wandb-mode disabled
)
if [[ "${MOTIONX_LEVEL_DRY_RUN:-0}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU"
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

wait_for_available_gpu
mkdir -p "$GROUP_ROOT" "$RETRIEVAL_EMBEDDINGS" "$GENERATION_AUDITS"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONNOUSERSITE=1

printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU" > "$COMMAND_RECORD"
printf ' %q' "${command[@]}" >> "$COMMAND_RECORD"
printf '\n' >> "$COMMAND_RECORD"
"${command[@]}" 2>&1 | tee "$LOG"

"$PYTHON_BIN" - \
  "$GROUP_ROOT/$MODEL_NAME/runs.csv" \
  "$GROUP_ROOT/$MODEL_NAME/sample_indices" \
  "$RETRIEVAL_EMBEDDINGS/$MODEL_NAME" \
  "$GENERATION_AUDITS/$MODEL_NAME" \
  "$START_SEED" "$END_SEED" <<'PY'
import csv
import sys
from pathlib import Path

runs_path, index_root, retrieval_root, audit_root = map(Path, sys.argv[1:5])
start, end = map(int, sys.argv[5:7])
with runs_path.open(newline="", encoding="utf-8") as handle:
    seeds = {int(row["seed"]) for row in csv.DictReader(handle)}
missing = sorted(set(range(start, end + 1)) - seeds)
if missing:
    raise SystemExit(f"missing completed seeds in {runs_path}: {missing}")
for seed in range(start, end + 1):
    required = (
        index_root / f"seed_{seed}.json",
        retrieval_root / f"seed{seed}.npz",
        audit_root / f"seed{seed}.json",
    )
    absent = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if absent:
        raise SystemExit(f"seed {seed} lacks audited artifacts: {absent}")
PY

for artifact in "$COMMAND_RECORD" "$LOG" "$REPORT" \
    "$GROUP_ROOT/protocol_manifest.json"; do
  if [[ ! -s "$artifact" ]]; then
    echo "completed seed range lacks required artifact: $artifact" >&2
    exit 1
  fi
done

date -Is > "$GROUP_ROOT/complete_seed${START_SEED}_${END_SEED}.txt"