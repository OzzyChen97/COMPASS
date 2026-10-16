#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[$(date '\''+%F %T'\'')] ERROR line=$LINENO exit=$?" >&2' ERR
if [[ "${TRACE:-0}" == "1" ]]; then
  set -x
fi

ROOT="${ROOT:-/mnt/workspace/Wilson/T2M-GPT}"
PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
GPU_ID="${GPU_ID:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
PAIR_MANIFEST="${PAIR_MANIFEST:-ablation/vlm_judge/pairs_20_each.jsonl}"
VISUAL_DIR="${VISUAL_DIR:-ablation/vlm_judge/visuals_20_each}"
JUDGE_OUT="${JUDGE_OUT:-ablation/vlm_judge/qwen35vl_results.jsonl}"
JUDGE_SUMMARY="${JUDGE_SUMMARY:-ablation/vlm_judge/qwen35vl_summary.csv}"
LOG_DIR="${LOG_DIR:-ablation/vlm_judge/logs}"
RUN_JUDGE="${RUN_JUDGE:-auto}"

cd "$ROOT"
mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

echo "[$(date '+%F %T')] build pair manifest"
"$PYTHON" tools/build_vlm_pair_manifest.py \
  --out "$PAIR_MANIFEST" \
  --per-type 20 \
  --seed 123

echo "[$(date '+%F %T')] generate visuals with $PYTHON on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
"$PYTHON" tools/generate_vlm_eval_visuals.py \
  --pair-manifest "$PAIR_MANIFEST" \
  --out-dir "$VISUAL_DIR" \
  --models t2mgpt t2mgpt_aug sato_pp \
  --gpu-id 0 \
  --resume

echo "[$(date '+%F %T')] visual generation finished"

if [[ "$RUN_JUDGE" == "0" || "$RUN_JUDGE" == "false" ]]; then
  echo "[$(date '+%F %T')] RUN_JUDGE=$RUN_JUDGE, skip VLM judge"
  exit 0
fi

if [[ -z "${QWEN_VLM_BASE_URL:-}" ]]; then
  echo "[$(date '+%F %T')] QWEN_VLM_BASE_URL is not set, skip VLM judge"
  exit 0
fi

if ! "$PYTHON" - <<'PY'
import os
import urllib.request
url = os.environ["QWEN_VLM_BASE_URL"].rstrip("/")
health = url[:-3] + "/health" if url.endswith("/v1") else url.rstrip("/") + "/health"
try:
    urllib.request.urlopen(health, timeout=5).read()
except Exception:
    raise SystemExit(1)
PY
then
  echo "[$(date '+%F %T')] QWEN_VLM_BASE_URL=$QWEN_VLM_BASE_URL is not healthy, skip VLM judge"
  exit 0
fi

echo "[$(date '+%F %T')] run Qwen3.5-VL judge"
"$PYTHON" tools/judge_vlm_visuals_qwen.py \
  --visual-manifest "$VISUAL_DIR/manifest.jsonl" \
  --out "$JUDGE_OUT" \
  --summary "$JUDGE_SUMMARY" \
  --resume

echo "[$(date '+%F %T')] all done"
