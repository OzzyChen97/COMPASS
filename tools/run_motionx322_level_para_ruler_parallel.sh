#!/usr/bin/env bash
# 新尺子 Motion-X-322 P1/P2/P3 四模型评测并行编排。
#
# 用法: tools/run_motionx322_level_para_ruler_parallel.sh GPU_A GPU_B
#   GPU_A/GPU_B: 两个空闲 GPU；任务轮询分配
#
# 任务矩阵:
#   t2m 家族: 2 models (t2mgpt, satopp) x 3 levels，各 20 种子
#   motiongpt 家族: 2 models (motiongpt, motiongpt_satopp) x 3 levels，
#     各 4 段 x 5 种子
# 完成后聚合。中断后重跑安全（各 runner 有断点续跑逻辑）。
set -euo pipefail

if [[ $# -lt 2 || $# -gt 8 ]]; then
  echo "usage: $0 GPU_A GPU_B [GPU_C ... GPU_H]" >&2
  exit 2
fi

GPUS=("$@")
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RULER="$ROOT/checkpoints/motionx322/text_mot_match_para_aware/model/finest.tar"
REPORT="$ROOT/ablation/ruler_retrain_322_report.md"
ASSET_ROOT="$ROOT/ablation/motionx322_level_rprecision"
RUN_ROOT="$ROOT/ablation/motionx322_level_rprecision_para_ruler/runs"
FINAL_ROOT="$ROOT/ablation/motionx322_level_rprecision_para_ruler/final"
PYTHON_BIN="${T2M_PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
MAX_WAIT_MINUTES="${MOTIONX_MAX_WAIT_MINUTES:-720}"

export MOTIONX_PARA_RULER_CHECKPOINT="$RULER"
export MOTIONX_LEVEL_SKIP_GPU_GUARD=1
export MOTIONX_LEVEL_ASSET_ROOT="$ASSET_ROOT"

wait_for_ruler() {
  local waited=0
  echo "waiting for ruler checkpoint: $RULER"
  while [[ ! -f "$RULER" ]]; do
    if (( waited >= MAX_WAIT_MINUTES * 60 )); then
      echo "timed out waiting for ruler training" >&2
      exit 1
    fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "ruler checkpoint found after $((waited / 60)) min"
}

check_acceptance() {
  [[ -f "$REPORT" ]] || { echo "missing training report: $REPORT" >&2; exit 1; }
  "$PYTHON_BIN" - "$REPORT" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"\*\*best epoch (\d+)\*\*, score=([\d.]+), clean_R1 drop=([-\d.]+)", text)
if not match:
    raise SystemExit("cannot parse acceptance line from training report")
drop = float(match.group(3))
# 322 噪声校准阈值: 同一标准尺子的 clean_R1 基线运行间波动约 2.5 点，
# 低于该噪声底的退化不可分辨（见 ablation/ruler_retrain_322_report.md 决策说明）。
threshold = 0.025
if drop >= threshold:
    raise SystemExit(f"acceptance gate FAILED: val clean R@1 dropped {drop} >= {threshold}")
print(f"acceptance gate PASSED: best epoch {match.group(1)}, "
      f"score {match.group(2)}, clean_R1 drop {drop} < {threshold}")
PY
}

run_t2m() {
  "$ROOT/tools/run_motionx322_level_para_ruler_t2mgpt_satopp.sh" \
    "$1" "$2" "$3" "$4" 20 "$RUN_ROOT" \
    > "$RUN_ROOT/log_t2m_${2}_${1}.out" 2>&1
}

run_motiongpt() {
  "$ROOT/tools/run_motionx322_level_para_ruler_motiongpt.sh" \
    "$1" "$2" "$3" "$4" "$5" "$RUN_ROOT" \
    > "$RUN_ROOT/log_motiongpt_${1}_${2}_${4}_${5}.out" 2>&1
}

wait_for_ruler
check_acceptance

mkdir -p "$RUN_ROOT"

echo "== t2m family (2 models x 3 levels, 20 seeds) =="
pids=()
i=0
for level in P1 P2 P3; do
  for model in t2mgpt satopp; do
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    run_t2m "$level" "$model" "$gpu" 123 &
    pids+=("$!")
    i=$((i + 1))
  done
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" -eq 0 ]] || { echo "t2m family phase failed" >&2; exit 1; }

echo "== motiongpt family (2 models x 3 levels x 4 segments) =="
pids=()
i=0
for level in P1 P2 P3; do
  for model in motiongpt motiongpt_satopp; do
    for seg in "123 127" "128 132" "133 137" "138 142"; do
      read -r s e <<< "$seg"
      gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
      run_motiongpt "$model" "$level" "$gpu" "$s" "$e" &
      pids+=("$!")
      i=$((i + 1))
    done
  done
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" -eq 0 ]] || { echo "motiongpt family phase failed" >&2; exit 1; }

echo "== aggregate =="
"$PYTHON_BIN" "$ROOT/tools/aggregate_motionx322_level_rprecision.py" \
  --run-root "$RUN_ROOT" \
  --asset-manifest "$ASSET_ROOT/assets_manifest_para_ruler.json" \
  --motiongpt-source-root "$ASSET_ROOT/motiongpt_sources" \
  --implementation-overrides-json "$ASSET_ROOT/implementation_overrides_para_ruler.json" \
  --para-ruler-evaluator-checkpoint "$ROOT/checkpoints/motionx322/text_mot_match_para_aware/model/finest.tar" \
  --output-dir "$FINAL_ROOT" \
  --start-seed 123 \
  --num-repeats 20

echo "pipeline complete: $FINAL_ROOT"
date -Is > "$ROOT/ablation/motionx322_level_rprecision_para_ruler/pipeline_complete.txt"