#!/usr/bin/env bash
# 新尺子（paraphrase-aware ruler）HumanML3D P1/P2/P3 全流程编排：
#   1) 等待尺子训练完成并通过验收门
#   2) 依次跑 T2M-GPT / SATO++ / MotionGPT 的 P1/P2/P3 评测（20 种子）
#   3) 聚合生成新表
#
# 用法:
#   tools/run_para_ruler_humanml_pipeline.sh GPU [--skip-motiongpt]
#   GPU: 评测任务使用的空闲 GPU（尺子训练本身不占太多资源）
#
# 环境变量:
#   HUMANML_LEVEL_SKIP_GPU_GUARD=1    跳过 GPU 空闲检查
#   HUMANML_PARA_RULER_CHECKPOINT     自定义尺子 checkpoint 路径
#   HUMANML_MAX_WAIT_MINUTES          等待训练完成的最大分钟数（默认 360）
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 GPU [--skip-motiongpt]" >&2
  exit 2
fi

GPU="$1"
SKIP_MOTIONGPT=0
if [[ $# -eq 2 && "$2" == "--skip-motiongpt" ]]; then
  SKIP_MOTIONGPT=1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RULER="$ROOT/checkpoints/t2m/text_mot_match_para_aware/model/finest.tar"
REPORT="$ROOT/ablation/ruler_retrain_report.md"
RUN_ROOT="$ROOT/ablation/humanml3d_table1_levels_para_ruler/runs"
FINAL_ROOT="$ROOT/ablation/humanml3d_table1_levels_para_ruler/final"
MAX_WAIT_MINUTES="${HUMANML_MAX_WAIT_MINUTES:-360}"
PYTHON_BIN="${T2MGPT_PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"

export HUMANML_PARA_RULER_CHECKPOINT="$RULER"

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
if drop >= 0.01:
    raise SystemExit(f"acceptance gate FAILED: val clean R@1 dropped {drop} >= 0.01")
print(f"acceptance gate PASSED: best epoch {match.group(1)}, "
      f"score {match.group(2)}, clean_R1 drop {drop}")
PY
}

run_t2m_satopp() {
  local level="$1" model="$2" start_seed="$3" repeats="$4"
  "$ROOT/tools/run_humanml3d_table1_level_para_ruler_t2mgpt_satopp.sh" \
    "$level" "$model" "$GPU" "$start_seed" "$repeats"
}

run_motiongpt() {
  local level="$1" start_seed="$2" end_seed="$3"
  "$ROOT/tools/run_humanml3d_table1_level_para_ruler_motiongpt.sh" \
    "$level" "$GPU" "$start_seed" "$end_seed"
}

aggregate() {
  "$PYTHON_BIN" "$ROOT/tools/aggregate_humanml3d_table1_level_rprecision.py" \
    --run-root "$RUN_ROOT" \
    --asset-manifest "$ROOT/ablation/humanml3d_table1_levels/assets_manifest.json" \
    --motiongpt-source-root "$ROOT/ablation/humanml3d_table1_levels/motiongpt_sources" \
    --output-dir "$FINAL_ROOT" \
    --start-seed 123 \
    --num-repeats 20
}

wait_for_ruler
check_acceptance

mkdir -p "$RUN_ROOT"

echo "== T2M-GPT / SATO++ P1/P2/P3 =="
for level in P1 P2 P3; do
  run_t2m_satopp "$level" t2mgpt 123 20
  run_t2m_satopp "$level" satopp 123 20
done

if (( SKIP_MOTIONGPT == 0 )); then
  echo "== MotionGPT P1/P2/P3 =="
  for level in P1 P2 P3; do
    run_motiongpt "$level" 123 127
    run_motiongpt "$level" 128 132
    run_motiongpt "$level" 133 137
    run_motiongpt "$level" 138 142
  done
fi

echo "== aggregate =="
aggregate
echo "pipeline complete: $FINAL_ROOT"
date -Is > "$ROOT/ablation/humanml3d_table1_levels_para_ruler/pipeline_complete.txt"