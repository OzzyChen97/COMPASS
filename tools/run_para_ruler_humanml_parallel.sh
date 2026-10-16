#!/usr/bin/env bash
# 新尺子 HumanML3D P1/P2/P3 评测并行编排（双卡）。
#
# 用法: tools/run_para_ruler_humanml_parallel.sh GPU_A GPU_B
#   GPU_A/GPU_B: 两个空闲 GPU；任务轮询分配
#
# 任务矩阵:
#   t2m/satopp: 2 models x 3 levels，每个 20 种子（约 100 分钟/任务）
#   motiongpt:  3 levels x 4 段（各 5 种子，约 25 分钟/任务）
# 完成后聚合。中断后重跑安全（driver 按 runs.csv 续跑已完成种子）。
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_A GPU_B" >&2
  exit 2
fi

GPU_A="$1"
GPU_B="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RULER="$ROOT/checkpoints/t2m/text_mot_match_para_aware/model/finest.tar"
REPORT="$ROOT/ablation/ruler_retrain_report.md"
RUN_ROOT="$ROOT/ablation/humanml3d_table1_levels_para_ruler/runs"
FINAL_ROOT="$ROOT/ablation/humanml3d_table1_levels_para_ruler/final"
PYTHON_BIN="${T2MGPT_PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"

export HUMANML_PARA_RULER_CHECKPOINT="$RULER"
export HUMANML_LEVEL_SKIP_GPU_GUARD=1

[[ -f "$RULER" ]] || { echo "ruler checkpoint missing: $RULER" >&2; exit 1; }
[[ -f "$REPORT" ]] || { echo "training report missing: $REPORT" >&2; exit 1; }

check_acceptance() {
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

run_job() {
  local gpu="$1" kind="$2" level="$3" arg1="$4" arg2="$5"
  if [[ "$kind" == "t2m" ]]; then
    "$ROOT/tools/run_humanml3d_table1_level_para_ruler_t2mgpt_satopp.sh" \
      "$level" "$arg1" "$gpu" "$arg2" 20 \
      > "$RUN_ROOT/log_${kind}_${arg1}_${level}.out" 2>&1
  else
    "$ROOT/tools/run_humanml3d_table1_level_para_ruler_motiongpt.sh" \
      "$level" "$gpu" "$arg1" "$arg2" \
      > "$RUN_ROOT/log_${kind}_${level}_${arg1}_${arg2}.out" 2>&1
  fi
}

check_acceptance

mkdir -p "$RUN_ROOT"

echo "== t2m/satopp (2 models x 3 levels, 20 seeds) =="
pids=()
i=0
for level in P1 P2 P3; do
  for model in t2mgpt satopp; do
    if (( i % 2 == 0 )); then gpu="$GPU_A"; else gpu="$GPU_B"; fi
    run_job "$gpu" t2m "$level" "$model" 123 \
      > "$RUN_ROOT/log_t2m_${model}_${level}.out" 2>&1 &
    pids+=("$!")
    i=$((i + 1))
  done
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" -eq 0 ]] || { echo "t2m/satopp phase failed" >&2; exit 1; }

echo "== motiongpt (3 levels x 4 segments, 5 seeds each) =="
pids=()
i=0
for level in P1 P2 P3; do
  for seg in "123 127" "128 132" "133 137" "138 142"; do
    read -r s e <<< "$seg"
    if (( i % 2 == 0 )); then gpu="$GPU_A"; else gpu="$GPU_B"; fi
    run_job "$gpu" motiongpt "$level" "$s" "$e" \
      > "$RUN_ROOT/log_motiongpt_${level}_${s}_${e}.out" 2>&1 &
    pids+=("$!")
    i=$((i + 1))
  done
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" -eq 0 ]] || { echo "motiongpt phase failed" >&2; exit 1; }

echo "== aggregate =="
"$PYTHON_BIN" "$ROOT/tools/aggregate_humanml3d_table1_level_rprecision.py" \
  --run-root "$RUN_ROOT" \
  --asset-manifest "$ROOT/ablation/humanml3d_table1_levels/assets_manifest.json" \
  --motiongpt-source-root "$ROOT/ablation/humanml3d_table1_levels/motiongpt_sources" \
  --output-dir "$FINAL_ROOT" \
  --start-seed 123 \
  --num-repeats 20

echo "pipeline complete: $FINAL_ROOT"
date -Is > "$ROOT/ablation/humanml3d_table1_levels_para_ruler/pipeline_complete.txt"