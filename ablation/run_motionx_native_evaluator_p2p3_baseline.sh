#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
SPEC="ablation/motionx_t2mgpt_sato/native_evaluator_p2p3_baseline_spec.json"
PAIR="dataset/Motion-X/test_perb_deterministic_p2p3.json"
OUT_ROOT="ablation/motionx_t2mgpt_sato/native_evaluator_p2p3_baseline_repeat20"
SHARD_ROOT="${OUT_ROOT}/shards"

mkdir -p "$SHARD_ROOT"
printf '%s\n' "$$" > "${OUT_ROOT}/pipeline.pid"
date -Is > "${OUT_ROOT}/started_at.txt"

pids=()
for gpu in 0 1 2 3; do
  start_seed=$((123 + gpu * 5))
  shard="${SHARD_ROOT}/gpu${gpu}"
  mkdir -p "$shard"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    ablation/repeat20_perturb_fast_batch.py \
    --spec-json "$SPEC" \
    --pair-data-path "$PAIR" \
    --out-dir "$shard" \
    --report-md "${shard}.md" \
    --start-seed "$start_seed" \
    --num-repeats 5 \
    --batch-size 32 \
    --humanml-fast-protocol \
    --cfg-scale 1.3 \
    --cfg-uncond zero \
    --sample-mode categorical \
    --wandb-mode disabled \
    > "${shard}_driver.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  date -Is > "${OUT_ROOT}/failed_at.txt"
  exit 1
fi

"$PYTHON" tools/merge_repeat_eval_shards.py \
  --spec-json "$SPEC" \
  --shard-root "$SHARD_ROOT" \
  --out-dir "${OUT_ROOT}/combined" \
  --pair-data-path "$PAIR" \
  --report-md "${OUT_ROOT}/MOTIONX_NATIVE_EVALUATOR_P2P3_BASELINE_REPEAT20.md" \
  --start-seed 123 \
  --num-repeats 20 \
  --cfg-scale 1.3 \
  > "${OUT_ROOT}/merge.log" 2>&1

date -Is > "${OUT_ROOT}/complete.txt"
