#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
PAIR="${MOTIONX322_TABLE3_PAIR:-dataset/Motion-X-322/test_perb_gemini31_max75_mixed_full.json}"
VQ_CKPT="${MOTIONX322_VQ_CKPT:-output_motionx322/VQVAE_MOTIONX322/net_best_fid.pth}"
BASELINE_CKPT="${MOTIONX322_BASELINE_CKPT:-output_motionx322/GPT_MOTIONX322_MAXTOK75_200K/net_best_fid.pth}"
SATO_CKPT="${MOTIONX322_SATO_CKPT:-ablation/motionx322_sato/max_token75_gemini31_human_recipe_100k_lr60k_bs132_seed123/sato_pp_iter_100000.pth}"
OUT_ROOT="${MOTIONX322_TABLE3_OUT_ROOT:-ablation/motionx322_sato/gemini31_100k_table3_7733_repeat20}"
SHARD_ROOT="$OUT_ROOT/shards"
COMBINED_ROOT="$OUT_ROOT/combined"
SPEC="$OUT_ROOT/table3_spec.json"
MANIFEST="$OUT_ROOT/motionx_table3_protocol_manifest.json"
REPORT="$OUT_ROOT/MOTIONX322_TABLE3_7733_REPEAT20.md"
GPU_LIST_STRING="${MOTIONX322_TABLE3_GPUS:-4 5 6 7}"
read -r -a GPU_LIST <<< "$GPU_LIST_STRING"

if [[ "${#GPU_LIST[@]}" -ne 4 ]]; then
  echo "MOTIONX322_TABLE3_GPUS must contain exactly four GPU ids" >&2
  exit 1
fi

for path in "$PAIR" "$VQ_CKPT" "$BASELINE_CKPT" "$SATO_CKPT"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

mkdir -p "$OUT_ROOT" "$SHARD_ROOT" "$COMBINED_ROOT"
printf '%s\n' "$$" > "$OUT_ROOT/pipeline.pid"
date -Is > "$OUT_ROOT/started_at.txt"

"$PYTHON" tools/write_sato_eval_spec.py \
  --dataname motionx322_75 \
  --vq-checkpoint "$VQ_CKPT" \
  --baseline-checkpoint "$BASELINE_CKPT" \
  --sato-checkpoint "$SATO_CKPT" \
  --baseline-name motionx322_t2mgpt_baseline_200k_best \
  --sato-name motionx322_gemini31_sato100k \
  --block-size 76 \
  --output "$SPEC"

pids=()
for index in 0 1 2 3; do
  gpu="${GPU_LIST[$index]}"
  start_seed=$((123 + index * 5))
  shard="$SHARD_ROOT/gpu${gpu}"
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
  date -Is > "$OUT_ROOT/failed_at.txt"
  exit 1
fi

"$PYTHON" tools/merge_repeat_eval_shards.py \
  --spec-json "$SPEC" \
  --shard-root "$SHARD_ROOT" \
  --out-dir "$COMBINED_ROOT" \
  --pair-data-path "$PAIR" \
  --report-md "$REPORT" \
  --start-seed 123 \
  --num-repeats 20 \
  --expected-matched-pairs 7733 \
  --expected-total-pairs 7733 \
  --cfg-scale 1.3 > "$OUT_ROOT/merge.log" 2>&1

"$PYTHON" tools/build_motionx_table3_manifest.py \
  --pair-file "$PAIR" \
  --summary-json "$COMBINED_ROOT/summary.json" \
  --spec-json "$SPEC" \
  --output "$MANIFEST" \
  --expected-samples 7733 \
  --cfg-scale 1.3 \
  --cfg-uncond zero > "$OUT_ROOT/manifest.log" 2>&1

date -Is > "$OUT_ROOT/complete.txt"
