#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
PAIR="${MOTIONX322_TABLE3_PAIR:-dataset/Motion-X-322/test_perb_gemini31_max75_mixed_full.json}"
VQ_CKPT="${MOTIONX322_VQ_CKPT:-output_motionx322/VQVAE_MOTIONX322/net_best_fid.pth}"
TRAIN_DIR="${MOTIONX322_SATO_TRAIN_DIR:-output_motionx322_sato/motionx322_sato_official_50k_seed123}"
TRANS_CKPT="${MOTIONX322_SATO_TRANS_CKPT:-${TRAIN_DIR}/net_best_fid.pth}"
CLIP_CKPT="${MOTIONX322_SATO_CLIP_CKPT:-${TRAIN_DIR}/clip_best.pth}"
OUT_ROOT="${MOTIONX322_SATO_TABLE3_OUT_ROOT:-output_motionx322_sato/table3_7733_repeat20_cfg1p3}"
MODEL_NAME="${MOTIONX322_SATO_MODEL_NAME:-motionx322_sato_official_50k_best_fid}"
MODEL_DESCRIPTION="${MOTIONX322_SATO_MODEL_DESCRIPTION:-Original SATO reproduced on Motion-X with a 50k budget}"
SPEC="${MOTIONX322_SATO_SPEC:-$OUT_ROOT/sato50k_spec.json}"
SHARD_ROOT="$OUT_ROOT/shards"
COMBINED_ROOT="$OUT_ROOT/combined"
REPORT="${MOTIONX322_SATO_REPORT:-$OUT_ROOT/MOTIONX322_SATO50K_TABLE3_REPEAT20.md}"
MANIFEST="$OUT_ROOT/protocol_manifest.json"
GPU_LIST_STRING="${MOTIONX322_TABLE3_GPUS:-4 5 6 7}"
read -r -a GPU_LIST <<< "$GPU_LIST_STRING"

if [[ "${#GPU_LIST[@]}" -lt 1 || "${#GPU_LIST[@]}" -gt 4 ]]; then
  echo "MOTIONX322_TABLE3_GPUS must contain between one and four GPU ids" >&2
  exit 1
fi
for path in "$PAIR" "$VQ_CKPT" "$TRANS_CKPT" "$CLIP_CKPT"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

mkdir -p "$OUT_ROOT" "$SHARD_ROOT" "$COMBINED_ROOT"
printf '%s\n' "$$" > "$OUT_ROOT/pipeline.pid"
date -Is > "$OUT_ROOT/started_at.txt"

"$PYTHON" tools/write_external_sato_eval_spec.py \
  --dataname motionx322_75 \
  --name "$MODEL_NAME" \
  --description "$MODEL_DESCRIPTION" \
  --vq-checkpoint "$VQ_CKPT" \
  --transformer-checkpoint "$TRANS_CKPT" \
  --clip-checkpoint "$CLIP_CKPT" \
  --block-size 76 \
  --output "$SPEC"

pids=()
gpu_count="${#GPU_LIST[@]}"
base_count=$((20 / gpu_count))
extra_count=$((20 % gpu_count))
next_seed=123
for ((index=0; index<gpu_count; index++)); do
  gpu="${GPU_LIST[$index]}"
  repeat_count="$base_count"
  if (( index < extra_count )); then
    repeat_count=$((repeat_count + 1))
  fi
  start_seed="$next_seed"
  next_seed=$((next_seed + repeat_count))
  shard="$SHARD_ROOT/gpu${gpu}"
  mkdir -p "$shard"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 \
    LD_LIBRARY_PATH="/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/lib/python3.10/site-packages/torch/lib:/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64" \
    "$PYTHON" ablation/repeat20_perturb_fast_batch.py \
      --spec-json "$SPEC" \
      --pair-data-path "$PAIR" \
      --out-dir "$shard" \
      --report-md "${shard}.md" \
      --start-seed "$start_seed" \
      --num-repeats "$repeat_count" \
      --batch-size 32 \
      --humanml-fast-protocol \
      --cfg-scale 1.3 \
      --cfg-uncond zero \
      --sample-mode categorical \
      --sample-temperature 1 \
      --top-k 0 \
      --top-p 1 \
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
  --expected-samples 7733 > "$OUT_ROOT/manifest.log" 2>&1

date -Is > "$OUT_ROOT/complete.txt"
