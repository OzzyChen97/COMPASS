#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/workspace/Wilson/T2M-GPT"
cd "$ROOT"
PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
PAIR="${MOTIONX322_TABLE3_PAIR:-dataset/Motion-X-322/test_perb_gemini31_max75_mixed_full.json}"
VQ_CKPT="${MOTIONX322_VQ_CKPT:-output_motionx322/VQVAE_MOTIONX322/net_best_fid.pth}"
TRAIN_DIR="${MOTIONX322_SATO_TRAIN_DIR:-output_motionx322_sato/motionx322_sato_token_aligned_earlystop_pgd_mild_v3_4gpu3457_seed123}"
OUT_ROOT="${MOTIONX322_SATO_30K45K_OUT_ROOT:-output_motionx322_sato/table3_7733_repeat20_cfg1p3_token_aligned_iter30k45k_20260828}"
EXTRACT_DIR="$OUT_ROOT/extracted_checkpoints"
SPEC="$OUT_ROOT/sato_token_aligned_iter30k45k_spec.json"
SHARD_ROOT="$OUT_ROOT/shards"
COMBINED_ROOT="$OUT_ROOT/combined"
REPORT="$OUT_ROOT/MOTIONX322_SATO_TOKEN_ALIGNED_ITER30K45K_TABLE3_REPEAT20.md"
MANIFEST="$OUT_ROOT/protocol_manifest.json"
STATE="$OUT_ROOT/state.tsv"
GPU_LIST_STRING="${MOTIONX322_TABLE3_GPUS:-1 2 3 4 7}"
read -r -a GPU_LIST <<< "$GPU_LIST_STRING"

record() {
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" | tee -a "$STATE"
}

on_error() {
  local code=$?
  record "failed_exit_${code}"
  date -Is > "$OUT_ROOT/failed_at.txt"
  exit "$code"
}
trap on_error ERR

if [[ "${#GPU_LIST[@]}" -lt 1 || "${#GPU_LIST[@]}" -gt 5 ]]; then
  echo "MOTIONX322_TABLE3_GPUS must contain between one and five GPU ids" >&2
  exit 1
fi
for path in "$PAIR" "$VQ_CKPT" "$TRAIN_DIR/train_state_iter_30000.pth" "$TRAIN_DIR/train_state_iter_45000.pth"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
if [[ -e "$OUT_ROOT/complete.txt" ]]; then
  echo "Output is already complete: $OUT_ROOT" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$EXTRACT_DIR" "$SHARD_ROOT" "$COMBINED_ROOT"
printf '%s\n' "$$" > "$OUT_ROOT/pipeline.pid"
date -Is > "$OUT_ROOT/started_at.txt"
record "start_token_aligned_sato_iter30k45k_table3_repeat20"

record "extract_iter_checkpoints"
PYTHONNOUSERSITE=1 "$PYTHON" - "$TRAIN_DIR" "$EXTRACT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import torch

train_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

items = []
for step in (30000, 45000):
    source = train_dir / f"train_state_iter_{step}.pth"
    checkpoint = torch.load(source, map_location="cpu", mmap=True)
    if int(checkpoint.get("iter", -1)) != step:
        raise ValueError(f"Unexpected iter in {source}: {checkpoint.get('iter')}")
    net_path = out_dir / f"net_iter_{step}.pth"
    clip_path = out_dir / f"clip_iter_{step}.pth"
    torch.save({"trans": checkpoint["trans"]}, net_path)
    torch.save(checkpoint["clip"], clip_path)
    items.append({
        "step": step,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "net": str(net_path),
        "net_sha256": sha256_file(net_path),
        "clip": str(clip_path),
        "clip_sha256": sha256_file(clip_path),
        "best_fid_recorded_in_state": float(checkpoint.get("best_fid", float("nan"))),
        "best_iter_recorded_in_state": int(checkpoint.get("best_iter", -1)),
    })
(out_dir / "extraction_manifest.json").write_text(
    json.dumps(items, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(out_dir / "extraction_manifest.json")
PY

record "write_eval_spec"
"$PYTHON" - "$SPEC" "$VQ_CKPT" "$EXTRACT_DIR" <<'PY'
import json
import sys
from pathlib import Path

spec_path = Path(sys.argv[1])
vq_checkpoint = sys.argv[2]
extract_dir = Path(sys.argv[3])
spec = []
for step in (30000, 45000):
    spec.append({
        "name": f"motionx322_sato_token_aligned_pgd_iter{step}",
        "kind": "baseline",
        "dataname": "motionx322_75",
        "resume_pth": vq_checkpoint,
        "resume_trans": str(extract_dir / f"net_iter_{step}.pth"),
        "clip_pth": str(extract_dir / f"clip_iter_{step}.pth"),
        "args": {"down_t": 2, "block_size": 76},
        "description": (
            "Token-aligned lexical SATO with mild PGD; exact periodic "
            f"training checkpoint at iter {step}"
        ),
    })
spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
print(spec_path)
PY

record "launch_shards_gpus_${GPU_LIST_STRING// /_}"
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
  record "launched_gpu${gpu}_seed${start_seed}_count${repeat_count}_pid${pids[-1]}"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  record "shards_failed"
  exit 1
fi
record "shards_complete"

record "merge_shards"
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

record "build_manifest"
"$PYTHON" tools/build_motionx_table3_manifest.py \
  --pair-file "$PAIR" \
  --summary-json "$COMBINED_ROOT/summary.json" \
  --spec-json "$SPEC" \
  --output "$MANIFEST" \
  --expected-samples 7733 \
  --cfg-scale 1.3 \
  --cfg-uncond zero > "$OUT_ROOT/manifest.log" 2>&1

date -Is > "$OUT_ROOT/complete.txt"
record "complete"