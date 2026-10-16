#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/workspace/Wilson/T2M-GPT"
cd "$ROOT"
PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
PAIR="${MOTIONX322_TABLE3_PAIR:-dataset/Motion-X-322/test_perb_gemini31_max75_mixed_full.json}"
VQ_CKPT="${MOTIONX322_VQ_CKPT:-output_motionx322/VQVAE_MOTIONX322/net_best_fid.pth}"
TRAIN_DIR="${MOTIONX322_SATO_TRAIN_DIR:-output_motionx322_sato/motionx322_sato_official_50k_seed123}"
OUT_ROOT="${MOTIONX322_SATO_RETEST_OUT_ROOT:-output_motionx322_sato/table3_7733_repeat20_cfg1p3_sato_iter30k_40k_20260823}"
EXTRACT_DIR="$OUT_ROOT/extracted_checkpoints"
SPEC="$OUT_ROOT/sato_iter30k40k_spec.json"
SHARD_ROOT="$OUT_ROOT/shards"
COMBINED_ROOT="$OUT_ROOT/combined"
REPORT="$OUT_ROOT/MOTIONX322_SATO_ITER30K40K_TABLE3_REPEAT20.md"
MANIFEST="$OUT_ROOT/protocol_manifest.json"
STATE="$OUT_ROOT/state.tsv"
SUMMARY_MD="$OUT_ROOT/summary_iter30k40k.md"
GPU_LIST_STRING="${MOTIONX322_TABLE3_GPUS:-1 2 3 4}"
read -r -a GPU_LIST <<< "$GPU_LIST_STRING"

record() {
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" | tee -a "$STATE"
}

on_error() {
  local code=$?
  record "failed_exit_${code}"
  exit "$code"
}
trap on_error ERR

if [[ "${#GPU_LIST[@]}" -ne 4 ]]; then
  echo "MOTIONX322_TABLE3_GPUS must contain exactly four GPU ids" >&2
  exit 1
fi
for path in "$PAIR" "$VQ_CKPT" "$TRAIN_DIR/train_state_iter_30000.pth" "$TRAIN_DIR/train_state_iter_40000.pth"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

mkdir -p "$OUT_ROOT" "$EXTRACT_DIR" "$SHARD_ROOT" "$COMBINED_ROOT"
printf '%s\n' "$$" > "$OUT_ROOT/pipeline.pid"
date -Is > "$OUT_ROOT/started_at.txt"
record "start_sato_iter30k40k_table3_repeat20"

record "extract_iter_checkpoints"
"$PYTHON" - "$TRAIN_DIR" "$EXTRACT_DIR" <<'PY'
import json, hashlib, sys
from pathlib import Path
import torch
train_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
items = []
for step in (30000, 40000):
    src = train_dir / f"train_state_iter_{step}.pth"
    ckpt = torch.load(src, map_location="cpu")
    if int(ckpt.get("iter", -1)) != step:
        raise SystemExit(f"unexpected iter in {src}: {ckpt.get('iter')}")
    net = out_dir / f"net_iter_{step}.pth"
    clip = out_dir / f"clip_iter_{step}.pth"
    torch.save({"trans": ckpt["trans"]}, net)
    torch.save(ckpt["clip"], clip)
    def sha(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    items.append({
        "step": step,
        "source": str(src),
        "net": str(net),
        "clip": str(clip),
        "source_sha256": sha(src),
        "net_sha256": sha(net),
        "clip_sha256": sha(clip),
        "best_fid_recorded_in_state": float(ckpt.get("best_fid", 0.0)),
        "best_iter_recorded_in_state": int(ckpt.get("best_iter", -1)),
    })
(out_dir / "extraction_manifest.json").write_text(json.dumps(items, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(out_dir / "extraction_manifest.json")
PY

record "write_eval_spec"
"$PYTHON" - "$SPEC" "$VQ_CKPT" "$EXTRACT_DIR" <<'PY'
import json, sys
from pathlib import Path
spec_path = Path(sys.argv[1])
vq = sys.argv[2]
extract = Path(sys.argv[3])
spec = []
for step in (30000, 40000):
    spec.append({
        "name": f"motionx322_sato_official_iter{step}",
        "kind": "baseline",
        "dataname": "motionx322_75",
        "resume_pth": vq,
        "resume_trans": str(extract / f"net_iter_{step}.pth"),
        "clip_pth": str(extract / f"clip_iter_{step}.pth"),
        "args": {"down_t": 2, "block_size": 76},
        "description": f"Original SATO reproduced on Motion-X; checkpoint train_state_iter_{step}.pth",
    })
spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
print(spec_path)
PY

record "launch_shards_gpus_${GPU_LIST_STRING// /_}"
pids=()
for index in 0 1 2 3; do
  gpu="${GPU_LIST[$index]}"
  start_seed=$((123 + index * 5))
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
      --num-repeats 5 \
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
  record "launched_gpu${gpu}_seed${start_seed}_pid${pids[-1]}"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  date -Is > "$OUT_ROOT/failed_at.txt"
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

record "write_short_summary"
"$PYTHON" - "$COMBINED_ROOT/summary.json" "$SUMMARY_MD" <<'PY'
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2])
metrics = ["FID", "FID_P", "FID_D", "Matching", "R@1", "R@2", "R@3", "Diversity"]
rows = []
for model, data in sorted(summary.get("summary", {}).items()):
    row = [model]
    for metric in metrics:
        value = data.get(metric, {})
        row.append(f"{float(value.get('mean', float('nan'))):.4f}±{float(value.get('std', float('nan'))):.4f}")
    rows.append(row)
lines = [
    "| Model | FID | FID_P | FID_D | Matching | R@1 | R@2 | R@3 | Diversity |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append("| " + " | ".join(row) + " |")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
PY

date -Is > "$OUT_ROOT/complete.txt"
record "complete"