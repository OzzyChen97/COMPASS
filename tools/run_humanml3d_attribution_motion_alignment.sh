#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---smoke}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${T2MGPT_PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
GPU="${ATTR_MOTION_GPU:-4}"
OUT_ROOT="${ATTR_MOTION_OUT:-$ROOT/ablation/attribution_motion_alignment}"
SATO_CKPT="$ROOT/output_GPT_Final/sato_pp_allloss_gpu0_teachl1_05_eval5k_resume104k_200k_single/sato_pp_iter_130000.pth"
BASELINE_TRANS="$ROOT/pretrained/VQTransformer_corruption05/net_best_fid.pth"
VQ_CKPT="$ROOT/pretrained/VQVAE/net_best_fid.pth"
DATA_ROOT="$ROOT/dataset/HumanML3D"
INDEX_ROOT="$ROOT/ablation/humanml3d_table1_levels/index_audit"
RUN_ROOT="$ROOT/ablation/humanml3d_table1_levels/runs/t2mgpt_satopp"

case "$MODE" in
  --smoke) LIMIT=64; SEEDS=1; RSA_PAIRS=5000; BOOTSTRAP=200; PERMUTATIONS=999; TAG=smoke ;;
  --full) LIMIT=0; SEEDS=20; RSA_PAIRS=100000; BOOTSTRAP=10000; PERMUTATIONS=100000; TAG=full ;;
  *) echo "usage: $0 --smoke|--full" >&2; exit 2 ;;
esac

for required in "$PYTHON_BIN" "$SATO_CKPT" "$BASELINE_TRANS" "$VQ_CKPT"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done
indices=(
  "$INDEX_ROOT/p1_sample_index_seed123.json"
  "$INDEX_ROOT/p2_sample_index_seed123.json"
  "$INDEX_ROOT/p3_sample_index_seed123.json"
)
for required in "${indices[@]}"; do
  [[ -e "$required" ]] || { echo "missing sample index: $required" >&2; exit 1; }
done

OUT="$OUT_ROOT/$TAG"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONNOUSERSITE=1
extract_common=(
  "$PYTHON_BIN" tools/extract_attribution_embeddings.py
  --reference-checkpoint "$SATO_CKPT"
  --vq-checkpoint "$VQ_CKPT"
  --data-root "$DATA_ROOT"
  --sample-index "${indices[@]}"
  --batch-size "${ATTR_MOTION_BATCH_SIZE:-64}"
  --gpu-id "$GPU"
)
if (( LIMIT > 0 )); then
  extract_common+=(--limit-per-level "$LIMIT")
fi

"${extract_common[@]}" \
  --transformer-checkpoint "$BASELINE_TRANS" \
  --model-label baseline \
  --output "$OUT/baseline_attribution.npz"
"${extract_common[@]}" \
  --model-label satopp \
  --output "$OUT/satopp_attribution.npz"

"$PYTHON_BIN" tools/analyze_attribution_motion_alignment.py \
  --baseline-attribution "$OUT/baseline_attribution.npz" \
  --satopp-attribution "$OUT/satopp_attribution.npz" \
  --run-root "$RUN_ROOT" \
  --output-dir "$OUT/analysis" \
  --num-seeds "$SEEDS" \
  --rsa-pairs "$RSA_PAIRS" \
  --bootstrap "$BOOTSTRAP" \
  --permutations "$PERMUTATIONS" \
  --seed 123

date -Is > "$OUT/complete.txt"
echo "$OUT/analysis/summary.json"