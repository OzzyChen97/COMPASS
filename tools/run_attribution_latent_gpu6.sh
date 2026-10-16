#!/usr/bin/env bash
# Reproduce the diagnostics without training or modifying checkpoints.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${T2MGPT_PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
OUT="${1:-$ROOT/ablation/attribution_latent_gpu6_$(date +%Y%m%d_%H%M%S)}"
cd "$ROOT"
if [[ -e "$OUT" ]]; then
  printf 'Refusing to overwrite existing output: %s\n' "$OUT" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$PY" -m pytest -q tests/test_attribution_latent_diagnostics.py tests/test_extract_attribution_embeddings.py tests/test_attribution_motion_alignment.py
"$PY" -u tools/audit_sato_gradient_connectivity.py --output "$OUT/gradient_audit.json" --include-create-graph-true
"$PY" -u tools/measure_attribution_latent.py --output "$OUT/main" --motions 300 --intervention-motions 100 --directions 3
"$PY" -u tools/analyze_attribution_latent.py --input "$OUT/main" --bootstrap 1000
printf 'Completed: %s\n' "$OUT/main/analysis/summary.json"