#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MOTIONGPT_PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/mgpt/bin/python}"
ASSET_ROOT="${MOTIONX_LEVEL_ASSET_ROOT:-$ROOT/ablation/motionx322_level_rprecision}"
PAIR_PREFIX="$ROOT/dataset/Motion-X-322/test_perb_gemini31_max75"
BASE_PARENT="$ROOT/MotionGPT/experiments/robustness/motionx322_official_fresh_r2_last100k_table3_20seed_20260820/manifest.json"
PLUGIN_PARENT="$ROOT/MotionGPT/experiments/robustness/motionx322_sato_plugin_pairgen1_preserve_only_bestcomposite_table3_20seed_20260823/manifest.json"

show_plan() {
  cat <<EOF
Prepare MotionGPT and MotionGPT+SATO++ level source manifests.
Freeze caption-locked P1/P2/P3 indices at 7,679 samples per level.
Verify the two MotionGPT-family loaders resolve byte-identical frozen indices.
Write $ASSET_ROOT/assets_manifest.json.
EOF
}

if [[ "$MODE" == "--dry-run" ]]; then
  show_plan
  exit 0
fi
if [[ "$MODE" != "--run" ]]; then
  echo "usage: $0 --dry-run|--run" >&2
  exit 2
fi
if [[ -e "$ASSET_ROOT/assets_manifest.json" && "${MOTIONX_LEVEL_REBUILD:-0}" != "1" ]]; then
  echo "refusing to replace frozen assets; set MOTIONX_LEVEL_REBUILD=1 to rebuild" >&2
  exit 1
fi
for required in "$PYTHON_BIN" "$BASE_PARENT" "$PLUGIN_PARENT"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done

cd "$ROOT"
if [[ "${MOTIONX_LEVEL_REBUILD:-0}" == "1" ]]; then
  rm -rf "$ASSET_ROOT/index_audit" "$ASSET_ROOT/motiongpt_sources"
fi
mkdir -p "$ASSET_ROOT/index_audit" "$ASSET_ROOT/motiongpt_sources"

for model_key in motiongpt motiongpt_satopp; do
  case "$model_key" in
    motiongpt) parent="$BASE_PARENT" ;;
    motiongpt_satopp) parent="$PLUGIN_PARENT" ;;
  esac
  for level in P1 P2 P3; do
    lower="${level,,}"
    output="$ASSET_ROOT/motiongpt_sources/$model_key/$lower"
    "$PYTHON_BIN" tools/prepare_motiongpt_motionx322_level_rprecision.py \
      --model-key "$model_key" \
      --parent-manifest "$parent" \
      --pair-file "${PAIR_PREFIX}_${lower}.json" \
      --level "$level" \
      --output-dir "$output"
  done
done

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/mnt/workspace/Wilson/miniconda3/envs/mgpt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export TMPDIR=/tmp
for level in P1 P2 P3; do
  lower="${level,,}"
  pair="${PAIR_PREFIX}_${lower}.json"
  base_source="$ASSET_ROOT/motiongpt_sources/motiongpt/$lower"
  base_index="$ASSET_ROOT/index_audit/${lower}_sample_index_seed123.json"
  common=(
    --dataset motionx322
    --expected-samples 7679
    --seed 123
    --batch-size 32
    --num-workers 0
    --loader-mode caption-locked
    --pair-file "$pair"
    --allowed-levels "$level"
    --caption-length-policy repair-all
    --pair-rng-mode sequential
    --sampling-kwargs-mode explicit
    --prompt-mode official-default
    --generation-order original-first
    --length-mode generated
    --index-only
  )
  "$PYTHON_BIN" MotionGPT/eval_loader_robustness.py \
    --manifest "$base_source/manifest.json" \
    --config "$base_source/configs/motiongpt_motionx322_seed123.yaml" \
    "${common[@]}" \
    --sample-index-output "$base_index" \
    --protocol-manifest "$ASSET_ROOT/index_audit/${lower}_protocol.json" \
    --output "$ASSET_ROOT/index_audit/${lower}_index_result.json"

  plugin_source="$ASSET_ROOT/motiongpt_sources/motiongpt_satopp/$lower"
  "$PYTHON_BIN" MotionGPT/eval_loader_robustness.py \
    --manifest "$plugin_source/manifest.json" \
    --config "$plugin_source/configs/motiongpt_satopp_motionx322_seed123.yaml" \
    "${common[@]}" \
    --expected-sample-index "$base_index" \
    --output "$ASSET_ROOT/index_audit/${lower}_plugin_index_verify.json"
done

"$PYTHON_BIN" tools/audit_motionx322_level_rprecision_assets.py \
  --pair-prefix "$PAIR_PREFIX" \
  --index-root "$ASSET_ROOT/index_audit" \
  --source-root "$ASSET_ROOT/motiongpt_sources" \
  --output "$ASSET_ROOT/assets_manifest.json" \
  > "$ASSET_ROOT/assets_audit.log"
date -Is > "$ASSET_ROOT/assets_complete.txt"
echo "Motion-X level assets frozen at $ASSET_ROOT"