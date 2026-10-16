#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MOTIONGPT_PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/mgpt/bin/python}"
ASSET_ROOT="$ROOT/ablation/humanml3d_table1_levels"
PAIR_PREFIX="$ROOT/dataset/HumanML3D/test_perb_table1_level"
BOOTSTRAP_ROOT="$ROOT/MotionGPT/experiments/robustness/humanml3d_official_fresh_last100k_table1_full4640_20seed_20260820"
BOOTSTRAP_MANIFEST="$BOOTSTRAP_ROOT/manifest.json"
BOOTSTRAP_CONFIG="$BOOTSTRAP_ROOT/configs/motiongpt_humanml3d_seed123.yaml"
ASSETS_CONFIG="$ROOT/MotionGPT/configs/assets_humanml3d_official_local.yaml"
CHECKPOINT="$ROOT/MotionGPT/experiments/mgpt/Instruct_H3D_OFFICIAL_FRESH_SEED123_20260818/checkpoints/last.ckpt"

commands() {
  cat <<EOF
python tools/finalize_test_perb_levels.py --dataset-root dataset/HumanML3D --pool dataset/HumanML3D/test_perb.json --output-prefix dataset/HumanML3D/test_perb_table1_level --seed 123 --single-level-only
for LEVEL in P1 P2 P3: MotionGPT index-only caption-locked repair-all audit (4,380 samples)
python tools/audit_humanml3d_table1_level_assets.py --output ablation/humanml3d_table1_levels/assets_manifest.json
for LEVEL in P1 P2 P3: tools/prepare_motiongpt_humanml3d_table1.py --level LEVEL
EOF
}

if [[ "$MODE" == "--dry-run" ]]; then
  commands
  exit 0
fi
if [[ "$MODE" != "--run" ]]; then
  echo "usage: $0 --dry-run|--run" >&2
  exit 2
fi
if [[ -e "$ASSET_ROOT/assets_manifest.json" && "${HUMANML_LEVEL_REBUILD:-0}" != "1" ]]; then
  echo "refusing to replace frozen assets; set HUMANML_LEVEL_REBUILD=1 to rebuild" >&2
  exit 1
fi
for required in \
  "$PYTHON_BIN" "$BOOTSTRAP_MANIFEST" "$BOOTSTRAP_CONFIG" \
  "$ASSETS_CONFIG" "$CHECKPOINT"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done

cd "$ROOT"
mkdir -p "$ASSET_ROOT/index_audit" "$ASSET_ROOT/motiongpt_sources"
python tools/finalize_test_perb_levels.py \
  --dataset-root dataset/HumanML3D \
  --pool dataset/HumanML3D/test_perb.json \
  --output-prefix dataset/HumanML3D/test_perb_table1_level \
  --seed 123 \
  --single-level-only

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/mnt/workspace/Wilson/miniconda3/envs/mgpt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
for level in P1 P2 P3; do
  lower="${level,,}"
  "$PYTHON_BIN" MotionGPT/eval_loader_humanml3d_table1.py \
    --manifest "$BOOTSTRAP_MANIFEST" \
    --config "$BOOTSTRAP_CONFIG" \
    --assets-cfg "$ASSETS_CONFIG" \
    --dataset humanml3d \
    --seed 123 \
    --batch-size 32 \
    --num-workers 0 \
    --loader-mode caption-locked \
    --pair-file "${PAIR_PREFIX}_${lower}.json" \
    --allowed-levels "$level" \
    --caption-length-policy repair-all \
    --pair-rng-mode sequential \
    --sampling-kwargs-mode explicit \
    --prompt-mode official-default \
    --generation-order original-first \
    --length-mode generated \
    --index-only \
    --sample-index-output "$ASSET_ROOT/index_audit/${lower}_sample_index_seed123.json" \
    --protocol-manifest "$ASSET_ROOT/index_audit/${lower}_protocol.json" \
    --output "$ASSET_ROOT/index_audit/${lower}_index_result.json"
done

python tools/audit_humanml3d_table1_level_assets.py \
  --pair-prefix "$PAIR_PREFIX" \
  --index-root "$ASSET_ROOT/index_audit" \
  --output "$ASSET_ROOT/assets_manifest.json" \
  > "$ASSET_ROOT/assets_audit.log"

for level in P1 P2 P3; do
  lower="${level,,}"
  source_root="$ASSET_ROOT/motiongpt_sources/$lower"
  rm -rf "$source_root"
  "$PYTHON_BIN" tools/prepare_motiongpt_humanml3d_table1.py \
    --checkpoint "$CHECKPOINT" \
    --base-config "$BOOTSTRAP_CONFIG" \
    --assets-config "$ASSETS_CONFIG" \
    --pair-file "${PAIR_PREFIX}_${lower}.json" \
    --fixed-sample-index "$ASSET_ROOT/index_audit/${lower}_sample_index_seed123.json" \
    --output-dir "$source_root" \
    --level "$level"
done

date -Is > "$ASSET_ROOT/assets_complete.txt"