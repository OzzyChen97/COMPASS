#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/workspace/Wilson/T2M-GPT"
MGPT_ROOT="$REPO_ROOT/MotionGPT"
RUN_ROOT="$MGPT_ROOT/experiments/humanml3d_official_fresh_20260818"
STAGE1_EXP="$MGPT_ROOT/experiments/mgpt/VQVAE_H3D_OFFICIAL_FRESH_SEED123_20260818"
STAGE2_EXP="$MGPT_ROOT/experiments/mgpt/Pretrain_H3D_OFFICIAL_FRESH_SEED123_20260818"
STAGE3_EXP="$MGPT_ROOT/experiments/mgpt/Instruct_H3D_OFFICIAL_FRESH_SEED123_20260818"
STAGE1_SELECTION="$STAGE1_EXP/checkpoints/best-validation.json"
STAGE2_SELECTION="$STAGE2_EXP/checkpoints/selected-validation.json"
STAGE2_LOG="$RUN_ROOT/stage2_launcher.log"
STAGE3_LOG="$RUN_ROOT/stage3_launcher.log"
STATE="$RUN_ROOT/stage2_to_stage3_state.tsv"
PYTHON_BIN="/mnt/workspace/Wilson/miniconda3/envs/mgpt/bin/python"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$MGPT_ROOT"
export LD_LIBRARY_PATH="/mnt/workspace/Wilson/miniconda3/envs/mgpt/lib/python3.11/site-packages/torch/lib:/mnt/workspace/Wilson/miniconda3/envs/mgpt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export TMPDIR="$RUN_ROOT"
export TOKENIZERS_PARALLELISM=false
record() { printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" >> "$STATE"; }
fail() { record "FAIL: $1"; printf '%s\n' "$1" >&2; exit 1; }

verify_sources() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$RUN_ROOT/source_manifest.json" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
for key in ("uses_t2mgpt_generator_checkpoint", "uses_released_motiongpt_generator_checkpoint"):
    if manifest.get("method", {}).get(key) is not False:
        raise SystemExit(f"manifest does not prove {key}=false")
for relative, expected in manifest["source_files"].items():
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"tracked source drift: {relative}")
for item in manifest["configs"].values():
    if hashlib.sha256(pathlib.Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
        raise SystemExit(f"generated config drift: {item['path']}")
PY
}

[[ ! -e "$STAGE3_EXP" ]] || fail "fresh Stage 3 experiment already exists: $STAGE3_EXP"
: > "$STATE"
record "waiting_for_stage2_clean_exit"
while ! rg -q 'Training ends!' "$STAGE2_LOG"; do
  if rg -q 'Traceback|RuntimeError|CUDA out of memory|non-finite validation|Detected nan|Detected NaN' "$STAGE2_LOG"; then
    fail "Stage 2 log contains a fatal error"
  fi
  sleep 30
done
if rg -q 'Traceback|RuntimeError|CUDA out of memory|non-finite validation|Detected nan|Detected NaN' "$STAGE2_LOG"; then
  fail "Stage 2 log contains a fatal error after exit"
fi
record "stage2_clean_exit"
verify_sources

cd "$REPO_ROOT"
"$PYTHON_BIN" tools/select_motiongpt_lm_best.py \
  --experiment-dir "$STAGE2_EXP" \
  --source-manifest "$RUN_ROOT/source_manifest.json" \
  --stage stage2 \
  --stage1-selection "$STAGE1_SELECTION" \
  --expected-samples 1504 \
  --validation-interval 10000 \
  --max-steps 200000 \
  --patience 5 \
  --min-validations 8 \
  --fid-tolerance 0.0001 \
  --output "$STAGE2_SELECTION"
record "stage2_validation_selection_pass"

readarray -t CHECKPOINTS < <("$PYTHON_BIN" - "$STAGE1_SELECTION" "$STAGE2_SELECTION" <<'PY'
import json, pathlib, sys
for path in sys.argv[1:]:
    selection = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if selection.get("split") != "val":
        raise SystemExit("non-validation predecessor selection")
    checkpoint = pathlib.Path(selection["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {checkpoint}")
    print(checkpoint)
PY
)
[[ "${#CHECKPOINTS[@]}" -eq 2 ]] || fail "could not resolve selected Stage 1/2 checkpoints"
verify_sources
record "launching_stage3"

cd "$MGPT_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec "$PYTHON_BIN" train.py \
  --cfg "$RUN_ROOT/configs/stage3.yaml" \
  --cfg_assets configs/assets_humanml3d_official_local.yaml \
  --pretrained_vae "${CHECKPOINTS[0]}" \
  --pretrained "${CHECKPOINTS[1]}" --nodebug > "$STAGE3_LOG" 2>&1
