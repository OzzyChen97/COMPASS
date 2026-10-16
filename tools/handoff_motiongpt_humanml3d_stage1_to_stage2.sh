#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/workspace/Wilson/T2M-GPT"
MGPT_ROOT="$REPO_ROOT/MotionGPT"
RUN_ROOT="$MGPT_ROOT/experiments/humanml3d_official_fresh_20260818"
STAGE1_EXP="$MGPT_ROOT/experiments/mgpt/VQVAE_H3D_OFFICIAL_FRESH_SEED123_20260818"
STAGE2_EXP="$MGPT_ROOT/experiments/mgpt/Pretrain_H3D_OFFICIAL_FRESH_SEED123_20260818"
DATA_ROOT="$REPO_ROOT/dataset/HumanML3D"
TOKEN_DIR="$DATA_ROOT/VQVAE_H3D_OFFICIAL_FRESH_SEED123_20260818"
STAGE1_LOG="$RUN_ROOT/stage1_launcher.log"
SELECTION="$STAGE1_EXP/checkpoints/best-validation.json"
TOKEN_AUDIT="$RUN_ROOT/token_audit.json"
STATE="$RUN_ROOT/stage1_to_stage2_state.tsv"
TOKEN_LOG="$RUN_ROOT/tokenize.log"
STAGE2_LOG="$RUN_ROOT/stage2_launcher.log"
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
manifest_path = pathlib.Path(sys.argv[2]).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
method = manifest.get("method", {})
if method.get("uses_t2mgpt_generator_checkpoint") is not False:
    raise SystemExit("manifest does not forbid T2M-GPT generator initialization")
if method.get("uses_released_motiongpt_generator_checkpoint") is not False:
    raise SystemExit("manifest does not forbid released MotionGPT initialization")
for relative, expected in manifest["source_files"].items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"tracked source drift: {relative}: {actual} != {expected}")
for stage in ("stage1", "token", "stage2", "stage3"):
    item = manifest["configs"][stage]
    actual = hashlib.sha256(pathlib.Path(item["path"]).read_bytes()).hexdigest()
    if actual != item["sha256"]:
        raise SystemExit(f"generated config drift: {stage}")
PY
}

[[ ! -e "$STAGE2_EXP" ]] || fail "fresh Stage 2 experiment already exists: $STAGE2_EXP"
[[ ! -e "$TOKEN_DIR" ]] || fail "fresh token directory already exists: $TOKEN_DIR"
: > "$STATE"
record "waiting_for_stage1_clean_exit"
while ! rg -q 'Training ends!' "$STAGE1_LOG"; do
  if rg -q 'Traceback|RuntimeError|CUDA out of memory|Monitored metric total/val.*not available|Detected nan|Detected NaN' "$STAGE1_LOG"; then
    fail "Stage 1 log contains a fatal error"
  fi
  sleep 30
done
rg -q 'Training ends!' "$STAGE1_LOG" || fail "Stage 1 did not exit cleanly"
if rg -q 'Traceback|RuntimeError|CUDA out of memory|Detected nan|Detected NaN' "$STAGE1_LOG"; then
  fail "Stage 1 log contains a fatal error after exit"
fi
record "stage1_clean_exit"
verify_sources
record "source_and_config_hashes_pass"

cd "$REPO_ROOT"
"$PYTHON_BIN" tools/select_motiongpt_stage1_best.py \
  --experiment-dir "$STAGE1_EXP" \
  --source-manifest "$RUN_ROOT/source_manifest.json" \
  --output "$SELECTION"
record "stage1_validation_selection_pass"

STAGE1_CKPT="$("$PYTHON_BIN" - "$SELECTION" <<'PY'
import json, pathlib, sys
selection = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if selection.get("split") != "val" or selection.get("monitor") != "total/val":
    raise SystemExit("Stage 1 selection is not validation-only total/val")
checkpoint = pathlib.Path(selection["checkpoint"]).resolve()
if not checkpoint.is_file():
    raise SystemExit(f"selected checkpoint is missing: {checkpoint}")
print(checkpoint)
PY
)"

cd "$MGPT_ROOT"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/get_motion_code.py \
  --cfg "$RUN_ROOT/configs/token.yaml" \
  --cfg_assets configs/assets_humanml3d_official_local.yaml \
  --pretrained_vae "$STAGE1_CKPT" --nodebug > "$TOKEN_LOG" 2>&1
rg -q 'Motion tokenization done' "$TOKEN_LOG" || fail "tokenization did not report completion"
record "tokenization_complete"

cd "$REPO_ROOT"
"$PYTHON_BIN" tools/audit_motiongpt_humanml3d_tokens.py \
  --data-root "$DATA_ROOT" \
  --token-dir "$TOKEN_DIR" \
  --stage1-selection "$SELECTION" \
  --source-manifest "$RUN_ROOT/source_manifest.json" \
  --output "$TOKEN_AUDIT"
record "token_audit_pass"
verify_sources
record "pre_stage2_source_and_config_hashes_pass"
record "launching_stage2"

cd "$MGPT_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec "$PYTHON_BIN" train.py \
  --cfg "$RUN_ROOT/configs/stage2.yaml" \
  --cfg_assets configs/assets_humanml3d_official_local.yaml \
  --pretrained_vae "$STAGE1_CKPT" --nodebug > "$STAGE2_LOG" 2>&1
