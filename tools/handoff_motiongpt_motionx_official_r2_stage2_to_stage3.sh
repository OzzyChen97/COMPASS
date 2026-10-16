#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/workspace/Wilson/T2M-GPT"
MGPT_ROOT="$REPO_ROOT/MotionGPT"
RUN_ROOT="$MGPT_ROOT/experiments/motionx322_official_fresh_r2_20260817"
STAGE1_EXP="$MGPT_ROOT/experiments/mgpt/VQVAE_MOTIONX322_OFFICIAL_FRESH_R2_SEED123_20260817"
STAGE2_EXP="$MGPT_ROOT/experiments/mgpt/Pretrain_MOTIONX322_OFFICIAL_FRESH_R2_SEED123_20260817"
STAGE3_EXP="$MGPT_ROOT/experiments/mgpt/Instruct_MOTIONX322_OFFICIAL_FRESH_R2_SEED123_20260817"
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

record() {
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" >> "$STATE"
}

fail() {
  record "FAIL: $1"
  printf '%s\n' "$1" >&2
  exit 1
}

verify_sources() {
  "$PYTHON_BIN" - "$REPO_ROOT" "$RUN_ROOT/source_manifest.json" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
manifest_path = pathlib.Path(sys.argv[2]).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("method", {}).get("uses_t2mgpt_generator_checkpoint") is not False:
    raise SystemExit("manifest does not prove baseline-free initialization")
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

if [[ -e "$STAGE3_EXP" ]]; then
  fail "fresh R2 Stage 3 experiment already exists: $STAGE3_EXP"
fi

: > "$STATE"
record "waiting_for_stage2_clean_exit"
while ! rg -q 'Training ends!' "$STAGE2_LOG"; do
  if rg -q 'Traceback|RuntimeError|CUDA out of memory|non-finite validation|Detected nan|Detected NaN' "$STAGE2_LOG"; then
    fail "Stage 2 log contains a fatal error"
  fi
  sleep 30
done

rg -q 'Training done' "$STAGE2_LOG" || fail "Stage 2 did not emit Training done"
if rg -q 'Traceback|RuntimeError|CUDA out of memory|non-finite validation|Detected nan|Detected NaN' "$STAGE2_LOG"; then
  fail "Stage 2 log contains a fatal error after exit"
fi
record "stage2_clean_exit"

verify_sources
record "source_and_config_hashes_pass"

cd "$REPO_ROOT"
"$PYTHON_BIN" tools/select_motiongpt_lm_best.py \
  --experiment-dir "$STAGE2_EXP" \
  --source-manifest "$RUN_ROOT/source_manifest.json" \
  --stage stage2 \
  --stage1-selection "$STAGE1_SELECTION" \
  --expected-samples 2552 \
  --validation-interval 10000 \
  --max-steps 200000 \
  --patience 5 \
  --min-validations 8 \
  --fid-tolerance 0.0001 \
  --output "$STAGE2_SELECTION"
record "stage2_validation_selection_pass"

readarray -t CHECKPOINTS < <("$PYTHON_BIN" - "$STAGE1_SELECTION" "$STAGE2_SELECTION" <<'PY'
import json
import pathlib
import sys

stage1 = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
stage2 = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
for label, selection in (("stage1", stage1), ("stage2", stage2)):
    if selection.get("split") != "val":
        raise SystemExit(f"{label} selection is not validation-only")
    if selection.get("uses_t2mgpt_generator_checkpoint") is not False:
        raise SystemExit(f"{label} selection does not prove baseline-free provenance")
    checkpoint = pathlib.Path(selection["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"{label} checkpoint is missing: {checkpoint}")
    print(checkpoint)
PY
)
if [[ "${#CHECKPOINTS[@]}" -ne 2 ]]; then
  fail "could not resolve selected Stage 1/2 checkpoints"
fi

verify_sources
record "pre_stage3_source_and_config_hashes_pass"
record "launching_stage3"

CONFIG="$RUN_ROOT/configs/stage3.yaml" \
ASSETS_CONFIG="configs/assets_motionx322.yaml" \
VAE_CHECKPOINT="${CHECKPOINTS[0]}" \
LM_CHECKPOINT="${CHECKPOINTS[1]}" \
GPU_SET="0,1,2,3" \
exec bash "$MGPT_ROOT/scripts/run_motionx322_stage3.sh" formal > "$STAGE3_LOG" 2>&1
