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
STAGE3_SELECTION="$STAGE3_EXP/checkpoints/selected-validation.json"
STAGE3_LOG="$RUN_ROOT/stage3_launcher.log"
EVAL_ROOT="$MGPT_ROOT/experiments/robustness/motionx322_official_fresh_r2_table3_20seed_20260817"
EVAL_LOG="$RUN_ROOT/table3_20seed_launcher.log"
STATE="$RUN_ROOT/stage3_to_table3_state.tsv"
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

if [[ -e "$EVAL_ROOT" ]]; then
  fail "fresh R2 Table-3 output already exists: $EVAL_ROOT"
fi

: > "$STATE"
record "waiting_for_stage3_clean_exit"
while ! rg -q 'Training ends!' "$STAGE3_LOG" 2>/dev/null; do
  if rg -q 'Traceback|RuntimeError|CUDA out of memory|non-finite validation|Detected nan|Detected NaN' "$STAGE3_LOG" 2>/dev/null; then
    fail "Stage 3 log contains a fatal error"
  fi
  sleep 30
done

rg -q 'Training done' "$STAGE3_LOG" || fail "Stage 3 did not emit Training done"
if rg -q 'Traceback|RuntimeError|CUDA out of memory|non-finite validation|Detected nan|Detected NaN' "$STAGE3_LOG"; then
  fail "Stage 3 log contains a fatal error after exit"
fi
record "stage3_clean_exit"

verify_sources
record "source_and_config_hashes_pass"

cd "$REPO_ROOT"
"$PYTHON_BIN" tools/select_motiongpt_lm_best.py \
  --experiment-dir "$STAGE3_EXP" \
  --source-manifest "$RUN_ROOT/source_manifest.json" \
  --stage stage3 \
  --stage1-selection "$STAGE1_SELECTION" \
  --predecessor-selection "$STAGE2_SELECTION" \
  --expected-samples 2552 \
  --validation-interval 5000 \
  --max-steps 100000 \
  --patience 6 \
  --min-validations 8 \
  --fid-tolerance 0.0001 \
  --output "$STAGE3_SELECTION"
record "stage3_validation_selection_pass"

STAGE3_CKPT="$($PYTHON_BIN - "$STAGE3_SELECTION" <<'PY'
import json
import pathlib
import sys

selection = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if selection.get("stage") != "stage3" or selection.get("split") != "val":
    raise SystemExit("formal evaluation requires a validation-selected Stage 3")
if selection.get("uses_t2mgpt_generator_checkpoint") is not False:
    raise SystemExit("Stage 3 selection does not prove baseline-free provenance")
checkpoint = pathlib.Path(selection["checkpoint"]).resolve()
if not checkpoint.is_file():
    raise SystemExit(f"selected Stage 3 checkpoint is missing: {checkpoint}")
print(checkpoint)
PY
)"

"$PYTHON_BIN" tools/prepare_motiongpt_motionx_table3.py \
  --checkpoint "$STAGE3_CKPT" \
  --base-config "$RUN_ROOT/configs/stage3.yaml" \
  --assets-config "$MGPT_ROOT/configs/assets_motionx322.yaml" \
  --output-dir "$EVAL_ROOT"
record "table3_manifest_and_configs_prepared"

verify_sources
record "pre_evaluation_source_and_config_hashes_pass"
record "launching_20seed_evaluation"

MOTIONGPT_PYTHON="$PYTHON_BIN" \
MOTIONGPT_MOTIONX_MANIFEST="$EVAL_ROOT/manifest.json" \
MOTIONGPT_MOTIONX_CONFIG_DIR="$EVAL_ROOT/configs" \
MOTIONGPT_MOTIONX_ASSETS_CFG="$MGPT_ROOT/configs/assets_motionx322.yaml" \
MOTIONGPT_MOTIONX_OUTPUT_ROOT="$EVAL_ROOT" \
MOTIONGPT_MOTIONX_GPUS="0 1 2 3" \
bash "$MGPT_ROOT/scripts/run_motionx322_table3_eval.sh" > "$EVAL_LOG" 2>&1

[[ -s "$EVAL_ROOT/complete.txt" ]] || fail "20-seed launcher lacks complete marker"
"$PYTHON_BIN" tools/audit_motiongpt_motionx_official_complete.py \
  --run-root "$RUN_ROOT" \
  --eval-root "$EVAL_ROOT" \
  --output "$RUN_ROOT/final_completion_audit.json"
record "evaluation_complete_and_strict_aggregate_pass"
