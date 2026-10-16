#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/workspace/Wilson/T2M-GPT"
MGPT_ROOT="$REPO_ROOT/MotionGPT"
RUN_ROOT="$MGPT_ROOT/experiments/motionx322_official_fresh_r2_20260817"
STAGE1_EXP="$MGPT_ROOT/experiments/mgpt/VQVAE_MOTIONX322_OFFICIAL_FRESH_R2_SEED123_20260817"
STAGE2_EXP="$MGPT_ROOT/experiments/mgpt/Pretrain_MOTIONX322_OFFICIAL_FRESH_R2_SEED123_20260817"
DATA_ROOT="$REPO_ROOT/dataset/Motion-X-322-MaxToken75"
TOKEN_DIR="$DATA_ROOT/VQVAE_MOTIONX322_OFFICIAL_FRESH_R2_SEED123_20260817"
STAGE1_LOG="$RUN_ROOT/stage1_launcher.log"
SELECTION="$STAGE1_EXP/checkpoints/best-validation.json"
TOKEN_AUDIT="$RUN_ROOT/token_audit.json"
STATE="$RUN_ROOT/handoff_state.tsv"
TOKEN_LOG="$RUN_ROOT/tokenize.log"
STAGE2_LOG="$RUN_ROOT/stage2_launcher.log"
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

if [[ -e "$STAGE2_EXP" ]]; then
  fail "fresh R2 Stage 2 experiment already exists: $STAGE2_EXP"
fi
if [[ -e "$TOKEN_DIR" ]]; then
  fail "fresh R2 token directory already exists before handoff: $TOKEN_DIR"
fi

: > "$STATE"
record "waiting_for_stage1_clean_exit"
while ! rg -q 'Training ends!' "$STAGE1_LOG"; do
  if rg -q 'Traceback|RuntimeError|CUDA out of memory|Monitored metric total/val.*not available|Detected nan|Detected NaN' "$STAGE1_LOG"; then
    fail "Stage 1 log contains a fatal error"
  fi
  sleep 30
done

rg -q 'Training done' "$STAGE1_LOG" || fail "Stage 1 did not emit Training done"
rg -q 'Training ends!' "$STAGE1_LOG" || fail "Stage 1 did not emit Training ends"
if rg -q 'Traceback|RuntimeError|CUDA out of memory|Monitored metric total/val.*not available|Detected nan|Detected NaN' "$STAGE1_LOG"; then
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

STAGE1_CKPT="$($PYTHON_BIN - "$SELECTION" <<'PY'
import json
import pathlib
import sys

selection = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if selection.get("split") != "val" or selection.get("monitor") != "total/val":
    raise SystemExit("Stage 1 selection is not validation-only total/val")
if selection.get("uses_t2mgpt_generator_checkpoint") is not False:
    raise SystemExit("Stage 1 selection does not prove baseline-free provenance")
checkpoint = pathlib.Path(selection["checkpoint"]).resolve()
if not checkpoint.is_file():
    raise SystemExit(f"selected checkpoint is missing: {checkpoint}")
print(checkpoint)
PY
)"

CONFIG="$RUN_ROOT/configs/token.yaml" \
ASSETS_CONFIG="configs/assets_motionx322.yaml" \
VAE_CHECKPOINT="$STAGE1_CKPT" \
GPU_SET="0" \
bash "$MGPT_ROOT/scripts/run_motionx322_tokenize.sh" > "$TOKEN_LOG" 2>&1
rg -q 'Motion tokenization done' "$TOKEN_LOG" || fail "tokenization did not report completion"
record "tokenization_complete"

"$PYTHON_BIN" tools/audit_motiongpt_motionx_tokens.py \
  --data-root "$DATA_ROOT" \
  --token-dir "$TOKEN_DIR" \
  --stage1-selection "$SELECTION" \
  --output "$TOKEN_AUDIT"
"$PYTHON_BIN" - "$TOKEN_AUDIT" <<'PY'
import json
import pathlib
import sys

audit = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if audit.get("status") != "pass":
    raise SystemExit("token audit did not pass")
if audit.get("missing_count") or audit.get("extra_count") or audit.get("invalid_token_count"):
    raise SystemExit("token audit coverage is not exact")
PY
record "token_audit_pass"

verify_sources
record "pre_stage2_source_and_config_hashes_pass"
record "launching_stage2"

CONFIG="$RUN_ROOT/configs/stage2.yaml" \
ASSETS_CONFIG="configs/assets_motionx322.yaml" \
VAE_CHECKPOINT="$STAGE1_CKPT" \
GPU_SET="0,1,2,3" \
exec bash "$MGPT_ROOT/scripts/run_motionx322_stage2.sh" formal > "$STAGE2_LOG" 2>&1
