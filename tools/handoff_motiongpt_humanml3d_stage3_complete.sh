#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mnt/workspace/Wilson/T2M-GPT"
MGPT_ROOT="$REPO_ROOT/MotionGPT"
RUN_ROOT="$MGPT_ROOT/experiments/humanml3d_official_fresh_20260818"
STAGE1_EXP="$MGPT_ROOT/experiments/mgpt/VQVAE_H3D_OFFICIAL_FRESH_SEED123_20260818"
STAGE2_EXP="$MGPT_ROOT/experiments/mgpt/Pretrain_H3D_OFFICIAL_FRESH_SEED123_20260818"
STAGE3_EXP="$MGPT_ROOT/experiments/mgpt/Instruct_H3D_OFFICIAL_FRESH_SEED123_20260818"
STAGE3_LOG="$RUN_ROOT/stage3_launcher.log"
STATE="$RUN_ROOT/stage3_complete_state.tsv"
OUTPUT="$RUN_ROOT/completion_audit.json"
PYTHON_BIN="/mnt/workspace/Wilson/miniconda3/envs/mgpt/bin/python"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$MGPT_ROOT"
export TMPDIR="$RUN_ROOT"
record() { printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" >> "$STATE"; }
fail() { record "FAIL: $1"; printf '%s\n' "$1" >&2; exit 1; }

: > "$STATE"
record "waiting_for_stage3_clean_exit"
while ! rg -q 'Training ends!' "$STAGE3_LOG"; do
  if rg -q 'Traceback|RuntimeError|CUDA out of memory|non-finite validation|Detected nan|Detected NaN' "$STAGE3_LOG"; then
    fail "Stage 3 log contains a fatal error"
  fi
  sleep 30
done
if rg -q 'Traceback|RuntimeError|CUDA out of memory|non-finite validation|Detected nan|Detected NaN' "$STAGE3_LOG"; then
  fail "Stage 3 log contains a fatal error after exit"
fi
record "stage3_clean_exit"

cd "$REPO_ROOT"
"$PYTHON_BIN" tools/select_motiongpt_lm_best.py \
  --experiment-dir "$STAGE3_EXP" \
  --source-manifest "$RUN_ROOT/source_manifest.json" \
  --stage stage3 \
  --stage1-selection "$STAGE1_EXP/checkpoints/best-validation.json" \
  --predecessor-selection "$STAGE2_EXP/checkpoints/selected-validation.json" \
  --expected-samples 1504 \
  --validation-interval 5000 \
  --max-steps 100000 \
  --patience 6 \
  --min-validations 8 \
  --fid-tolerance 0.0001 \
  --output "$STAGE3_EXP/checkpoints/selected-validation.json"
record "stage3_validation_selection_pass"

"$PYTHON_BIN" tools/audit_motiongpt_humanml3d_official_complete.py \
  --repo-root "$REPO_ROOT" --run-root "$RUN_ROOT" --output "$OUTPUT"
record "completion_audit_pass"
