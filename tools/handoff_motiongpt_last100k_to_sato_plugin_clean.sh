#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/workspace/Wilson/T2M-GPT"
MGPT_ROOT="$ROOT/MotionGPT"
PYTHON_BIN="/mnt/workspace/Wilson/miniconda3/envs/mgpt/bin/python"

BASELINE_PIPELINE="$MGPT_ROOT/experiments/motiongpt_continue_last_to_100k_noearly_20260820"
RUN_ROOT="$MGPT_ROOT/experiments/motionx322_sato_plugin_lastbaseline_clean_100k_noearly_20260820"
EXPERIMENT_NAME="SATO_PP_MOTIONX322_PLUGIN_LASTBASELINE_CLEAN_NOEARLY_100K_SEED123_20260820"
EXPERIMENT="$MGPT_ROOT/experiments/mgpt_sato_pp/$EXPERIMENT_NAME"
EVAL_ROOT="$MGPT_ROOT/experiments/robustness/motionx322_sato_plugin_lastbaseline_clean100k_table3_20seed_20260820"

BASELINE_CKPT="$MGPT_ROOT/experiments/mgpt/Instruct_MOTIONX322_OFFICIAL_FRESH_R2_SEED123_20260817/checkpoints/last.ckpt"
BASELINE_EVAL="$MGPT_ROOT/experiments/robustness/motionx322_official_fresh_r2_last100k_table3_20seed_20260820"
BASELINE_MANIFEST="$BASELINE_EVAL/manifest.json"
BASELINE_SUMMARY="$BASELINE_EVAL/combined/summary.json"
STAGE1_SELECTION="$MGPT_ROOT/experiments/mgpt/VQVAE_MOTIONX322_OFFICIAL_FRESH_R2_SEED123_20260817/checkpoints/best-validation.json"
PAIR_FILE="$ROOT/dataset/Motion-X-322/preserving_pairs_gemini31_human_recipe_filtered.jsonl"
ASSETS_CFG="$MGPT_ROOT/configs/assets_motionx322.yaml"

STATE="$RUN_ROOT/handoff_state.tsv"
HEARTBEAT="$RUN_ROOT/handoff_heartbeat.txt"
TRAIN_LOG="$RUN_ROOT/train_launcher.log"
SOURCE_MANIFEST="$RUN_ROOT/source_manifest.json"
SELECTED="$EXPERIMENT/checkpoints/selected-last.json"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/mnt/workspace/Wilson/miniconda3/envs/mgpt/lib/python3.11/site-packages/torch/lib:/mnt/workspace/Wilson/miniconda3/envs/mgpt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export TMPDIR="/tmp/mgptc100k"

mkdir -p "$RUN_ROOT/tmp" "$TMPDIR"
: > "$STATE"

record() {
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" >> "$STATE"
}

fail() {
  record "FAIL:$1"
  printf '%s\n' "$1" >&2
  exit 1
}

checkpoint_step() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
import zipfile
import pickletools
from pathlib import Path

path = Path(sys.argv[1])
with zipfile.ZipFile(path) as archive:
    data_name = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
    operations = list(pickletools.genops(archive.read(data_name)))
integer_ops = {"BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"}
for index, (operation, argument, _) in enumerate(operations):
    if argument != "global_step":
        continue
    for value_operation, value_argument, _ in operations[index + 1:index + 8]:
        if value_operation.name in integer_ops:
            print(int(value_argument))
            raise SystemExit(0)
print(-1)
PY
}

wait_for_baseline_eval() {
  record "waiting_for_motionx_last100k_baseline_eval"
  while true; do
    date -u +%FT%TZ > "$HEARTBEAT"
    if [[ -s "$BASELINE_PIPELINE/test_handoff_complete.txt" && -s "$BASELINE_EVAL/complete.txt" && -s "$BASELINE_SUMMARY" && -s "$BASELINE_MANIFEST" ]]; then
      break
    fi
    if rg -qi 'FAIL|Traceback|CUDA out of memory|RuntimeError|ChildFailedError|ProcessExitedException' "$BASELINE_PIPELINE"/*.tsv "$BASELINE_PIPELINE"/*.log 2>/dev/null; then
      fail "baseline training/test handoff has failure marker"
    fi
    sleep 60
  done
}

verify_fresh_targets() {
  [[ -s "$BASELINE_CKPT" ]] || fail "missing baseline checkpoint: $BASELINE_CKPT"
  local step
  step="$(checkpoint_step "$BASELINE_CKPT")"
  record "motionx322_baseline_last_step=${step}"
  [[ "$step" == "100000" ]] || fail "Motion-X baseline last.ckpt is step ${step}, expected 100000"
  [[ ! -e "$EXPERIMENT" ]] || fail "plugin experiment already exists: $EXPERIMENT"
  [[ ! -e "$EVAL_ROOT" ]] || fail "plugin eval output already exists: $EVAL_ROOT"
}

prepare_source() {
  record "prepare_clean_lastbaseline_plugin_source"
  cd "$ROOT"
  "$PYTHON_BIN" tools/prepare_motiongpt_motionx_sato_plugin_last_baseline.py \
    --repo-root "$ROOT" \
    --template-config "$MGPT_ROOT/configs/config_motionx322_sato_pp.yaml" \
    --baseline-checkpoint "$BASELINE_CKPT" \
    --baseline-manifest "$BASELINE_MANIFEST" \
    --baseline-summary "$BASELINE_SUMMARY" \
    --stage1-selection "$STAGE1_SELECTION" \
    --pair-file "$PAIR_FILE" \
    --run-root "$RUN_ROOT" \
    --experiment-name "$EXPERIMENT_NAME" \
    --max-steps 100000 \
    --overwrite
}

run_training() {
  record "train_clean_lastbaseline_plugin:gpu0-3:no_earlystop"
  cd "$MGPT_ROOT"
  MASTER_PORT=29633 "$PYTHON_BIN" train.py \
    --cfg "$RUN_ROOT/config.yaml" \
    --cfg_assets configs/assets_motionx322.yaml \
    --device 0 1 2 3 \
    --nodebug > "$TRAIN_LOG" 2>&1
  if rg -qi 'Traceback|CUDA out of memory|ProcessExitedException|ChildFailedError|NCCL WARN|sato/(total|base):[[:space:]]*(nan|inf)|loss[^[:alnum:]]*[=:][^[:space:]]*(nan|inf)' "$TRAIN_LOG"; then
    fail "plugin training error signature"
  fi
  if ! rg -q 'Training ends!' "$TRAIN_LOG"; then
    fail "plugin training missing terminal marker"
  fi
  record "done_clean_lastbaseline_plugin_training"
}

run_eval_and_compare() {
  record "select_clean_plugin_last"
  cd "$ROOT"
  "$PYTHON_BIN" tools/select_motiongpt_sato_plugin_last_clean.py \
    --experiment-dir "$EXPERIMENT" \
    --source-manifest "$SOURCE_MANIFEST" \
    --max-steps 100000

  record "prepare_clean_plugin_table3_last"
  "$PYTHON_BIN" tools/prepare_motiongpt_motionx_sato_plugin_eval.py \
    --checkpoint "$EXPERIMENT/checkpoints/last.ckpt" \
    --plugin-selection "$SELECTED" \
    --source-manifest "$SOURCE_MANIFEST" \
    --baseline-manifest "$BASELINE_MANIFEST" \
    --base-config "$RUN_ROOT/config.yaml" \
    --assets-config "$ASSETS_CFG" \
    --output-dir "$EVAL_ROOT"

  record "eval_clean_plugin_table3_last:gpu0-3"
  MOTIONGPT_PYTHON="$PYTHON_BIN" \
  MOTIONGPT_MOTIONX_MANIFEST="$EVAL_ROOT/manifest.json" \
  MOTIONGPT_MOTIONX_CONFIG_DIR="$EVAL_ROOT/configs" \
  MOTIONGPT_MOTIONX_CONFIG_PATTERN='motiongpt_sato_pp_motionx322_seed%s.yaml' \
  MOTIONGPT_MOTIONX_ASSETS_CFG="$ASSETS_CFG" \
  MOTIONGPT_MOTIONX_OUTPUT_ROOT="$EVAL_ROOT" \
  MOTIONGPT_MOTIONX_GPUS="0 1 2 3" \
  bash "$MGPT_ROOT/scripts/run_motionx322_table3_eval.sh" \
    > "$RUN_ROOT/table3_eval.log" 2>&1

  record "compare_clean_plugin_with_lastbaseline"
  "$PYTHON_BIN" tools/compare_motiongpt_motionx_sato_plugin.py \
    --baseline-summary "$BASELINE_SUMMARY" \
    --plugin-summary "$EVAL_ROOT/combined/summary.json" \
    --baseline-manifest "$BASELINE_MANIFEST" \
    --plugin-manifest "$EVAL_ROOT/manifest.json" \
    --baseline-run-dir "$BASELINE_EVAL" \
    --plugin-run-dir "$EVAL_ROOT" \
    --output "$RUN_ROOT/comparison.json" \
    --markdown-output "$RUN_ROOT/comparison.md"
}

wait_for_baseline_eval
verify_fresh_targets
prepare_source
run_training
run_eval_and_compare
record "complete"
date -u +%FT%TZ > "$RUN_ROOT/complete.txt"
