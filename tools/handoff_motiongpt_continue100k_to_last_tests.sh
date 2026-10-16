#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/workspace/Wilson/T2M-GPT"
MGPT_ROOT="$ROOT/MotionGPT"
RUN_ROOT="$MGPT_ROOT/experiments/motiongpt_continue_last_to_100k_noearly_20260820"
PYTHON_BIN="/mnt/workspace/Wilson/miniconda3/envs/mgpt/bin/python"
STATE="$RUN_ROOT/test_handoff_state.tsv"
HEARTBEAT="$RUN_ROOT/test_handoff_heartbeat.txt"
SHORT_TMPDIR="/tmp/mgptc100k"

H3D_EXP="$MGPT_ROOT/experiments/mgpt/Instruct_H3D_OFFICIAL_FRESH_SEED123_20260818"
MX_EXP="$MGPT_ROOT/experiments/mgpt/Instruct_MOTIONX322_OFFICIAL_FRESH_R2_SEED123_20260817"
H3D_CKPT="$H3D_EXP/checkpoints/last.ckpt"
MX_CKPT="$MX_EXP/checkpoints/last.ckpt"

H3D_EVAL="$MGPT_ROOT/experiments/robustness/humanml3d_official_fresh_last100k_table1_full4640_20seed_20260820"
MX_EVAL="$MGPT_ROOT/experiments/robustness/motionx322_official_fresh_r2_last100k_table3_20seed_20260820"

H3D_PAIR="$MGPT_ROOT/experiments/robustness/humanml3d_caption_locked_sato_mix_repair_all_4640_20260810/pairs_sato_mix_4640.json"
H3D_INDEX="$MGPT_ROOT/experiments/robustness/humanml3d_caption_locked_sato_mix_repair_all_4640_20260810/sample_index_seed123.json"

record() {
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" >> "$STATE"
}

mkdir -p "$SHORT_TMPDIR"
export TMPDIR="$SHORT_TMPDIR"

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

wait_for_training_sequence() {
  record "waiting_for_training_sequence_complete"
  while true; do
    date -u +%FT%TZ > "$HEARTBEAT"
    if rg -q '^.*\tcomplete$' "$RUN_ROOT/state.tsv" 2>/dev/null; then
      break
    fi
    if rg -qi 'FAIL|Traceback|CUDA out of memory|RuntimeError|ChildFailedError|ProcessExitedException' "$RUN_ROOT/state.tsv" "$RUN_ROOT"/*.log 2>/dev/null; then
      fail "training sequence has failure marker"
    fi
    sleep 60
  done
  record "training_sequence_complete"
}

verify_100k() {
  local label="$1" ckpt="$2" step
  [[ -s "$ckpt" ]] || fail "missing checkpoint: $ckpt"
  step="$(checkpoint_step "$ckpt")"
  record "${label}_last_step=${step}"
  [[ "$step" == "100000" ]] || fail "${label} last.ckpt is step ${step}, expected 100000"
}

run_humanml3d_table1() {
  if [[ -e "$H3D_EVAL" ]]; then
    fail "HumanML3D output already exists: $H3D_EVAL"
  fi
  record "prepare_humanml3d_table1_last100k"
  cd "$ROOT"
  "$PYTHON_BIN" tools/prepare_motiongpt_humanml3d_table1.py \
    --checkpoint "$H3D_CKPT" \
    --base-config "$MGPT_ROOT/experiments/humanml3d_official_fresh_20260818/configs/stage3.yaml" \
    --assets-config "$MGPT_ROOT/configs/assets_humanml3d_official_local.yaml" \
    --pair-file "$H3D_PAIR" \
    --fixed-sample-index "$H3D_INDEX" \
    --output-dir "$H3D_EVAL"

  record "eval_humanml3d_table1_last100k"
  mkdir -p "$H3D_EVAL/logs"
  pids=()
  for shard in 0 1 2 3; do
    case "$shard" in
      0) start=123; end=127 ;;
      1) start=128; end=132 ;;
      2) start=133; end=137 ;;
      3) start=138; end=142 ;;
    esac
    MOTIONGPT_HUMANML_MANIFEST="$H3D_EVAL/manifest.json" \
    MOTIONGPT_HUMANML_CONFIG_ROOT="$H3D_EVAL/configs" \
    MOTIONGPT_HUMANML_CONFIG_PATTERN='motiongpt_humanml3d_seed%s.yaml' \
    OUTPUT_ROOT="$H3D_EVAL" \
    PAIR_FILE="$H3D_PAIR" \
    ALLOWED_LEVELS='P1 P2 P3 fallback' \
    CAPTION_LENGTH_POLICY='repair-all' \
    FIXED_SAMPLE_INDEX="$H3D_INDEX" \
    NUM_WORKERS=0 \
    bash "$MGPT_ROOT/scripts/run_humanml3d_loader_eval_shard.sh" "$shard" "$start" "$end" \
      > "$H3D_EVAL/logs/shard${shard}_driver.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  [[ "$failed" -eq 0 ]] || fail "HumanML3D Table-1 eval shard failed"
  "$PYTHON_BIN" tools/aggregate_motiongpt_loader_eval.py \
    --dataset humanml3d --run-dir "$H3D_EVAL" --output-dir "$H3D_EVAL/combined" \
    --expected-samples 4640 --require-caption-locked \
    --expected-unique-pairs 4353 --expected-levels P1 P2 P3 fallback \
    --source-manifest "$H3D_EVAL/manifest.json"
  date -u +%FT%TZ > "$H3D_EVAL/complete.txt"
  record "complete_humanml3d_table1_last100k"
}

run_motionx_table3() {
  if [[ -e "$MX_EVAL" ]]; then
    fail "Motion-X output already exists: $MX_EVAL"
  fi
  record "prepare_motionx_table3_last100k"
  cd "$ROOT"
  "$PYTHON_BIN" tools/prepare_motiongpt_motionx_table3.py \
    --checkpoint "$MX_CKPT" \
    --selection-policy last \
    --base-config "$MGPT_ROOT/experiments/motionx322_official_fresh_r2_20260817/configs/stage3.yaml" \
    --assets-config "$MGPT_ROOT/configs/assets_motionx322.yaml" \
    --output-dir "$MX_EVAL"

  record "eval_motionx_table3_last100k"
  MOTIONGPT_PYTHON="$PYTHON_BIN" \
  MOTIONGPT_MOTIONX_MANIFEST="$MX_EVAL/manifest.json" \
  MOTIONGPT_MOTIONX_CONFIG_DIR="$MX_EVAL/configs" \
  MOTIONGPT_MOTIONX_ASSETS_CFG="$MGPT_ROOT/configs/assets_motionx322.yaml" \
  MOTIONGPT_MOTIONX_OUTPUT_ROOT="$MX_EVAL" \
  MOTIONGPT_MOTIONX_GPUS="0 1 2 3" \
  bash "$MGPT_ROOT/scripts/run_motionx322_table3_eval.sh" \
    > "$RUN_ROOT/motionx_last100k_table3_eval.log" 2>&1
  [[ -s "$MX_EVAL/complete.txt" ]] || fail "Motion-X Table-3 eval missing complete marker"
  record "complete_motionx_table3_last100k"
}

: > "$STATE"
wait_for_training_sequence
verify_100k humanml3d "$H3D_CKPT"
verify_100k motionx322 "$MX_CKPT"
run_humanml3d_table1
run_motionx_table3
record "complete"
date -u +%FT%TZ > "$RUN_ROOT/test_handoff_complete.txt"
