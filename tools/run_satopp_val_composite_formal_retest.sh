#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
RUN_DIR="ablation/motionx322_sato/max_token75_gemini31_human_recipe_100k_lr60k_bs132_seed123"
OUT_ROOT="${SATOPP_COMPOSITE_OUT_ROOT:-ablation/motionx322_sato/val_selected_fid_fidp_fidd_retest_20260823}"
VAL_PAIR="$OUT_ROOT/validation/val_pairs_train_gemini_exact_mixed.json"
VAL_PAIR_META="$VAL_PAIR.meta.json"
CANDIDATE_MANIFEST="$OUT_ROOT/selection/checkpoint_candidates.json"
PROTOCOL_MANIFEST="$OUT_ROOT/selection/protocol_manifest.json"
VAL_SHARD_ROOT="$OUT_ROOT/validation/shards"
SELECTION_DIR="$OUT_ROOT/selection/result"
FORMAL_SPEC="$OUT_ROOT/formal/selected_spec.json"
FORMAL_ROOT="$OUT_ROOT/formal/table3_repeat20"
FORMAL_SHARD_ROOT="$FORMAL_ROOT/shards"
FORMAL_COMBINED="$FORMAL_ROOT/combined"
FORMAL_REPORT="$FORMAL_ROOT/SATOPP_VAL_COMPOSITE_SELECTED_TABLE3_REPEAT20.md"
TEST_PAIR="dataset/Motion-X-322/test_perb_gemini31_max75_mixed_full.json"
DATA_ROOT="dataset/Motion-X-322-MaxToken75"
SOURCE_ROOT="dataset/Motion-X-322"
SOURCE_POOL="dataset/Motion-X-322/preserving_pairs_gemini31_human_recipe_filtered.jsonl"
GPU_IDS=(1 2 3 4)
VAL_SEEDS=(123 124 125)
FORMAL_START_SEED=123
FORMAL_REPEATS=20
FORMAL_REPEATS_PER_GPU=5
MIN_FREE_MIB="${SATOPP_MIN_FREE_MIB:-70000}"
MAX_UTIL_PCT="${SATOPP_MAX_UTIL_PCT:-20}"
WAIT_SECONDS="${SATOPP_GPU_WAIT_SECONDS:-60}"

mkdir -p "$OUT_ROOT"

state() {
  printf '%s\t%s\n' "$(date -Is)" "$1" | tee "$OUT_ROOT/state.tsv"
}

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

write_validation_specs() {
  "$PYTHON" - "$CANDIDATE_MANIFEST" "$OUT_ROOT/validation/specs" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
for candidate in manifest['candidates']:
    spec = [{
        'name': candidate['candidate_id'],
        'kind': 'sato_pp',
        'checkpoint': candidate['canonical_path'],
        'description': 'Frozen SATO++ validation-selection candidate',
    }]
    (out / f"{candidate['candidate_id']}.json").write_text(
        json.dumps(spec, indent=2) + '\n', encoding='utf-8')
PY
}

verify_frozen_inputs() {
  "$PYTHON" - "$PROTOCOL_MANIFEST" "$CANDIDATE_MANIFEST" "$VAL_PAIR" "$TEST_PAIR" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024), b''): h.update(chunk)
    return h.hexdigest()
p=json.loads(Path(sys.argv[1]).read_text())
checks={
    'candidate_manifest_sha256': sha(sys.argv[2]),
    'validation_pair_sha256': sha(sys.argv[3]),
    'test_pair_sha256': sha(sys.argv[4]),
}
expected={
    'candidate_manifest_sha256': p['candidate_manifest_sha256'],
    'validation_pair_sha256': p['validation']['pair_file_sha256'],
    'test_pair_sha256': p['formal_test']['pair_file_sha256'],
}
if checks != expected:
    raise SystemExit(f'Frozen input mismatch: actual={checks}, expected={expected}')
if p.get('test_metrics_used_for_selection') is not False:
    raise SystemExit('Protocol does not forbid test-metric checkpoint selection')
print(json.dumps(checks, indent=2))
PY
}

prepare() {
  require_file "$RUN_DIR/sato_pp_best_fid.pth"
  require_file "$TEST_PAIR"
  require_file "$SOURCE_POOL"
  require_file "$DATA_ROOT/val.txt"
  mkdir -p "$OUT_ROOT/validation" "$OUT_ROOT/selection"

  if [[ ! -f "$VAL_PAIR" ]]; then
    state "building_validation_pairs"
    "$PYTHON" tools/build_motionx322_val_pairs_from_train_pool.py \
      --dataset-root "$DATA_ROOT" \
      --split val \
      --source-dataset-root "$SOURCE_ROOT" \
      --source-pool "$SOURCE_POOL" \
      --output "$VAL_PAIR" \
      --seed 123 \
      --weights 0.46 0.27 0.27 \
      > "$OUT_ROOT/validation/build_pairs.log" 2>&1
  fi
  require_file "$VAL_PAIR"
  require_file "$VAL_PAIR_META"

  if [[ ! -f "$CANDIDATE_MANIFEST" ]]; then
    state "hashing_checkpoint_candidates"
    "$PYTHON" tools/build_satopp_checkpoint_manifest.py \
      --run-dir "$RUN_DIR" \
      --output "$CANDIDATE_MANIFEST" \
      > "$OUT_ROOT/selection/build_candidates.log" 2>&1
  fi
  require_file "$CANDIDATE_MANIFEST"
  write_validation_specs

  if [[ ! -f "$PROTOCOL_MANIFEST" ]]; then
    state "freezing_validation_selection_protocol"
    "$PYTHON" tools/freeze_satopp_validation_protocol.py \
      --candidate-manifest "$CANDIDATE_MANIFEST" \
      --validation-pairs "$VAL_PAIR" \
      --test-pairs "$TEST_PAIR" \
      --dataset-root "$DATA_ROOT" \
      --validation-shard-root "$VAL_SHARD_ROOT" \
      --output "$PROTOCOL_MANIFEST" \
      --validation-seeds "${VAL_SEEDS[@]}" \
      --formal-start-seed "$FORMAL_START_SEED" \
      --formal-repeats "$FORMAL_REPEATS" \
      > "$OUT_ROOT/selection/freeze_protocol.log" 2>&1
  fi
  require_file "$PROTOCOL_MANIFEST"
  verify_frozen_inputs > "$OUT_ROOT/selection/verify_frozen_inputs.log"
  state "protocol_frozen_ready"
}

gpus_available() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | \
    "$PYTHON" -c '
import sys
wanted={1,2,3,4}; minimum=int(sys.argv[1]); max_util=int(sys.argv[2]); seen=set()
for line in sys.stdin:
    index, free, util=[int(value.strip()) for value in line.split(",")]
    if index in wanted:
        seen.add(index)
        if free < minimum or util > max_util:
            raise SystemExit(1)
raise SystemExit(0 if seen == wanted else 1)
' "$MIN_FREE_MIB" "$MAX_UTIL_PCT"
}

wait_for_gpus() {
  while ! gpus_available; do
    state "waiting_for_gpus_1_4"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits > "$OUT_ROOT/last_gpu_wait_snapshot.csv"
    sleep "$WAIT_SECONDS"
  done
  state "gpus_1_4_available"
}

run_validation_gpu() {
  local gpu="$1"
  local shard_index="$2"
  local shard="$VAL_SHARD_ROOT/gpu${gpu}"
  local tmp="/tmp/sv${gpu}"
  mkdir -p "$shard" "$tmp"

  mapfile -t candidate_ids < <(
    "$PYTHON" - "$CANDIDATE_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
for row in json.loads(Path(sys.argv[1]).read_text())['candidates']:
    print(row['candidate_id'])
PY
  )
  local index=0
  for candidate_id in "${candidate_ids[@]}"; do
    if (( index % ${#GPU_IDS[@]} != shard_index )); then
      index=$((index + 1))
      continue
    fi
    local spec="$OUT_ROOT/validation/specs/${candidate_id}.json"
    local out="$shard/${candidate_id}"
    local report="$shard/${candidate_id}.md"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$gpu" TMPDIR="$tmp" "$PYTHON" \
      ablation/repeat20_perturb_fast_batch.py \
      --spec-json "$spec" \
      --pair-data-path "$VAL_PAIR" \
      --out-dir "$out" \
      --report-md "$report" \
      --dataset-split val \
      --start-seed 123 \
      --num-repeats 3 \
      --batch-size 32 \
      --num-workers 0 \
      --caption-locked \
      --caption-locked-length-policy strict \
      --sampling-pair-rng-mode sequential \
      --fast-batch \
      --cfg-scale 1.3 \
      --cfg-uncond zero \
      --sample-mode categorical \
      --sample-temperature 1.0 \
      --top-k 0 \
      --top-p 1.0 \
      --wandb-mode disabled \
      > "$shard/${candidate_id}_driver.log" 2>&1
    index=$((index + 1))
  done
}

run_validation() {
  prepare
  wait_for_gpus
  state "validation_sweep_running"
  mkdir -p "$VAL_SHARD_ROOT"
  local pids=()
  local shard_index=0
  for gpu in "${GPU_IDS[@]}"; do
    run_validation_gpu "$gpu" "$shard_index" &
    pids+=("$!")
    shard_index=$((shard_index + 1))
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if (( failed )); then
    state "validation_sweep_failed"
    return 1
  fi
  state "validation_sweep_complete"
}

select_checkpoint() {
  prepare
  state "selecting_validation_composite_checkpoint"
  "$PYTHON" tools/select_satopp_validation_composite.py \
    --candidate-manifest "$CANDIDATE_MANIFEST" \
    --protocol-manifest "$PROTOCOL_MANIFEST" \
    --shard-root "$VAL_SHARD_ROOT" \
    --output-dir "$SELECTION_DIR" \
    --formal-spec "$FORMAL_SPEC" \
    > "$OUT_ROOT/selection/select_checkpoint.log" 2>&1
  require_file "$SELECTION_DIR/selection_result.json"
  require_file "$FORMAL_SPEC"
  state "validation_checkpoint_locked"
}

run_formal_gpu() {
  local gpu="$1"
  local start_seed="$2"
  local shard="$FORMAL_SHARD_ROOT/gpu${gpu}"
  local tmp="/tmp/sf${gpu}"
  mkdir -p "$shard" "$tmp"
  CUDA_VISIBLE_DEVICES="$gpu" TMPDIR="$tmp" "$PYTHON" \
    ablation/repeat20_perturb_fast_batch.py \
    --spec-json "$FORMAL_SPEC" \
    --pair-data-path "$TEST_PAIR" \
    --out-dir "$shard" \
    --report-md "${shard}.md" \
    --dataset-split test \
    --start-seed "$start_seed" \
    --num-repeats "$FORMAL_REPEATS_PER_GPU" \
    --batch-size 32 \
    --num-workers 8 \
    --humanml-fast-protocol \
    --cfg-scale 1.3 \
    --cfg-uncond zero \
    --sample-mode categorical \
    --sample-temperature 1.0 \
    --top-k 0 \
    --top-p 1.0 \
    --wandb-mode disabled \
    > "${shard}_driver.log" 2>&1
}

run_formal() {
  require_file "$SELECTION_DIR/selection_result.json"
  require_file "$FORMAL_SPEC"
  if [[ -f "$FORMAL_ROOT/complete.txt" ]]; then
    state "formal_test_already_complete"
    return 0
  fi
  wait_for_gpus
  state "formal_table3_repeat20_running"
  mkdir -p "$FORMAL_SHARD_ROOT"
  local pids=()
  local offset=0
  for gpu in "${GPU_IDS[@]}"; do
    run_formal_gpu "$gpu" "$((FORMAL_START_SEED + offset))" &
    pids+=("$!")
    offset=$((offset + FORMAL_REPEATS_PER_GPU))
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if (( failed )); then
    state "formal_table3_repeat20_failed"
    return 1
  fi

  "$PYTHON" tools/merge_repeat_eval_shards.py \
    --spec-json "$FORMAL_SPEC" \
    --shard-root "$FORMAL_SHARD_ROOT" \
    --out-dir "$FORMAL_COMBINED" \
    --pair-data-path "$TEST_PAIR" \
    --report-md "$FORMAL_REPORT" \
    --start-seed "$FORMAL_START_SEED" \
    --num-repeats "$FORMAL_REPEATS" \
    --expected-matched-pairs 7733 \
    --expected-total-pairs 7733 \
    --cfg-scale 1.3 \
    > "$FORMAL_ROOT/merge.log" 2>&1
  "$PYTHON" tools/summarize_satopp_val_composite_formal.py \
    --root "$OUT_ROOT" \
    --previous-summary \
      ablation/motionx322_sato/gemini31_100k_full_test_human_protocol_cfg1p3_repeat20/combined/summary.json \
    > "$FORMAL_ROOT/summarize.log" 2>&1
  date -Is > "$FORMAL_ROOT/complete.txt"
  state "formal_table3_repeat20_complete"
}

case "$STAGE" in
  prepare)
    prepare
    ;;
  validation)
    run_validation
    ;;
  select)
    select_checkpoint
    ;;
  formal)
    run_formal
    ;;
  all)
    run_validation
    select_checkpoint
    run_formal
    ;;
  *)
    echo "Usage: $0 [prepare|validation|select|formal|all]" >&2
    exit 2
    ;;
esac