#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ROOT="${MOTIONX_LEVEL_ASSET_ROOT:-$ROOT/ablation/motionx322_level_rprecision}"
ASSET_MANIFEST="$ASSET_ROOT/assets_manifest.json"
RUN_ROOT="${MOTIONX_LEVEL_RUN_ROOT:-$ASSET_ROOT/runs}"
FINAL_ROOT="${MOTIONX_LEVEL_FINAL_ROOT:-$ASSET_ROOT/final}"
PYTHON_BIN="${T2M_PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
GPU_LIST_STRING="${MOTIONX_LEVEL_GPUS:-0 1 2 3 4 5 6 7}"
read -r -a GPUS <<< "$GPU_LIST_STRING"

if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "MOTIONX_LEVEL_GPUS must contain at least one GPU" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU id: $gpu" >&2; exit 2; }
  (( gpu <= 7 )) || { echo "GPU id out of range: $gpu" >&2; exit 2; }
done

prepare_assets() {
  if [[ -s "$ASSET_MANIFEST" ]]; then
    echo "frozen assets already exist; validating them"
    asset_audit
    return 0
  fi
  bash "$ROOT/tools/prepare_motionx322_level_rprecision_assets.sh" --run
  asset_audit
}

asset_audit() {
  [[ -s "$ASSET_MANIFEST" ]] || {
    echo "missing frozen asset manifest: $ASSET_MANIFEST" >&2
    return 1
  }
  local candidate
  candidate="$(mktemp /tmp/motionx-level-assets.XXXXXX.json)"
  trap 'rm -f "$candidate"' RETURN
  "$PYTHON_BIN" "$ROOT/tools/audit_motionx322_level_rprecision_assets.py" \
    --output "$candidate" > "$ASSET_ROOT/assets_audit_latest.log"
  if ! cmp -s "$candidate" "$ASSET_MANIFEST"; then
    echo "frozen Motion-X assets differ from assets_manifest.json" >&2
    diff -u "$ASSET_MANIFEST" "$candidate" >&2 || true
    return 1
  fi
  rm -f "$candidate"
  trap - RETURN
}

launch_job() {
  local job="$1" gpu="$2" log_root="$3"
  local kind level model start count end
  read -r kind level model start count <<< "$job"
  end=$((start + count - 1))
  mkdir -p "$log_root"
  if [[ "$kind" == "t2m" ]]; then
    bash "$ROOT/tools/run_motionx322_level_t2mgpt_satopp.sh" \
      "$level" "$model" "$gpu" "$start" "$count" "$RUN_ROOT" \
      > "$log_root/${kind}_${model}_${level}_seed${start}_n${count}_gpu${gpu}.log" 2>&1 &
  elif [[ "$kind" == "motiongpt" ]]; then
    bash "$ROOT/tools/run_motionx322_level_motiongpt.sh" \
      "$model" "$level" "$gpu" "$start" "$end" "$RUN_ROOT" \
      > "$log_root/${kind}_${model}_${level}_seed${start}_${end}_gpu${gpu}.log" 2>&1 &
  else
    echo "unknown job kind: $kind" >&2
    return 2
  fi
  LAUNCHED_PID=$!
}

job_is_complete() {
  local job="$1"
  local kind level model start count end lower
  read -r kind level model start count <<< "$job"
  end=$((start + count - 1))
  lower="${level,,}"
  if [[ "$kind" == "t2m" ]]; then
    [[ -s "$RUN_ROOT/t2m_family/$lower/$model/complete_seed${start}_${end}.txt" ]]
  elif [[ "$kind" == "motiongpt" ]]; then
    compgen -G \
      "$RUN_ROOT/motiongpt_family/$model/$lower/complete_gpu*_seed${start}_${end}.txt" \
      >/dev/null
  else
    return 1
  fi
}

run_jobs() {
  local phase="$1"
  shift
  local requested_jobs=("$@") jobs=() candidate
  for candidate in "${requested_jobs[@]}"; do
    if job_is_complete "$candidate"; then
      echo "skipped completed job=$candidate"
    else
      jobs+=("$candidate")
    fi
  done
  if (( ${#jobs[@]} == 0 )); then
    echo "$phase has no pending jobs"
    return 0
  fi

  local next=0 active=0 failed=0 finished status gpu job
  local log_root="$RUN_ROOT/orchestrator/$phase"
  declare -A pid_gpu=()
  declare -A pid_job=()

  terminate_children() {
    local pid
    for pid in "${!pid_job[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
  }
  trap terminate_children INT TERM

  while (( next < ${#jobs[@]} && active < ${#GPUS[@]} )); do
    gpu="${GPUS[$active]}"
    job="${jobs[$next]}"
    launch_job "$job" "$gpu" "$log_root"
    pid_gpu[$LAUNCHED_PID]="$gpu"
    pid_job[$LAUNCHED_PID]="$job"
    echo "launched pid=$LAUNCHED_PID gpu=$gpu job=$job"
    next=$((next + 1))
    active=$((active + 1))
  done

  while (( active > 0 )); do
    status=0
    wait -n -p finished || status=$?
    gpu="${pid_gpu[$finished]}"
    job="${pid_job[$finished]}"
    unset 'pid_gpu[$finished]' 'pid_job[$finished]'
    active=$((active - 1))
    if (( status != 0 )); then
      echo "failed status=$status pid=$finished gpu=$gpu job=$job" >&2
      failed=1
    else
      echo "completed pid=$finished gpu=$gpu job=$job"
    fi
    if (( failed == 0 && next < ${#jobs[@]} )); then
      job="${jobs[$next]}"
      launch_job "$job" "$gpu" "$log_root"
      pid_gpu[$LAUNCHED_PID]="$gpu"
      pid_job[$LAUNCHED_PID]="$job"
      echo "launched pid=$LAUNCHED_PID gpu=$gpu job=$job"
      next=$((next + 1))
      active=$((active + 1))
    fi
  done
  trap - INT TERM

  if (( failed != 0 )); then
    echo "$phase stopped before scheduling all jobs" >&2
    return 1
  fi
  if (( next != ${#jobs[@]} )); then
    echo "$phase scheduled $next/${#jobs[@]} jobs" >&2
    return 1
  fi
}

aggregate_results() {
  local repeats="$1" output="$2"
  "$PYTHON_BIN" "$ROOT/tools/aggregate_motionx322_level_rprecision.py" \
    --run-root "$RUN_ROOT" \
    --asset-manifest "$ASSET_MANIFEST" \
    --motiongpt-source-root "$ASSET_ROOT/motiongpt_sources" \
    --output-dir "$output" \
    --start-seed 123 \
    --num-repeats "$repeats"
}

smoke_jobs=()
for level in P1 P2 P3; do
  smoke_jobs+=("t2m $level t2mgpt 123 1")
  smoke_jobs+=("motiongpt $level motiongpt 123 1")
  smoke_jobs+=("t2m $level satopp 123 1")
  smoke_jobs+=("motiongpt $level motiongpt_satopp 123 1")
done

formal_jobs=()
for level in P1 P2 P3; do
  formal_jobs+=("t2m $level t2mgpt 123 20")
  formal_jobs+=("t2m $level satopp 123 20")
done
for level in P1 P2 P3; do
  for model in motiongpt motiongpt_satopp; do
    formal_jobs+=("motiongpt $level $model 123 5")
    formal_jobs+=("motiongpt $level $model 128 5")
    formal_jobs+=("motiongpt $level $model 133 5")
    formal_jobs+=("motiongpt $level $model 138 5")
  done
done

case "$MODE" in
  --dry-run)
    echo "Asset root: $ASSET_ROOT"
    echo "Run root: $RUN_ROOT"
    echo "GPUs: ${GPUS[*]}"
    echo "Smoke jobs (${#smoke_jobs[@]}):"
    printf '  %s\n' "${smoke_jobs[@]}"
    echo "Formal jobs (${#formal_jobs[@]}):"
    printf '  %s\n' "${formal_jobs[@]}"
    ;;
  --prepare)
    prepare_assets
    ;;
  --smoke)
    asset_audit
    run_jobs smoke "${smoke_jobs[@]}"
    aggregate_results 1 "$ASSET_ROOT/smoke_final"
    date -Is > "$ASSET_ROOT/smoke_complete.txt"
    ;;
  --full)
    [[ -s "$ASSET_ROOT/smoke_complete.txt" ]] || {
      echo "run --smoke successfully before --full" >&2
      exit 1
    }
    asset_audit
    run_jobs formal "${formal_jobs[@]}"
    aggregate_results 20 "$FINAL_ROOT"
    date -Is > "$ASSET_ROOT/full_complete.txt"
    ;;
  --aggregate-smoke)
    asset_audit
    aggregate_results 1 "$ASSET_ROOT/smoke_final"
    ;;
  --aggregate-full)
    asset_audit
    aggregate_results 20 "$FINAL_ROOT"
    ;;
  --all)
    bash "$0" --prepare
    bash "$0" --smoke
    bash "$0" --full
    ;;
  *)
    echo "usage: $0 --dry-run|--prepare|--smoke|--full|--aggregate-smoke|--aggregate-full|--all" >&2
    exit 2
    ;;
esac