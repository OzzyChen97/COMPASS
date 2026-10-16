#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---dry-run}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${HUMANML_LEVEL_RUN_ROOT:-$ROOT/ablation/humanml3d_table1_levels/runs}"
FINAL_ROOT="${HUMANML_LEVEL_FINAL_ROOT:-$ROOT/ablation/humanml3d_table1_levels/final}"
GPU_LIST_STRING="${HUMANML_LEVEL_GPUS:-1 2 3 4 7}"
read -r -a GPUS <<< "$GPU_LIST_STRING"

if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "HUMANML_LEVEL_GPUS must contain at least one GPU" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU id: $gpu" >&2; exit 2; }
done

asset_audit() {
  python "$ROOT/tools/audit_humanml3d_table1_level_assets.py" \
    --output "$ROOT/ablation/humanml3d_table1_levels/assets_manifest.json" \
    > "$ROOT/ablation/humanml3d_table1_levels/assets_audit.log"
}

launch_job() {
  local job="$1" gpu="$2" log_root="$3"
  local kind level model start count end
  read -r kind level model start count <<< "$job"
  mkdir -p "$log_root"
  if [[ "$kind" == "t2m" ]]; then
    "$ROOT/tools/run_humanml3d_table1_level_t2mgpt_satopp.sh" \
      "$level" "$model" "$gpu" "$start" "$count" "$RUN_ROOT" \
      > "$log_root/${kind}_${model}_${level}_seed${start}_n${count}_gpu${gpu}.log" 2>&1 &
  elif [[ "$kind" == "motiongpt" ]]; then
    end=$((start + count - 1))
    "$ROOT/tools/run_humanml3d_table1_level_motiongpt.sh" \
      "$level" "$gpu" "$start" "$end" "$RUN_ROOT" \
      > "$log_root/${kind}_${level}_seed${start}_${end}_gpu${gpu}.log" 2>&1 &
  else
    echo "unknown job kind: $kind" >&2
    return 2
  fi
  LAUNCHED_PID=$!
}

job_is_complete() {
  local job="$1"
  local kind level model start count end lower model_name marker
  read -r kind level model start count <<< "$job"
  end=$((start + count - 1))
  lower="${level,,}"

  if [[ "$kind" == "t2m" ]]; then
    case "$model" in
      t2mgpt) model_name="baseline_t2mgpt_original" ;;
      satopp) model_name="sato_pp_allloss_l1_iter130k" ;;
      *) return 1 ;;
    esac
    marker="$RUN_ROOT/t2mgpt_satopp/$lower/$model_name/complete_seed${start}_${end}.txt"
    [[ -s "$marker" ]]
  elif [[ "$kind" == "motiongpt" ]]; then
    compgen -G "$RUN_ROOT/motiongpt/$lower/complete_gpu*_seed${start}_${end}.txt" \
      >/dev/null
  else
    return 1
  fi
}

run_jobs() {
  local phase="$1"
  shift
  local requested_jobs=("$@")
  local jobs=()
  local candidate
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
  python "$ROOT/tools/aggregate_humanml3d_table1_level_rprecision.py" \
    --run-root "$RUN_ROOT" \
    --asset-manifest "$ROOT/ablation/humanml3d_table1_levels/assets_manifest.json" \
    --motiongpt-source-root "$ROOT/ablation/humanml3d_table1_levels/motiongpt_sources" \
    --output-dir "$output" \
    --start-seed 123 \
    --num-repeats "$repeats"
}

smoke_jobs=()
for level in P1 P2 P3; do
  smoke_jobs+=("t2m $level t2mgpt 123 1")
  smoke_jobs+=("t2m $level satopp 123 1")
  smoke_jobs+=("motiongpt $level motiongpt 123 1")
done

formal_jobs=()
for level in P1 P2 P3; do
  formal_jobs+=("t2m $level t2mgpt 123 20")
  formal_jobs+=("t2m $level satopp 123 20")
done
for level in P1 P2 P3; do
  formal_jobs+=("motiongpt $level motiongpt 123 5")
  formal_jobs+=("motiongpt $level motiongpt 128 5")
  formal_jobs+=("motiongpt $level motiongpt 133 5")
  formal_jobs+=("motiongpt $level motiongpt 138 5")
done

case "$MODE" in
  --dry-run)
    echo "GPUs: ${GPUS[*]}"
    echo "Smoke jobs (${#smoke_jobs[@]}):"
    printf '  %s\n' "${smoke_jobs[@]}"
    echo "Formal jobs (${#formal_jobs[@]}):"
    printf '  %s\n' "${formal_jobs[@]}"
    ;;
  --smoke)
    asset_audit
    run_jobs smoke "${smoke_jobs[@]}"
    aggregate_results 1 "$ROOT/ablation/humanml3d_table1_levels/smoke_final"
    date -Is > "$ROOT/ablation/humanml3d_table1_levels/smoke_complete.txt"
    ;;
  --full)
    [[ -f "$ROOT/ablation/humanml3d_table1_levels/smoke_complete.txt" ]] || {
      echo "run --smoke successfully before --full" >&2
      exit 1
    }
    asset_audit
    run_jobs formal "${formal_jobs[@]}"
    aggregate_results 20 "$FINAL_ROOT"
    date -Is > "$ROOT/ablation/humanml3d_table1_levels/full_complete.txt"
    ;;
  --all)
    "$0" --smoke
    "$0" --full
    ;;
  *)
    echo "usage: $0 --dry-run|--smoke|--full|--all" >&2
    exit 2
    ;;
esac