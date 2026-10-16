#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
STAGE="${1:-status}"
RUN_ROOT="${RUN_ROOT:-ablation/var_cov_loss_ablation}"
SEED="${SEED:-123}"
START_SEED=123
NUM_REPEATS=20
ALLOW_STREAM_RESET_RESUME="${ALLOW_STREAM_RESET_RESUME:-0}"

H_PAIR_TRAIN=dataset/HumanML3D/preserving_pairs_filtered.jsonl
H_PAIR_TEST=dataset/HumanML3D/test_perb_deterministic.json
H_VQ=pretrained/VQVAE/net_best_fid.pth
H_GPT=pretrained/VQTransformer_corruption05/net_best_fid.pth
H_FULL="${H_FULL:-ablation/loss_full_allloss_l1_150k/sato_pp_last.pth}"

M_PAIR_TRAIN=dataset/Motion-X-322/preserving_pairs_gemini31_human_recipe_filtered.jsonl
M_PAIR_TEST=dataset/Motion-X-322/test_perb_gemini31_max75_mixed_full.json
M_VQ=output_motionx322/VQVAE_MOTIONX322/net_best_fid.pth
M_GPT=output_motionx322/GPT_MOTIONX322_MAXTOK75_200K/net_best_fid.pth
M_FULL=ablation/motionx322_sato/max_token75_gemini31_human_recipe_100k_lr60k_bs132_seed123/sato_pp_last.pth

require_file() {
  [[ -s "$1" ]] || { echo "missing required file: $1" >&2; exit 1; }
}

wait_training_lock_released() {
  local out="$1" lock_fd
  exec {lock_fd}>"$out/train.lock"
  echo "waiting for training process to release $out/train.lock"
  flock "$lock_fd"
  flock -u "$lock_fd"
  exec {lock_fd}>&-
}

variant_weights() {
  case "$1" in
    full)    printf '%s %s\n' 0.5 0.04 ;;
    no_var)  printf '%s %s\n' 0.0 0.04 ;;
    no_cov)  printf '%s %s\n' 0.5 0.0 ;;
    no_both) printf '%s %s\n' 0.0 0.0 ;;
    *) echo "unknown variant: $1" >&2; return 2 ;;
  esac
}

write_meta() {
  local out="$1" dataset="$2" variant="$3" gpu="$4" var="$5" cov="$6"
  {
    echo "dataset=$dataset"
    echo "variant=$variant"
    echo "gpu=$gpu"
    echo "seed=$SEED"
    echo "lambda_var=$var"
    echo "lambda_cov=$cov"
    echo "started_at=$(date -Is)"
    echo "git_head=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  } > "$out/run_meta.txt"
}

train_human() {
  local variant="$1" gpu="$2"
  local var cov exp out code
  read -r var cov < <(variant_weights "$variant")
  exp="humanml3d_${variant}_seed${SEED}"
  out="$RUN_ROOT/$exp"
  if [[ "$variant" == no_both && "$gpu" == 2 && -s "$RUN_ROOT/motionx322_no_both_seed${SEED}/sato_pp_final.pth" ]]; then
    gpu=0
  fi
  mkdir -p "$out"
  # Serialize invocations of the same experiment.  This lets manually
  # backfilled runs share GPUs with the main schedule without a later
  # schedule stage launching a duplicate process.
  local lock_fd
  exec {lock_fd}>"$out/train.lock"
  flock "$lock_fd"
  if [[ -s "$out/sato_pp_final.pth" ]]; then echo "skip completed $exp"; return; fi
  local resume_args=()
  if [[ -s "$out/sato_pp_last.pth" ]]; then
    if [[ "$ALLOW_STREAM_RESET_RESUME" != 1 ]]; then
      echo "refusing non-exact resume for controlled run $exp: DataLoader stream state is not checkpointed" >&2
      echo "restart the variant from iteration 0, or set ALLOW_STREAM_RESET_RESUME=1 only for a non-formal recovery" >&2
      return 3
    fi
    resume_args=(--resume-sato-pp "$out/sato_pp_last.pth")
  fi
  write_meta "$out" humanml3d "$variant" "$gpu" "$var" "$cov"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON" train_sato_pp.py \
    --dataname t2m --resume-pth "$H_VQ" --resume-trans "$H_GPT" \
    --pair-data-path "$H_PAIR_TRAIN" --out-dir "$RUN_ROOT" --exp-name "$exp" \
    "${resume_args[@]}" \
    --vq-name VQVAE --batch-size 22 --num-workers 4 --total-iter 150000 \
    --down-t 2 --block-size 51 --embed-dim-gpt 1024 --num-layers 9 --n-head-gpt 16 \
    --pkeep 0.5 --lr 2e-4 --lr-scheduler 60000 --gamma 0.05 \
    --teacher-loss-type l1 --lambda-teach 0.5 --lambda-align 0.2 \
    --lambda-var "$var" --lambda-cov "$cov" --lambda-diff 0.5 --lambda-loc 0.1 \
    --preserve-ratio 0.5 --attribution-K 16 --tau 1.0 --freeze-clip \
    --eval-iter 5000 --save-iter 5000 --milestone-save-iter 0 --print-iter 50 \
    --seed "$SEED" --wandb-project SATO++-VarCov-Ablation --wandb-mode offline \
    --wandb-group "HumanML3D-var-cov-seed${SEED}" --wandb-name "$exp" \
    --wandb-tags "humanml3d,var-cov-ablation,${variant},seed${SEED}" \
    > "$out/train.log" 2>&1
  code=$?; set -e; echo "$code" > "$out/train_exit_code.txt"; return "$code"
}

train_motionx() {
  local variant="$1" gpus="$2"
  local var cov exp out code
  read -r var cov < <(variant_weights "$variant")
  exp="motionx322_${variant}_seed${SEED}"
  out="$RUN_ROOT/$exp"
  mkdir -p "$out"
  local lock_fd
  exec {lock_fd}>"$out/train.lock"
  flock "$lock_fd"
  if [[ -s "$out/sato_pp_final.pth" ]]; then echo "skip completed $exp"; return; fi
  local resume_args=()
  if [[ -s "$out/sato_pp_last.pth" ]]; then
    if [[ "$ALLOW_STREAM_RESET_RESUME" != 1 ]]; then
      echo "refusing non-exact resume for controlled run $exp: DataLoader stream state is not checkpointed" >&2
      echo "restart the variant from iteration 0, or set ALLOW_STREAM_RESET_RESUME=1 only for a non-formal recovery" >&2
      return 3
    fi
    resume_args=(--resume-sato-pp "$out/sato_pp_last.pth")
  fi
  write_meta "$out" motionx322_75 "$variant" "$gpus" "$var" "$cov"
  set +e
  CUDA_VISIBLE_DEVICES="$gpus" PYTHONUNBUFFERED=1 "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node 2 train_sato_pp_ddp.py \
    --dataname motionx322_75 --resume-pth "$M_VQ" --resume-trans "$M_GPT" \
    --pair-data-path "$M_PAIR_TRAIN" --out-dir "$RUN_ROOT" --exp-name "$exp" \
    "${resume_args[@]}" \
    --vq-name VQVAE --batch-size 66 --grad-accum-steps 1 --num-workers 4 \
    --nb-code 512 --code-dim 512 --output-emb-width 512 --down-t 2 --stride-t 2 \
    --width 512 --depth 3 --dilation-growth-rate 3 --quantizer ema_reset \
    --block-size 76 --embed-dim-gpt 1024 --clip-dim 512 --num-layers 9 \
    --n-head-gpt 16 --ff-rate 4 --drop-out-rate 0.1 --total-iter 100000 \
    --lr 2e-4 --lr-scheduler 60000 --gamma 0.05 --optimizer adamw --weight-decay 1e-6 \
    --pkeep 0.5 --teacher-loss-type l1 --lambda-teach 0.5 --lambda-align 0.2 \
    --lambda-var "$var" --lambda-cov "$cov" --lambda-diff 0.5 --lambda-loc 0.1 \
    --preserve-ratio 0.5 --attribution-K 16 --tau 1.0 --freeze-clip \
    --eval-iter 5000 --save-iter 5000 --milestone-save-iter 0 --print-iter 50 \
    --seed "$SEED" --wandb-project SATO++-VarCov-Ablation --wandb-mode offline \
    --wandb-group "MotionX322-var-cov-seed${SEED}" --wandb-name "$exp" \
    --wandb-tags "motionx322,var-cov-ablation,${variant},seed${SEED}" \
    > "$out/train.log" 2>&1
  code=$?; set -e; echo "$code" > "$out/train_exit_code.txt"; return "$code"
}

launch_all() {
  mkdir -p "$RUN_ROOT"
  for f in "$H_PAIR_TRAIN" "$H_PAIR_TEST" "$H_VQ" "$H_GPT" "$H_FULL" \
           "$M_PAIR_TRAIN" "$M_PAIR_TEST" "$M_VQ" "$M_GPT" "$M_FULL"; do require_file "$f"; done
  local driver="$RUN_ROOT/schedule_wrapper.sh"
  {
    printf '%s\n' '#!/usr/bin/env bash' 'set +e'
    printf 'bash %q schedule\n' "$0"
    printf '%s\n' 'code=$?'
    printf 'echo "$code" > %q\n' "$RUN_ROOT/schedule_exit_code.txt"
    printf '%s\n' 'exit "$code"'
  } > "$driver"
  chmod +x "$driver"
  nohup setsid bash "$driver" > "$RUN_ROOT/schedule.driver.log" 2>&1 < /dev/null & echo $! > "$RUN_ROOT/schedule.pid"
  date -Is > "$RUN_ROOT/launched_at.txt"
}

schedule() {
  # Motion-X Full used two ranks with per-rank batch 66. Run two variants in
  # parallel, then the third, before starting the single-GPU HumanML3D runs.
  bash "$0" train-motionx no_var 0,1 & local p0=$!
  bash "$0" train-motionx no_cov 2,3 & local p1=$!
  wait "$p0"; wait "$p1"
  bash "$0" train-motionx no_both 0,1
  bash "$0" train-human no_var 2 & p0=$!
  bash "$0" train-human no_cov 3 & p1=$!
  bash "$0" train-human no_both 0 & local p2=$!
  wait "$p0"; wait "$p1"; wait "$p2"
  bash "$0" eval-all
}

write_eval_spec() {
  local dataset="$1" spec="$2" full="$3"
  local dataname vq gpt prefix
  if [[ "$dataset" == humanml3d ]]; then
    dataname=t2m; vq="$H_VQ"; gpt="$H_GPT"; prefix=humanml3d
  else
    dataname=motionx322_75; vq="$M_VQ"; gpt="$M_GPT"; prefix=motionx322
  fi
  "$PYTHON" - "$spec" "$dataname" "$vq" "$gpt" "$prefix" "$full" "$RUN_ROOT" "$SEED" <<'PY'
import json, sys
from pathlib import Path
spec, dataname, vq, gpt, prefix, full, root, seed = sys.argv[1:]
rows = [{"name": f"{prefix}_full", "kind": "sato_pp", "checkpoint": full}]
for variant in ("no_var", "no_cov", "no_both"):
    rows.append({
        "name": f"{prefix}_{variant}", "kind": "sato_pp",
        "checkpoint": str(Path(root) / f"{prefix}_{variant}_seed{seed}" / "sato_pp_final.pth"),
    })
Path(spec).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
PY
}

eval_dataset() {
  local dataset="$1" pair full protocol eval_root spec
  if [[ "$dataset" == humanml3d ]]; then
    pair="$H_PAIR_TEST"; protocol=caption; eval_root="$RUN_ROOT/eval_humanml3d"
    full="$RUN_ROOT/humanml3d_full_seed${SEED}/sato_pp_final.pth"
    require_file "$full"
    for variant in no_var no_cov no_both; do
      require_file "$RUN_ROOT/humanml3d_${variant}_seed${SEED}/sato_pp_final.pth"
    done
    # A final checkpoint is written after the 150k validation but just before
    # the training wrapper exits. Wait for all wrapper-held locks so formal
    # evaluation cannot contend with a trainer that is still shutting down.
    for variant in full no_var no_cov no_both; do
      wait_training_lock_released \
        "$RUN_ROOT/humanml3d_${variant}_seed${SEED}"
    done
    mkdir -p "$RUN_ROOT/tmp"
    TMPDIR="${TMPDIR:-$RUN_ROOT/tmp}" PYTHONDONTWRITEBYTECODE=1 "$PYTHON" \
      tools/validate_var_cov_checkpoints.py --run-root "$RUN_ROOT" \
      --dataset-prefix humanml3d --seed "$SEED" --iteration 150000
  else
    pair="$M_PAIR_TEST"; full="$M_FULL"; protocol=humanml-fast; eval_root="$RUN_ROOT/eval_motionx322"
  fi
  mkdir -p "$eval_root"
  spec="$eval_root/spec.json"
  write_eval_spec "$dataset" "$spec" "$full"
  local prefix="$dataset" models=(full no_var no_cov no_both) pids=() failed=0
  for gpu in 0 1 2 3; do
    local model="${prefix}_${models[$gpu]}" out="$eval_root/gpu${gpu}"
    local protocol_args=()
    if [[ "$protocol" == caption ]]; then protocol_args=(--caption-locked --fast-batch); else protocol_args=(--humanml-fast-protocol); fi
    # runs.csv is append-only and the evaluator skips seeds already present,
    # so retries resume the same formal 20-seed shard instead of duplicating it.
    (
      for attempt in 1 2 3; do
        printf '%s attempt=%d model=%s gpu=%d\n' "$(date -Is)" "$attempt" "$model" "$gpu" \
          >> "${out}_driver.log"
        if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" ablation/repeat20_perturb_fast_batch.py \
          --spec-json "$spec" --model "$model" --pair-data-path "$pair" \
          --out-dir "$out" --report-md "${out}.md" --start-seed "$START_SEED" \
          --num-repeats "$NUM_REPEATS" --batch-size 32 "${protocol_args[@]}" \
          --cfg-scale 1.3 --cfg-uncond zero --sample-mode categorical --wandb-mode disabled \
          >> "${out}_driver.log" 2>&1; then
          exit 0
        fi
      done
      exit 1
    ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  if (( failed )); then date -Is > "$eval_root/failed_at.txt"; return 1; fi
  "$PYTHON" tools/summarize_var_cov_ablation.py \
    --eval-root "$eval_root" --dataset "$dataset" \
    --no-second-order-attribution
  date -Is > "$eval_root/complete.txt"
}

eval_all() {
  eval_dataset motionx322
  eval_dataset humanml3d
}

audit_human() {
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON" tools/audit_var_cov_run_state.py \
    --run-root "$RUN_ROOT" --seed "$SEED" --total-iter 150000 \
    --json-out "$RUN_ROOT/health_latest.json"
}

status() {
  printf '%-38s %-10s %-8s %s\n' experiment state iter exit
  for dataset in humanml3d motionx322; do
    local variants=(no_var no_cov no_both)
    [[ "$dataset" == humanml3d ]] && variants=(full no_var no_cov no_both)
    for variant in "${variants[@]}"; do
      local exp="${dataset}_${variant}_seed${SEED}" out="$RUN_ROOT/${dataset}_${variant}_seed${SEED}" state=pending iter=- exit=-
      [[ -s "$out/train.log" ]] && iter=$(sed -n 's/.*Train[.] Iter \([0-9][0-9]*\):.*/\1/p' "$out/train.log" | tail -1)
      [[ -s "$out/sato_pp_final.pth" ]] && state=complete
      [[ -s "$out/train_exit_code.txt" ]] && exit=$(cat "$out/train_exit_code.txt")
      [[ "$state" != complete && "$iter" != - ]] && state=running_or_stopped
      printf '%-38s %-10s %-8s %s\n' "$exp" "$state" "${iter:--}" "$exit"
    done
  done
}

case "$STAGE" in
  launch) launch_all ;;
  train-human) train_human "${2:?variant}" "${3:?gpu}" ;;
  train-motionx) train_motionx "${2:?variant}" "${3:?gpu}" ;;
  eval-dataset) eval_dataset "${2:?dataset}" ;;
  eval-all) eval_all ;;
  audit-human) audit_human ;;
  schedule) schedule ;;
  status) status ;;
  *) echo "usage: $0 {launch|schedule|status|audit-human|train-human VARIANT GPU|train-motionx VARIANT GPU_LIST|eval-dataset DATASET|eval-all}" >&2; exit 2 ;;
esac
