#!/usr/bin/env bash
set -euo pipefail

cd /mnt/workspace/Wilson/T2M-GPT

PYTHON=/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python
BASE_OUT=ablation
LAUNCH_STAGGER_SECONDS=${LAUNCH_STAGGER_SECONDS:-30}

COMMON_ARGS=(
  --dataname t2m
  --batch-size 22
  --total-iter 150000
  --resume-pth pretrained/VQVAE/net_best_fid.pth
  --resume-trans pretrained/VQTransformer_corruption05/net_best_fid.pth
  --out-dir "$BASE_OUT"
  --vq-name VQVAE
  --down-t 2
  --block-size 51
  --embed-dim-gpt 1024
  --num-layers 9
  --n-head-gpt 16
  --pkeep 0.5
  --print-iter 50
  --eval-iter 5000
  --save-iter 5000
  --milestone-save-iter 10000
  --lr 2e-4
  --lr-scheduler 60000
  --gamma 0.05
  --pair-data-path dataset/HumanML3D/preserving_pairs_filtered.jsonl
  --preserve-ratio 0.5
  --teacher-loss-type l1
  --freeze-clip
  --wandb-project 'SATO++'
  --wandb-tags loss_ablation_150k,current_loss_ablation
)

launch_run() {
  local gpu="$1"
  local exp_name="$2"
  local lambda_teach="$3"
  local lambda_align="$4"
  local lambda_var="$5"
  local lambda_cov="$6"
  local lambda_diff="$7"
  local lambda_loc="$8"

  local out_dir="$BASE_OUT/$exp_name"
  local out_abs="/mnt/workspace/Wilson/T2M-GPT/$out_dir"
  mkdir -p "$out_dir"

  if [[ -s "$out_dir/pid.txt" ]] && kill -0 "$(cat "$out_dir/pid.txt")" 2>/dev/null; then
    echo "skip $exp_name: already running pid=$(cat "$out_dir/pid.txt")"
    return 0
  fi

  echo "launch $exp_name on GPU $gpu"
  rm -f "$out_dir/exit_code.txt"

  local run_script="$out_dir/run.sh"
  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'set -o pipefail'
    printf 'cd %q\n' '/mnt/workspace/Wilson/T2M-GPT'
    printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$gpu"
    printf '%s\n' "export WANDB_PROJECT='SATO++'"
    printf '%s\n' 'export PYTHONUNBUFFERED=1'
    printf 'exec_cmd=('
    printf ' %q' "$PYTHON" train_sato_pp.py \
      "${COMMON_ARGS[@]}" \
      --exp-name "$exp_name" \
      --wandb-name "$exp_name" \
      --lambda-teach "$lambda_teach" \
      --lambda-align "$lambda_align" \
      --lambda-var "$lambda_var" \
      --lambda-cov "$lambda_cov" \
      --lambda-diff "$lambda_diff" \
      --lambda-loc "$lambda_loc"
    printf '%s\n' ' )'
    printf '%s\n' '"${exec_cmd[@]}"'
    printf '%s\n' 'code=$?'
    printf 'echo "$code" > %q\n' "$out_abs/exit_code.txt"
    printf '%s\n' 'exit "$code"'
  } > "$run_script"
  chmod +x "$run_script"

  nohup setsid bash "$run_script" > "$out_dir/nohup.log" 2>&1 < /dev/null &

  local pid=$!
  echo "$pid" > "$out_dir/pid.txt"
  {
    echo "pid=$pid"
    echo "gpu=$gpu"
    echo "exp_name=$exp_name"
    echo "started_at=$(date -Is)"
    echo "wandb_project=SATO++"
    echo "wandb_name=$exp_name"
    echo "run_script=$run_script"
    echo "cmd=CUDA_VISIBLE_DEVICES=$gpu WANDB_PROJECT=SATO++ PYTHONUNBUFFERED=1 $PYTHON train_sato_pp.py ${COMMON_ARGS[*]} --exp-name $exp_name --wandb-name $exp_name --lambda-teach $lambda_teach --lambda-align $lambda_align --lambda-var $lambda_var --lambda-cov $lambda_cov --lambda-diff $lambda_diff --lambda-loc $lambda_loc"
  } > "$out_dir/launch_meta.txt"

  sleep "$LAUNCH_STAGGER_SECONDS"
}

launch_run 0 loss_full_allloss_l1_150k 0.5 0.2 0.5 0.04 0.5 0.1
launch_run 1 loss_no_teacher_l1_150k 0.0 0.2 0.5 0.04 0.5 0.1
launch_run 2 loss_no_preserve_l1_150k 0.5 0.0 0.0 0.0 0.5 0.1
launch_run 3 loss_no_alter_l1_150k 0.5 0.2 0.5 0.04 0.0 0.0
launch_run 3 loss_no_attribute_l1_150k 0.5 0.0 0.0 0.0 0.0 0.0

echo "launched loss ablation 150k runs"
