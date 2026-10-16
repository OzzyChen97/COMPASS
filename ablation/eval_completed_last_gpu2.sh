#!/usr/bin/env bash
set -euo pipefail

cd /mnt/workspace/Wilson/T2M-GPT

PYTHON_BIN=/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python
PAIR_DATA=dataset/HumanML3D/test_perb_deterministic.json

export CUDA_VISIBLE_DEVICES=2

run_eval() {
  local exp_name="$1"
  local exp_dir="ablation/${exp_name}"
  local out_dir="${exp_dir}/eval_last_test_perb_gpu2"

  mkdir -p "${out_dir}"
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] start ${exp_name}" | tee -a ablation/eval_completed_last_gpu2.log

  "${PYTHON_BIN}" eval_sato_pp_batch_checkpoints.py \
    --checkpoint-dir "${exp_dir}" \
    --checkpoint-names sato_pp_last.pth \
    --pair-data-path "${PAIR_DATA}" \
    --out-dir "${out_dir}" 2>&1 | tee "${out_dir}/launcher.log"

  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] done ${exp_name}" | tee -a ablation/eval_completed_last_gpu2.log
}

run_eval loss_no_preserve_l1_150k
run_eval loss_no_attribute_l1_150k

