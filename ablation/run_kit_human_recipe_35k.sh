#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
SEED="${SEED:-123}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${KIT_SATO_NPROC_PER_NODE:-4}"
REPEATS="${KIT_EVAL_REPEATS:-20}"
SATO_ROOT="ablation/kit_t2mgpt_sato"
SOURCE_EXP_NAME="humanml3d_recipe_35k_lrdrop25k_seed${SEED}"
SOURCE_EXP_DIR="${SATO_ROOT}/${SOURCE_EXP_NAME}"
RESUME_CKPT="${SOURCE_EXP_DIR}/sato_pp_best_fid.pth"
SOURCE_BEST_CKPT="${SOURCE_EXP_DIR}/sato_pp_best_fid.pth"
EXP_NAME="humanml3d_recipe_35k_lrmilestones25k30k32p5k_seed${SEED}"
EXP_DIR="${SATO_ROOT}/${EXP_NAME}"
PAIR_TEST="dataset/KIT-ML/test_perb_gemini31_mixed_full.json"
VQ_CKPT="output_kit/VQVAE_KIT/net_best_fid.pth"
GPT_CKPT="output_kit/GPT_KIT/net_best_fid.pth"
FINAL_CKPT="${EXP_DIR}/sato_pp_final.pth"
BEST_CKPT="${EXP_DIR}/sato_pp_best_fid.pth"
EVAL_ROOT="${SATO_ROOT}/${EXP_NAME}_caption_locked_repeat${REPEATS}"

mkdir -p "$EXP_DIR"
printf '%s\n' "$$" > "${EXP_DIR}/pipeline.pid"
date -Is > "${EXP_DIR}/pipeline_started_at.txt"
echo "[$(date -Is)] Resuming KIT at 30k: milestones=25000,30000,32500 gamma=0.05"

if [[ ! -f "$RESUME_CKPT" ]]; then
  echo "Missing 30k resume checkpoint: $RESUME_CKPT" >&2
  exit 1
fi

GPUS="$GPUS" \
KIT_SATO_NPROC_PER_NODE="$NPROC" \
KIT_SATO_BATCH_SIZE=33 \
KIT_SATO_TOTAL_ITER=35000 \
KIT_SATO_LR_SCHEDULER="25000 30000 32500" \
KIT_SATO_EXP_NAME="$EXP_NAME" \
KIT_SATO_RESUME="$RESUME_CKPT" \
SATO_WANDB_GROUP="KIT-HumanML3D-recipe-multistep-seed${SEED}" \
SATO_WANDB_NAME="KIT-HumanML3D-recipe-30k-to35k-lr25k30k32p5k-globalbs132-seed${SEED}" \
SATO_WANDB_TAGS="kit,t2m-gpt,sato++,humanml3d-recipe,no-second-order,35k,resume30k,lr-25k-30k-32p5k,ddp,4gpu,globalbs132,seed${SEED}" \
SEED="$SEED" \
bash ablation/launch_kit_t2mgpt_sato.sh train-sato

if [[ ! -f "$FINAL_CKPT" ]]; then
  echo "Missing expected checkpoint: $FINAL_CKPT" >&2
  exit 1
fi
if [[ ! -f "$BEST_CKPT" ]]; then
  BEST_CKPT="$SOURCE_BEST_CKPT"
fi
if [[ ! -f "$BEST_CKPT" ]]; then
  echo "Missing expected best checkpoint: $BEST_CKPT" >&2
  exit 1
fi

FINAL_SPEC="${EXP_DIR}/eval_final_spec.json"
BEST_SPEC="${EXP_DIR}/eval_best_spec.json"
"$PYTHON" tools/write_sato_eval_spec.py \
  --dataname kit --vq-checkpoint "$VQ_CKPT" \
  --baseline-checkpoint "$GPT_CKPT" --sato-checkpoint "$FINAL_CKPT" \
  --sato-name kit_sato_humanml3d_recipe_multistep_final35k --output "$FINAL_SPEC"
"$PYTHON" tools/write_sato_eval_spec.py \
  --dataname kit --vq-checkpoint "$VQ_CKPT" \
  --baseline-checkpoint "$GPT_CKPT" --sato-checkpoint "$BEST_CKPT" \
  --sato-name kit_sato_humanml3d_recipe_multistep_best --output "$BEST_SPEC"

run_eval() {
  local gpu="$1" spec="$2" model="$3" output="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" ablation/repeat20_perturb_fast_batch.py \
    --spec-json "$spec" --model "$model" --pair-data-path "$PAIR_TEST" \
    --out-dir "$output" --report-md "${output}.md" \
    --start-seed 123 --num-repeats "$REPEATS" --batch-size 32 \
    --caption-locked > "${output}_driver.log" 2>&1
}

run_eval 0 "$FINAL_SPEC" kit_t2mgpt_baseline "${EVAL_ROOT}_baseline" &
pid_baseline="$!"
run_eval 1 "$FINAL_SPEC" kit_sato_humanml3d_recipe_multistep_final35k "${EVAL_ROOT}_final35k" &
pid_final="$!"
run_eval 2 "$BEST_SPEC" kit_sato_humanml3d_recipe_multistep_best "${EVAL_ROOT}_best" &
pid_best="$!"

failed=0
for pid in "$pid_baseline" "$pid_final" "$pid_best"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

REPORT_STEM="${SATO_ROOT}/KIT_HUMANML3D_RECIPE_35K_LR25K_30K_32P5K"
"$PYTHON" tools/summarize_human_recipe_eval.py \
  --baseline "${EVAL_ROOT}_baseline/summary.json" \
  --candidate "final35k=${EVAL_ROOT}_final35k/summary.json" \
  --candidate "best=${EVAL_ROOT}_best/summary.json" \
  --output-json "${REPORT_STEM}.json" \
  --output-md "${REPORT_STEM}.md"

date -Is > "${EXP_DIR}/train_and_eval_complete.txt"
echo "[$(date -Is)] KIT HumanML3D-recipe 35k train/eval complete"
