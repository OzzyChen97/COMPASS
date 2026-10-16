#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
REPEATS="${PERTURB_EVAL_REPEATS:-20}"
CFG_SCALE="${PERTURB_EVAL_CFG_SCALE:-1.3}"
OUT_ROOT="${PERTURB_EVAL_ROOT:-ablation/gemini31_test_perb_latest_cfg1p3_repeat20_20260725}"

KIT_POOL="dataset/KIT-ML/test_perb.json"
KIT_MIX="dataset/KIT-ML/test_perb_gemini31_mixed_human_ratio.json"
KIT_VQ="output_kit/VQVAE_KIT/net_best_fid.pth"
KIT_GPT="output_kit/GPT_KIT/net_best_fid.pth"
KIT_SATO="ablation/kit_t2mgpt_sato/humanml3d_recipe_85k_seed123/sato_pp_last.pth"

MOTIONX_POOL="dataset/Motion-X-322/test_perb.json"
MOTIONX_MIX="dataset/Motion-X-322/test_perb_gemini31_mixed_human_ratio.json"
MOTIONX_VQ="output_motionx322/VQVAE_MOTIONX322/net_best_fid.pth"
MOTIONX_GPT="output_motionx322/GPT_MOTIONX322_MAXTOK75_200K/net_best_fid.pth"
MOTIONX_SATO="ablation/motionx322_sato/max_token75_humanml3d_recipe_150k_bs132_seed123/sato_pp_last.pth"

for path in \
  "$KIT_POOL" "$KIT_VQ" "$KIT_GPT" "$KIT_SATO" \
  "$MOTIONX_POOL" "$MOTIONX_VQ" "$MOTIONX_GPT" "$MOTIONX_SATO"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

mkdir -p "$OUT_ROOT"
date -Is > "${OUT_ROOT}/started_at.txt"

"$PYTHON" tools/build_mixed_perturbation_pool.py \
  --input "$KIT_POOL" \
  --output "$KIT_MIX" \
  --weights 0.46 0.27 0.27 \
  --seed 123

"$PYTHON" tools/build_mixed_perturbation_pool.py \
  --input "$MOTIONX_POOL" \
  --output "$MOTIONX_MIX" \
  --weights 0.46 0.27 0.27 \
  --seed 123

KIT_SPEC="${OUT_ROOT}/kit_latest_baseline_spec.json"
MOTIONX_SPEC="${OUT_ROOT}/motionx_latest_baseline_spec.json"

"$PYTHON" tools/write_sato_eval_spec.py \
  --dataname kit \
  --vq-checkpoint "$KIT_VQ" \
  --baseline-checkpoint "$KIT_GPT" \
  --sato-checkpoint "$KIT_SATO" \
  --baseline-name kit_t2mgpt_baseline_best \
  --sato-name kit_sato_latest_85k \
  --block-size 51 \
  --output "$KIT_SPEC"

"$PYTHON" tools/write_sato_eval_spec.py \
  --dataname motionx322_75 \
  --vq-checkpoint "$MOTIONX_VQ" \
  --baseline-checkpoint "$MOTIONX_GPT" \
  --sato-checkpoint "$MOTIONX_SATO" \
  --baseline-name motionx322_t2mgpt_baseline_best \
  --sato-name motionx322_sato_latest_110k \
  --block-size 76 \
  --output "$MOTIONX_SPEC"

run_eval() {
  local spec="$1" model="$2" pair_file="$3" output="$4"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" \
    ablation/repeat20_perturb_fast_batch.py \
    --spec-json "$spec" \
    --model "$model" \
    --pair-data-path "$pair_file" \
    --out-dir "$output" \
    --report-md "${output}.md" \
    --start-seed 123 \
    --num-repeats "$REPEATS" \
    --batch-size 32 \
    --caption-locked \
    --fast-batch \
    --cfg-scale "$CFG_SCALE" \
    --sample-mode categorical \
    --wandb-mode disabled \
    > "${output}_driver.log" 2>&1
}

run_eval "$KIT_SPEC" kit_sato_latest_85k "$KIT_MIX" "${OUT_ROOT}/kit_latest_85k" &
pid_kit_sato="$!"
run_eval "$KIT_SPEC" kit_t2mgpt_baseline_best "$KIT_MIX" "${OUT_ROOT}/kit_baseline" &
pid_kit_baseline="$!"
run_eval "$MOTIONX_SPEC" motionx322_sato_latest_110k "$MOTIONX_MIX" "${OUT_ROOT}/motionx_latest_110k" &
pid_motionx_sato="$!"
run_eval "$MOTIONX_SPEC" motionx322_t2mgpt_baseline_best "$MOTIONX_MIX" "${OUT_ROOT}/motionx_baseline" &
pid_motionx_baseline="$!"

failed=0
for pid in "$pid_kit_sato" "$pid_kit_baseline" "$pid_motionx_sato" "$pid_motionx_baseline"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  date -Is > "${OUT_ROOT}/failed_at.txt"
  exit 1
fi

date -Is > "${OUT_ROOT}/complete.txt"
