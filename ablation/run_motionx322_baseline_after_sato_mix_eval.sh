#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
CURRENT_ROOT="${MOTIONX322_SATO_EVAL_ROOT:-ablation/motionx322_sato/best10k_last110k_level_mix_repeat20}"
CURRENT_COMPLETE="${CURRENT_ROOT}/complete.txt"
VQ_CKPT="${MOTIONX322_VQ_CKPT:-output_motionx322/VQVAE_MOTIONX322/net_best_fid.pth}"
GPT_CKPT="${MOTIONX322_GPT_CKPT:-output_motionx322/GPT_MOTIONX322_MAXTOK75_200K/net_best_fid.pth}"
SATO_CKPT="${MOTIONX322_SATO_CKPT:-ablation/motionx322_sato/max_token75_humanml3d_recipe_150k_bs132_seed123/sato_pp_best_fid.pth}"
P2P3_PAIR="${MOTIONX322_P2P3_PAIR:-dataset/Motion-X-322/test_perb_gemini31_p2p3.json}"
MIXED_PAIR="${MOTIONX322_MIXED_PAIR:-dataset/Motion-X-322/test_perb_gemini31_mixed_full.json}"
OUT_ROOT="${MOTIONX322_BASELINE_EVAL_ROOT:-ablation/motionx322_sato/gpt_baseline_level_mix_repeat20}"
SPEC_PATH="${OUT_ROOT}/gpt_baseline_spec.json"

mkdir -p "$OUT_ROOT"
date -Is > "${OUT_ROOT}/queued_at.txt"

while [[ ! -f "$CURRENT_COMPLETE" ]]; do
  if ! pgrep -f '[r]un_motionx322_checkpoint_level_mix_eval.sh' >/dev/null; then
    echo "SATO evaluation stopped without completion marker: $CURRENT_COMPLETE" >&2
    date -Is > "${OUT_ROOT}/blocked_at.txt"
    exit 1
  fi
  sleep 60
done

for path in "$VQ_CKPT" "$GPT_CKPT" "$SATO_CKPT" "$P2P3_PAIR" "$MIXED_PAIR"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

date -Is > "${OUT_ROOT}/started_at.txt"
"$PYTHON" tools/write_sato_eval_spec.py \
  --dataname motionx322_75 \
  --vq-checkpoint "$VQ_CKPT" \
  --baseline-checkpoint "$GPT_CKPT" \
  --sato-checkpoint "$SATO_CKPT" \
  --baseline-name motionx322_t2mgpt_best \
  --sato-name unused_sato_entry \
  --block-size 76 \
  --output "$SPEC_PATH"

levels=(p2p3_balanced p123_human_ratio)
pair_files=("$P2P3_PAIR" "$MIXED_PAIR")

for index in 0 1; do
  level="${levels[$index]}"
  pair_file="${pair_files[$index]}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" \
    ablation/repeat20_perturb_fast_batch.py \
    --spec-json "$SPEC_PATH" \
    --model motionx322_t2mgpt_best \
    --pair-data-path "$pair_file" \
    --out-dir "${OUT_ROOT}/${level}" \
    --report-md "${OUT_ROOT}/${level}.md" \
    --start-seed 123 \
    --num-repeats 20 \
    --batch-size 32 \
    --caption-locked \
    --fast-batch \
    --cfg-scale 1.3 \
    --sample-mode categorical \
    --wandb-mode disabled \
    > "${OUT_ROOT}/${level}_driver.log" 2>&1
done

date -Is > "${OUT_ROOT}/complete.txt"
