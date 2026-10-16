#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/workspace/Wilson/T2M-GPT
OUT_ROOT="${OUT_ROOT:-${ROOT}/MotionGPT/experiments/robustness/fixed_caption_100x10_20260806}"
MODEL="${1:?usage: $0 motiongpt GPU START_REPEAT END_REPEAT | sato_pp GPU | sato50k GPU}"
GPU="${2:?missing GPU}"
mkdir -p "${OUT_ROOT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU}"

if [[ "${MODEL}" == "motiongpt" ]]; then
  START_REPEAT="${3:?missing start repeat}"
  END_REPEAT="${4:?missing end repeat}"
  PYTHON_BIN=/mnt/workspace/Wilson/miniconda3/envs/mgpt/bin/python
  MGPT_ROOT="${ROOT}/MotionGPT"
  MANIFEST="${MGPT_ROOT}/experiments/robustness/humanml3d_baseline_final/manifest.json"
  CONFIG="${MGPT_ROOT}/experiments/robustness/humanml3d_baseline_final/configs/motiongpt_baseline_seed123.yaml"
  INDEX="${MGPT_ROOT}/experiments/robustness/humanml3d_sato_aligned_official_20seed_gpu0_4_20260806/seed123_sample_index.json"
  mkdir -p "${OUT_ROOT}/motiongpt"
  export LD_LIBRARY_PATH=/mnt/workspace/Wilson/miniconda3/envs/mgpt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
  export PYTHONNOUSERSITE=1
  for repeat in $(seq "${START_REPEAT}" "${END_REPEAT}"); do
    generation_seed=$((1000 + repeat))
    "${PYTHON_BIN}" "${MGPT_ROOT}/eval_loader_robustness.py" \
      --manifest "${MANIFEST}" \
      --config "${CONFIG}" \
      --seed 123 \
      --generation-seed "${generation_seed}" \
      --device cuda:0 \
      --batch-size 32 \
      --num-workers 8 \
      --limit 100 \
      --pair-rng-mode sequential \
      --sampling-kwargs-mode explicit \
      --prompt-mode official-default \
      --generation-order original-first \
      --length-mode generated \
      --expected-sample-index "${INDEX}" \
      --embedding-output "${OUT_ROOT}/motiongpt/repeat_${repeat}.npz" \
      --output "${OUT_ROOT}/motiongpt/repeat_${repeat}.json" \
      >"${OUT_ROOT}/logs/motiongpt_repeat_${repeat}.log" 2>&1
  done
elif [[ "${MODEL}" == "sato_pp" ]]; then
  PYTHON_BIN=/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python
  "${PYTHON_BIN}" "${ROOT}/ablation/repeat20_perturb_fast_batch.py" \
    --out-dir "${OUT_ROOT}/sato_pp_runs" \
    --report-md "${OUT_ROOT}/sato_pp_runs.md" \
    --start-seed 1000 \
    --num-repeats 10 \
    --loader-seed 123 \
    --sample-limit 100 \
    --batch-size 32 \
    --fast-batch \
    --sampling-pair-rng-mode sequential \
    --diagnostics-dir "${OUT_ROOT}/sato_pp" \
    --model sato_pp_allloss_l1_iter130k \
    --cfg-scale 1.0 \
    --sample-mode categorical \
    --sample-temperature 1.0 \
    --top-k 0 \
    --top-p 1.0 \
    --wandb-mode disabled \
    >"${OUT_ROOT}/logs/sato_pp.log" 2>&1
elif [[ "${MODEL}" == "sato50k" ]]; then
  PYTHON_BIN=/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python
  "${PYTHON_BIN}" "${ROOT}/ablation/repeat20_perturb_fast_batch.py" \
    --out-dir "${OUT_ROOT}/sato_runs" \
    --report-md "${OUT_ROOT}/sato_runs.md" \
    --start-seed 1000 \
    --num-repeats 10 \
    --loader-seed 123 \
    --sample-limit 100 \
    --batch-size 32 \
    --fast-batch \
    --sampling-pair-rng-mode sequential \
    --diagnostics-dir "${OUT_ROOT}/sato" \
    --external-sato-name sato_reproduced_50k \
    --external-sato-vq pretrained/VQVAE/net_best_fid.pth \
    --external-sato-trans Stable-Text-to-Motion-Framework-official/experiments/sato_h3d_repro_100k_seed123_ddp4/eval_iter_50000/net_iter_50000.pth \
    --external-sato-clip Stable-Text-to-Motion-Framework-official/experiments/sato_h3d_repro_100k_seed123_ddp4/eval_iter_50000/clip_iter_50000.pth \
    --model sato_reproduced_50k \
    --cfg-scale 1.0 \
    --sample-mode categorical \
    --sample-temperature 1.0 \
    --top-k 0 \
    --top-p 1.0 \
    --wandb-mode disabled \
    >"${OUT_ROOT}/logs/sato50k.log" 2>&1
else
  echo "unknown model: ${MODEL}" >&2
  exit 2
fi
