#!/usr/bin/env bash
set -euo pipefail

# Independent Motion-X reproduction of the SATO paper's final RSR recipe.
# This launcher never enables PGD or early stopping and never reuses the
# token-aligned experiment directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SATO_DIR="${ROOT}/Stable-Text-to-Motion-Framework-official"
PYTHON_BIN="${PYTHON_BIN:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "${PYTHON_BIN}")/torchrun}"
MODE="${1:-formal}"
GPUS="${GPUS:-3,4,5,7}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NPROC="${NPROC:-${#GPU_LIST[@]}}"
GLOBAL_BATCH_SIZE="${MOTIONX322_SATO_PAPER_GLOBAL_BATCH_SIZE:-64}"
SEED="${SEED:-123}"

if (( NPROC < 1 )); then
  echo "NPROC must be positive" >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE % NPROC != 0 )); then
  echo "Global batch size ${GLOBAL_BATCH_SIZE} must be divisible by NPROC ${NPROC}" >&2
  exit 2
fi
LOCAL_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NPROC))

case "${MODE}" in
  smoke)
    TOTAL_ITER="${MOTIONX322_SATO_PAPER_TOTAL_ITER:-2}"
    DATASET_LIMIT="${MOTIONX322_SATO_PAPER_DATASET_LIMIT:-64}"
    PRINT_ITER=1
    EVAL_ITER=1000000
    SAVE_ITER=1000000
    DEFAULT_EXP_NAME="motionx322_sato_paper_faithful_rsr_smoke_seed${SEED}"
    DRY_RUN=0
    ;;
  smoke-dry-run)
    TOTAL_ITER="${MOTIONX322_SATO_PAPER_TOTAL_ITER:-2}"
    DATASET_LIMIT="${MOTIONX322_SATO_PAPER_DATASET_LIMIT:-64}"
    PRINT_ITER=1
    EVAL_ITER=1000000
    SAVE_ITER=1000000
    DEFAULT_EXP_NAME="motionx322_sato_paper_faithful_rsr_smoke_seed${SEED}"
    DRY_RUN=1
    ;;
  formal)
    TOTAL_ITER=100000
    DATASET_LIMIT=0
    PRINT_ITER="${MOTIONX322_SATO_PAPER_PRINT_ITER:-100}"
    EVAL_ITER="${MOTIONX322_SATO_PAPER_EVAL_ITER:-5000}"
    SAVE_ITER="${MOTIONX322_SATO_PAPER_SAVE_ITER:-5000}"
    DEFAULT_EXP_NAME="motionx322_sato_paper_faithful_rsr_100k_4gpu3457_seed${SEED}"
    DRY_RUN=0
    ;;
  formal-dry-run|--dry-run)
    TOTAL_ITER=100000
    DATASET_LIMIT=0
    PRINT_ITER="${MOTIONX322_SATO_PAPER_PRINT_ITER:-100}"
    EVAL_ITER="${MOTIONX322_SATO_PAPER_EVAL_ITER:-5000}"
    SAVE_ITER="${MOTIONX322_SATO_PAPER_SAVE_ITER:-5000}"
    DEFAULT_EXP_NAME="motionx322_sato_paper_faithful_rsr_100k_4gpu3457_seed${SEED}"
    DRY_RUN=1
    ;;
  *)
    echo "usage: $0 {smoke|smoke-dry-run|formal|formal-dry-run}" >&2
    exit 2
    ;;
esac

PAIR_JSONL="${MOTIONX322_SATO_PAPER_PAIR_JSONL:-${ROOT}/dataset/Motion-X-322/paper_faithful_rsr_pairs_sato_train.jsonl}"
PAIR_MANIFEST="${MOTIONX322_SATO_PAPER_PAIR_MANIFEST:-${ROOT}/dataset/Motion-X-322/paper_faithful_rsr_pairs_sato_train.manifest.json}"
DATA_ROOT="${MOTIONX322_SATO_PAPER_DATA_ROOT:-${ROOT}/dataset/Motion-X-322-MaxToken75}"
VQ_CKPT="${MOTIONX322_SATO_PAPER_VQ_CKPT:-${ROOT}/output_motionx322/VQVAE_MOTIONX322/net_best_fid.pth}"
T2M_CKPT="${MOTIONX322_SATO_PAPER_T2M_CKPT:-${ROOT}/output_motionx322/GPT_MOTIONX322_MAXTOK75_200K/net_best_fid.pth}"
CLIP_CKPT="${MOTIONX322_SATO_PAPER_CLIP_CKPT:-${SATO_DIR}/pretrained/ViT-B-32.pt}"
OUT_DIR="${MOTIONX322_SATO_PAPER_OUT_DIR:-${ROOT}/output_motionx322_sato}"
EXP_NAME="${MOTIONX322_SATO_PAPER_EXP_NAME:-${DEFAULT_EXP_NAME}}"
EXP_DIR="${OUT_DIR}/${EXP_NAME}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 1; }
[[ -x "${TORCHRUN}" ]] || { echo "Missing torchrun: ${TORCHRUN}" >&2; exit 1; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing dataset root: ${DATA_ROOT}" >&2; exit 1; }
[[ -s "${PAIR_JSONL}" ]] || { echo "Missing paper-faithful RSR pairs: ${PAIR_JSONL}" >&2; exit 1; }
[[ -s "${PAIR_MANIFEST}" ]] || { echo "Missing pair manifest: ${PAIR_MANIFEST}" >&2; exit 1; }
[[ -f "${VQ_CKPT}" ]] || { echo "Missing VQ checkpoint: ${VQ_CKPT}" >&2; exit 1; }
[[ -f "${T2M_CKPT}" ]] || { echo "Missing transformer checkpoint: ${T2M_CKPT}" >&2; exit 1; }
[[ -f "${CLIP_CKPT}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_CKPT}" >&2; exit 1; }

COMMAND=("${TORCHRUN}" --standalone --nproc_per_node="${NPROC}" "${SATO_DIR}/train.py"
  --dataname motionx322_75
  --batch-size "${LOCAL_BATCH_SIZE}"
  --total-iter "${TOTAL_ITER}"
  --warm-up-iter 1000
  --lr 1e-4
  --lr-scheduler 60000
  --gamma 0.05
  --optimizer adamw
  --weight-decay 1e-6
  --print-iter "${PRINT_ITER}"
  --eval-iter "${EVAL_ITER}"
  --save-iter "${SAVE_ITER}"
  --seed "${SEED}"
  --pkeep 0.5
  --vq-name VQVAE
  --skip-tokenize
  --pair-jsonl "${PAIR_JSONL}"
  --paper-faithful-rsr
  --K 2
  --teacher-loss-weight 0.1
  --attention-loss-weight 0.2
  --robust-loss-weight 0.05
  --resume-pth "${VQ_CKPT}"
  --resume-trans "${T2M_CKPT}"
  --clip_path "${CLIP_CKPT}"
  --block-size 76
  --num-workers "${MOTIONX322_SATO_PAPER_NUM_WORKERS:-0}"
  --dataset-limit "${DATASET_LIMIT}"
  --out-dir "${OUT_DIR}"
  --exp-name "${EXP_NAME}")

printf 'CUDA_VISIBLE_DEVICES=%s ' "${GPUS}"
printf '%q ' "${COMMAND[@]}"
printf '\n'
if (( DRY_RUN )); then
  exit 0
fi

if [[ -e "${EXP_DIR}/training_complete.json" || -e "${EXP_DIR}/run.log" ]]; then
  echo "Refusing to overwrite existing experiment: ${EXP_DIR}" >&2
  exit 1
fi
mkdir -p "${EXP_DIR}"
printf 'CUDA_VISIBLE_DEVICES=%s ' "${GPUS}" > "${EXP_DIR}/command.txt"
printf '%q ' "${COMMAND[@]}" >> "${EXP_DIR}/command.txt"
printf '\n' >> "${EXP_DIR}/command.txt"

export ROOT SATO_DIR PAIR_JSONL PAIR_MANIFEST DATA_ROOT VQ_CKPT T2M_CKPT CLIP_CKPT
export OUT_DIR EXP_NAME EXP_DIR GPUS NPROC GLOBAL_BATCH_SIZE LOCAL_BATCH_SIZE SEED TOTAL_ITER
"${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

root = Path(os.environ['ROOT'])
sato = Path(os.environ['SATO_DIR'])
manifest = {
    'schema_version': 1,
    'experiment': 'Motion-X SATO paper-faithful RSR',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'paper': {
        'title': 'SATO: Stable Text-to-Motion Framework',
        'arxiv': '2405.01461v3',
        'final_perturbation_method': 'Random Synonym Replacement (RSR)',
        'loss': 'L_trans + lambda1*teacher + lambda2*TopK + lambda3*robust',
        'lambda1_teacher': 0.1,
        'lambda2_attention': 0.2,
        'lambda3_robust': 0.05,
        'top_k': 2,
        'batch_size': int(os.environ['GLOBAL_BATCH_SIZE']),
        'optimizer': 'AdamW',
        'betas': [0.9, 0.99],
        'learning_rate': 1e-4,
        'total_iterations': int(os.environ['TOTAL_ITER']),
    },
    'released_source': {
        'repository': 'https://github.com/sato-team/Stable-Text-to-Motion-Framework.git',
        'commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=sato, text=True).strip(),
        'working_tree_diff_sha256': hashlib.sha256(subprocess.check_output(
            ['git', 'diff', '--binary'], cwd=sato)).hexdigest(),
    },
    'motionx_required_adaptation': {
        'feature_dimension': 322,
        'fps': 30,
        'vq_unit_length': 4,
        'maximum_frames': 300,
        'maximum_motion_tokens': 75,
        'block_size_with_eos': 76,
        'distributed_gpus': os.environ['GPUS'],
        'world_size': int(os.environ['NPROC']),
        'local_batch_size': int(os.environ['LOCAL_BATCH_SIZE']),
        'seed': int(os.environ['SEED']),
    },
    'recipe_decisions': {
        'pgd_enabled': False,
        'early_stopping_enabled': False,
        'student_clip_trainable': True,
        'teacher_clip_frozen': True,
        'teacher_transformer_frozen': True,
        'vqvae_frozen': True,
        'generation_cross_entropy_input': 'original text x',
        'rsr_pair_input': 'synonym-perturbed text x_prime',
        'clip_attention_position_logic': 'released code whitespace-count EOT heuristic',
        'strict_bpe_alignment': False,
        'warmup_iterations': 1000,
        'warmup_source': 'T2M-GPT architecture default; paper states linear warm-up but omits duration',
        'lr_milestone': 60000,
        'lr_gamma': 0.05,
        'scheduler_source': 'released SATO/T2M-GPT options',
        'weight_decay': 1e-6,
        'adam_epsilon': 1e-3,
    },
    'artifacts': {},
}
for name, value in {
    'pair_jsonl': os.environ['PAIR_JSONL'],
    'pair_manifest': os.environ['PAIR_MANIFEST'],
    'vq_checkpoint': os.environ['VQ_CKPT'],
    'transformer_checkpoint': os.environ['T2M_CKPT'],
    'clip_checkpoint': os.environ['CLIP_CKPT'],
    'train_split': root / 'dataset/Motion-X-322/train.txt',
    'val_split': root / 'dataset/Motion-X-322/val.txt',
    'test_split': root / 'dataset/Motion-X-322/test.txt',
}.items():
    path = Path(value).resolve()
    manifest['artifacts'][name] = {'path': str(path), 'sha256': sha256(path)}

out = Path(os.environ['EXP_DIR']) / 'experiment_manifest.json'
out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

cd "${SATO_DIR}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/lib/python3.10/site-packages/torch/lib:/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
exec "${COMMAND[@]}"