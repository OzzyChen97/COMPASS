#!/usr/bin/env bash
# Fixed-budget, train/validation-only Motion-X adaptation controls on GPUs 4/5.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
OUT="${PILOT_OUT:-$ROOT/ablation/motionx_text_adaptation_pilot_20260905}"
ASSETS="${PILOT_ASSETS:-$ROOT/ablation/motionx_text_adaptation_pilot_20260905/assets}"
STEPS="${PILOT_STEPS:-1000}"
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'PILOT_STEPS must be positive' >&2; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1
for gpu in 4 5; do
  memory="$(timeout 20s nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
  [[ "$memory" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || { echo "Invalid GPU $gpu memory reading" >&2; exit 1; }
  if (( memory > 2048 )); then
    echo "GPU $gpu is occupied (${memory} MiB); refusing to launch" >&2
    exit 1
  fi
done
[[ -s "$ASSETS/manifest.json" ]] || { echo 'Prepare assets first' >&2; exit 1; }
for existing in "$OUT/main" "$OUT/validation" "$OUT/queues" "$OUT/pilot_summary.csv"; do
  [[ ! -e "$existing" ]] || { echo "Refusing to overwrite $existing" >&2; exit 1; }
done
mkdir -p "$OUT/queues"
train_arm() {
  local arm="$1"
  "$PY" -u tools/train_motionx_text_adaptation_pilot.py \
    --arm "$arm" --train-pairs "$ASSETS/train_preserve.jsonl" \
    --out-dir "$OUT/main/$arm" --steps "$STEPS" --batch-size 16 --seed 123 \
    --trans-lr 2e-5 --clip-lr 1e-6 --save-every 2000 --log-every 50
}
eval_arm() {
  local arm="$1"
  local extra=()
  if [[ "$arm" != baseline ]]; then
    extra=(--checkpoint "$OUT/main/$arm/motionx_text_adaptation_pilot_last.pth")
  fi
  "$PY" -u tools/eval_motionx_text_adaptation_pilot.py \
    --name "$arm" "${extra[@]}" --assets "$ASSETS" \
    --output "$OUT/validation/$arm" --seeds 123 124 125
}
queue4() {
  export CUDA_VISIBLE_DEVICES=4
  train_arm generator
  eval_arm generator
  eval_arm baseline
  echo QUEUE4_COMPLETE
}
queue5() {
  export CUDA_VISIBLE_DEVICES=5
  train_arm text
  train_arm joint
  eval_arm text
  eval_arm joint
  echo QUEUE5_COMPLETE
}
queue4 > "$OUT/queues/gpu4.log" 2>&1 & p4=$!
queue5 > "$OUT/queues/gpu5.log" 2>&1 & p5=$!
printf 'GPU4_PID=%s GPU5_PID=%s\n' "$p4" "$p5"
failed=0
wait "$p4" || failed=1
wait "$p5" || failed=1
if (( failed )); then
  echo 'PILOT_PIPELINE_FAILED: inspect queues/gpu4.log and gpu5.log' >&2
  exit 1
fi
"$PY" - "$OUT" "$STEPS" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
train = {a: json.loads((root / 'main' / a / 'manifest.json').read_text()) for a in ('generator', 'text', 'joint')}
assert len({m['schedule_sha256'] for m in train.values()}) == 1, 'Training schedules differ'
assert all(m['frozen_hashes_initial'] == train['generator']['frozen_hashes_initial'] for m in train.values()), 'Initial weights differ'
summary = []
for arm in ('baseline', 'generator', 'text', 'joint'):
    manifest = json.loads((root / 'validation' / arm / 'manifest.json').read_text())
    assert manifest['samples'] == 661 and manifest['generation_seeds'] == [123, 124, 125]
    summary.extend(json.loads((root / 'validation' / arm / 'summary.json').read_text()))
with (root / 'pilot_summary.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
(root / 'pilot_summary.json').write_text(json.dumps({'results': summary, 'training_schedule_matched': True, 'initial_weights_matched': True, 'training_steps': int(sys.argv[2]), 'training_seed': 123, 'note': 'Diagnostic plain clean+paraphrase supervised CE, not SATO++ training. Fixed endpoints, three generation seeds; no test selection.'}, indent=2) + '\n')
PY
echo MOTIONX_TEXT_PILOT_COMPLETE