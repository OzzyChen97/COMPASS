#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/workspace/Wilson/T2M-GPT
OUT_ROOT="${OUT_ROOT:-${ROOT}/MotionGPT/experiments/robustness/fixed_caption_100x10_20260806}"
export OUT_ROOT
mkdir -p "${OUT_ROOT}/logs"

"${ROOT}/tools/run_fixed_caption_diagnostic.sh" motiongpt 0 0 2 &
pid0=$!
"${ROOT}/tools/run_fixed_caption_diagnostic.sh" motiongpt 1 3 5 &
pid1=$!
"${ROOT}/tools/run_fixed_caption_diagnostic.sh" motiongpt 2 6 7 &
pid2=$!
"${ROOT}/tools/run_fixed_caption_diagnostic.sh" motiongpt 3 8 9 &
pid3=$!
"${ROOT}/tools/run_fixed_caption_diagnostic.sh" sato_pp 4 &
pid4=$!

status=0
for pid in "${pid0}" "${pid1}" "${pid2}" "${pid3}" "${pid4}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  echo "one or more diagnostic shards failed; inspect ${OUT_ROOT}/logs" >&2
  exit "${status}"
fi

/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python \
  "${ROOT}/tools/analyze_fixed_caption_repeats.py" \
  --model "MotionGPT=${OUT_ROOT}/motiongpt/repeat_*.npz" \
  --model "SATO++=${OUT_ROOT}/sato_pp/sato_pp_allloss_l1_iter130k/seed*.npz" \
  --output-dir "${OUT_ROOT}/final_motiongpt_vs_sato_pp"
