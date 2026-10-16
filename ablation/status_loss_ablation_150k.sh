#!/usr/bin/env bash
set -euo pipefail

cd /mnt/workspace/Wilson/T2M-GPT

echo "== GPU =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo
echo "== Runs =="
for exp in \
  loss_full_allloss_l1_150k \
  loss_no_teacher_l1_150k \
  loss_no_preserve_l1_150k \
  loss_no_alter_l1_150k \
  loss_no_attribute_l1_150k
do
  out="ablation/$exp"
  pid_file="$out/pid.txt"
  pid=""
  state="not-started"
  if [[ -s "$pid_file" ]]; then
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      state="running"
    else
      state="exited"
    fi
  fi
  exit_code="NA"
  if [[ -s "$out/exit_code.txt" ]]; then
    exit_code=$(cat "$out/exit_code.txt")
  fi
  iter=$(rg -o "Train\\. Iter [0-9]+" "$out/run.log" "$out/nohup.log" 2>/dev/null | rg -o "[0-9]+" | tail -n 1 || true)
  best=$(rg "Best\\. Iter|Saved full SATO\\+\\+ best-FID|Eva\\. Iter" "$out/run.log" "$out/nohup.log" 2>/dev/null | tail -n 1 || true)
  printf "%-32s pid=%-8s state=%-10s exit=%-4s last_iter=%s\n" "$exp" "$pid" "$state" "$exit_code" "${iter:-NA}"
  if [[ -n "$best" ]]; then
    echo "  $best"
  fi
done
