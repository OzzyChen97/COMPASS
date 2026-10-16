#!/usr/bin/env python
"""Read-only health audit for the formal SATO++ var/cov experiment."""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path


VARIANTS = ("full", "no_var", "no_cov", "no_both")
ITER_RE = re.compile(r"Train\. Iter (\d+):")
ERROR_RE = re.compile(r"Traceback|CUDA out|RuntimeError|Error:")


def last_iteration(path: Path):
    if not path.is_file():
        return None
    value = None
    for match in ITER_RE.finditer(path.read_text(errors="replace")):
        value = int(match.group(1))
    return value


def pid_alive(path: Path):
    if not path.is_file():
        return None, False
    try:
        pid = int(path.read_text().strip())
    except ValueError:
        return None, False
    return pid, Path(f"/proc/{pid}").exists()


def find_training_pid(exp_name: str):
    needle = f"--exp-name {exp_name}"
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except (OSError, PermissionError):
            continue
        if "train_sato_pp.py" in command and needle in command:
            return int(proc.name)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--total-iter", type=int, default=150000)
    parser.add_argument("--max-log-age-sec", type=float, default=1200)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    rows = []
    failures = []
    for variant in VARIANTS:
        exp = args.run_root / f"humanml3d_{variant}_seed{args.seed}"
        log = exp / "train.log"
        final = exp / "sato_pp_final.pth"
        exit_file = exp / "train_exit_code.txt"
        launcher_pid = args.run_root / f"humanml3d_{variant}_launcher.pid"
        launcher, launcher_alive = pid_alive(launcher_pid)
        training_pid = find_training_pid(exp.name)
        alive = training_pid is not None
        iteration = last_iteration(log)
        age = None if not log.exists() else max(0.0, time.time() - log.stat().st_mtime)
        exit_code = exit_file.read_text().strip() if exit_file.is_file() else ""
        complete = final.is_file() and final.stat().st_size > 0

        if complete:
            state = "complete"
            if iteration != args.total_iter:
                failures.append(f"{variant}: final exists but last logged iteration is {iteration}")
            if exit_code and exit_code != "0":
                failures.append(f"{variant}: final exists with exit code {exit_code}")
        elif exit_code:
            state = "failed"
            failures.append(f"{variant}: exited before final with code {exit_code}")
        elif alive:
            state = "running_or_validating"
            if age is not None and age > args.max_log_age_sec:
                failures.append(f"{variant}: log is stale by {age:.0f}s")
        else:
            state = "unknown_or_stopped"
            failures.append(f"{variant}: launcher absent and no final checkpoint")

        rows.append({
            "variant": variant,
            "state": state,
            "iteration": "" if iteration is None else iteration,
            "log_age_sec": "" if age is None else f"{age:.1f}",
            "launcher_pid": "" if launcher is None else launcher,
            "launcher_alive": launcher_alive,
            "training_pid": "" if training_pid is None else training_pid,
            "exit_code": exit_code,
            "final": complete,
        })

    writer = csv.DictWriter(os.sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    if args.json_out is not None:
        payload = {
            "run_root": str(args.run_root),
            "seed": args.seed,
            "total_iter": args.total_iter,
            "checked_at_unix": time.time(),
            "rows": rows,
            "failures": failures,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=os.sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
