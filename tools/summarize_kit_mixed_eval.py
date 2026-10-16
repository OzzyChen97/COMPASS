#!/usr/bin/env python
"""Summarize corrected KIT mixed-set baseline/15k/80k evaluations."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


METRICS = ("fid", "fid_p", "fid_d", "matching", "top1", "top2", "top3")
ORIGINAL_METRICS = ("fid", "matching", "top1", "top2", "top3")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sato15", type=Path, required=True)
    parser.add_argument("--sato80", type=Path, required=True)
    parser.add_argument("--audit", action="append", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_runs(path):
    return json.loads(path.read_text(encoding="utf-8"))["runs"]


def aggregate(name, path):
    runs = load_runs(path)
    row = {"model": name, "n": len(runs), "source": str(path)}
    for metric in METRICS:
        values = [float(run[metric]) for run in runs]
        row[f"{metric}_mean"] = mean(values)
        row[f"{metric}_std"] = stdev(values)
    return row, {int(run["seed"]): run for run in runs}


def paired_delta(name, baseline, candidate):
    seeds = sorted(baseline.keys() & candidate.keys())
    row = {"comparison": name, "n": len(seeds)}
    t_critical = 2.093 if len(seeds) == 20 else 1.96
    for metric in METRICS:
        values = [
            float(candidate[seed][metric]) - float(baseline[seed][metric])
            for seed in seeds
        ]
        value = mean(values)
        half = t_critical * stdev(values) / math.sqrt(len(values))
        row[f"{metric}_delta"] = value
        row[f"{metric}_ci_low"] = value - half
        row[f"{metric}_ci_high"] = value + half
    return row


def audit_invariance(paths):
    rows = [load_runs(path)[0] for path in paths]
    metrics = {
        metric: [float(row[metric]) for row in rows]
        for metric in ORIGINAL_METRICS
    }
    maximum_spread = {metric: max(values) - min(values) for metric, values in metrics.items()}
    return {
        "sources": [str(path) for path in paths],
        "values": metrics,
        "maximum_spread": maximum_spread,
        "passed": all(spread < 1e-10 for spread in maximum_spread.values()),
    }


def fmt(value):
    return f"{value:.4f}"


def main():
    args = parse_args()
    baseline_row, baseline_runs = aggregate("baseline", args.baseline)
    sato15_row, sato15_runs = aggregate("sato15k", args.sato15)
    sato80_row, sato80_runs = aggregate("sato80k", args.sato80)
    aggregates = [baseline_row, sato15_row, sato80_row]
    deltas = [
        paired_delta("sato15k_minus_baseline", baseline_runs, sato15_runs),
        paired_delta("sato80k_minus_baseline", baseline_runs, sato80_runs),
        paired_delta("sato80k_minus_sato15k", sato15_runs, sato80_runs),
    ]
    audit = audit_invariance(args.audit)
    output = {"aggregates": aggregates, "paired_deltas": deltas, "rng_audit": audit}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)

    lines = [
        "# KIT-ML Corrected Mixed Perturbation Evaluation",
        "",
        "Mixed set: one pair per motion, HumanML3D level ratio (P1=336, P2=199, P3=199). Repeat20 values are mean +/- sample standard deviation.",
        "",
        f"RNG isolation audit: `{'PASS' if audit['passed'] else 'FAIL'}`; maximum standard-metric spread across P1/P2/P3 = `{max(audit['maximum_spread'].values()):.12g}`.",
        "",
        "| Model | n | FID | FID_P | FID_D | Matching | Top1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['model']} | {row['n']} | "
            f"{fmt(row['fid_mean'])} +/- {fmt(row['fid_std'])} | "
            f"{fmt(row['fid_p_mean'])} +/- {fmt(row['fid_p_std'])} | "
            f"{fmt(row['fid_d_mean'])} +/- {fmt(row['fid_d_std'])} | "
            f"{fmt(row['matching_mean'])} +/- {fmt(row['matching_std'])} | "
            f"{fmt(row['top1_mean'])} +/- {fmt(row['top1_std'])} |"
        )
    lines.extend([
        "",
        "## Paired Mean Deltas",
        "",
        "| Comparison | Delta FID | Delta FID_P | Delta FID_D | Delta Matching | Delta Top1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in deltas:
        lines.append(
            f"| {row['comparison']} | {fmt(row['fid_delta'])} | "
            f"{fmt(row['fid_p_delta'])} | {fmt(row['fid_d_delta'])} | "
            f"{fmt(row['matching_delta'])} | {fmt(row['top1_delta'])} |"
        )
    lines.append("")
    lines.append("The JSON file contains paired 95% confidence intervals and full RNG audit values.")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
