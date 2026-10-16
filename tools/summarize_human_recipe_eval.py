#!/usr/bin/env python
"""Summarize paired Repeat-N baseline and Human-recipe SATO evaluations."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


METRICS = ("fid", "fid_p", "fid_d", "matching", "top1", "top2", "top3")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate in NAME=summary.json form.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--protocol-description",
        default="Caption-locked, full-set, legacy autoregressive Repeat-N evaluation.",
    )
    return parser.parse_args()


def load_runs(path):
    rows = json.loads(path.read_text(encoding="utf-8"))["runs"]
    return {int(row["seed"]): row for row in rows}


def aggregate(name, runs):
    row = {"model": name, "n": len(runs)}
    values_by_metric = {}
    for metric in METRICS:
        values = [float(run[metric]) for run in runs.values()]
        values_by_metric[metric] = values
        row[f"{metric}_mean"] = mean(values)
        row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
    return row, values_by_metric


def paired_delta(name, baseline, candidate):
    seeds = sorted(baseline.keys() & candidate.keys())
    if len(seeds) < 2:
        raise ValueError(f"Need at least two paired seeds for {name}")
    # Repeat20 uses t(19); fall back to a normal approximation for other N.
    critical = 2.093 if len(seeds) == 20 else 1.96
    row = {"comparison": f"{name}_minus_baseline", "n": len(seeds)}
    for metric in METRICS:
        values = [
            float(candidate[seed][metric]) - float(baseline[seed][metric])
            for seed in seeds
        ]
        delta = mean(values)
        half_width = critical * stdev(values) / math.sqrt(len(values))
        row[f"{metric}_delta"] = delta
        row[f"{metric}_ci_low"] = delta - half_width
        row[f"{metric}_ci_high"] = delta + half_width
    return row


def fmt(value):
    return f"{value:.4f}"


def main():
    args = parse_args()
    baseline_runs = load_runs(args.baseline)
    aggregates = [aggregate("baseline", baseline_runs)[0]]
    deltas = []

    for value in args.candidate:
        if "=" not in value:
            raise ValueError("--candidate must be NAME=summary.json")
        name, raw_path = value.split("=", 1)
        candidate_runs = load_runs(Path(raw_path))
        aggregates.append(aggregate(name, candidate_runs)[0])
        deltas.append(paired_delta(name, baseline_runs, candidate_runs))

    output = {"aggregates": aggregates, "paired_deltas": deltas}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# KIT HumanML3D-Recipe 35k Evaluation",
        "",
        args.protocol_description,
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
        "## Paired Deltas",
        "",
        "| Comparison | Delta FID | Delta FID_P | Delta FID_D | 95% CI FID_P |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in deltas:
        lines.append(
            f"| {row['comparison']} | {fmt(row['fid_delta'])} | "
            f"{fmt(row['fid_p_delta'])} | {fmt(row['fid_d_delta'])} | "
            f"[{fmt(row['fid_p_ci_low'])}, {fmt(row['fid_p_ci_high'])}] |"
        )
    lines.append("")
    lines.append("The JSON file contains paired confidence intervals for every metric.")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
