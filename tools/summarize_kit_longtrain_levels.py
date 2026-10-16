#!/usr/bin/env python
"""Summarize paired KIT long-training evaluations across perturbation levels."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


METRICS = ("fid", "fid_p", "fid_d", "matching", "top1", "top2", "top3")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="LEVEL:LABEL=SUMMARY_JSON",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def parse_source(value):
    if ":" not in value or "=" not in value:
        raise ValueError(f"Expected LEVEL:LABEL=SUMMARY_JSON, got {value}")
    level, remainder = value.split(":", 1)
    label, raw_path = remainder.split("=", 1)
    path = Path(raw_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    models = {str(run["model"]) for run in runs}
    if len(models) != 1:
        raise ValueError(f"Expected exactly one model in {path}, got {sorted(models)}")
    by_seed = {int(run["seed"]): run for run in runs}
    if len(by_seed) != len(runs):
        raise ValueError(f"Duplicate seeds in {path}")
    return level, label, path, by_seed


def aggregate(level, label, path, runs):
    row = {
        "level": level,
        "label": label,
        "n": len(runs),
        "source": str(path),
    }
    for metric in METRICS:
        values = [float(run[metric]) for run in runs.values()]
        row[f"{metric}_mean"] = mean(values)
        row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
    return row


def paired_delta(level, reference_label, reference, candidate_label, candidate):
    seeds = sorted(reference.keys() & candidate.keys())
    if len(seeds) < 2:
        raise ValueError(
            f"Need at least two paired seeds for {level}: "
            f"{candidate_label} vs {reference_label}"
        )
    critical = 2.093 if len(seeds) == 20 else 1.96
    row = {
        "level": level,
        "reference": reference_label,
        "candidate": candidate_label,
        "n": len(seeds),
    }
    for metric in METRICS:
        values = [
            float(candidate[seed][metric]) - float(reference[seed][metric])
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
    data = {}
    aggregates = []
    for value in args.source:
        level, label, path, runs = parse_source(value)
        if label in data.setdefault(level, {}):
            raise ValueError(f"Duplicate source for {level}:{label}")
        data[level][label] = runs
        aggregates.append(aggregate(level, label, path, runs))

    deltas = []
    for level, models in data.items():
        for reference_label in ("baseline", "35k"):
            if reference_label not in models:
                raise ValueError(f"Missing {level}:{reference_label}")
            for candidate_label, candidate in models.items():
                if candidate_label == reference_label:
                    continue
                if reference_label == "baseline" and candidate_label == "35k":
                    pass
                elif reference_label == "baseline" and candidate_label == "baseline":
                    continue
                elif reference_label == "35k" and candidate_label in ("baseline", "35k"):
                    continue
                deltas.append(
                    paired_delta(
                        level,
                        reference_label,
                        models[reference_label],
                        candidate_label,
                        candidate,
                    )
                )

    output = {"aggregates": aggregates, "paired_deltas": deltas}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)

    label_order = {"baseline": 0, "35k": 1, "60k": 2, "70k": 3, "80k": 4, "85k": 5}
    level_order = {"P1P2P3": 0, "P2P3": 1, "P3": 2}
    aggregates.sort(
        key=lambda row: (
            level_order.get(row["level"], 99),
            label_order.get(row["label"], 99),
        )
    )
    lines = [
        "# KIT SATO++ Long-Training Perturbation Evaluation",
        "",
        "HumanML3D-reference fast categorical Repeat20. Each level uses 734 pair rows; the standard drop-last loader evaluates 704 rows per seed.",
        "",
        "| Level | Checkpoint | FID | FID_P | FID_D | Matching | Top1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['level']} | {row['label']} | "
            f"{fmt(row['fid_mean'])} +/- {fmt(row['fid_std'])} | "
            f"{fmt(row['fid_p_mean'])} +/- {fmt(row['fid_p_std'])} | "
            f"{fmt(row['fid_d_mean'])} +/- {fmt(row['fid_d_std'])} | "
            f"{fmt(row['matching_mean'])} +/- {fmt(row['matching_std'])} | "
            f"{fmt(row['top1_mean'])} +/- {fmt(row['top1_std'])} |"
        )
    lines.extend(
        [
            "",
            "## Later Checkpoints Minus 35k",
            "",
            "Negative is better for FID/FID_P/FID_D/Matching; positive is better for Top1.",
            "",
            "| Level | Checkpoint | Delta FID | Delta FID_P | Delta FID_D | Delta Matching | Delta Top1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in deltas:
        if row["reference"] != "35k":
            continue
        lines.append(
            f"| {row['level']} | {row['candidate']} | "
            f"{fmt(row['fid_delta'])} | {fmt(row['fid_p_delta'])} | "
            f"{fmt(row['fid_d_delta'])} | {fmt(row['matching_delta'])} | "
            f"{fmt(row['top1_delta'])} |"
        )
    lines.extend(
        [
            "",
            "The JSON file contains paired 95% confidence intervals against both baseline and 35k.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
