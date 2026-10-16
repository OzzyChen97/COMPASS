#!/usr/bin/env python
"""Compare two SATO++ Repeat20 checkpoints with paired-seed statistics."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


METRICS = ("fid", "fid_p", "fid_d", "matching", "top1", "top2", "top3")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="append", required=True, metavar="LEVEL=JSON")
    parser.add_argument("--candidate", action="append", required=True, metavar="LEVEL=JSON")
    parser.add_argument("--reference-name", default="15k-best")
    parser.add_argument("--candidate-name", default="80k-last")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def parse_paths(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LEVEL=JSON, got {value}")
        level, path = value.split("=", 1)
        result[level.upper()] = Path(path)
    return result


def load_sato(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload["runs"] if row["kind"] == "sato_pp"]
    if not rows:
        raise ValueError(f"No sato_pp rows in {path}")
    return {int(row["seed"]): row for row in rows}


def fmt(value):
    return f"{value:.4f}"


def main():
    args = parse_args()
    references = parse_paths(args.reference)
    candidates = parse_paths(args.candidate)
    if references.keys() != candidates.keys():
        raise ValueError("Reference and candidate levels must match")

    results = []
    for level in references:
        reference = load_sato(references[level])
        candidate = load_sato(candidates[level])
        seeds = sorted(reference.keys() & candidate.keys())
        if len(seeds) < 2:
            raise ValueError(f"Not enough paired seeds for {level}")
        row = {
            "level": level,
            "n": len(seeds),
            "reference": args.reference_name,
            "candidate": args.candidate_name,
            "reference_source": str(references[level]),
            "candidate_source": str(candidates[level]),
        }
        t_critical = 2.093 if len(seeds) == 20 else 1.96
        for metric in METRICS:
            ref_values = [float(reference[seed][metric]) for seed in seeds]
            cand_values = [float(candidate[seed][metric]) for seed in seeds]
            deltas = [cand - ref for ref, cand in zip(ref_values, cand_values)]
            delta_mean = mean(deltas)
            ci_half = t_critical * stdev(deltas) / math.sqrt(len(deltas))
            row[f"{metric}_reference_mean"] = mean(ref_values)
            row[f"{metric}_reference_std"] = stdev(ref_values)
            row[f"{metric}_candidate_mean"] = mean(cand_values)
            row[f"{metric}_candidate_std"] = stdev(cand_values)
            row[f"{metric}_delta"] = delta_mean
            row[f"{metric}_delta_ci_low"] = delta_mean - ci_half
            row[f"{metric}_delta_ci_high"] = delta_mean + ci_half
        results.append(row)

    output = {
        "reference": args.reference_name,
        "candidate": args.candidate_name,
        "delta_definition": "candidate_minus_reference",
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    lines = [
        f"# KIT-ML SATO++ {args.reference_name} vs {args.candidate_name}",
        "",
        f"Repeat20 paired by seed. Delta is {args.candidate_name} minus "
        f"{args.reference_name}. Lower is better for FID, FID_P, FID_D "
        "and Matching; higher is better for Top-k.",
        "",
        "| Level | Checkpoint | FID | FID_P | FID_D | Matching | Top1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        for prefix, name in (("reference", args.reference_name), ("candidate", args.candidate_name)):
            lines.append(
                f"| {row['level']} | {name} | "
                f"{fmt(row[f'fid_{prefix}_mean'])} +/- {fmt(row[f'fid_{prefix}_std'])} | "
                f"{fmt(row[f'fid_p_{prefix}_mean'])} +/- {fmt(row[f'fid_p_{prefix}_std'])} | "
                f"{fmt(row[f'fid_d_{prefix}_mean'])} +/- {fmt(row[f'fid_d_{prefix}_std'])} | "
                f"{fmt(row[f'matching_{prefix}_mean'])} +/- {fmt(row[f'matching_{prefix}_std'])} | "
                f"{fmt(row[f'top1_{prefix}_mean'])} +/- {fmt(row[f'top1_{prefix}_std'])} |"
            )
    lines.extend(
        [
            "",
            f"## {args.candidate_name} Minus {args.reference_name} Paired Delta",
            "",
            "| Level | Delta FID | Delta FID_P | Delta FID_D | Delta Matching | Delta Top1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in results:
        lines.append(
            f"| {row['level']} | {fmt(row['fid_delta'])} | {fmt(row['fid_p_delta'])} | "
            f"{fmt(row['fid_d_delta'])} | {fmt(row['matching_delta'])} | "
            f"{fmt(row['top1_delta'])} |"
        )
    lines.append("")
    lines.append("The JSON file contains paired 95% confidence intervals for every metric.")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
