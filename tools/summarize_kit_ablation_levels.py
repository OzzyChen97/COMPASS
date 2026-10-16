#!/usr/bin/env python
"""Combine P1/P2/P3 KIT-ML baseline vs SATO++ evaluation summaries."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


METRICS = ("fid", "fid_p", "fid_d", "matching", "top1", "top2", "top3")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LEVEL=SUMMARY_JSON",
        help="Evaluation level and repeat-eval summary.json path.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="KIT-ML T2M-GPT Baseline vs Full SATO++",
    )
    return parser.parse_args()


def as_float(value):
    return float(value) if value not in (None, "") else None


def load_level(value):
    if "=" not in value:
        raise ValueError(f"Expected LEVEL=SUMMARY_JSON, got: {value}")
    level, raw_path = value.split("=", 1)
    path = Path(raw_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for source in payload["summary_wide"]:
        row = {
            "level": level.upper(),
            "model": source["model"],
            "kind": source["kind"],
            "checkpoint": source.get("checkpoint", ""),
            "n": int(source["n"]),
            "source": str(path),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = as_float(source.get(f"{metric}_mean"))
            row[f"{metric}_std"] = as_float(source.get(f"{metric}_std"))
        rows.append(row)
    return level.upper(), rows


def build_deltas(level, rows):
    by_kind = {row["kind"]: row for row in rows}
    if "baseline" not in by_kind or "sato_pp" not in by_kind:
        raise ValueError(f"{level} must contain one baseline and one sato_pp row")
    baseline = by_kind["baseline"]
    sato = by_kind["sato_pp"]
    delta = {"level": level, "comparison": "sato_pp_minus_baseline"}
    for metric in METRICS:
        delta[f"{metric}_delta"] = sato[f"{metric}_mean"] - baseline[f"{metric}_mean"]
    return delta


def fmt(value):
    return f"{value:.4f}"


def write_markdown(path, rows, deltas, overall, title):
    results_heading = "Results" if len(deltas) == 1 else "Per-Level Results"
    overall_heading = "Overall Delta" if len(deltas) == 1 else "Mean Delta Across Levels"
    lines = [
        f"# {title}",
        "",
        "All values are Repeat20 mean +/- sample standard deviation. Lower is better for FID, FID_P, FID_D and Matching; higher is better for Top-k retrieval.",
        "",
        f"## {results_heading}",
        "",
        "| Level | Model | n | FID | FID_P | FID_D | Matching | Top1 | Top2 | Top3 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['level']} | {row['kind']} | {row['n']} | "
            f"{fmt(row['fid_mean'])} +/- {fmt(row['fid_std'])} | "
            f"{fmt(row['fid_p_mean'])} +/- {fmt(row['fid_p_std'])} | "
            f"{fmt(row['fid_d_mean'])} +/- {fmt(row['fid_d_std'])} | "
            f"{fmt(row['matching_mean'])} +/- {fmt(row['matching_std'])} | "
            f"{fmt(row['top1_mean'])} +/- {fmt(row['top1_std'])} | "
            f"{fmt(row['top2_mean'])} +/- {fmt(row['top2_std'])} | "
            f"{fmt(row['top3_mean'])} +/- {fmt(row['top3_std'])} |"
        )
    lines.extend(
        [
            "",
            "## SATO++ Minus Baseline",
            "",
            "| Level | Delta FID | Delta FID_P | Delta FID_D | Delta Matching | Delta Top1 | Delta Top2 | Delta Top3 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for delta in deltas:
        lines.append(
            f"| {delta['level']} | {fmt(delta['fid_delta'])} | "
            f"{fmt(delta['fid_p_delta'])} | {fmt(delta['fid_d_delta'])} | "
            f"{fmt(delta['matching_delta'])} | {fmt(delta['top1_delta'])} | "
            f"{fmt(delta['top2_delta'])} | {fmt(delta['top3_delta'])} |"
        )
    lines.extend(
        [
            "",
            f"## {overall_heading}",
            "",
            "| Delta FID | Delta FID_P | Delta FID_D | Delta Matching | Delta Top1 | Delta Top2 | Delta Top3 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {fmt(overall['fid_delta'])} | {fmt(overall['fid_p_delta'])} | "
            f"{fmt(overall['fid_d_delta'])} | {fmt(overall['matching_delta'])} | "
            f"{fmt(overall['top1_delta'])} | {fmt(overall['top2_delta'])} | "
            f"{fmt(overall['top3_delta'])} |",
            "",
            "The JSON and CSV next to this report contain all selected metrics and source paths.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rows = []
    deltas = []
    for value in args.input:
        level, level_rows = load_level(value)
        rows.extend(level_rows)
        deltas.append(build_deltas(level, level_rows))
    overall = {
        f"{metric}_delta": mean(delta[f"{metric}_delta"] for delta in deltas)
        for metric in METRICS
    }
    output = {"results": rows, "deltas": deltas, "mean_delta_across_levels": overall}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    fields = list(rows[0])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(args.output_md, rows, deltas, overall, args.title)
    print(args.output_md)


if __name__ == "__main__":
    main()
