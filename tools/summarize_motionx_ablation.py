#!/usr/bin/env python
"""Combine Motion-X baseline/full SATO++ Repeat20 level summaries."""

import argparse
import csv
import json
from pathlib import Path


METRICS = ("fid", "fid_p", "fid_d", "diversity", "top1", "top2", "top3", "matching")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="LEVEL=summary.json")
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def load_source(value):
    level, path = value.split("=", 1)
    payload = json.loads(Path(path).read_text())
    rows = {row["model"]: row for row in payload["summary_wide"]}
    return level, path, rows


def main():
    args = parse_args()
    combined = []
    markdown = [
        "# Motion-X T2M-GPT Baseline vs Full SATO++",
        "",
        "| Level | Model | FID | FID_P | FID_D | Diversity | Top1 | Top2 | Top3 | Matching |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in args.source:
        level, path, rows = load_source(source)
        for model, row in sorted(rows.items()):
            output = {"level": level, "model": model, "source": path, "n": row["n"]}
            cells = []
            for metric in METRICS:
                mean = float(row[f"{metric}_mean"])
                std = float(row[f"{metric}_std"])
                output[f"{metric}_mean"] = mean
                output[f"{metric}_std"] = std
                cells.append(f"{mean:.4f}+/-{std:.4f}")
            combined.append(output)
            markdown.append(f"| {level} | `{model}` | " + " | ".join(cells) + " |")

        if len(rows) == 2:
            baseline_name = next(name for name in rows if "baseline" in name)
            sato_name = next(name for name in rows if name != baseline_name)
            delta = {"level": level, "model": "sato_minus_baseline", "source": path, "n": rows[sato_name]["n"]}
            cells = []
            for metric in METRICS:
                value = float(rows[sato_name][f"{metric}_mean"]) - float(rows[baseline_name][f"{metric}_mean"])
                delta[f"{metric}_mean"] = value
                delta[f"{metric}_std"] = ""
                cells.append(f"{value:+.4f}")
            combined.append(delta)
            markdown.append(f"| {level} | `SATO++ - baseline` | " + " | ".join(cells) + " |")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    fieldnames = ["level", "model", "source", "n"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
