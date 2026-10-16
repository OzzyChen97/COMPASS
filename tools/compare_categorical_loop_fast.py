#!/usr/bin/env python
"""Compare categorical legacy-loop and fast-batch single-seed results."""

import argparse
import json
from pathlib import Path


METRICS = ("fid", "fid_p", "fid_d", "matching", "top1", "top2", "top3")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", required=True, metavar="NAME=LOOP_JSON=FAST_JSON")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["runs"]
    if len(rows) != 1:
        raise ValueError(f"Expected one run in {path}, found {len(rows)}")
    return rows[0]


def main():
    args = parse_args()
    results = []
    for value in args.pair:
        name, loop_path, fast_path = value.split("=", 2)
        loop = load(loop_path)
        fast = load(fast_path)
        row = {"model": name, "loop_source": loop_path, "fast_source": fast_path}
        for metric in METRICS:
            row[f"{metric}_loop"] = float(loop[metric])
            row[f"{metric}_fast"] = float(fast[metric])
            row[f"{metric}_delta"] = float(fast[metric]) - float(loop[metric])
        results.append(row)

    output = {"delta_definition": "fast_minus_loop", "results": results}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Categorical Fast-Batch vs Legacy Loop",
        "",
        "Single seed 123 with corrected per-batch paired RNG. Delta is fast minus loop.",
        "",
        "| Model | FID loop | FID fast | Delta | FID_P loop | FID_P fast | Delta | FID_D loop | FID_D fast | Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['model']} | {row['fid_loop']:.4f} | {row['fid_fast']:.4f} | {row['fid_delta']:+.4f} | "
            f"{row['fid_p_loop']:.4f} | {row['fid_p_fast']:.4f} | {row['fid_p_delta']:+.4f} | "
            f"{row['fid_d_loop']:.4f} | {row['fid_d_fast']:.4f} | {row['fid_d_delta']:+.4f} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
