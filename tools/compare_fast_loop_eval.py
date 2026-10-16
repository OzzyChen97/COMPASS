#!/usr/bin/env python
"""Compare one-run exact-loop and fast-batch evaluation metrics."""

import argparse
import json
from pathlib import Path


METRICS = (
    "fid", "fid_p", "fid_d", "diversity_real", "diversity",
    "top1", "top2", "top3", "matching", "matched_pairs", "total_pairs",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-loop", type=Path, required=True)
    parser.add_argument("--baseline-fast", type=Path, required=True)
    parser.add_argument("--sato-loop", type=Path, required=True)
    parser.add_argument("--sato-fast", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load(path):
    rows = json.loads(path.read_text(encoding="utf-8"))["runs"]
    if len(rows) != 1:
        raise ValueError(f"Expected one run in {path}, found {len(rows)}")
    return rows[0]


def compare(name, loop_path, fast_path, tolerance):
    loop = load(loop_path)
    fast = load(fast_path)
    differences = {}
    for metric in METRICS:
        loop_value = float(loop[metric])
        fast_value = float(fast[metric])
        differences[metric] = {
            "loop": loop_value,
            "fast": fast_value,
            "delta": fast_value - loop_value,
            "abs_delta": abs(fast_value - loop_value),
        }
    return {
        "model": name,
        "loop_source": str(loop_path),
        "fast_source": str(fast_path),
        "differences": differences,
        "passed": all(item["abs_delta"] <= tolerance for item in differences.values()),
    }


def main():
    args = parse_args()
    results = [
        compare("baseline", args.baseline_loop, args.baseline_fast, args.tolerance),
        compare("sato80k", args.sato_loop, args.sato_fast, args.tolerance),
    ]
    output = {"tolerance": args.tolerance, "results": results}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Fast-Batch vs Legacy Loop Greedy Parity",
        "",
        f"Tolerance: `{args.tolerance}`",
        "",
        "| Model | Status | FID loop | FID fast | Delta FID | FID_P loop | FID_P fast | Delta FID_P |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        fid = result["differences"]["fid"]
        fid_p = result["differences"]["fid_p"]
        lines.append(
            f"| {result['model']} | {'PASS' if result['passed'] else 'FAIL'} | "
            f"{fid['loop']:.8f} | {fid['fast']:.8f} | {fid['delta']:+.8f} | "
            f"{fid_p['loop']:.8f} | {fid_p['fast']:.8f} | {fid_p['delta']:+.8f} |"
        )
    lines.append("")
    lines.append("The JSON file contains every metric and absolute difference.")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
