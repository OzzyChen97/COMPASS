#!/usr/bin/env python
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def load_by_motion(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["motion_id"]): row for row in rows}


def main():
    parser = argparse.ArgumentParser(
        description="Build a deterministic one-pair-per-motion P2/P3 mixture."
    )
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--p3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    by_level = {"P2": load_by_motion(args.p2), "P3": load_by_motion(args.p3)}
    motion_ids = sorted(set(by_level["P2"]) | set(by_level["P3"]))
    output = []
    counts = Counter()
    fallback = Counter()

    for motion_id in motion_ids:
        digest = hashlib.sha256(
            f"{args.seed}:{motion_id}:P2P3".encode("utf-8")
        ).digest()
        level = "P2" if digest[0] < 128 else "P3"
        row = by_level[level].get(motion_id)
        if row is None:
            other = "P3" if level == "P2" else "P2"
            row = by_level[other].get(motion_id)
            level = other
            fallback[level] += 1
        if row is None:
            continue
        output.append(row)
        counts[level] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "inputs": {"P2": str(args.p2), "P3": str(args.p3)},
                "total": len(output),
                "counts": dict(sorted(counts.items())),
                "fallback_counts": dict(sorted(fallback.items())),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(meta_path)


if __name__ == "__main__":
    main()
