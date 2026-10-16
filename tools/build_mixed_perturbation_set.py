#!/usr/bin/env python
"""Build a one-pair-per-motion P1/P2/P3 mixture with fixed proportions."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


LEVELS = ("P1", "P2", "P3")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", type=Path, required=True)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--p3", type=Path, required=True)
    parser.add_argument("--weights", nargs=3, type=int, default=(1988, 1178, 1178))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def deterministic_key(motion_id, seed):
    return hashlib.sha256(f"{seed}:{motion_id}".encode("utf-8")).digest()


def main():
    args = parse_args()
    paths = {"P1": args.p1, "P2": args.p2, "P3": args.p3}
    rows = {level: json.loads(path.read_text(encoding="utf-8")) for level, path in paths.items()}
    by_level = {
        level: {str(row["motion_id"]): row for row in level_rows}
        for level, level_rows in rows.items()
    }
    motion_ids = list(by_level["P1"])
    expected = set(motion_ids)
    for level in LEVELS[1:]:
        if set(by_level[level]) != expected:
            raise ValueError(f"Motion IDs differ for {level}")

    motion_ids.sort(key=lambda motion_id: deterministic_key(motion_id, args.seed))
    total = len(motion_ids)
    weight_sum = sum(args.weights)
    counts = [round(total * weight / weight_sum) for weight in args.weights]
    counts[-1] += total - sum(counts)

    mixed = []
    offset = 0
    for level, count in zip(LEVELS, counts):
        selected = motion_ids[offset:offset + count]
        mixed.extend(dict(by_level[level][motion_id]) for motion_id in selected)
        offset += count
    mixed.sort(key=lambda row: deterministic_key(str(row["motion_id"]), args.seed + 1))
    for pair_id, row in enumerate(mixed, start=1):
        row["pair_id"] = pair_id

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mixed, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "weights": dict(zip(LEVELS, args.weights)),
        "counts": dict(Counter(row["level"] for row in mixed)),
        "total": len(mixed),
        "one_pair_per_motion": len({row["motion_id"] for row in mixed}) == len(mixed),
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
