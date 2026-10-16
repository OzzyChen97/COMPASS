#!/usr/bin/env python
"""Select one deterministic P1/P2/P3 pair per motion from a raw pair pool."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


LEVELS = ("P1", "P2", "P3")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", nargs=3, type=float, default=(0.46, 0.27, 0.27))
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def digest(seed, *parts):
    payload = ":".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def main():
    args = parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    by_motion = defaultdict(lambda: defaultdict(list))
    for row in rows:
        level = row.get("level")
        if level in LEVELS and row.get("relation") == "preserve":
            by_motion[str(row["motion_id"])][level].append(row)

    complete_ids = sorted(
        motion_id
        for motion_id, candidates in by_motion.items()
        if all(candidates[level] for level in LEVELS)
    )
    excluded = sorted(set(by_motion) - set(complete_ids))
    complete_ids.sort(key=lambda motion_id: digest(args.seed, "level", motion_id))

    weight_sum = sum(args.weights)
    if weight_sum <= 0:
        raise ValueError("--weights must have a positive sum")
    counts = [round(len(complete_ids) * weight / weight_sum) for weight in args.weights]
    counts[-1] += len(complete_ids) - sum(counts)

    selected = []
    offset = 0
    for level, count in zip(LEVELS, counts):
        for motion_id in complete_ids[offset:offset + count]:
            candidates = by_motion[motion_id][level]
            candidates = sorted(
                candidates,
                key=lambda row: digest(
                    args.seed,
                    "candidate",
                    motion_id,
                    level,
                    row.get("text_orig", ""),
                    row.get("text_pair", ""),
                ),
            )
            selected.append(dict(candidates[0]))
        offset += count

    selected.sort(key=lambda row: digest(args.seed, "output", row["motion_id"]))
    for pair_id, row in enumerate(selected, start=1):
        row["pair_id"] = pair_id

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "weights": dict(zip(LEVELS, args.weights)),
        "input_pairs": len(rows),
        "input_motions": len(by_motion),
        "complete_motions": len(complete_ids),
        "excluded_incomplete_motions": len(excluded),
        "excluded_motion_ids": excluded,
        "counts": dict(Counter(row["level"] for row in selected)),
        "one_pair_per_motion": len({row["motion_id"] for row in selected}) == len(selected),
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
