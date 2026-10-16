#!/usr/bin/env python3
"""Build an all-pair caption-locked mix with SATO++'s exact level counts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def stable_key(seed: int, row: dict, replica: int) -> str:
    value = f"{seed}:{row['level']}:{row['pair_id']}:{row['motion_id']}:{replica}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--sato-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=20260810)
    parser.add_argument(
        "--allowed-levels",
        nargs="+",
        help="optional level allowlist, e.g. P1 P2 P3",
    )
    args = parser.parse_args()

    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
    index_payload = json.loads(args.sato_index.read_text(encoding="utf-8"))
    samples = index_payload["samples"]
    if args.allowed_levels:
        allowed = set(args.allowed_levels)
        pairs = [row for row in pairs if str(row.get("level")) in allowed]
        samples = [row for row in samples if str(row.get("level")) in allowed]
    target = Counter(str(row["level"]) for row in samples)
    base = Counter(str(row["level"]) for row in pairs)
    multiplicity = Counter(str(row["motion_id"]) for row in samples)
    by_motion = {str(row["motion_id"]): row for row in pairs}
    if len(by_motion) != len(pairs):
        raise RuntimeError("source pair definitions must have unique motion IDs")

    expanded = []
    for row in pairs:
        expanded.append({**row, "mix_replica": 0,
                         "sato_loader_multiplicity": multiplicity[row["motion_id"]]})

    selected_extra = Counter()
    for level in sorted(target):
        required = target[level] - base[level]
        if required < 0:
            raise RuntimeError(f"SATO++ target for {level} is below all-pair count")
        candidates = []
        for row in pairs:
            if str(row["level"]) != level:
                continue
            for replica in range(1, max(1, multiplicity[row["motion_id"]])):
                candidates.append((stable_key(args.selection_seed, row, replica), row, replica))
        candidates.sort(key=lambda item: item[0])
        if len(candidates) < required:
            raise RuntimeError(
                f"not enough SATO++ duplicate slots for {level}: {len(candidates)} < {required}")
        for _, row, replica in candidates[:required]:
            expanded.append({**row, "mix_replica": replica,
                             "sato_loader_multiplicity": multiplicity[row["motion_id"]]})
            selected_extra[level] += 1

    final_counts = Counter(str(row["level"]) for row in expanded)
    if final_counts != target or len(expanded) != len(samples):
        raise RuntimeError(
            f"mix mismatch: final={dict(final_counts)} target={dict(target)} "
            f"count={len(expanded)}/{len(samples)}")
    if {row["pair_id"] for row in expanded} != {row["pair_id"] for row in pairs}:
        raise RuntimeError("expanded mix does not retain every source pair definition")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(expanded, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    manifest = {
        "schema_version": "humanml3d-caption-locked-sato-mix/v1",
        "source_pairs": str(args.pairs.resolve()),
        "source_pair_count": len(pairs),
        "all_source_pairs_retained": True,
        "sato_reference_index": str(args.sato_index.resolve()),
        "sato_reference_index_hash": index_payload.get("sample_index_sha256"),
        "selection_seed": args.selection_seed,
        "allowed_levels": sorted(target),
        "target_level_counts": dict(sorted(target.items())),
        "base_level_counts": dict(sorted(base.items())),
        "selected_extra_counts": dict(sorted(selected_extra.items())),
        "output_count": len(expanded),
        "mix_rule": (
            "include every pair definition once, then fill only SATO++ duplicate "
            "slots within each level using a deterministic hash ranking"
        ),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
