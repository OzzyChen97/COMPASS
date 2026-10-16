#!/usr/bin/env python3
"""Audit strict exclusions and the explicit all-pairs repair policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def parse_tag(value: str) -> float:
    parsed = float(value)
    return 0.0 if np.isnan(parsed) else parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-file",
        type=Path,
        default=REPO_ROOT / "dataset/HumanML3D/test_perb_deterministic.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_pairs = json.loads(args.pair_file.read_text(encoding="utf-8"))
    pairs = [row for row in source_pairs if row.get("level") in {"P1", "P2", "P3"}]
    data_root = REPO_ROOT / "dataset/HumanML3D"

    records = []
    counts = Counter()
    level_counts = Counter()
    for pair in pairs:
        motion_id = str(pair["motion_id"])
        annotations = []
        for line in (data_root / "texts" / f"{motion_id}.txt").read_text(
                encoding="utf-8").splitlines():
            fields = line.split("#")
            if len(fields) >= 4 and normalize(fields[0]) == normalize(pair["text_orig"]):
                annotations.append({
                    "from_tag": parse_tag(fields[2]),
                    "to_tag": parse_tag(fields[3]),
                })
        if not annotations:
            raise RuntimeError(f"caption annotation missing for pair {pair.get('pair_id')}")
        annotation = next((row for row in annotations
                           if row["from_tag"] == 0.0 and row["to_tag"] == 0.0),
                          annotations[0])
        full_length = len(np.load(data_root / "new_joint_vecs" / f"{motion_id}.npy",
                                  mmap_mode="r"))
        if annotation["from_tag"] == 0.0 and annotation["to_tag"] == 0.0:
            start, end = 0, full_length
        else:
            start = max(0, min(int(annotation["from_tag"] * 20.0), full_length))
            end = max(0, min(int(annotation["to_tag"] * 20.0), full_length))
        original_start, original_end = start, end
        original_length = max(0, end - start)
        repaired_length = original_length
        repair_type = "none"
        if original_length == 0:
            start, end = 0, full_length
            repaired_length = full_length
            repair_type = "invalid-segment-use-full-motion"
        if repaired_length < 40:
            if full_length >= 40 and repaired_length > 0:
                center = (start + end) // 2
                start = max(0, min(center - 20, full_length - 40))
                end = start + 40
                repaired_length = 40
                repair_type = "expand-short-segment-to-minimum"
            else:
                repair_type = "keep-native-short"
        if repaired_length >= 200:
            repaired_length = 196
            repair_type = "resample-long-to-maximum"

        counts[repair_type] += 1
        level_counts[str(pair["level"])] += 1
        if repair_type != "none":
            records.append({
                "pair_id": pair.get("pair_id"),
                "motion_id": motion_id,
                "level": pair.get("level"),
                "caption": pair["text_orig"],
                "perturbed_caption": pair.get("text_pair"),
                "policy": "repair-all",
                "repair_type": repair_type,
                "original_start": original_start,
                "original_end": original_end,
                "original_length": original_length,
                "repaired_start": start,
                "repaired_end": end,
                "repaired_length": repaired_length,
                "full_motion_length": full_length,
            })

    report = {
        "schema_version": "caption-locked-length-repair-audit/v1",
        "pair_file": str(args.pair_file.resolve()),
        "source_pair_count": len(source_pairs),
        "p123_pair_count": len(pairs),
        "strict": {
            "input_pairs": len(pairs),
            "resolved_pairs": counts["none"],
            "skipped": {
                "motion_too_short": sum(
                    value for key, value in counts.items()
                    if key in {"keep-native-short", "expand-short-segment-to-minimum",
                               "invalid-segment-use-full-motion"}),
                "motion_too_long": counts["resample-long-to-maximum"],
            },
        },
        "repair_all": {
            "input_pairs": len(pairs),
            "resolved_pairs": sum(counts.values()),
            "repairs": dict(sorted(counts.items())),
            "skipped": {},
        },
        "repair_counts": dict(sorted(counts.items())),
        "level_counts": dict(sorted(level_counts.items())),
        "repaired_record_count": len(records),
        "repaired_records_sha256": canonical_hash(records),
        "repaired_records": records,
        "limitations": [
            "repair-all is a robustness protocol, not the official HumanML3D test protocol",
            "native short source motions are retained without temporal interpolation",
            "motions of at least 200 frames are linearly resampled to 196 frames",
        ],
    }
    if sum(counts.values()) != len(pairs):
        raise RuntimeError(
            f"repair-all resolved {sum(counts.values())}/{len(pairs)} pairs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "p123_pair_count", "strict", "repair_counts",
        "repaired_record_count", "repaired_records_sha256")}, indent=2))


if __name__ == "__main__":
    main()
