#!/usr/bin/env python
"""Build a split-only Motion-X 322D subset capped by VQ token length."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


LINK_NAMES = ("new_joint_vecs", "texts", "VQVAE", "Mean.npy", "Std.npy")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-motion-tokens", type=int, default=75)
    parser.add_argument("--unit-length", type=int, default=4)
    return parser.parse_args()


def load_manifest(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def token_length(frames, unit_length):
    return (int(frames) + unit_length - 1) // unit_length


def replace_symlink(link, target):
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise FileExistsError(
            f"Refusing to replace non-symlink in generated subset: {link}")
    link.symlink_to(os.path.relpath(target.resolve(), link.parent.resolve()))


def build_subset(source, output, max_motion_tokens, unit_length):
    if max_motion_tokens < 1 or unit_length < 1:
        raise ValueError("Token cap and unit length must be positive")
    source = source.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(source / "manifest.jsonl")
    kept = [
        row for row in rows
        if token_length(row["frames"], unit_length) <= max_motion_tokens
    ]
    excluded = [
        row for row in rows
        if token_length(row["frames"], unit_length) > max_motion_tokens
    ]
    if not kept:
        raise RuntimeError("Token cap removed every sequence")

    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    for split in ("train", "val", "test"):
        ids = [row["motion_id"] for row in kept if row["split"] == split]
        (output / f"{split}.txt").write_text(
            "\n".join(ids) + "\n", encoding="utf-8")

    for name in LINK_NAMES:
        target = source / name
        if not target.exists():
            raise FileNotFoundError(target)
        replace_symlink(output / name, target)

    source_meta = json.loads(
        (source / "dataset_meta.json").read_text(encoding="utf-8"))
    lengths = np.asarray([int(row["frames"]) for row in kept], dtype=np.int64)
    kept_counts = Counter(row["split"] for row in kept)
    excluded_counts = Counter(row["split"] for row in excluded)
    metadata = dict(source_meta)
    metadata.update({
        "dataset": f"motionx322_{max_motion_tokens}",
        "source_dataset_root": str(source),
        "generation_subset": True,
        "max_motion_tokens": max_motion_tokens,
        "unit_length": unit_length,
        "max_frames": int(lengths.max()),
        "min_frames": int(lengths.min()),
        "median_frames": float(np.median(lengths)),
        "p95_frames": float(np.percentile(lengths, 95)),
        "selected_exact_intersection": len(kept),
        "split_counts": dict(kept_counts),
        "excluded_over_token_cap": len(excluded),
        "excluded_over_token_cap_by_split": dict(excluded_counts),
    })
    (output / "dataset_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    summary = {
        "source_count": len(rows),
        "kept_count": len(kept),
        "excluded_count": len(excluded),
        "excluded_percent": 100.0 * len(excluded) / len(rows),
        "kept_by_split": dict(kept_counts),
        "excluded_by_split": dict(excluded_counts),
        "max_frames": int(lengths.max()),
        "max_motion_tokens": max(
            token_length(row["frames"], unit_length) for row in kept),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    args = parse_args()
    build_subset(
        args.source, args.output, args.max_motion_tokens, args.unit_length)


if __name__ == "__main__":
    main()
