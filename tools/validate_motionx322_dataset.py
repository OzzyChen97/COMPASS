#!/usr/bin/env python
"""Validate native Motion-X 322D data and generated SATO++ text pairs."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


LEVELS = {"P1", "P2", "P3"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("dataset/Motion-X-322"))
    parser.add_argument("--train-pairs", type=Path)
    parser.add_argument("--train-raw", type=Path)
    parser.add_argument("--train-decisions", type=Path)
    parser.add_argument("--test-prefix", type=Path)
    parser.add_argument("--vq-name")
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--full-scan", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_ids(root):
    result = {}
    for split in ("train", "val", "test"):
        values = [
            line.strip()
            for line in (root / f"{split}.txt").read_text().splitlines()
            if line.strip()
        ]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate IDs in {split}.txt")
        result[split] = values
    if set(result["train"]) & set(result["val"]):
        raise ValueError("train/val split overlap")
    if set(result["train"]) & set(result["test"]):
        raise ValueError("train/test split overlap")
    if set(result["val"]) & set(result["test"]):
        raise ValueError("val/test split overlap")
    return result


def read_json_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            return json.load(handle)
        return [json.loads(line) for line in handle if line.strip()]


def validate_train_pairs(path, expected_ids):
    rows = read_json_rows(path)
    levels = defaultdict(set)
    pair_ids = set()
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, int) or pair_id in pair_ids:
            raise ValueError(f"Invalid or duplicate train pair_id: {pair_id!r}")
        pair_ids.add(pair_id)
        motion_id = str(row.get("motion_id", ""))
        if motion_id not in expected_ids:
            raise ValueError(f"Train pair outside train split: {motion_id}")
        relation = row.get("relation")
        if relation == "preserve":
            level = row.get("level")
            if level not in LEVELS:
                raise ValueError(f"Invalid preserving level: {level!r}")
            levels[motion_id].add(level)
        elif relation != "alter":
            raise ValueError(f"Invalid train pair relation: {relation!r}")
        if not row.get("text_orig") or not row.get("text_pair"):
            raise ValueError("Empty train pair text")
    complete = sum(levels[motion_id] == LEVELS for motion_id in expected_ids)
    covered = sum(bool(levels[motion_id]) for motion_id in expected_ids)
    return {
        "rows": len(rows),
        "covered_motions": covered,
        "complete_p1_p2_p3_motions": complete,
        "identity_fallback_motions": len(expected_ids) - covered,
    }


def validate_filter_decisions(raw_path, decisions_path, filtered_path):
    raw_rows = read_json_rows(raw_path)
    filtered_rows = read_json_rows(filtered_path)

    raw_ids = [row.get("pair_id") for row in raw_rows]
    if any(not isinstance(pair_id, int) for pair_id in raw_ids):
        raise ValueError("Raw train pairs contain invalid pair_id")
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("Raw train pairs contain duplicate pair_id")

    decisions = {}
    with decisions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = row.get("pair_id")
            passed = row.get("passed")
            if not isinstance(pair_id, int) or not isinstance(passed, bool):
                raise ValueError("Invalid filter decision row")
            if pair_id in decisions:
                raise ValueError(f"Duplicate filter decision: {pair_id}")
            decisions[pair_id] = passed

    if set(decisions) != set(raw_ids):
        missing = len(set(raw_ids) - set(decisions))
        extra = len(set(decisions) - set(raw_ids))
        raise ValueError(
            f"Filter decisions do not match raw pairs: missing={missing}, extra={extra}")

    filtered_ids = [row.get("pair_id") for row in filtered_rows]
    expected_filtered_ids = {
        pair_id for pair_id, passed in decisions.items() if passed
    }
    if len(filtered_ids) != len(set(filtered_ids)):
        raise ValueError("Filtered train pairs contain duplicate pair_id")
    if set(filtered_ids) != expected_filtered_ids:
        raise ValueError("Filtered rows do not exactly match passed decisions")

    return {
        "raw_rows": len(raw_rows),
        "decision_rows": len(decisions),
        "passed_rows": len(expected_filtered_ids),
        "rejected_rows": len(decisions) - len(expected_filtered_ids),
    }


def validate_test_levels(prefix, expected_ids):
    counts = {}
    for level in sorted(LEVELS):
        path = Path(f"{prefix}_{level.lower()}.json")
        rows = read_json_rows(path)
        ids = [str(row.get("motion_id", "")) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate motion IDs in {path}")
        if set(ids) != expected_ids:
            raise ValueError(
                f"{path} coverage mismatch: {len(ids)}/{len(expected_ids)}")
        if any(row.get("level") != level for row in rows):
            raise ValueError(f"Incorrect level annotation in {path}")
        if any(not row.get("text_orig") or not row.get("text_pair") for row in rows):
            raise ValueError(f"Empty text in {path}")
        counts[level] = len(rows)
    return counts


def main():
    args = parse_args()
    root = args.dataset_root
    metadata = json.loads((root / "dataset_meta.json").read_text())
    if metadata.get("representation") != "smplx322":
        raise ValueError("Expected native smplx322 representation")
    if metadata.get("dim") != 322 or metadata.get("fps") != 30:
        raise ValueError("Expected native 322D motions at 30 FPS")
    if metadata.get("motions_unchanged") is not True:
        raise ValueError("Dataset metadata does not guarantee unchanged motions")

    splits = split_ids(root)
    expected_counts = metadata.get("split_counts", {})
    actual_counts = {name: len(values) for name, values in splits.items()}
    if actual_counts != expected_counts:
        raise ValueError(
            f"Split counts differ from metadata: {actual_counts} != {expected_counts}")

    manifest = {}
    with (root / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["source_path"].lower().startswith("humanml/"):
                raise ValueError(f"HumanML overlap retained: {row['source_path']}")
            manifest[row["motion_id"]] = row
    all_ids = set().union(*(set(values) for values in splits.values()))
    if set(manifest) != all_ids:
        raise ValueError("Manifest and split IDs differ")

    scan_ids = sorted(all_ids) if args.full_scan else sorted(all_ids)[:128]
    for motion_id in scan_ids:
        path = root / "new_joint_vecs" / f"{motion_id}.npy"
        motion = np.load(path, mmap_mode="r", allow_pickle=False)
        if motion.ndim != 2 or motion.shape[1] != 322:
            raise ValueError(f"Invalid motion shape {motion_id}: {motion.shape}")
        if motion.dtype != np.float32:
            raise ValueError(f"Invalid motion dtype {motion_id}: {motion.dtype}")
        if motion.shape[0] != manifest[motion_id]["frames"]:
            raise ValueError(f"Motion length differs from manifest: {motion_id}")
        if args.full_scan and sha256(path) != manifest[motion_id]["sha256"]:
            raise ValueError(f"Raw byte hash mismatch: {motion_id}")
        if not (root / "texts" / f"{motion_id}.txt").is_file():
            raise FileNotFoundError(f"Missing text: {motion_id}")

    report = {
        "dataset_root": str(root.resolve()),
        "split_counts": actual_counts,
        "motions_scanned": len(scan_ids),
        "full_scan": args.full_scan,
        "train_pair_rows": None,
        "train_filter_decisions": None,
        "test_level_counts": None,
    }
    if args.train_pairs:
        report["train_pair_rows"] = validate_train_pairs(
            args.train_pairs, set(splits["train"]))
    if args.train_raw or args.train_decisions:
        if not (args.train_pairs and args.train_raw and args.train_decisions):
            raise ValueError(
                "--train-pairs, --train-raw and --train-decisions are required together")
        report["train_filter_decisions"] = validate_filter_decisions(
            args.train_raw, args.train_decisions, args.train_pairs)
    if args.test_prefix:
        report["test_level_counts"] = validate_test_levels(
            args.test_prefix, set(splits["test"]))
    if args.vq_name:
        token_dir = root / args.vq_name
        token_lengths = []
        for motion_id in splits["train"]:
            path = token_dir / f"{motion_id}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing motion tokens: {motion_id}")
            tokens = np.load(path, mmap_mode="r", allow_pickle=False)
            if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
                raise ValueError(f"Invalid token shape {motion_id}: {tokens.shape}")
            if int(tokens.min()) < 0 or int(tokens.max()) >= args.codebook_size:
                raise ValueError(f"Token outside codebook range: {motion_id}")
            token_lengths.append(int(tokens.shape[1]))
        report["motion_tokens"] = {
            "directory": str(token_dir.resolve()),
            "count": len(token_lengths),
            "min_length": min(token_lengths),
            "max_length": max(token_lengths),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
