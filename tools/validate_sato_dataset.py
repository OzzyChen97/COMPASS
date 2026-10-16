#!/usr/bin/env python
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a T2M-GPT/SATO++ dataset package.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-dim", type=int, required=True)
    parser.add_argument("--vq-name", default="VQVAE")
    parser.add_argument("--train-pairs", type=Path, default=None)
    parser.add_argument("--test-pairs", type=Path, default=None)
    parser.add_argument("--max-missing-fraction", type=float, default=0.05)
    parser.add_argument("--require-disjoint-train-val", action="store_true")
    parser.add_argument("--full-scan", action="store_true")
    return parser.parse_args()


def load_id_list(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_pairs(path):
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            return json.load(handle)
        return [json.loads(line) for line in handle if line.strip()]


def validate_pair_file(path, allowed_ids):
    rows = load_pairs(path)
    required = {"motion_id", "text_orig", "text_pair", "relation"}
    invalid_schema = sum(not required.issubset(row) for row in rows)
    invalid_relation = sum(row.get("relation") not in {"preserve", "alter"} for row in rows)
    outside_split = sum(str(row.get("motion_id")) not in allowed_ids for row in rows)
    empty_text = sum(not row.get("text_orig") or not row.get("text_pair") for row in rows)
    duplicates = len(rows) - len({
        (
            str(row.get("motion_id")),
            row.get("text_orig"),
            row.get("text_pair"),
            row.get("relation"),
            row.get("edit_type", ""),
        )
        for row in rows
    })
    stats = {
        "path": str(path),
        "rows": len(rows),
        "relations": dict(Counter(row.get("relation") for row in rows)),
        "unique_motions": len({str(row.get("motion_id")) for row in rows}),
        "invalid_schema": invalid_schema,
        "invalid_relation": invalid_relation,
        "outside_split": outside_split,
        "empty_text": empty_text,
        "duplicates": duplicates,
    }
    if any((invalid_schema, invalid_relation, outside_split, empty_text, duplicates)):
        raise RuntimeError(f"Invalid pair file: {json.dumps(stats, indent=2)}")
    return stats


def main():
    args = parse_args()
    root = args.dataset_root
    required = [
        root / "new_joint_vecs",
        root / "texts",
        root / "Mean.npy",
        root / "Std.npy",
        root / "train.txt",
        root / "val.txt",
        root / "test.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset artifacts: {missing}")

    split_lists = {
        name: load_id_list(root / f"{name}.txt") for name in ("train", "val", "test")
    }
    split_duplicates = {
        name: len(values) - len(set(values)) for name, values in split_lists.items()
    }
    if any(split_duplicates.values()):
        raise RuntimeError(f"Duplicate IDs within dataset splits: {split_duplicates}")
    split_ids = {name: set(values) for name, values in split_lists.items()}
    overlaps = {
        "train_val": len(split_ids["train"] & split_ids["val"]),
        "train_test": len(split_ids["train"] & split_ids["test"]),
        "val_test": len(split_ids["val"] & split_ids["test"]),
    }
    forbidden_overlap = overlaps["train_test"] or overlaps["val_test"]
    if args.require_disjoint_train_val:
        forbidden_overlap = forbidden_overlap or overlaps["train_val"]
    if forbidden_overlap:
        raise RuntimeError(f"Dataset split leakage detected: {overlaps}")

    all_ids = set().union(*split_ids.values())
    missing_motion = []
    missing_text = []
    invalid_motion = []
    invalid_text = []
    dimensions = Counter()
    lengths = Counter()
    for motion_id in sorted(all_ids):
        motion_path = root / "new_joint_vecs" / f"{motion_id}.npy"
        text_path = root / "texts" / f"{motion_id}.txt"
        if not motion_path.exists():
            missing_motion.append(motion_id)
        else:
            try:
                motion = np.load(motion_path, mmap_mode="r", allow_pickle=False)
                if motion.ndim != 2:
                    raise ValueError(f"shape={motion.shape}")
                dimensions[int(motion.shape[-1])] += 1
                lengths[int(motion.shape[0])] += 1
                if not 40 <= int(motion.shape[0]) <= 196:
                    raise ValueError(f"length={motion.shape[0]}")
                if args.full_scan and not np.isfinite(np.asarray(motion)).all():
                    raise ValueError("non-finite values")
            except Exception as exc:
                invalid_motion.append({"motion_id": motion_id, "error": repr(exc)})
        if not text_path.exists():
            missing_text.append(motion_id)
        else:
            lines = [line for line in text_path.read_text(
                encoding="utf-8", errors="strict"
            ).splitlines() if line.strip()]
            if not lines or any(len(line.split("#")) < 4 for line in lines):
                invalid_text.append(motion_id)

    if dimensions and set(dimensions) != {args.expected_dim}:
        raise RuntimeError(f"Unexpected motion dimensions: {dict(dimensions)}")
    if invalid_motion:
        raise RuntimeError(
            f"Invalid motion files: count={len(invalid_motion)}, examples={invalid_motion[:10]}"
        )
    if invalid_text:
        raise RuntimeError(
            f"Invalid text files: count={len(invalid_text)}, examples={invalid_text[:10]}"
        )
    missing_fraction = max(len(missing_motion), len(missing_text)) / len(all_ids) if all_ids else 0.0
    if missing_fraction > args.max_missing_fraction:
        raise RuntimeError(
            f"Too many missing motions/texts: motions={len(missing_motion)}, "
            f"texts={len(missing_text)}, fraction={missing_fraction:.4f}"
        )

    motion_files = {path.stem for path in (root / "new_joint_vecs").glob("*.npy")}
    text_files = {path.stem for path in (root / "texts").glob("*.txt")}
    orphan_motion = sorted(motion_files - all_ids)
    orphan_text = sorted(text_files - all_ids)
    if orphan_motion or orphan_text:
        raise RuntimeError(
            f"Orphan dataset files: motions={len(orphan_motion)}, texts={len(orphan_text)}"
        )

    mean = np.load(root / "Mean.npy", allow_pickle=False)
    std = np.load(root / "Std.npy", allow_pickle=False)
    if mean.shape != (args.expected_dim,) or std.shape != (args.expected_dim,):
        raise RuntimeError(f"Invalid Mean/Std shapes: mean={mean.shape}, std={std.shape}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise RuntimeError("Mean/Std contain non-finite or non-positive values")

    id_map_path = root / "motion_id_map.jsonl"
    id_map_stats = None
    if id_map_path.is_file():
        rows = [json.loads(line) for line in id_map_path.read_text().splitlines() if line.strip()]
        mapped_ids = [str(row.get("motion_id")) for row in rows]
        duplicate_map_ids = len(mapped_ids) - len(set(mapped_ids))
        outside_map = len(set(mapped_ids) - all_ids)
        missing_map = len(all_ids - set(mapped_ids))
        id_map_stats = {
            "rows": len(rows),
            "duplicate_ids": duplicate_map_ids,
            "outside_splits": outside_map,
            "missing_from_map": missing_map,
        }
        if duplicate_map_ids or outside_map or missing_map:
            raise RuntimeError(f"Invalid motion_id_map: {id_map_stats}")

    token_dir = root / args.vq_name
    token_coverage = None
    if token_dir.is_dir():
        min_motion_len = 24 if args.expected_dim == 251 else 40
        eligible_train_ids = []
        for motion_id in split_ids["train"]:
            motion_path = root / "new_joint_vecs" / f"{motion_id}.npy"
            if not motion_path.exists():
                continue
            motion_len = int(np.load(motion_path, mmap_mode="r").shape[0])
            if min_motion_len <= motion_len < 200:
                eligible_train_ids.append(motion_id)
        available = sum((token_dir / f"{motion_id}.npy").exists() for motion_id in eligible_train_ids)
        token_coverage = {
            "available": available,
            "expected": len(eligible_train_ids),
            "coverage": available / len(eligible_train_ids) if eligible_train_ids else 0.0,
        }
        if available != len(eligible_train_ids):
            raise RuntimeError(f"Incomplete token coverage: {token_coverage}")

    report = {
        "dataset_root": str(root.resolve()),
        "splits": {name: len(ids) for name, ids in split_ids.items()},
        "split_duplicates": split_duplicates,
        "overlaps": overlaps,
        "motion_dimensions": dict(dimensions),
        "motion_length_min": min(lengths) if lengths else None,
        "motion_length_max": max(lengths) if lengths else None,
        "missing_motion_files": len(missing_motion),
        "missing_text_files": len(missing_text),
        "invalid_motion_files": len(invalid_motion),
        "invalid_text_files": len(invalid_text),
        "orphan_motion_files": len(orphan_motion),
        "orphan_text_files": len(orphan_text),
        "missing_fraction": missing_fraction,
        "mean_finite": bool(np.isfinite(mean).all()),
        "std_finite": bool(np.isfinite(std).all()),
        "std_min": float(std.min()),
        "motion_id_map": id_map_stats,
        "token_coverage": token_coverage,
    }
    if args.train_pairs:
        report["train_pairs"] = validate_pair_file(args.train_pairs, split_ids["train"])
    if args.test_pairs:
        report["test_pairs"] = validate_pair_file(args.test_pairs, split_ids["test"])

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
