#!/usr/bin/env python3
"""Audit frozen HumanML3D P1/P2/P3 evaluation assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


LEVELS = ("P1", "P2", "P3")


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def pair_signature(row: dict[str, Any]) -> tuple[str, str, str, str, object]:
    return (
        str(row["motion_id"]),
        normalized(row["text_orig"]),
        str(row["text_pair"]),
        str(row["level"]),
        row.get("pair_id"),
    )


def sample_signature(row: dict[str, Any]) -> tuple[str, str, str, str, object]:
    return (
        str(row["motion_id"]),
        normalized(row["caption"]),
        str(row["perturbed_caption"]),
        str(row["level"]),
        row.get("pair_id"),
    )


def audit_level_assets(
    pair_paths: dict[str, Path],
    index_paths: dict[str, Path],
    *,
    expected_count: int,
    batch_size: int = 32,
) -> dict[str, Any]:
    if set(pair_paths) != set(LEVELS) or set(index_paths) != set(LEVELS):
        raise ValueError("pair/index paths must contain exactly P1, P2, and P3")
    if expected_count < 3 or batch_size < 3:
        raise ValueError("R@3 requires expected_count and batch_size of at least three")

    source_orders: dict[str, list[tuple[str, str]]] = {}
    level_assets: dict[str, dict[str, Any]] = {}
    for level in LEVELS:
        pair_path = pair_paths[level].resolve()
        index_path = index_paths[level].resolve()
        pairs = json.loads(pair_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        samples = index.get("samples")
        if not isinstance(pairs, list) or not isinstance(samples, list):
            raise ValueError(f"{level} pair/index payload must contain JSON lists")
        if len(pairs) != expected_count or len(samples) != expected_count:
            raise ValueError(
                f"{level} count mismatch: pairs={len(pairs)}, "
                f"samples={len(samples)}, expected={expected_count}"
            )
        if int(index.get("count", -1)) != expected_count:
            raise ValueError(f"{level} index count field is not {expected_count}")
        computed_index_hash = canonical_sha256(samples)
        if index.get("sample_index_sha256") != computed_index_hash:
            raise ValueError(f"{level} sample-index hash mismatch")
        if Counter(pair_signature(row) for row in pairs) != Counter(
            sample_signature(row) for row in samples
        ):
            raise ValueError(f"{level} pair/index identities differ")
        if Counter(str(row.get("level")) for row in pairs) != {level: expected_count}:
            raise ValueError(f"{level} pair file contains another level")
        if Counter(str(row.get("level")) for row in samples) != {level: expected_count}:
            raise ValueError(f"{level} sample index contains another level")
        if any(str(row.get("relation")) != "preserve" for row in pairs):
            raise ValueError(f"{level} contains a non-preserving pair")
        if any(
            row.get("caption_pair_exact") is not True
            or str(row.get("loader_name")) != str(row.get("motion_id"))
            or normalized(row.get("caption")) != normalized(row.get("pair_text_orig"))
            or row.get("length_repair", {}).get("policy") != "repair-all"
            for row in samples
        ):
            raise ValueError(f"{level} sample index violates caption-locked repair-all")

        pair_ids = [row.get("pair_id") for row in samples]
        motion_ids = [str(row.get("motion_id")) for row in samples]
        if None in pair_ids or len(set(pair_ids)) != expected_count:
            raise ValueError(f"{level} pair IDs are missing or duplicated")
        if len(set(motion_ids)) != expected_count:
            raise ValueError(f"{level} motion IDs are not unique")

        source_order = [
            (str(row["motion_id"]), normalized(row["caption"])) for row in samples
        ]
        source_orders[level] = source_order
        level_assets[level] = {
            "pair_path": str(pair_path),
            "pair_file_sha256": file_sha256(pair_path),
            "pair_records_sha256": canonical_sha256(pairs),
            "index_path": str(index_path),
            "index_file_sha256": file_sha256(index_path),
            "sample_index_sha256": computed_index_hash,
            "protocol_fingerprint": str(index.get("protocol_fingerprint", "")),
            "count": expected_count,
            "unique_pair_ids": len(set(pair_ids)),
            "unique_motion_ids": len(set(motion_ids)),
            "source_order_sha256": canonical_sha256(source_order),
            "level_counts": {level: expected_count},
        }

    first_order = source_orders["P1"]
    if any(source_orders[level] != first_order for level in LEVELS[1:]):
        raise ValueError("P1/P2/P3 do not share the same motion/caption source order")

    tail = expected_count % batch_size
    batch_sizes = [batch_size] * (expected_count // batch_size)
    if tail:
        batch_sizes.append(tail)
    if batch_sizes[-1] < 3:
        raise ValueError("final evaluator batch is too small for R@3")

    manifest = {
        "schema_version": "humanml3d-table1-independent-level-assets/v1",
        "levels": list(LEVELS),
        "expected_samples_per_level": expected_count,
        "batch_size": batch_size,
        "batch_count": math.ceil(expected_count / batch_size),
        "batch_sizes": batch_sizes,
        "retrieval_scope": "within each independent level-specific evaluator batch",
        "caption_locked": True,
        "length_policy": "repair-all",
        "shuffle": False,
        "drop_last": False,
        "common_source_order_sha256": canonical_sha256(first_order),
        "assets": level_assets,
    }
    manifest["fingerprint"] = canonical_sha256(manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-prefix",
        type=Path,
        default=Path("dataset/HumanML3D/test_perb_table1_level"),
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path("ablation/humanml3d_table1_levels/index_audit"),
    )
    parser.add_argument("--expected-count", type=int, default=4380)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_paths = {
        level: Path(f"{args.pair_prefix}_{level.lower()}.json") for level in LEVELS
    }
    index_paths = {
        level: args.index_root / f"{level.lower()}_sample_index_seed123.json"
        for level in LEVELS
    }
    manifest = audit_level_assets(
        pair_paths,
        index_paths,
        expected_count=args.expected_count,
        batch_size=args.batch_size,
    )
    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()