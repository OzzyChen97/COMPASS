#!/usr/bin/env python3
"""Freeze and verify the Motion-X SATO++ seed-123 Table-3 parity run."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_MODEL = "motionx322_gemini31_sato100k"
EXPECTED_SEED = 123
EXPECTED_SAMPLES = 7_733
EXPECTED_PROTOCOL = {
    "cfg_scale": 1.3,
    "cfg_uncond": "zero",
    "sampling": {
        "mode": "categorical", "temperature": 1.0, "top_k": 0, "top_p": 1.0,
    },
}
METRICS = ("fid", "fid_p", "fid_d", "matching", "top1", "top2", "top3", "diversity")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("runs", payload.get("rows", []))


def one_row(path: Path, model: str = EXPECTED_MODEL, seed: int = EXPECTED_SEED) -> dict:
    rows = [row for row in load_rows(path) if row.get("model") == model and int(row["seed"]) == seed]
    if len(rows) != 1:
        raise ValueError(f"expected one {model} seed {seed} row in {path}, got {len(rows)}")
    return rows[0]


def prepare_manifest(args) -> dict:
    pair_rows = json.loads(args.pair_file.read_text(encoding="utf-8"))
    motion_ids = [str(row.get("motion_id", "")) for row in pair_rows]
    if len(pair_rows) != EXPECTED_SAMPLES or len(set(motion_ids)) != EXPECTED_SAMPLES:
        raise ValueError("pair file is not 7733 rows / 7733 unique motion ids")
    frozen = one_row(args.frozen_summary)
    if int(frozen["matched_pairs"]) != EXPECTED_SAMPLES or int(frozen["total_pairs"]) != EXPECTED_SAMPLES:
        raise ValueError("frozen seed-123 row is not 7733/7733")
    spec = json.loads(args.spec_json.read_text(encoding="utf-8"))
    entries = [row for row in spec if row.get("name") == EXPECTED_MODEL]
    if len(entries) != 1:
        raise ValueError(f"spec must contain exactly one {EXPECTED_MODEL} entry")
    checkpoint = Path(entries[0]["checkpoint"])
    evaluator_files = [
        args.evaluator_root / "Comp_v6_KLD005/opt.txt",
        args.evaluator_root / "Decomp_SP001_SM001_H512/meta/mean.npy",
        args.evaluator_root / "Decomp_SP001_SM001_H512/meta/std.npy",
        args.evaluator_root / "Decomp_SP001_SM001_H512/model/finest.tar",
        args.evaluator_root / "text_mot_match/model/finest.tar",
    ]
    return {
        "schema_version": "motionx322-sato-pp-seed123-parity/v1",
        "model": EXPECTED_MODEL,
        "seed": EXPECTED_SEED,
        "expected_samples": EXPECTED_SAMPLES,
        "matched_pairs_contract": "7733/7733",
        "protocol": EXPECTED_PROTOCOL,
        "pair_file": {
            **identity(args.pair_file),
            "unique_motion_ids": len(set(motion_ids)),
            "level_counts": dict(sorted(Counter(row.get("level") for row in pair_rows).items())),
        },
        "checkpoint": identity(checkpoint),
        "evaluator_files": [identity(path) for path in evaluator_files],
        "frozen_summary": identity(args.frozen_summary),
        "frozen_seed123_metrics": {key: float(frozen[key]) for key in METRICS},
    }


def verify_current(args, manifest: dict) -> dict:
    current = one_row(args.current_summary)
    if int(current["matched_pairs"]) != EXPECTED_SAMPLES or int(current["total_pairs"]) != EXPECTED_SAMPLES:
        raise ValueError("current parity row is not 7733/7733")
    index_path = args.current_out / EXPECTED_MODEL / "sample_indices" / "seed_123.json"
    sample_index = json.loads(index_path.read_text(encoding="utf-8"))
    if sample_index.get("sample_count") != EXPECTED_SAMPLES:
        raise ValueError("current sample index is not 7733 samples")
    samples = sample_index.get("samples", [])
    if len(samples) != EXPECTED_SAMPLES:
        raise ValueError("current sample-index record count is not 7733")
    loader_names = [str(row.get("loader_name", "")) for row in samples]
    motion_ids = [str(row.get("motion_id", "")) for row in samples]
    if len(set(loader_names)) != EXPECTED_SAMPLES or len(set(motion_ids)) != EXPECTED_SAMPLES:
        raise ValueError("current sample index does not contain 7733 unique motions")
    if any(value in {"", "<missing>"} for value in motion_ids):
        raise ValueError("current sample index contains an unmatched motion id")
    current_levels = Counter(str(row.get("level", "")) for row in samples)
    if dict(sorted(current_levels.items())) != manifest["pair_file"]["level_counts"]:
        raise ValueError("current sample-index P1/P2/P3 counts differ from frozen pairs")
    current_metrics = {key: float(current[key]) for key in METRICS}
    deltas = {
        key: current_metrics[key] - manifest["frozen_seed123_metrics"][key]
        for key in METRICS
    }
    failed = {key: value for key, value in deltas.items() if abs(value) > args.atol}
    result = {
        **manifest,
        "current_summary": identity(args.current_summary),
        "sample_index": identity(index_path),
        "sample_index_sha256": sample_index.get("sample_index_sha256"),
        "current_metrics": current_metrics,
        "metric_deltas": deltas,
        "absolute_tolerance": args.atol,
        "parity_passed": not failed,
    }
    if failed:
        raise ValueError(f"seed-123 parity exceeded tolerance: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--spec-json", type=Path, required=True)
    parser.add_argument("--frozen-summary", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, default=Path("checkpoints/motionx322"))
    parser.add_argument("--current-out", type=Path)
    parser.add_argument("--current-summary", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_manifest(args)
    if args.current_out is not None or args.current_summary is not None:
        if args.current_out is None or args.current_summary is None:
            parser.error("--current-out and --current-summary must be supplied together")
        manifest = verify_current(args, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
