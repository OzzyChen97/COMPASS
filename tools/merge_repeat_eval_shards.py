#!/usr/bin/env python
"""Merge independently evaluated seed shards into standard Repeat20 outputs."""

import argparse
import hashlib
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablation import repeat20_perturb_fast_batch as repeat_eval


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recompute_retrieval_metrics(
    text: np.ndarray,
    motion: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, float], list[int]]:
    if text.ndim != 2 or motion.ndim != 2 or text.shape != motion.shape:
        raise ValueError("perturbed text/motion embedding shapes differ")
    counts = np.zeros(3, dtype=np.int64)
    matching_sum = 0.0
    batch_sizes = []
    for offset in range(0, len(text), batch_size):
        stop = min(offset + batch_size, len(text))
        current = stop - offset
        if current < 3:
            raise ValueError("retrieval batch is too small for R@3")
        batch_sizes.append(current)
        text_batch = text[offset:stop]
        motion_batch = motion[offset:stop]
        squared = (
            -2 * np.dot(text_batch, motion_batch.T)
            + np.sum(np.square(text_batch), axis=1, keepdims=True)
            + np.sum(np.square(motion_batch), axis=1)
        )
        distances = np.sqrt(np.maximum(squared, 0.0))
        ranking = np.argsort(distances, axis=1)
        target = np.arange(current)
        correct = np.zeros(current, dtype=bool)
        for top_index in range(3):
            correct = correct | (ranking[:, top_index] == target)
            counts[top_index] += int(correct.sum())
        matching_sum += float(np.trace(distances))
    return {
        "top1_pert": float(counts[0] / len(text)),
        "top2_pert": float(counts[1] / len(text)),
        "top3_pert": float(counts[2] / len(text)),
        "matching_pert": float(matching_sum / len(text)),
    }, batch_sizes


def audit_seed_diagnostics(
    diagnostic_path: Path,
    sample_index_path: Path,
    row: dict,
    expected_samples: int,
    batch_size: int,
    require_full_batches: bool,
) -> dict:
    if not diagnostic_path.is_file():
        raise ValueError(f"missing diagnostic archive: {diagnostic_path}")
    if not sample_index_path.is_file():
        raise ValueError(f"missing sample index: {sample_index_path}")
    sample_index = json.loads(sample_index_path.read_text(encoding="utf-8"))
    samples = sample_index.get("samples")
    if not isinstance(samples, list) or len(samples) != expected_samples:
        raise ValueError(f"sample index count mismatch: {sample_index_path}")
    computed_sample_hash = canonical_sha256(samples)
    recorded_sample_hash = str(sample_index.get("sample_index_sha256", ""))
    if (
        recorded_sample_hash != computed_sample_hash
        or str(row.get("sample_index_sha256", "")) != computed_sample_hash
    ):
        raise ValueError(f"sample index checksum mismatch: {sample_index_path}")

    with np.load(diagnostic_path, allow_pickle=False) as dump:
        required = {
            "perturbed_text_embedding",
            "perturbed_embedding",
            "loader_name",
            "motion_id",
            "caption",
            "perturbed_caption",
            "level",
        }
        missing = sorted(required - set(dump.files))
        if missing:
            raise ValueError(f"{diagnostic_path} lacks {missing}")
        text = np.asarray(dump["perturbed_text_embedding"])
        motion = np.asarray(dump["perturbed_embedding"])
        loader_names = np.asarray(dump["loader_name"]).astype(str).tolist()
        motion_ids = np.asarray(dump["motion_id"]).astype(str).tolist()
        captions = np.asarray(dump["caption"]).astype(str).tolist()
        perturbed_captions = np.asarray(
            dump["perturbed_caption"]
        ).astype(str).tolist()
        levels = np.asarray(dump["level"]).astype(str).tolist()

    arrays = (
        loader_names,
        motion_ids,
        captions,
        perturbed_captions,
        levels,
    )
    if any(len(values) != expected_samples for values in arrays):
        raise ValueError(f"diagnostic sample count mismatch: {diagnostic_path}")
    for offset, sample in enumerate(samples):
        expected = (
            str(sample["loader_name"]),
            str(sample["motion_id"]),
            str(sample["caption"]),
            str(sample["perturbed_caption"]),
            str(sample.get("level")),
        )
        actual = tuple(values[offset] for values in arrays)
        if actual != expected:
            raise ValueError(
                f"diagnostic/sample identity mismatch at seed {row['seed']} "
                f"offset {offset}"
            )

    recomputed, batch_sizes = recompute_retrieval_metrics(
        text, motion, batch_size
    )
    if require_full_batches and any(size != batch_size for size in batch_sizes):
        raise ValueError(f"seed {row['seed']} contains a retrieval tail batch")
    errors = {
        metric: abs(float(row[metric]) - value)
        for metric, value in recomputed.items()
    }
    if any(error > 1e-12 for error in errors.values()):
        raise ValueError(
            f"seed {row['seed']} embedding recomputation mismatch: {errors}"
        )
    return {
        "seed": int(row["seed"]),
        "diagnostic_path": str(diagnostic_path.resolve()),
        "diagnostic_sha256": file_sha256(diagnostic_path),
        "sample_index_sha256": computed_sample_hash,
        "sample_count": expected_samples,
        "batch_count": len(batch_sizes),
        "batch_sizes": batch_sizes,
        "top1_count": int(round(recomputed["top1_pert"] * expected_samples)),
        "top2_count": int(round(recomputed["top2_pert"] * expected_samples)),
        "top3_count": int(round(recomputed["top3_pert"] * expected_samples)),
        "max_abs_metric_error": max(errors.values()),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-json", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pair-data-path", required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=123)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument("--expected-matched-pairs", type=int, default=None)
    parser.add_argument("--expected-total-pairs", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=1.3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--diagnostics-root", type=Path)
    parser.add_argument(
        "--require-full-retrieval-batches", action="store_true"
    )
    parser.add_argument(
        "--strict-audit-output",
        type=Path,
        help="defaults to OUT_DIR/strict_audit.json when diagnostics are audited",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    specs = json.loads(args.spec_json.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    expected_matched_pairs = args.expected_matched_pairs
    expected_total_pairs = args.expected_total_pairs
    strict_audits = []

    for spec in specs:
        rows_by_seed = {}
        pattern = f"*/{spec['name']}/runs.csv"
        for csv_path in sorted(args.shard_root.glob(pattern)):
            for row in repeat_eval.read_rows(csv_path):
                seed = int(row["seed"])
                if seed in rows_by_seed:
                    raise ValueError(
                        f"Duplicate seed {seed} for {spec['name']}: {csv_path}"
                    )
                if expected_matched_pairs is not None:
                    if int(row["matched_pairs"]) != expected_matched_pairs:
                        raise ValueError(
                            f"matched_pairs mismatch for {spec['name']} seed {seed}: "
                            f"{row['matched_pairs']} != {expected_matched_pairs}"
                        )
                if expected_total_pairs is not None:
                    if int(row["total_pairs"]) != expected_total_pairs:
                        raise ValueError(
                            f"total_pairs mismatch for {spec['name']} seed {seed}: "
                            f"{row['total_pairs']} != {expected_total_pairs}"
                        )
                if int(row.get("matched_pairs", "0")) != int(row.get("total_pairs", "0")):
                    raise ValueError(
                        f"Matched pairs mismatch for {spec['name']} seed {seed}: "
                        f"{row['matched_pairs']}/{row['total_pairs']}"
                    )
                rows_by_seed[seed] = row

        expected = set(range(args.start_seed, args.start_seed + args.num_repeats))
        missing = sorted(expected - set(rows_by_seed))
        extra = sorted(set(rows_by_seed) - expected)
        if missing or extra:
            raise ValueError(
                f"Seed mismatch for {spec['name']}: missing={missing}, extra={extra}"
            )

        rows = []
        for run_idx, seed in enumerate(sorted(rows_by_seed), start=1):
            row = dict(rows_by_seed[seed])
            row["run_idx"] = run_idx
            rows.append(row)
        if args.diagnostics_root is not None:
            for row in rows:
                seed = int(row["seed"])
                diagnostic_matches = list(args.diagnostics_root.glob(
                    f"*/{spec['name']}/seed{seed}.npz"
                ))
                sample_index_matches = list(args.shard_root.glob(
                    f"*/{spec['name']}/sample_indices/seed_{seed}.json"
                ))
                if len(diagnostic_matches) != 1:
                    raise ValueError(
                        f"expected one diagnostic archive for seed {seed}, got "
                        f"{diagnostic_matches}"
                    )
                if len(sample_index_matches) != 1:
                    raise ValueError(
                        f"expected one sample index for seed {seed}, got "
                        f"{sample_index_matches}"
                    )
                expected_samples = int(
                    expected_total_pairs
                    if expected_total_pairs is not None
                    else row["total_pairs"]
                )
                strict_audits.append(audit_seed_diagnostics(
                    diagnostic_matches[0],
                    sample_index_matches[0],
                    row,
                    expected_samples,
                    args.batch_size,
                    args.require_full_retrieval_batches,
                ))
        repeat_eval.write_csv(
            args.out_dir / spec["name"] / "runs.csv", repeat_eval.RUN_FIELDS, rows
        )

    report_args = Namespace(
        out_dir=str(args.out_dir),
        pair_data_path=args.pair_data_path,
        dataset_split="test",
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        spacy_model="en_core_web_sm",
        start_seed=args.start_seed,
        num_repeats=args.num_repeats,
        fast_batch=True,
        caption_locked=False,
        caption_locked_length_policy="strict",
        expected_sample_index=None,
        humanml_fast_protocol=True,
        cfg_scale=args.cfg_scale,
        cfg_uncond="zero",
        sample_mode="categorical",
        sample_temperature=1.0,
        top_k=0,
        top_p=1.0,
    )
    rows, _, _ = repeat_eval.write_outputs(
        args.out_dir, args.report_md, specs, report_args
    )
    if args.diagnostics_root is not None:
        expected_audits = len(specs) * args.num_repeats
        if len(strict_audits) != expected_audits:
            raise ValueError(
                f"strict audit covered {len(strict_audits)}/{expected_audits} runs"
            )
        audit_path = args.strict_audit_output or (
            args.out_dir / "strict_audit.json"
        )
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps({
                "schema_version": "satopp-repeat-embedding-audit/v1",
                "model_count": len(specs),
                "seed_count": args.num_repeats,
                "sample_count_per_seed": expected_total_pairs,
                "batch_size": args.batch_size,
                "require_full_retrieval_batches": (
                    args.require_full_retrieval_batches
                ),
                "max_abs_metric_error": max(
                    float(item["max_abs_metric_error"])
                    for item in strict_audits
                ),
                "runs": strict_audits,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"Merged {len(rows)} runs into {args.out_dir}")


if __name__ == "__main__":
    main()
