#!/usr/bin/env python
"""Freeze SATO++ validation selection and formal-test rules before evaluation."""

import argparse
import hashlib
import json
from pathlib import Path


FORMULA = (
    "mean((metric-min_across_candidates)/(max_across_candidates-"
    "min_across_candidates)) for metric in [FID,FID_P,FID_D]"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--validation-pairs", type=Path, required=True)
    parser.add_argument("--test-pairs", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--validation-shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=(123, 124, 125))
    parser.add_argument("--formal-start-seed", type=int, default=123)
    parser.add_argument("--formal-repeats", type=int, default=20)
    return parser.parse_args()


def sha256_file(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_pair_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            return json.load(handle)
        return [json.loads(line) for line in handle if line.strip()]


def main():
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen protocol: {output}")
    if args.validation_shard_root.exists() and any(
        args.validation_shard_root.rglob("runs.csv")
    ):
        raise RuntimeError("Validation metrics already exist; cannot freeze post hoc")

    candidate_manifest = json.loads(
        args.candidate_manifest.read_text(encoding="utf-8")
    )
    val_rows = load_pair_rows(args.validation_pairs)
    test_rows = load_pair_rows(args.test_pairs)
    val_ids = [str(row["motion_id"]) for row in val_rows]
    test_ids = [str(row["motion_id"]) for row in test_rows]
    if len(set(val_ids)) != len(val_ids):
        raise ValueError("Validation pair file must contain one pair per motion")
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("Formal test pair file must contain one pair per motion")
    if set(val_ids) & set(test_ids):
        raise ValueError("Validation and formal-test pair motion IDs overlap")

    dataset_root = args.dataset_root.resolve()
    split_ids = {
        split: [
            line.strip()
            for line in (dataset_root / f"{split}.txt").read_text().splitlines()
            if line.strip()
        ]
        for split in ("train", "val", "test")
    }
    if not set(val_ids) <= set(split_ids["val"]):
        raise ValueError("Validation pairs contain IDs outside the validation split")
    if set(val_ids) & set(split_ids["test"]):
        raise ValueError("Validation pairs overlap the dataset test split")
    if set(test_ids) != set(split_ids["test"]):
        raise ValueError(
            "Formal test pair file must cover the full generation-subset test split"
        )

    validation_meta_path = args.validation_pairs.with_suffix(
        args.validation_pairs.suffix + ".meta.json"
    )
    validation_meta = (
        json.loads(validation_meta_path.read_text(encoding="utf-8"))
        if validation_meta_path.is_file()
        else None
    )
    if not validation_meta or validation_meta.get("test_pair_files_used") != []:
        raise ValueError("Validation-pair metadata does not prove test-pair isolation")

    repo_root = Path(__file__).resolve().parents[1]
    code_paths = [
        repo_root / "ablation/repeat20_perturb_fast_batch.py",
        repo_root / "dataset/dataset_TM_eval.py",
        repo_root / "utils/sato_pp_eval.py",
        repo_root / "models/evaluator_wrapper.py",
    ]
    payload = {
        "schema_version": 1,
        "selection_status": "frozen_before_validation_metrics",
        "selection_type": "post_training_independent_validation_selection",
        "candidate_manifest": str(args.candidate_manifest.resolve()),
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
        "candidate_ids": [
            row["candidate_id"] for row in candidate_manifest["candidates"]
        ],
        "candidate_count": len(candidate_manifest["candidates"]),
        "checkpoint_selection_rule": {
            "metrics": ["FID", "FID_P", "FID_D"],
            "direction": "lower_is_better",
            "aggregation_over_validation_seeds": "arithmetic_mean_per_metric",
            "normalization": "candidate-wise min-max per metric",
            "formula": FORMULA,
            "weights": {"FID": 1 / 3, "FID_P": 1 / 3, "FID_D": 1 / 3},
            "zero_range_value": 0.0,
            "tie_break": ["lower_validation_fid", "earlier_iter", "candidate_id"],
        },
        "validation": {
            "dataset_root": str(dataset_root),
            "dataset_name": "motionx322_75",
            "split": "val",
            "pair_path": str(args.validation_pairs.resolve()),
            "pair_file_sha256": sha256_file(args.validation_pairs),
            "pair_rows_sha256": canonical_sha256(val_rows),
            "sample_count": len(val_rows),
            "sample_motion_ids_sha256": canonical_sha256(val_ids),
            "pair_metadata": validation_meta,
            "seeds": list(args.validation_seeds),
            "cfg_scale": 1.3,
            "cfg_uncond": "zero",
            "sampling": {
                "mode": "categorical",
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "pair_rng_order": "sequential_original_then_perturbed",
            },
            "loader": {
                "caption_locked": True,
                "caption_locked_length_policy": "strict",
                "shuffle": False,
                "drop_last": False,
                "batch_size": 32,
                "num_workers": 0,
            },
        },
        "formal_test": {
            "dataset_root": str(dataset_root),
            "dataset_name": "motionx322_75",
            "split": "test",
            "protocol": "Motion-X-322 Table 3",
            "pair_path": str(args.test_pairs.resolve()),
            "pair_file_sha256": sha256_file(args.test_pairs),
            "pair_rows_sha256": canonical_sha256(test_rows),
            "sample_count": len(test_rows),
            "sample_motion_ids_sha256": canonical_sha256(test_ids),
            "seeds": list(
                range(
                    args.formal_start_seed,
                    args.formal_start_seed + args.formal_repeats,
                )
            ),
            "cfg_scale": 1.3,
            "cfg_uncond": "zero",
            "sampling": {
                "mode": "categorical",
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
            },
            "loader": {
                "humanml_fast_protocol": True,
                "batch_size": 32,
                "num_workers": 8,
            },
            "checkpoint_evaluations_after_selection": 1,
        },
        "dataset_split_counts": {
            split: len(values) for split, values in split_ids.items()
        },
        "dataset_split_id_sha256": {
            split: canonical_sha256(values) for split, values in split_ids.items()
        },
        "dataset_split_intersections": {
            "train_val": len(set(split_ids["train"]) & set(split_ids["val"])),
            "train_test": len(set(split_ids["train"]) & set(split_ids["test"])),
            "val_test": len(set(split_ids["val"]) & set(split_ids["test"])),
        },
        "code_sha256": {
            str(path.relative_to(repo_root)): sha256_file(path) for path in code_paths
        },
        "test_metrics_used_for_selection": False,
    }
    if any(payload["dataset_split_intersections"].values()):
        raise ValueError("Dataset splits overlap")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()