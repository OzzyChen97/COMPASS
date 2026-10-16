#!/usr/bin/env python3
"""Strictly audit and aggregate MotionGPT shared-loader repeat evaluations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


METRICS = (
    "FID", "FID_P", "FID_D", "Matching", "R@1", "R@2", "R@3",
    "top1_pert", "top2_pert", "top3_pert", "matching_pert", "Diversity",
)
DATASETS = {"humanml3d": 4_640, "motionx322": 7_733}
EXPECTED_SAMPLING = {
    "mode": "categorical", "temperature": 1.0, "top_k": 0, "top_p": 1.0,
    "kwargs_mode": "explicit",
    "implementation": "explicit Hugging Face generate kwargs",
}
EXPECTED_GENERATION_LENGTH_MODE = "generated"
EXPECTED_EVALUATOR_LENGTH_MODE = "generated_length_capped_to_batch_seq"
EXPECTED_PAIR_RNG_MODE = "sequential"
EXPECTED_GENERATION_ORDER = "original-first"
EXPECTED_PROMPT_MODE = "official-default"
RETRIEVAL_METRIC_ABS_TOLERANCES = {
    # R-Precision is derived from integer hit counts and should be identical.
    "top1_pert": 1e-12,
    "top2_pert": 1e-12,
    "top3_pert": 1e-12,
    # Matching is a float32 distance reduction; independent NumPy/Torch
    # accumulation can differ by a few float32 ULPs after 4,380 samples.
    "matching_pert": 1e-6,
}


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


def retrieval_metric_failures(
    metric_errors: dict[str, float],
) -> dict[str, float]:
    """Keep count-derived R@ exact while allowing float32 reduction noise."""
    return {
        key: error for key, error in metric_errors.items()
        if error > RETRIEVAL_METRIC_ABS_TOLERANCES[key]
    }


def recompute_perturbed_retrieval(
    embedding_path: Path,
    expected_count: int,
    expected_batch_size: int,
) -> tuple[dict[str, float], dict[str, object]]:
    with np.load(embedding_path, allow_pickle=False) as dump:
        required = {
            "perturbed_text_embedding",
            "perturbed_motion_embedding",
            "batch_sizes",
            "loader_name",
            "seed",
            "sample_count",
            "batch_size",
            "sample_index_sha256",
            "protocol_fingerprint",
            "schema_version",
        }
        missing = sorted(required - set(dump.files))
        if missing:
            raise ValueError(
                f"retrieval embedding dump {embedding_path} lacks {missing}"
            )
        text = np.asarray(dump["perturbed_text_embedding"])
        motion = np.asarray(dump["perturbed_motion_embedding"])
        batch_sizes = np.asarray(dump["batch_sizes"], dtype=np.int64)
        loader_names = np.asarray(dump["loader_name"]).astype(str).tolist()
        metadata = {
            "schema_version": str(np.asarray(dump["schema_version"]).item()),
            "seed": int(np.asarray(dump["seed"]).item()),
            "sample_count": int(np.asarray(dump["sample_count"]).item()),
            "batch_size": int(np.asarray(dump["batch_size"]).item()),
            "sample_index_sha256": str(
                np.asarray(dump["sample_index_sha256"]).item()
            ),
            "protocol_fingerprint": str(
                np.asarray(dump["protocol_fingerprint"]).item()
            ),
            "name_order_sha256": canonical_sha256(loader_names),
        }

    if metadata["schema_version"] != "tm2t-perturbed-retrieval-embeddings/v1":
        raise ValueError("retrieval embedding dump schema mismatch")
    if text.ndim != 2 or motion.ndim != 2 or text.shape != motion.shape:
        raise ValueError("perturbed text/motion embedding shapes differ")
    if text.shape[0] != expected_count or len(loader_names) != expected_count:
        raise ValueError("retrieval embedding dump sample count mismatch")
    expected_batches = [expected_batch_size] * (
        expected_count // expected_batch_size
    )
    tail = expected_count % expected_batch_size
    if tail:
        expected_batches.append(tail)
    if batch_sizes.tolist() != expected_batches:
        raise ValueError(
            "retrieval embedding dump evaluator batch boundaries mismatch"
        )
    if (
        metadata["sample_count"] != expected_count
        or metadata["batch_size"] != expected_batch_size
    ):
        raise ValueError("retrieval embedding dump protocol counts mismatch")

    counts = np.zeros(3, dtype=np.int64)
    matching_sum = 0.0
    offset = 0
    for batch_size in batch_sizes.tolist():
        stop = offset + int(batch_size)
        text_batch = text[offset:stop]
        motion_batch = motion[offset:stop]
        squared = (
            -2 * np.dot(text_batch, motion_batch.T)
            + np.sum(np.square(text_batch), axis=1, keepdims=True)
            + np.sum(np.square(motion_batch), axis=1)
        )
        distances = np.sqrt(np.maximum(squared, 0.0))
        ranking = np.argsort(distances, axis=1)
        target = np.arange(int(batch_size))
        correct = np.zeros(int(batch_size), dtype=bool)
        for top_index in range(3):
            correct = correct | (ranking[:, top_index] == target)
            counts[top_index] += int(correct.sum())
        matching_sum += float(np.trace(distances))
        offset = stop
    if offset != expected_count:
        raise ValueError("retrieval embedding dump did not consume every sample")
    metrics = {
        "top1_pert": float(counts[0] / expected_count),
        "top2_pert": float(counts[1] / expected_count),
        "top3_pert": float(counts[2] / expected_count),
        "matching_pert": float(matching_sum / expected_count),
    }
    metadata.update({
        "batch_count": len(batch_sizes),
        "counts": counts.tolist(),
    })
    return metrics, metadata


def validate_generation_audit(audit: dict, samples: list[dict], seed: int) -> None:
    expected_count = len(samples)
    for role, text_key in (("original", "caption"), ("perturbed", "perturbed_caption")):
        rows = audit.get(role, [])
        if len(rows) != expected_count:
            raise ValueError(f"seed {seed} {role} generation audit is incomplete")
        for offset, (row, sample) in enumerate(zip(rows, samples)):
            if int(row.get("sample_offset", -1)) != offset:
                raise ValueError(f"seed {seed} {role} audit order mismatch at {offset}")
            if row.get("generation_role") != role:
                raise ValueError(f"seed {seed} {role} audit role mismatch at {offset}")
            for key in ("loader_name", "motion_id", "level"):
                if str(row.get(key)) != str(sample.get(key)):
                    raise ValueError(
                        f"seed {seed} {role} audit {key} mismatch at {offset}")
            expected_text = str(sample[text_key])
            if row.get("input_text") != expected_text:
                raise ValueError(f"seed {seed} {role} input text mismatch at {offset}")
            prompt = row.get("fulfilled_prompt")
            if not isinstance(prompt, str) or expected_text not in prompt:
                raise ValueError(f"seed {seed} {role} prompt mismatch at {offset}")
            tokens = row.get("motion_tokens")
            if not isinstance(tokens, list) or int(row.get("motion_token_count", -1)) != len(tokens):
                raise ValueError(f"seed {seed} {role} token audit mismatch at {offset}")
            for key in ("generated_frame_count", "evaluator_frame_count"):
                if int(row.get(key, 0)) < 1:
                    raise ValueError(f"seed {seed} {role} lacks {key} at {offset}")


def aggregate(
    run_dir: Path,
    dataset: str,
    expected_samples: int | None = None,
    require_caption_locked: bool = False,
    expected_unique_pairs: int | None = None,
    expected_levels: list[str] | None = None,
    source_manifest: Path | None = None,
    require_retrieval_embeddings: bool = False,
    start_seed: int = 123,
    num_repeats: int = 20,
) -> tuple[list[dict], dict]:
    if num_repeats < 1:
        raise ValueError("num_repeats must be positive")
    expected_seeds = list(range(start_seed, start_seed + num_repeats))
    if min(expected_seeds) < 123 or max(expected_seeds) > 142:
        raise ValueError("evaluation seeds must be an ordered subset of 123..142")
    expected_count = int(expected_samples or DATASETS[dataset])
    expected_retrieval_metadata = {
        "sample_count": expected_count,
        "batch_count": math.ceil(expected_count / 32),
        "batch_size": 32,
        "text_source": "resolved text_pair perturbation",
        "motion_source": (
            "motion generated from the same resolved text_pair perturbation"
        ),
        "evaluator": "standard TM2T TextEncoderBiGRUCo/MotionEncoderBiGRUCo",
        "word_vectorizer": "glove/our_vab",
        "spacy_model": "en_core_web_sm",
        "max_text_len": 20,
        "alphabetic_only": dataset != "motionx322",
        "caption_preprocessing": (
            "strip whitespace; remove hyphens; keep alphabetic tokens; "
            "lemmatize nouns/verbs except literal lowercase left"
            if dataset != "motionx322"
            else (
                "strip whitespace; remove hyphens; skip spaces/punctuation; "
                "lemmatize nouns/verbs except literal lowercase left"
            )
        ),
        "retrieval_scope": "within each evaluator-loader batch",
    }
    rows = []
    identities = {
        "checkpoint": set(),
        "evaluator": set(),
        "protocol": set(),
    }
    sample_index_hashes: set[str] = set()
    generation_replay_hashes: set[str] = set()
    generation_replay_count = 0
    generation_sources: set[str] = set()
    level_counts: Counter[str] | None = None
    retrieval_embedding_audits: list[dict[str, object]] = []
    for seed in expected_seeds:
        result_path = run_dir / f"seed{seed}.json"
        index_path = run_dir / "sample_indices" / f"seed{seed}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if result.get("dataset") != dataset or int(result.get("seed", -1)) != seed:
            raise ValueError(f"seed/dataset mismatch in {result_path}")
        if result.get("matched_count") != expected_count or result.get("total_count") != expected_count:
            raise ValueError(f"seed {seed} is not {expected_count}/{expected_count}")
        if result.get("unmatched_count") != 0 or result.get("runtime_fallback_count") != 0:
            raise ValueError(f"seed {seed} contains missing matches or runtime fallback")
        if index.get("count") != expected_count:
            raise ValueError(f"seed {seed} sample index count mismatch")
        samples = index.get("samples")
        if not isinstance(samples, list) or len(samples) != expected_count:
            raise ValueError(f"seed {seed} frozen sample index is incomplete")
        if require_caption_locked:
            required_levels = expected_levels or ["P1", "P2", "P3"]
            loader_protocol = result.get("protocol", {}).get("loader", {})
            if loader_protocol.get("caption_locked") is not True:
                raise ValueError(f"seed {seed} is not caption-locked")
            if loader_protocol.get("pair_match") != "exact normalized (motion_id, text_orig)":
                raise ValueError(f"seed {seed} does not use caption-aware pair matching")
            if result.get("caption_matched_count") != expected_count:
                raise ValueError(f"seed {seed} caption matching is incomplete")
            if result.get("caption_mismatch_count") != 0:
                raise ValueError(f"seed {seed} contains caption-pair mismatches")
            if loader_protocol.get("allowed_levels") != required_levels:
                raise ValueError(
                    f"seed {seed} does not use the required levels {required_levels}"
                )
            if any(
                sample.get("caption_pair_exact") is not True
                or " ".join(str(sample.get("caption", "")).strip().lower().split())
                != " ".join(str(sample.get("pair_text_orig", "")).strip().lower().split())
                for sample in samples
            ):
                raise ValueError(f"seed {seed} contains a caption/text_orig mismatch")
            pair_ids = [sample.get("pair_id") for sample in samples]
            if None in pair_ids:
                raise ValueError(f"seed {seed} contains an unresolved pair ID")
            required_unique = int(expected_unique_pairs or expected_count)
            if len(set(pair_ids)) != required_unique:
                raise ValueError(
                    f"seed {seed} resolved {len(set(pair_ids))} unique pair IDs; "
                    f"expected {required_unique}"
                )
            current_levels = Counter(str(sample.get("level")) for sample in samples)
            if set(current_levels) != set(required_levels):
                raise ValueError(
                    f"seed {seed} level set differs from {required_levels}"
                )
            if level_counts is None:
                level_counts = current_levels
            elif current_levels != level_counts:
                raise ValueError(f"seed {seed} level counts differ from the frozen index")
        if result.get("sample_index_sha256") != index.get("sample_index_sha256"):
            raise ValueError(f"seed {seed} sample index hash mismatch")
        if result["protocol"].get("sampling") != EXPECTED_SAMPLING:
            raise ValueError(f"seed {seed} sampling protocol mismatch")
        if result["protocol"].get("pair_rng_mode") != EXPECTED_PAIR_RNG_MODE:
            raise ValueError(f"seed {seed} pair RNG mode is not sequential")
        if result["protocol"].get("generation_order") != EXPECTED_GENERATION_ORDER:
            raise ValueError(f"seed {seed} generation order is not original-first")
        if result["protocol"].get("prompt", {}).get("mode") != EXPECTED_PROMPT_MODE:
            raise ValueError(f"seed {seed} prompt mode is not official-default")
        if result["protocol"].get("generation_length_mode") != EXPECTED_GENERATION_LENGTH_MODE:
            raise ValueError(f"seed {seed} generation length mode is not generated")
        if result["protocol"].get("evaluator_length_mode") != EXPECTED_EVALUATOR_LENGTH_MODE:
            raise ValueError(
                f"seed {seed} evaluator length mode is not "
                "generated_length_capped_to_batch_seq"
            )
        audit_path = result.get("generation_audit")
        if not audit_path or not Path(audit_path).is_file():
            raise ValueError(f"seed {seed} lacks a complete generation audit")
        audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        validate_generation_audit(audit, samples, seed)
        replay = result.get("generation_replay")
        if replay is None:
            generation_sources.add("live_generation")
        else:
            generation_sources.add("audited_token_replay")
            if not isinstance(replay, dict):
                raise ValueError(f"seed {seed} generation replay metadata is invalid")
            replay_path = Path(str(replay.get("path", ""))).resolve()
            if replay.get("mode") != "exact audited motion-token replay":
                raise ValueError(f"seed {seed} generation replay mode mismatch")
            if replay_path != Path(audit_path).resolve():
                raise ValueError(f"seed {seed} generation replay path mismatch")
            if int(replay.get("original_count", -1)) != expected_count:
                raise ValueError(f"seed {seed} generation replay original count mismatch")
            if int(replay.get("perturbed_count", -1)) != expected_count:
                raise ValueError(f"seed {seed} generation replay perturbed count mismatch")
            replay_sha256 = str(replay.get("sha256", ""))
            if replay_sha256 != file_sha256(replay_path):
                raise ValueError(f"seed {seed} generation replay checksum mismatch")
            generation_replay_hashes.add(replay_sha256)
            generation_replay_count += 1
        perturbed_retrieval = result.get("perturbed_text_retrieval")
        if not isinstance(perturbed_retrieval, dict):
            raise ValueError(f"seed {seed} lacks perturbed-text retrieval metadata")
        if perturbed_retrieval != expected_retrieval_metadata:
            raise ValueError(
                f"seed {seed} perturbed-text retrieval protocol mismatch"
            )
        metric_values = result.get("metrics")
        if not isinstance(metric_values, dict):
            raise ValueError(f"seed {seed} lacks metrics")
        missing_metrics = [metric for metric in METRICS if metric not in metric_values]
        if missing_metrics:
            raise ValueError(
                f"seed {seed} lacks metrics: {', '.join(missing_metrics)}"
            )
        if not all(math.isfinite(float(metric_values[metric])) for metric in METRICS):
            raise ValueError(f"seed {seed} contains a non-finite metric")
        retrieval_dump = result.get("retrieval_embedding_dump")
        if require_retrieval_embeddings and not isinstance(retrieval_dump, dict):
            raise ValueError(f"seed {seed} lacks a retrieval embedding dump")
        if isinstance(retrieval_dump, dict):
            embedding_path = Path(str(retrieval_dump.get("path", ""))).resolve()
            if not embedding_path.is_file():
                raise ValueError(
                    f"seed {seed} retrieval embedding dump does not exist"
                )
            if retrieval_dump.get("schema_version") != (
                "tm2t-perturbed-retrieval-embeddings/v1"
            ):
                raise ValueError(f"seed {seed} retrieval embedding schema mismatch")
            embedding_sha256 = file_sha256(embedding_path)
            if retrieval_dump.get("sha256") != embedding_sha256:
                raise ValueError(f"seed {seed} retrieval embedding checksum mismatch")
            recomputed, embedding_metadata = recompute_perturbed_retrieval(
                embedding_path, expected_count, 32
            )
            expected_embedding_metadata = {
                "seed": seed,
                "sample_count": expected_count,
                "batch_size": 32,
                "sample_index_sha256": result["sample_index_sha256"],
                "protocol_fingerprint": result["protocol_fingerprint"],
                "name_order_sha256": result["name_order_sha256"],
            }
            for key, expected_value in expected_embedding_metadata.items():
                if embedding_metadata[key] != expected_value:
                    raise ValueError(
                        f"seed {seed} retrieval embedding {key} mismatch"
                    )
            metric_errors = {
                key: abs(float(recomputed[key]) - float(metric_values[key]))
                for key in recomputed
            }
            metric_failures = retrieval_metric_failures(metric_errors)
            if metric_failures:
                raise ValueError(
                    f"seed {seed} retrieval embedding metric recomputation mismatch: "
                    f"{metric_errors}"
                )
            retrieval_embedding_audits.append({
                "seed": seed,
                "path": str(embedding_path),
                "sha256": embedding_sha256,
                "batch_count": embedding_metadata["batch_count"],
                "counts": embedding_metadata["counts"],
                "max_abs_error": max(metric_errors.values()),
            })
        identities["checkpoint"].add(result["artifacts"]["checkpoint"]["sha256"])
        identities["evaluator"].add(result["artifacts"]["evaluator_checkpoint"]["sha256"])
        identities["protocol"].add(result["protocol_fingerprint"])
        sample_index_hashes.add(result["sample_index_sha256"])
        rows.append({
            "seed": seed,
            **result["metrics"],
            "matched_count": expected_count,
            "total_count": expected_count,
            "sample_index_sha256": result["sample_index_sha256"],
        })
    if any(len(values) != 1 for values in identities.values()):
        raise ValueError("checkpoint, evaluator, or protocol differs across seeds")
    if len(generation_sources) != 1:
        raise ValueError("live generation and audited token replay are mixed across seeds")
    generation_source = next(iter(generation_sources))
    if (
        generation_source == "audited_token_replay"
        and generation_replay_count != len(rows)
    ):
        raise ValueError("not every seed has a verified generation replay audit")
    source_identity = None
    if source_manifest is not None:
        source_manifest = source_manifest.resolve()
        source = json.loads(source_manifest.read_text(encoding="utf-8"))
        source_seeds = source.get("protocol", {}).get("seeds")
        if not isinstance(source_seeds, list) or any(
            seed not in source_seeds for seed in expected_seeds
        ):
            raise ValueError(
                "source manifest does not contain every requested evaluation seed"
            )
        if source.get("protocol", {}).get("samples_per_seed") != expected_count:
            raise ValueError("source manifest sample-count contract mismatch")
        if source.get("model", {}).get("uses_t2mgpt_generator_checkpoint") is not False:
            raise ValueError("source manifest does not prove baseline-free provenance")
        expected_checkpoint = source.get("model", {}).get("checkpoint_sha256")
        actual_checkpoint = next(iter(identities["checkpoint"]))
        if not expected_checkpoint or actual_checkpoint != expected_checkpoint:
            raise ValueError(
                "evaluated checkpoint hash differs from validation-selected source manifest"
            )
        source_identity = {
            "path": str(source_manifest),
            "checkpoint_sha256": expected_checkpoint,
        }
    if require_caption_locked and len(sample_index_hashes) != 1:
        raise ValueError("caption-locked sample index differs across seeds")
    if require_retrieval_embeddings and len(retrieval_embedding_audits) != len(rows):
        raise ValueError("not every seed has an independently verified embedding dump")
    summary = {}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        mean = float(np.mean(values))
        sample_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        sample_variance = (
            float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
        )
        population_std = float(np.std(values, ddof=0))
        population_variance = float(np.var(values, ddof=0))
        ci95 = float(1.96 * population_std / math.sqrt(len(values)))
        summary[metric] = {
            "mean": mean,
            "std": population_std,
            "variance": population_variance,
            "population_std": population_std,
            "population_variance": population_variance,
            "sample_std": sample_std,
            "sample_variance": sample_variance,
            "ci95": ci95,
            "ci95_low": mean - ci95,
            "ci95_high": mean + ci95,
        }
    return rows, {
        "dataset": dataset,
        "n": len(rows),
        "seeds": expected_seeds,
        "samples_per_seed": expected_count,
        "std_ddof": 0,
        "variance_ddof": 0,
        "sample_std_ddof": 1,
        "sample_variance_ddof": 1,
        "ci95_definition": "1.96 * population_std / sqrt(n)",
        "ci95_std_ddof": 0,
        "perturbed_text_retrieval": expected_retrieval_metadata,
        "retrieval_embedding_audit": {
            "required": require_retrieval_embeddings,
            "verified_seed_count": len(retrieval_embedding_audits),
            "max_abs_metric_error": (
                max(
                    float(row["max_abs_error"])
                    for row in retrieval_embedding_audits
                )
                if retrieval_embedding_audits else None
            ),
            "seeds": retrieval_embedding_audits,
        },
        "generation_source": {
            "mode": generation_source,
            "audited_token_replay_count": generation_replay_count,
            "distinct_audit_checksum_count": len(generation_replay_hashes),
            "checksums_verified": generation_source == "audited_token_replay",
        },
        "sampling": EXPECTED_SAMPLING,
        "identities": {
            **{key: next(iter(value)) for key, value in identities.items()},
            "sample_index": (
                next(iter(sample_index_hashes))
                if len(sample_index_hashes) == 1 else None
            ),
        },
        "source_manifest": source_identity,
        "level_counts": None if level_counts is None else dict(sorted(level_counts.items())),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--require-caption-locked", action="store_true")
    parser.add_argument("--expected-unique-pairs", type=int)
    parser.add_argument("--expected-levels", nargs="+")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--start-seed", type=int, default=123)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument(
        "--require-retrieval-embeddings",
        action="store_true",
        help=(
            "require each seed's lightweight TM2T embedding NPZ and independently "
            "recompute matching_pert/top1_pert/top2_pert/top3_pert"
        ),
    )
    args = parser.parse_args()
    rows, summary = aggregate(
        args.run_dir,
        args.dataset,
        expected_samples=args.expected_samples,
        require_caption_locked=args.require_caption_locked,
        expected_unique_pairs=args.expected_unique_pairs,
        expected_levels=args.expected_levels,
        source_manifest=args.source_manifest,
        require_retrieval_embeddings=args.require_retrieval_embeddings,
        start_seed=args.start_seed,
        num_repeats=args.num_repeats,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["seed", *METRICS, "matched_count", "total_count", "sample_index_sha256"]
    with (args.output_dir / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
