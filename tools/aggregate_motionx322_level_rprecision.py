#!/usr/bin/env python3
"""Strictly audit and aggregate independent Motion-X-322 P1/P2/P3 R-Precision."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_motionx322_level_rprecision_assets import (  # noqa: E402
    audit_assets as audit_frozen_assets,
)
from tools.aggregate_motiongpt_loader_eval import (  # noqa: E402
    aggregate as aggregate_motiongpt,
    canonical_sha256,
    file_sha256 as _uncached_file_sha256,
    retrieval_metric_failures,
)


LEVELS = ("P1", "P2", "P3")
MODEL_ORDER = ("t2mgpt", "satopp", "motiongpt", "motiongpt_satopp")
MODEL_LABELS = {
    "t2mgpt": "T2M-GPT",
    "motiongpt": "MotionGPT",
    "satopp": "SATO++ (T2M-GPT backbone)",
    "motiongpt_satopp": "MotionGPT + SATO++ plug-in",
}
T2M_INTERNAL_MODELS = {
    "t2mgpt": "motionx322_t2mgpt_baseline_200k_best",
    "satopp": "motionx322_gemini31_sato100k",
}
METRICS = ("top1_pert", "top2_pert", "top3_pert")
EXPECTED_SAMPLES = 7_679
BATCH_SIZE = 32
T2M_PYTHON = Path("/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python")
MOTIONGPT_PYTHON = Path("/mnt/workspace/Wilson/miniconda3/envs/mgpt/bin/python")
_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def file_sha256(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    digest = _FILE_HASH_CACHE.get(key)
    if digest is None:
        digest = _uncached_file_sha256(resolved)
        _FILE_HASH_CACHE[key] = digest
    return digest


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_cli_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def finite_monotonic(metrics: dict[str, float], identity: str) -> None:
    values = [float(metrics[name]) for name in METRICS]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{identity} contains a non-finite R-Precision value")
    if not (0.0 <= values[0] <= values[1] <= values[2] <= 1.0):
        raise ValueError(f"{identity} violates 0 <= R@1 <= R@2 <= R@3 <= 1")


def expected_batch_sizes(expected_samples: int) -> list[int]:
    sizes = [BATCH_SIZE] * (expected_samples // BATCH_SIZE)
    tail = expected_samples % BATCH_SIZE
    if tail:
        sizes.append(tail)
    return sizes


def validate_asset_manifest(
    asset_manifest: dict[str, Any],
    *,
    source_root: Path,
    implementation_overrides_json: Path | None = None,
) -> None:
    if asset_manifest.get("schema_version") != (
        "motionx322-independent-level-rprecision-assets/v1"
    ):
        raise ValueError("Motion-X asset manifest schema mismatch")
    recorded_fingerprint = str(asset_manifest.get("fingerprint", ""))
    computed_fingerprint = canonical_sha256({
        key: value for key, value in asset_manifest.items() if key != "fingerprint"
    })
    if recorded_fingerprint != computed_fingerprint:
        raise ValueError("Motion-X asset manifest fingerprint mismatch")
    if (
        asset_manifest.get("dataset") != "motionx322_75"
        or asset_manifest.get("levels") != list(LEVELS)
        or int(asset_manifest.get("expected_samples_per_level", -1))
        != EXPECTED_SAMPLES
        or int(asset_manifest.get("excluded_incomplete_levels", -1)) != 54
        or asset_manifest.get("caption_locked") is not True
        or asset_manifest.get("length_policy") != "repair-all"
        or asset_manifest.get("shuffle") is not False
        or asset_manifest.get("drop_last") is not False
        or int(asset_manifest.get("batch_size", -1)) != BATCH_SIZE
        or asset_manifest.get("batch_sizes")
        != expected_batch_sizes(EXPECTED_SAMPLES)
    ):
        raise ValueError("Motion-X asset manifest protocol contract mismatch")

    pair_path = Path(
        asset_manifest["pairs_and_indices"]["P1"]["pair_path"]
    ).resolve()
    suffix = "_p1.json"
    if not str(pair_path).endswith(suffix):
        raise ValueError("cannot derive Motion-X level pair prefix")
    pair_prefix = Path(str(pair_path)[: -len(suffix)])
    index_root = Path(
        asset_manifest["pairs_and_indices"]["P1"]["index_path"]
    ).resolve().parent
    implementation_overrides = None
    if implementation_overrides_json is not None:
        import json as _json

        implementation_overrides = _json.loads(
            implementation_overrides_json.read_text(encoding="utf-8")
        )
    fresh = audit_frozen_assets(
        pair_prefix=pair_prefix,
        index_root=index_root,
        source_root=source_root.resolve(),
        levels_meta_path=Path(asset_manifest["levels_metadata"]["path"]),
        dataset_meta_path=Path(asset_manifest["dataset_metadata"]["path"]),
        t2m_spec_path=Path(asset_manifest["t2m_family_spec"]["path"]),
        expected_count=EXPECTED_SAMPLES,
        batch_size=BATCH_SIZE,
        implementation_overrides=implementation_overrides,
    )
    if fresh != asset_manifest:
        raise ValueError(
            "frozen Motion-X assets changed after assets_manifest.json was written"
        )


def evaluator_identity(asset_manifest: dict[str, Any]) -> dict[str, Any]:
    matches = [
        artifact for artifact in asset_manifest.get("evaluator_files", [])
        if str(artifact.get("path", "")).endswith(
            "/checkpoints/motionx322/text_mot_match/model/finest.tar"
        )
    ]
    if len(matches) != 1:
        raise ValueError("asset manifest does not identify one Motion-X TM2T evaluator")
    artifact = matches[0]
    path = Path(str(artifact["path"])).resolve()
    if not path.is_file() or file_sha256(path) != artifact.get("sha256"):
        raise ValueError(f"Motion-X evaluator checksum mismatch: {path}")
    return {
        "path": str(path),
        "size": int(artifact["size"]),
        "sha256": str(artifact["sha256"]),
    }


def validate_t2m_generation_audit(
    path: Path,
    sample_index_path: Path,
    *,
    seed: int,
    expected_samples: int,
    expected_level: str,
    expected_sample_index_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    index = json.loads(sample_index_path.read_text(encoding="utf-8"))
    samples = index.get("samples")
    if payload.get("schema_version") != "t2mgpt-satopp-generation-audit/v1":
        raise ValueError(f"generation audit schema mismatch: {path}")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError(f"generation audit seed mismatch: {path}")
    if int(payload.get("sample_count", -1)) != expected_samples:
        raise ValueError(f"generation audit count mismatch: {path}")
    if not isinstance(samples, list) or len(samples) != expected_samples:
        raise ValueError(f"sample index count mismatch: {sample_index_path}")
    if (
        canonical_sha256(samples) != expected_sample_index_sha256
        or index.get("sample_index_sha256") != expected_sample_index_sha256
        or int(index.get("sample_count", -1)) != expected_samples
        or int(index.get("seed", -1)) != seed
        or {str(sample.get("level")) for sample in samples} != {expected_level}
    ):
        raise ValueError(f"sample index identity mismatch: {sample_index_path}")
    for role, text_key in (
        ("original", "caption"),
        ("perturbed", "perturbed_caption"),
    ):
        rows = payload.get(role)
        if not isinstance(rows, list) or len(rows) != expected_samples:
            raise ValueError(f"{role} generation audit is incomplete: {path}")
        for offset, (row, sample) in enumerate(zip(rows, samples)):
            if int(row.get("sample_offset", -1)) != offset:
                raise ValueError(f"{role} generation audit order mismatch at {offset}")
            if row.get("generation_role") != role:
                raise ValueError(f"{role} generation audit role mismatch at {offset}")
            for key in ("loader_name", "motion_id", "level"):
                if str(row.get(key)) != str(sample.get(key)):
                    raise ValueError(
                        f"{role} generation audit {key} mismatch at {offset}"
                    )
            if str(row.get("input_text")) != str(sample.get(text_key)):
                raise ValueError(f"{role} generation text mismatch at {offset}")
            tokens = row.get("motion_tokens")
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(not isinstance(token, int) for token in tokens)
                or int(row.get("motion_token_count", -1)) != len(tokens)
            ):
                raise ValueError(f"{role} generation token audit mismatch at {offset}")
            if int(row.get("reference_frame_count", -1)) != int(
                sample.get("motion_length", -2)
            ):
                raise ValueError(f"{role} reference length mismatch at {offset}")
            for key in ("generated_frame_count", "evaluator_frame_count"):
                if int(row.get(key, 0)) < 1:
                    raise ValueError(f"{role} generation audit lacks {key} at {offset}")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "seed": seed,
        "sample_count": expected_samples,
        "original_tokens_sha256": canonical_sha256(
            [row["motion_tokens"] for row in payload["original"]]
        ),
        "perturbed_tokens_sha256": canonical_sha256(
            [row["motion_tokens"] for row in payload["perturbed"]]
        ),
    }


def audit_t2m_seed_embeddings(
    embedding_path: Path,
    sample_index_path: Path,
    row: dict[str, Any],
    *,
    expected_samples: int,
    expected_level: str,
    expected_sample_index_sha256: str,
    expected_protocol_fingerprint: str,
) -> dict[str, Any]:
    sample_index = json.loads(sample_index_path.read_text(encoding="utf-8"))
    samples = sample_index.get("samples")
    if not isinstance(samples, list) or len(samples) != expected_samples:
        raise ValueError(f"sample index count mismatch: {sample_index_path}")
    sample_hash = canonical_sha256(samples)
    if (
        sample_hash != expected_sample_index_sha256
        or sample_index.get("sample_index_sha256") != sample_hash
        or row.get("sample_index_sha256") != sample_hash
        or sample_index.get("model") != row.get("model")
        or int(sample_index.get("seed", -1)) != int(row["seed"])
        or int(sample_index.get("sample_count", -1)) != expected_samples
        or {str(sample.get("level")) for sample in samples} != {expected_level}
    ):
        raise ValueError(f"sample index identity mismatch: {sample_index_path}")

    with np.load(embedding_path, allow_pickle=False) as dump:
        required = {
            "perturbed_text_embedding", "perturbed_motion_embedding",
            "batch_sizes", "loader_name", "motion_id", "caption",
            "perturbed_caption", "level", "seed", "sample_count",
            "batch_size", "sample_index_sha256", "protocol_fingerprint",
            "schema_version",
        }
        missing = sorted(required - set(dump.files))
        if missing:
            raise ValueError(f"{embedding_path} lacks {missing}")
        text = np.asarray(dump["perturbed_text_embedding"])
        motion = np.asarray(dump["perturbed_motion_embedding"])
        batch_sizes = np.asarray(dump["batch_sizes"], dtype=np.int64).tolist()
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
        }
        identity_arrays = (
            np.asarray(dump["loader_name"]).astype(str).tolist(),
            np.asarray(dump["motion_id"]).astype(str).tolist(),
            np.asarray(dump["caption"]).astype(str).tolist(),
            np.asarray(dump["perturbed_caption"]).astype(str).tolist(),
            np.asarray(dump["level"]).astype(str).tolist(),
        )
    expected_sizes = expected_batch_sizes(expected_samples)
    expected_metadata = {
        "schema_version": "tm2t-perturbed-retrieval-embeddings/v1",
        "seed": int(row["seed"]),
        "sample_count": expected_samples,
        "batch_size": BATCH_SIZE,
        "sample_index_sha256": sample_hash,
        "protocol_fingerprint": expected_protocol_fingerprint,
    }
    if metadata != expected_metadata:
        raise ValueError(
            f"retrieval embedding metadata mismatch: {metadata!r} != "
            f"{expected_metadata!r}"
        )
    if batch_sizes != expected_sizes:
        raise ValueError(
            f"retrieval embedding batch boundaries mismatch: {embedding_path}"
        )
    if text.ndim != 2 or motion.ndim != 2 or text.shape != motion.shape:
        raise ValueError(f"embedding shape mismatch: {embedding_path}")
    if text.shape[1] < 1 or not np.isfinite(text).all() or not np.isfinite(motion).all():
        raise ValueError(f"embedding dump contains invalid values: {embedding_path}")
    if text.shape[0] != expected_samples or any(
        len(values) != expected_samples for values in identity_arrays
    ):
        raise ValueError(f"diagnostic sample count mismatch: {embedding_path}")
    for offset, sample in enumerate(samples):
        expected = (
            str(sample["loader_name"]),
            str(sample["motion_id"]),
            str(sample["caption"]),
            str(sample["perturbed_caption"]),
            str(sample["level"]),
        )
        if tuple(values[offset] for values in identity_arrays) != expected:
            raise ValueError(
                f"diagnostic/sample identity mismatch at seed {row['seed']} "
                f"offset {offset}"
            )

    counts = np.zeros(3, dtype=np.int64)
    matching_sum = 0.0
    sizes = expected_sizes
    offset = 0
    for size in sizes:
        stop = offset + size
        text_batch = text[offset:stop]
        motion_batch = motion[offset:stop]
        squared = (
            -2 * np.dot(text_batch, motion_batch.T)
            + np.sum(np.square(text_batch), axis=1, keepdims=True)
            + np.sum(np.square(motion_batch), axis=1)
        )
        distances = np.sqrt(np.maximum(squared, 0.0))
        ranking = np.argsort(distances, axis=1)
        targets = np.arange(size)
        correct = np.zeros(size, dtype=bool)
        for top_index in range(3):
            correct |= ranking[:, top_index] == targets
            counts[top_index] += int(correct.sum())
        matching_sum += float(np.trace(distances))
        offset = stop
    if offset != expected_samples:
        raise ValueError(f"embedding batches did not consume every sample: {embedding_path}")
    recomputed = {
        "top1_pert": float(counts[0] / expected_samples),
        "top2_pert": float(counts[1] / expected_samples),
        "top3_pert": float(counts[2] / expected_samples),
        "matching_pert": float(matching_sum / expected_samples),
    }
    errors = {
        metric: abs(float(row[metric]) - value)
        for metric, value in recomputed.items()
    }
    if retrieval_metric_failures(errors):
        raise ValueError(
            f"seed {row['seed']} embedding recomputation mismatch: {errors}"
        )
    return {
        "seed": int(row["seed"]),
        "embedding_path": str(embedding_path.resolve()),
        "embedding_sha256": file_sha256(embedding_path),
        "sample_index_sha256": sample_hash,
        "sample_count": expected_samples,
        "batch_count": len(sizes),
        "batch_sizes": sizes,
        "counts": counts.tolist(),
        "max_abs_metric_error": max(errors.values()),
    }


def t2m_expected_artifacts(
    asset_manifest: dict[str, Any], model_key: str
) -> tuple[list[dict[str, Any]], Path, str]:
    model = asset_manifest["t2m_family_spec"]["models"][
        T2M_INTERNAL_MODELS[model_key]
    ]
    artifacts = model.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"asset manifest lacks T2M-family artifacts for {model_key}")
    expected_primary_role = "transformer" if model_key == "t2mgpt" else "satopp_bundle"
    primary = None
    verified: list[dict[str, Any]] = []
    for artifact in artifacts:
        path = Path(str(artifact["path"])).resolve()
        expected_hash = str(artifact["sha256"])
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"{model_key} artifact checksum mismatch: {path}")
        verified.append({
            "role": str(artifact["role"]),
            "path": str(path),
            "size": int(artifact["size"]),
            "sha256": expected_hash,
        })
        if artifact["role"] == expected_primary_role:
            primary = (path, expected_hash)
    if primary is None:
        raise ValueError(f"{model_key} lacks primary {expected_primary_role} artifact")
    return verified, primary[0], primary[1]


def validate_t2m_command(
    command_path: Path,
    *,
    group_root: Path,
    model_key: str,
    level: str,
    seeds: list[int],
    asset_manifest: dict[str, Any],
    para_ruler_evaluator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    command_text = command_path.read_text(encoding="utf-8").strip()
    tokens = shlex.split(command_text)
    if len(tokens) < 3 or re.fullmatch(
        r"CUDA_VISIBLE_DEVICES=\d+", tokens[0]
    ) is None:
        raise ValueError(f"malformed T2M-family command record: {command_path}")
    if resolve_cli_path(tokens[1]) != T2M_PYTHON.resolve():
        raise ValueError(f"{model_key}/{level} used the wrong Python environment")
    if resolve_cli_path(tokens[2]) != (
        REPO_ROOT / "ablation/repeat20_perturb_fast_batch.py"
    ).resolve():
        raise ValueError(f"{model_key}/{level} used the wrong evaluation driver")

    def option(name: str) -> str:
        positions = [index for index, token in enumerate(tokens) if token == name]
        if len(positions) != 1 or positions[0] + 1 >= len(tokens):
            raise ValueError(
                f"{command_path} must contain exactly one value for {name}"
            )
        return tokens[positions[0] + 1]

    internal = T2M_INTERNAL_MODELS[model_key]
    expected_pair = Path(
        asset_manifest["pairs_and_indices"][level]["pair_path"]
    ).resolve()
    expected_index = Path(
        asset_manifest["pairs_and_indices"][level]["index_path"]
    ).resolve()
    expected_spec = Path(asset_manifest["t2m_family_spec"]["path"]).resolve()
    path_options = {
        "--spec-json": expected_spec,
        "--pair-data-path": expected_pair,
        "--expected-sample-index": expected_index,
        "--out-dir": group_root.resolve(),
        "--report-md": (group_root / "report.md").resolve(),
        "--retrieval-embedding-dir": (
            group_root / "retrieval_embeddings"
        ).resolve(),
        "--generation-audit-dir": (group_root / "generation_audits").resolve(),
    }
    for flag, expected in path_options.items():
        actual = resolve_cli_path(option(flag))
        if actual != expected:
            raise ValueError(
                f"{model_key}/{level} command {flag} mismatch: {actual} != {expected}"
            )
    scalar_options = {
        "--allowed-levels": level,
        "--model": internal,
        "--start-seed": str(seeds[0]),
        "--num-repeats": str(len(seeds)),
        "--batch-size": str(BATCH_SIZE),
        "--num-workers": "0",
        "--dataset-split": "test",
        "--spacy-model": "en_core_web_sm",
        "--cfg-scale": "1.3",
        "--cfg-uncond": "zero",
        "--caption-locked-length-policy": "repair-all",
        "--sample-mode": "categorical",
        "--sample-temperature": "1",
        "--top-k": "0",
        "--top-p": "1",
        "--sampling-pair-rng-mode": "sequential",
        "--wandb-mode": "disabled",
    }
    if para_ruler_evaluator is not None:
        scalar_options["--evaluator-checkpoint"] = str(
            para_ruler_evaluator["path"]
        )
    for flag, expected in scalar_options.items():
        actual = option(flag)
        if actual != expected:
            raise ValueError(
                f"{model_key}/{level} command {flag} mismatch: {actual!r} != {expected!r}"
            )
    required_flags = {
        "--fast-batch",
        "--caption-locked",
    }
    missing_flags = sorted(
        flag for flag in required_flags if tokens.count(flag) != 1
    )
    if missing_flags:
        raise ValueError(
            f"{model_key}/{level} command must contain each required flag once: "
            f"{missing_flags}"
        )
    allowed_flags = set(path_options) | set(scalar_options) | required_flags
    unknown_flags = sorted({
        token for token in tokens[3:]
        if token.startswith("--") and token not in allowed_flags
    })
    if unknown_flags:
        raise ValueError(
            f"{model_key}/{level} command contains unsupported flags "
            f"{unknown_flags}"
        )
    return {
        "path": str(command_path.resolve()),
        "sha256": file_sha256(command_path),
        "command": tokens,
    }


def validate_t2m_commands(
    group_root: Path,
    *,
    model_key: str,
    level: str,
    seeds: list[int],
    asset_manifest: dict[str, Any],
    para_ruler_evaluator: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    command_paths = sorted(group_root.glob("command_seed*_*.txt"))
    if not command_paths:
        raise ValueError(f"{model_key}/{level} has no command records")
    audited = []
    covered_seeds: set[int] = set()
    for command_path in command_paths:
        match = re.fullmatch(r"command_seed(\d+)_(\d+)\.txt", command_path.name)
        if match is None:
            raise ValueError(f"malformed T2M-family command filename: {command_path}")
        start_seed, end_seed = (int(value) for value in match.groups())
        if start_seed < 123 or end_seed > 142 or start_seed > end_seed:
            raise ValueError(f"invalid T2M-family command seed range: {command_path}")
        command_seeds = list(range(start_seed, end_seed + 1))
        audited.append(validate_t2m_command(
            command_path,
            group_root=group_root,
            model_key=model_key,
            level=level,
            seeds=command_seeds,
            asset_manifest=asset_manifest,
            para_ruler_evaluator=para_ruler_evaluator,
        ))
        covered_seeds.update(command_seeds)
    missing = sorted(set(seeds) - covered_seeds)
    if missing:
        raise ValueError(
            f"{model_key}/{level} command records do not cover seeds {missing}"
        )
    return audited


def collect_t2m_family(
    run_root: Path,
    asset_manifest: dict[str, Any],
    level: str,
    model_key: str,
    seeds: list[int],
    expected_samples: int,
    expected_index_hash: str,
    para_ruler_evaluator: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    internal = T2M_INTERNAL_MODELS[model_key]
    group_root = run_root / "t2m_family" / level.lower() / model_key
    model_root = group_root / internal
    rows_path = model_root / "runs.csv"
    raw_rows = read_csv(rows_path)
    raw_seed_values = [int(row["seed"]) for row in raw_rows]
    if len(raw_seed_values) != len(set(raw_seed_values)):
        raise ValueError(f"{model_key}/{level} contains duplicate seed rows")
    rows_by_seed = {
        int(row["seed"]): row for row in raw_rows
        if int(row["seed"]) in seeds
    }
    if set(rows_by_seed) != set(seeds):
        raise ValueError(
            f"{model_key}/{level} lacks seeds {sorted(set(seeds) - set(rows_by_seed))}"
        )

    protocol = json.loads(
        (group_root / "protocol_manifest.json").read_text(encoding="utf-8")
    )
    expected_index_path = Path(
        asset_manifest["pairs_and_indices"][level]["index_path"]
    ).resolve()
    expected_protocol = {
        "schema_version": 1,
        "sample_count": expected_samples,
        "sample_index_path": str(expected_index_path),
        "sample_index_sha256": expected_index_hash,
        "dataset_split": "test",
        "level_counts": {level: expected_samples},
        "caption_locked": True,
        "length_policy": "repair-all",
        "cfg_scale": 1.3,
        "sampling": {
            "mode": "categorical", "temperature": 1.0, "top_k": 0, "top_p": 1.0,
        },
        "rng": "sequential",
        "generation_order": "original-first",
        "evaluator_length": "generated_length",
        "evaluator_motion_tensor_frames": 300,
        "perturbed_text_retrieval": {
            "sample_count": expected_samples,
            "batch_count": math.ceil(expected_samples / BATCH_SIZE),
            "batch_size": BATCH_SIZE,
            "text_source": "resolved text_pair perturbation",
            "motion_source": (
                "motion generated from the same resolved text_pair perturbation"
            ),
            "evaluator": (
                str(para_ruler_evaluator["path"])
                if para_ruler_evaluator
                else (
                    "standard TM2T TextEncoderBiGRUCo/MotionEncoderBiGRUCo"
                )
            ),
            **(
                {"evaluator_checkpoint_sha256": para_ruler_evaluator["sha256"]}
                if para_ruler_evaluator
                else {}
            ),
            "word_vectorizer": "glove/our_vab",
            "spacy_model": "en_core_web_sm",
            "max_text_len": 20,
            "alphabetic_only": False,
            "caption_preprocessing": (
                "strip whitespace; remove hyphens; skip spaces/punctuation; "
                "lemmatize nouns/verbs except literal lowercase left"
            ),
            "retrieval_scope": "within each evaluator-loader batch",
        },
        "metric_scope": {
            "top1/top2/top3/matching": (
                "original caption TM2T embedding vs original-caption generation"
            ),
            "top1_pert/top2_pert/top3_pert/matching_pert": (
                "perturbed caption TM2T embedding vs perturbed-caption generation"
            ),
        },
        "models": [internal],
    }
    actual_protocol = {
        key: value for key, value in protocol.items() if key != "fingerprint"
    }
    if actual_protocol != expected_protocol:
        raise ValueError(f"{model_key}/{level} protocol contract mismatch")
    expected_protocol_fingerprint = canonical_sha256(expected_protocol)
    if protocol.get("fingerprint") != expected_protocol_fingerprint:
        raise ValueError(f"{model_key}/{level} protocol fingerprint mismatch")

    expected_artifacts, expected_checkpoint_path, expected_checkpoint_hash = (
        t2m_expected_artifacts(asset_manifest, model_key)
    )
    expected_evaluator = (
        para_ruler_evaluator if para_ruler_evaluator
        else evaluator_identity(asset_manifest)
    )
    command_audits = validate_t2m_commands(
        group_root,
        model_key=model_key,
        level=level,
        seeds=seeds,
        asset_manifest=asset_manifest,
        para_ruler_evaluator=para_ruler_evaluator,
    )
    collected = []
    embedding_audits = []
    generation_audits = []
    expected_kind = asset_manifest["t2m_family_spec"]["models"][internal]["kind"]
    expected_iter = 100_000 if model_key == "satopp" else 0
    for seed in seeds:
        row = rows_by_seed[seed]
        identity = f"{model_key}/{level}/seed{seed}"
        row_identity = {
            "model": row.get("model"),
            "kind": row.get("kind"),
            "run_idx": int(row.get("run_idx", -1)),
            "seed": int(row.get("seed", -1)),
            "checkpoint": row.get("checkpoint"),
            "iter": int(row.get("iter", -1)),
        }
        expected_row_identity = {
            "model": internal,
            "kind": expected_kind,
            "run_idx": seed - 122,
            "seed": seed,
            "checkpoint": expected_checkpoint_path.name,
            "iter": expected_iter,
        }
        if row_identity != expected_row_identity:
            raise ValueError(
                f"{identity} run identity mismatch: "
                f"{row_identity!r} != {expected_row_identity!r}"
            )
        for key in ("matched_pairs", "total_pairs", "perturbed_retrieval_samples"):
            if int(row[key]) != expected_samples:
                raise ValueError(f"{identity} {key} is not {expected_samples}")
        if row.get("sample_index_sha256") != expected_index_hash:
            raise ValueError(f"{identity} frozen-index hash mismatch")
        row_checkpoint_path = Path(str(row["path"])).resolve()
        if row_checkpoint_path != expected_checkpoint_path:
            raise ValueError(f"{identity} checkpoint path mismatch")
        if file_sha256(row_checkpoint_path) != expected_checkpoint_hash:
            raise ValueError(f"{identity} checkpoint checksum mismatch")
        metrics = {name: float(row[name]) for name in METRICS}
        finite_monotonic(metrics, identity)
        sample_index_path = model_root / "sample_indices" / f"seed_{seed}.json"
        embedding_audit = audit_t2m_seed_embeddings(
            group_root / "retrieval_embeddings" / internal / f"seed{seed}.npz",
            sample_index_path,
            row,
            expected_samples=expected_samples,
            expected_level=level,
            expected_sample_index_sha256=expected_index_hash,
            expected_protocol_fingerprint=expected_protocol_fingerprint,
        )
        generation_audit = validate_t2m_generation_audit(
            group_root / "generation_audits" / internal / f"seed{seed}.json",
            sample_index_path,
            seed=seed,
            expected_samples=expected_samples,
            expected_level=level,
            expected_sample_index_sha256=expected_index_hash,
        )
        embedding_audits.append(embedding_audit)
        generation_audits.append(generation_audit)
        collected.append({
            "model": model_key,
            "method": MODEL_LABELS[model_key],
            "level": level,
            "seed": seed,
            "samples": expected_samples,
            **metrics,
            "sample_index_sha256": expected_index_hash,
            "checkpoint_sha256": expected_checkpoint_hash,
            "evaluator_sha256": expected_evaluator["sha256"],
            "online_offline_max_abs_error": float(
                embedding_audit["max_abs_metric_error"]
            ),
        })
    return collected, {
        "protocol_fingerprint": protocol.get("fingerprint"),
        "checkpoint_sha256": expected_checkpoint_hash,
        "checkpoint_artifacts": expected_artifacts,
        "evaluator": expected_evaluator,
        "commands": command_audits,
        "verified_seed_count": len(embedding_audits),
        "max_abs_metric_error": max(
            float(audit["max_abs_metric_error"]) for audit in embedding_audits
        ),
        "embedding_audits": embedding_audits,
        "generation_audits": generation_audits,
    }


def validate_motiongpt_seed_sidecars(
    *,
    run_dir: Path,
    source_manifest_path: Path,
    source_asset: dict[str, Any],
    asset_manifest: dict[str, Any],
    model_key: str,
    level: str,
    seed: int,
    expected_samples: int,
    expected_index_hash: str,
) -> dict[str, Any]:
    source_manifest_path = source_manifest_path.resolve()
    if source_manifest_path != Path(source_asset["manifest_path"]).resolve():
        raise ValueError(f"{model_key}/{level} source manifest path mismatch")
    if file_sha256(source_manifest_path) != source_asset["manifest_file_sha256"]:
        raise ValueError(f"{model_key}/{level} source manifest file checksum mismatch")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    recorded_source_hash = source.get("manifest_sha256")
    computed_source_hash = canonical_sha256({
        key: value for key, value in source.items() if key != "manifest_sha256"
    })
    if (
        recorded_source_hash != computed_source_hash
        or recorded_source_hash != source_asset["manifest_sha256"]
    ):
        raise ValueError(f"{model_key}/{level} source manifest checksum mismatch")

    config_rows = {
        int(row["seed"]): row for row in source_asset.get("configs", [])
    }
    if seed not in config_rows:
        raise ValueError(f"{model_key}/{level}/seed{seed} lacks a frozen config")
    config_identity = config_rows[seed]
    config_path = Path(config_identity["path"]).resolve()
    if not config_path.is_file() or file_sha256(config_path) != config_identity["sha256"]:
        raise ValueError(f"{model_key}/{level}/seed{seed} config checksum mismatch")

    sample_index_path = run_dir / "sample_indices" / f"seed{seed}.json"
    frozen_index_asset = asset_manifest["pairs_and_indices"][level]
    frozen_index_path = Path(frozen_index_asset["index_path"]).resolve()
    if file_sha256(sample_index_path) != frozen_index_asset["index_file_sha256"]:
        raise ValueError(
            f"{model_key}/{level}/seed{seed} run index is not the frozen index file"
        )
    if file_sha256(frozen_index_path) != frozen_index_asset["index_file_sha256"]:
        raise ValueError(f"{model_key}/{level} frozen index file checksum mismatch")
    sample_index = json.loads(sample_index_path.read_text(encoding="utf-8"))
    samples = sample_index.get("samples")
    if (
        not isinstance(samples, list)
        or len(samples) != expected_samples
        or canonical_sha256(samples) != expected_index_hash
        or sample_index.get("sample_index_sha256") != expected_index_hash
        or sample_index.get("protocol_fingerprint") is None
        or int(sample_index.get("count", -1)) != expected_samples
        or sample_index.get("name_order_sha256") != canonical_sha256([
            str(sample["loader_name"]) for sample in samples
        ])
    ):
        raise ValueError(f"{model_key}/{level}/seed{seed} run index mismatch")

    command_path = run_dir / f"command_seed{seed}.txt"
    command_text = command_path.read_text(encoding="utf-8").strip()
    tokens = shlex.split(command_text)
    if len(tokens) < 3 or re.fullmatch(
        r"CUDA_VISIBLE_DEVICES=\d+", tokens[0]
    ) is None:
        raise ValueError(f"malformed MotionGPT command record: {command_path}")
    if resolve_cli_path(tokens[1]) != MOTIONGPT_PYTHON.resolve():
        raise ValueError(f"{model_key}/{level}/seed{seed} used the wrong Python environment")
    if resolve_cli_path(tokens[2]) != (
        REPO_ROOT / "MotionGPT/eval_loader_robustness.py"
    ).resolve():
        raise ValueError(f"{model_key}/{level}/seed{seed} used the wrong evaluator")

    def option(name: str) -> str:
        positions = [index for index, token in enumerate(tokens) if token == name]
        if len(positions) != 1 or positions[0] + 1 >= len(tokens):
            raise ValueError(
                f"{command_path} must contain exactly one value for {name}"
            )
        return tokens[positions[0] + 1]

    expected_paths = {
        "--manifest": source_manifest_path,
        "--config": config_path,
        "--pair-file": Path(
            asset_manifest["pairs_and_indices"][level]["pair_path"]
        ).resolve(),
        "--expected-sample-index": sample_index_path.resolve(),
        "--generation-audit-output": (
            run_dir / "generation_audits" / f"seed{seed}.json"
        ).resolve(),
        "--retrieval-embedding-output": (
            run_dir / "retrieval_embeddings" / f"seed{seed}.npz"
        ).resolve(),
        "--diagnostics-output": (
            run_dir / "audits" / f"seed{seed}_diagnostics.json"
        ).resolve(),
        "--output": (run_dir / f"seed{seed}.json").resolve(),
    }
    for flag, expected in expected_paths.items():
        actual = resolve_cli_path(option(flag))
        if actual != expected:
            raise ValueError(
                f"{model_key}/{level}/seed{seed} command {flag} mismatch"
            )
    expected_scalars = {
        "--dataset": "motionx322",
        "--expected-samples": str(expected_samples),
        "--loader-mode": "caption-locked",
        "--allowed-levels": level,
        "--caption-length-policy": "repair-all",
        "--seed": str(seed),
        "--device": "cuda:0",
        "--batch-size": str(BATCH_SIZE),
        "--num-workers": "0",
        "--spacy-model": "en_core_web_sm",
        "--sample-mode": "categorical",
        "--temperature": "1",
        "--top-k": "0",
        "--top-p": "1",
        "--pair-rng-mode": "sequential",
        "--sampling-kwargs-mode": "explicit",
        "--prompt-mode": "official-default",
        "--generation-order": "original-first",
        "--length-mode": "generated",
        "--generation-audit-batches": "0",
    }
    for flag, expected in expected_scalars.items():
        actual = option(flag)
        if actual != expected:
            raise ValueError(
                f"{model_key}/{level}/seed{seed} command {flag} mismatch: "
                f"{actual!r} != {expected!r}"
            )
    allowed_flags = set(expected_paths) | set(expected_scalars)
    unknown_flags = sorted({
        token for token in tokens[3:]
        if token.startswith("--") and token not in allowed_flags
    })
    if unknown_flags:
        raise ValueError(
            f"{model_key}/{level}/seed{seed} command contains unsupported flags "
            f"{unknown_flags}"
        )

    result_path = run_dir / f"seed{seed}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema_version") != "motiongpt-motionx322-loader-result/v1":
        raise ValueError(f"{model_key}/{level}/seed{seed} result schema mismatch")
    if result.get("source_manifest_fingerprint") != source_asset["protocol_fingerprint"]:
        raise ValueError(f"{model_key}/{level}/seed{seed} source fingerprint mismatch")
    if (
        result.get("dataset") != "motionx322"
        or int(result.get("seed", -1)) != seed
        or result.get("smoke") is not False
        or result.get("index_only") is not False
        or int(result.get("matched_count", -1)) != expected_samples
        or int(result.get("total_count", -1)) != expected_samples
        or int(result.get("formal_sample_count", -1)) != expected_samples
        or int(result.get("unmatched_count", -1)) != 0
        or int(result.get("runtime_fallback_count", -1)) != 0
        or int(result.get("caption_matched_count", -1)) != expected_samples
        or int(result.get("caption_mismatch_count", -1)) != 0
        or result.get("sample_index_sha256") != expected_index_hash
        or result.get("name_order_sha256") != sample_index["name_order_sha256"]
        or result.get("protocol_fingerprint")
        != sample_index["protocol_fingerprint"]
        or canonical_sha256(result.get("protocol"))
        != result.get("protocol_fingerprint")
        or result.get("model") != source.get("model")
    ):
        raise ValueError(f"{model_key}/{level}/seed{seed} result identity mismatch")
    result_artifacts = result.get("artifacts")
    expected_checkpoint = source_asset["checkpoint"]
    expected_evaluator = evaluator_identity(asset_manifest)
    expected_result_artifacts = {
        "checkpoint": {
            "path": expected_checkpoint["path"],
            "sha256": expected_checkpoint["sha256"],
        },
        "evaluator_checkpoint": {
            "path": expected_evaluator["path"],
            "sha256": expected_evaluator["sha256"],
        },
    }
    if result_artifacts != expected_result_artifacts:
        raise ValueError(f"{model_key}/{level}/seed{seed} result artifact mismatch")
    expected_load = source_asset.get("expected_checkpoint_load")
    actual_load = result.get("checkpoint_load_audit")
    if not isinstance(expected_load, dict) or not isinstance(actual_load, dict):
        raise ValueError(f"{model_key}/{level}/seed{seed} checkpoint load audit missing")
    for key in ("missing_key_count", "unexpected_key_count"):
        if int(actual_load.get(key, -1)) != int(expected_load.get(key, -2)):
            raise ValueError(
                f"{model_key}/{level}/seed{seed} checkpoint load {key} mismatch"
            )
    if (
        len(actual_load.get("missing_keys", [])) != int(actual_load["missing_key_count"])
        or len(actual_load.get("unexpected_keys", []))
        != int(actual_load["unexpected_key_count"])
    ):
        raise ValueError(f"{model_key}/{level}/seed{seed} checkpoint key audit mismatch")

    generation_audit_path = (
        run_dir / "generation_audits" / f"seed{seed}.json"
    ).resolve()
    if Path(str(result.get("generation_audit", ""))).resolve() != generation_audit_path:
        raise ValueError(f"{model_key}/{level}/seed{seed} generation audit path mismatch")
    generation_audit = json.loads(
        generation_audit_path.read_text(encoding="utf-8")
    )
    for role in ("original", "perturbed"):
        role_rows = generation_audit.get(role)
        if not isinstance(role_rows, list) or len(role_rows) != expected_samples:
            raise ValueError(
                f"{model_key}/{level}/seed{seed} {role} generation audit incomplete"
            )
        for offset, (audit_row, sample) in enumerate(zip(role_rows, samples)):
            if (
                int(audit_row.get("reference_frame_count", -1))
                != int(sample["motion_length"])
                or not isinstance(audit_row.get("motion_tokens"), list)
                or int(audit_row.get("motion_token_count", -1))
                != len(audit_row["motion_tokens"])
                or any(
                    not isinstance(token, int)
                    for token in audit_row["motion_tokens"]
                )
            ):
                raise ValueError(
                    f"{model_key}/{level}/seed{seed} {role} generation audit "
                    f"payload mismatch at {offset}"
                )

    diagnostics_path = run_dir / "audits" / f"seed{seed}_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    batches = diagnostics.get("batches")
    if (
        diagnostics.get("schema_version")
        != "motiongpt-motionx322-loader-diagnostics/v1"
        or int(diagnostics.get("seed", -1)) != seed
        or int(diagnostics.get("count", -1)) != expected_samples
        or not isinstance(batches, list)
        or [int(batch.get("batch_index", -1)) for batch in batches]
        != list(range(len(expected_batch_sizes(expected_samples))))
        or [int(batch.get("count", -1)) for batch in batches]
        != expected_batch_sizes(expected_samples)
    ):
        raise ValueError(f"{model_key}/{level}/seed{seed} diagnostics mismatch")
    counts = [
        sum(int(batch["temp_R_perturbed_text_generated"][rank]) for batch in batches)
        for rank in range(3)
    ]
    if counts != [
        int(value) for value in diagnostics["perturbed_text_generated_R_counts"]
    ]:
        raise ValueError(f"{model_key}/{level}/seed{seed} diagnostic counts mismatch")
    result_metrics = result.get("metrics", {})
    for rank, metric in enumerate(METRICS):
        if abs(float(result_metrics[metric]) - counts[rank] / expected_samples) > 1e-12:
            raise ValueError(
                f"{model_key}/{level}/seed{seed} diagnostic {metric} mismatch"
            )
    if abs(
        float(result_metrics["matching_pert"])
        - float(diagnostics["perturbed_text_generated_matching"])
    ) > 1e-12:
        raise ValueError(
            f"{model_key}/{level}/seed{seed} diagnostic matching mismatch"
        )
    return {
        "command_path": str(command_path.resolve()),
        "command_sha256": file_sha256(command_path),
        "config": {
            "path": str(config_path),
            "sha256": str(config_identity["sha256"]),
        },
        "diagnostics_path": str(diagnostics_path.resolve()),
        "diagnostics_sha256": file_sha256(diagnostics_path),
        "generation_audit": {
            "path": str(generation_audit_path),
            "sha256": file_sha256(generation_audit_path),
            "original_tokens_sha256": canonical_sha256([
                row["motion_tokens"] for row in generation_audit["original"]
            ]),
            "perturbed_tokens_sha256": canonical_sha256([
                row["motion_tokens"] for row in generation_audit["perturbed"]
            ]),
        },
        "checkpoint_load_audit": actual_load,
    }


def collect_motiongpt_family(
    run_root: Path,
    source_root: Path,
    asset_manifest: dict[str, Any],
    level: str,
    model_key: str,
    seeds: list[int],
    expected_samples: int,
    expected_index_hash: str,
    para_ruler_evaluator: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = run_root / "motiongpt_family" / model_key / level.lower()
    source_manifest = source_root / model_key / level.lower() / "manifest.json"
    source_asset = asset_manifest["motiongpt_sources"][model_key][level]
    # MotionGPT 家族的 dump 只记录标准 evaluator 身份（loader 不从实际加载
    # 文件派生指纹），因此身份校验始终使用标准 evaluator；实际评测通过
    # T2MGPT_RULER_CHECKPOINT 环境变量加载了 para-ruler（见 runner 脚本）。
    expected_evaluator = evaluator_identity(asset_manifest)
    rows, summary = aggregate_motiongpt(
        run_dir,
        "motionx322",
        expected_samples=expected_samples,
        require_caption_locked=True,
        expected_unique_pairs=expected_samples,
        expected_levels=[level],
        source_manifest=source_manifest,
        require_retrieval_embeddings=True,
        start_seed=seeds[0],
        num_repeats=len(seeds),
    )
    if summary["identities"]["sample_index"] != expected_index_hash:
        raise ValueError(f"{model_key}/{level} frozen-index hash mismatch")
    if summary["identities"]["checkpoint"] != source_asset["checkpoint"]["sha256"]:
        raise ValueError(f"{model_key}/{level} frozen checkpoint hash mismatch")
    if summary["identities"]["evaluator"] != expected_evaluator["sha256"]:
        raise ValueError(f"{model_key}/{level} evaluator hash mismatch")
    embedding_audits = {
        int(row["seed"]): row
        for row in summary["retrieval_embedding_audit"]["seeds"]
    }
    sidecar_audits = []
    collected = []
    for row in rows:
        metrics = {name: float(row[name]) for name in METRICS}
        seed = int(row["seed"])
        identity = f"{model_key}/{level}/seed{seed}"
        finite_monotonic(metrics, identity)
        sidecar_audit = validate_motiongpt_seed_sidecars(
            run_dir=run_dir,
            source_manifest_path=source_manifest,
            source_asset=source_asset,
            asset_manifest=asset_manifest,
            model_key=model_key,
            level=level,
            seed=seed,
            expected_samples=expected_samples,
            expected_index_hash=expected_index_hash,
        )
        sidecar_audits.append(sidecar_audit)
        if seed not in embedding_audits:
            raise ValueError(f"{identity} lacks an embedding recomputation audit")
        collected.append({
            "model": model_key,
            "method": MODEL_LABELS[model_key],
            "level": level,
            "seed": seed,
            "samples": expected_samples,
            **metrics,
            "sample_index_sha256": row["sample_index_sha256"],
            "checkpoint_sha256": summary["identities"]["checkpoint"],
            # 实际评测通过 T2MGPT_RULER_CHECKPOINT 加载了 para-ruler；
            # dump 的身份字段仍记录标准 evaluator（loader 不派生实际指纹），
            # 此处按 para-ruler 协议归一化，保证四模型共享同一尺子的溯源一致。
            "evaluator_sha256": (
                para_ruler_evaluator["sha256"]
                if para_ruler_evaluator
                else summary["identities"]["evaluator"]
            ),
            "online_offline_max_abs_error": float(
                embedding_audits[seed]["max_abs_error"]
            ),
        })
    summary["source_asset"] = source_asset
    summary["evaluator"] = (
        para_ruler_evaluator if para_ruler_evaluator else expected_evaluator
    )
    if para_ruler_evaluator is not None:
        summary["dump_recorded_evaluator"] = {
            "path": expected_evaluator["path"],
            "sha256": expected_evaluator["sha256"],
            "note": (
                "dump identity records the standard evaluator because the "
                "loader does not derive the actual checkpoint fingerprint; the "
                "runs loaded the para-ruler via T2MGPT_RULER_CHECKPOINT."
            ),
        }
    summary["sidecar_audits"] = sidecar_audits
    return collected, summary


def summarize_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["level"]), str(row["model"]))].append(row)
    output = []
    for level in LEVELS:
        for model in MODEL_ORDER:
            values = grouped[(level, model)]
            if not values:
                raise ValueError(f"no runs collected for {model}/{level}")
            checkpoint_hashes = {str(value["checkpoint_sha256"]) for value in values}
            evaluator_hashes = {str(value["evaluator_sha256"]) for value in values}
            if len(checkpoint_hashes) != 1 or len(evaluator_hashes) != 1:
                raise ValueError(f"artifact identity drift for {model}/{level}")
            row: dict[str, Any] = {
                "level": level,
                "model": model,
                "method": MODEL_LABELS[model],
                "n": len(values),
                "samples_per_seed": int(values[0]["samples"]),
                "sample_index_sha256": values[0]["sample_index_sha256"],
                "checkpoint_sha256": next(iter(checkpoint_hashes)),
                "evaluator_sha256": next(iter(evaluator_hashes)),
                "online_offline_max_abs_error": max(
                    float(value["online_offline_max_abs_error"])
                    for value in values
                ),
            }
            for metric in METRICS:
                array = np.asarray(
                    [float(value[metric]) for value in values], dtype=np.float64
                )
                mean = float(array.mean())
                population_std = float(array.std(ddof=0))
                sample_std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
                ci95 = float(1.96 * population_std / math.sqrt(len(array)))
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sample_std"] = sample_std
                row[f"{metric}_population_std"] = population_std
                row[f"{metric}_ci95"] = ci95
                row[f"{metric}_ci95_low"] = mean - ci95
                row[f"{metric}_ci95_high"] = mean + ci95
            output.append(row)
    return output


def format_metric(row: dict[str, Any], metric: str) -> str:
    return f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_sample_std']:.4f}"


def write_outputs(
    output_dir: Path,
    all_runs: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    protocol_manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_fields = [
        "model", "method", "level", "seed", "samples", *METRICS,
        "sample_index_sha256", "checkpoint_sha256", "evaluator_sha256",
        "online_offline_max_abs_error",
    ]
    with (output_dir / "all_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(all_runs)
    summary_fields = list(summary_rows[0])
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "statistics": {
                    "mean_over_seeds": True,
                    "sample_std_ddof": 1,
                    "ci95": "1.96 * population_std / sqrt(n)",
                },
                "rows": summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# Motion-X-322 independent P1/P2/P3 perturbed-caption R-Precision",
        "",
        "Values are mean ± sample standard deviation over the evaluated seeds.",
        "Each level uses an independent caption-locked 7,679-motion pool.",
        "The independent-level protocol excludes 54 incomplete motions and is not ",
        "directly mergeable with the existing 7,733-motion mixed-pool results.",
        "The standalone SATO++ model and the MotionGPT SATO++ plug-in are distinct systems.",
        "",
        "| Level | Method | R@1_P | R@2_P | R@3_P |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    by_key = {(row["level"], row["model"]): row for row in summary_rows}
    for level in LEVELS:
        for model in MODEL_ORDER:
            row = by_key[(level, model)]
            markdown.append(
                f"| {level} | {row['method']} | "
                f"{format_metric(row, 'top1_pert')} | "
                f"{format_metric(row, 'top2_pert')} | "
                f"{format_metric(row, 'top3_pert')} |"
            )
    (output_dir / "table.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    latex = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llccc@{}}",
        r"\toprule",
        r"Level & Method & R@1$_P$ $\uparrow$ & R@2$_P$ $\uparrow$ & R@3$_P$ $\uparrow$ \\",
        r"\midrule",
    ]
    for level_index, level in enumerate(LEVELS):
        best = {
            metric: max(
                by_key[(level, model)][f"{metric}_mean"] for model in MODEL_ORDER
            )
            for metric in METRICS
        }
        for model_index, model in enumerate(MODEL_ORDER):
            row = by_key[(level, model)]
            values = []
            for metric in METRICS:
                value = format_metric(row, metric).replace(" ± ", r" $\pm$ ")
                if math.isclose(
                    float(row[f"{metric}_mean"]), float(best[metric]),
                    rel_tol=0.0, abs_tol=1e-15,
                ):
                    value = r"\textbf{" + value + "}"
                values.append(value)
            prefix = level if model_index == 0 else ""
            latex.append(
                f"{prefix} & {row['method']} & {values[0]} & {values[1]} & {values[2]} \\\\"
            )
        if level_index != len(LEVELS) - 1:
            latex.append(r"\midrule")
    latex.extend([r"\bottomrule", r"\end{tabular*}"])
    (output_dir / "table.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )

    manifest_payload = {
        key: value for key, value in protocol_manifest.items()
        if key not in {"fingerprint", "outputs"}
    }
    manifest_payload["outputs"] = {
        name: file_identity(output_dir / name)
        for name in (
            "all_runs.csv",
            "summary.csv",
            "summary.json",
            "table.md",
            "table.tex",
        )
    }
    manifest_payload["fingerprint"] = canonical_sha256(manifest_payload)
    (output_dir / "protocol_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("ablation/motionx322_level_rprecision/runs"),
    )
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        default=Path("ablation/motionx322_level_rprecision/assets_manifest.json"),
    )
    parser.add_argument(
        "--motiongpt-source-root",
        type=Path,
        default=Path(
            "ablation/motionx322_level_rprecision/motiongpt_sources"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=123)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument(
        "--implementation-overrides-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON map {repo-relative path: allowed sha256} for "
            "implementation files changed by a documented protocol extension."
        ),
    )
    parser.add_argument(
        "--para-ruler-evaluator-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional paraphrase-aware ruler matcher checkpoint used by the "
            "para-ruler evaluation protocol (e.g. checkpoints/motionx322/"
            "text_mot_match_para_aware/model/finest.tar)."
        ),
    )
    args = parser.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.num_repeats))
    if not seeds or min(seeds) < 123 or max(seeds) > 142:
        parser.error("seeds must be an ordered subset of 123..142")
    asset_manifest_path = args.asset_manifest.resolve()
    asset_manifest = json.loads(
        asset_manifest_path.read_text(encoding="utf-8")
    )
    validate_asset_manifest(
        asset_manifest,
        source_root=args.motiongpt_source_root,
        implementation_overrides_json=args.implementation_overrides_json,
    )
    para_ruler_evaluator = None
    if args.para_ruler_evaluator_checkpoint is not None:
        ckpt = args.para_ruler_evaluator_checkpoint.resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        para_ruler_evaluator = {
            "path": str(ckpt),
            "size": ckpt.stat().st_size,
            "sha256": file_sha256(ckpt),
        }
    expected_samples = int(asset_manifest["expected_samples_per_level"])
    if expected_samples != EXPECTED_SAMPLES:
        raise ValueError("asset manifest sample-count mismatch")

    all_runs: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for level in LEVELS:
        expected_index_hash = asset_manifest["pairs_and_indices"][level][
            "sample_index_sha256"
        ]
        for model_key in ("t2mgpt", "satopp"):
            rows, audit = collect_t2m_family(
                args.run_root,
                asset_manifest,
                level,
                model_key,
                seeds,
                expected_samples,
                expected_index_hash,
                para_ruler_evaluator=para_ruler_evaluator,
            )
            all_runs.extend(rows)
            audits[f"{model_key}/{level}"] = audit
        for model_key in ("motiongpt", "motiongpt_satopp"):
            rows, audit = collect_motiongpt_family(
                args.run_root,
                args.motiongpt_source_root,
                asset_manifest,
                level,
                model_key,
                seeds,
                expected_samples,
                expected_index_hash,
                para_ruler_evaluator=para_ruler_evaluator,
            )
            all_runs.extend(rows)
            audits[f"{model_key}/{level}"] = audit

    if len(all_runs) != len(MODEL_ORDER) * len(LEVELS) * len(seeds):
        raise ValueError("aggregated run count mismatch")
    all_runs.sort(key=lambda row: (
        LEVELS.index(str(row["level"])),
        MODEL_ORDER.index(str(row["model"])),
        int(row["seed"]),
    ))
    evaluator_hashes = {str(row["evaluator_sha256"]) for row in all_runs}
    if len(evaluator_hashes) != 1:
        raise ValueError("the four systems do not share one Motion-X evaluator")
    summary_rows = summarize_runs(all_runs)
    implementation_paths = (
        REPO_ROOT / "MotionGPT/eval_loader_robustness.py",
        REPO_ROOT / "ablation/repeat20_perturb_fast_batch.py",
        REPO_ROOT / "dataset/config.py",
        REPO_ROOT / "dataset/dataset_TM_eval.py",
        REPO_ROOT / "models/evaluator_wrapper.py",
        REPO_ROOT / "tools/aggregate_motiongpt_loader_eval.py",
        REPO_ROOT / "tools/aggregate_motionx322_level_rprecision.py",
        REPO_ROOT / "tools/audit_motionx322_level_rprecision_assets.py",
        REPO_ROOT / "tools/prepare_motiongpt_motionx322_level_rprecision.py",
        REPO_ROOT / "tools/prepare_motionx322_level_rprecision_assets.sh",
        REPO_ROOT / "tools/run_motionx322_level_motiongpt.sh",
        REPO_ROOT / "tools/run_motionx322_level_rprecision.sh",
        REPO_ROOT / "tools/run_motionx322_level_t2mgpt_satopp.sh",
        REPO_ROOT / "utils/sato_pp_eval.py",
    )
    protocol_manifest = {
        "schema_version": "motionx322-independent-level-rprecision-results/v1",
        "asset_manifest": {
            "path": str(asset_manifest_path),
            "sha256": file_sha256(asset_manifest_path),
            "fingerprint": asset_manifest["fingerprint"],
        },
        "levels": list(LEVELS),
        "models": list(MODEL_ORDER),
        "model_labels": dict(MODEL_LABELS),
        "system_identity": {
            "t2mgpt": asset_manifest["t2m_family_spec"]["models"][
                T2M_INTERNAL_MODELS["t2mgpt"]
            ],
            "satopp": asset_manifest["t2m_family_spec"]["models"][
                T2M_INTERNAL_MODELS["satopp"]
            ],
            "motiongpt": asset_manifest["motiongpt_sources"]["motiongpt"]["P1"],
            "motiongpt_satopp": asset_manifest["motiongpt_sources"]
            ["motiongpt_satopp"]["P1"],
        },
        "seeds": seeds,
        "expected_samples_per_level": expected_samples,
        "excluded_incomplete_level_motions": 54,
        "candidate_pools": "three independent caption-locked P1/P2/P3 pools",
        "protocol_boundary": (
            "the 7,679-motion pure-level protocol must not be merged directly with "
            "the prior 7,733-motion mixed-pool Motion-X table"
        ),
        "batch_sizes": expected_batch_sizes(expected_samples),
        "retrieval_scope": "within each independent level-specific evaluator batch",
        "metrics": list(METRICS),
        "generation_settings": {
            "t2m_family": {
                "models": ["t2mgpt", "satopp"],
                "cfg_scale": 1.3,
                "cfg_unconditional": "zero",
                "sampling": {
                    "mode": "categorical",
                    "temperature": 1.0,
                    "top_k": 0,
                    "top_p": 1.0,
                },
            },
            "motiongpt_family": {
                "models": ["motiongpt", "motiongpt_satopp"],
                "guidance": "official model-native default; no T2M-GPT CFG override",
                "sampling": {
                    "mode": "categorical",
                    "temperature": 1.0,
                    "top_k": 0,
                    "top_p": 1.0,
                },
            },
        },
        "shared_evaluator_sha256": next(iter(evaluator_hashes)),
        "implementation_files": [
            file_identity(path) for path in implementation_paths
        ],
        "statistics": {
            "sample_std_ddof": 1,
            "ci95": "1.96 * population_std / sqrt(n)",
        },
        "audits": audits,
    }
    write_outputs(args.output_dir, all_runs, summary_rows, protocol_manifest)


if __name__ == "__main__":
    main()