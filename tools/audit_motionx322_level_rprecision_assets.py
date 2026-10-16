#!/usr/bin/env python3
"""Audit frozen Motion-X-322 P1/P2/P3 R-Precision assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


LEVELS = ("P1", "P2", "P3")
MOTIONGPT_MODELS = ("motiongpt", "motiongpt_satopp")
EXPECTED_COUNT = 7_679
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MOTIONGPT_CHECKPOINTS = {
    "motiongpt": REPO_ROOT / (
        "MotionGPT/experiments/mgpt/"
        "Instruct_MOTIONX322_OFFICIAL_FRESH_R2_SEED123_20260817/"
        "checkpoints/last.ckpt"
    ),
    "motiongpt_satopp": REPO_ROOT / (
        "MotionGPT/experiments/mgpt_sato_pp/"
        "SATO_PP_MOTIONX322_PAIRGEN1_PRESERVE_ONLY_100K_SEED123_20260822/"
        "checkpoints/best-fid-fidp-fidd.ckpt"
    ),
}
EXPECTED_MOTIONGPT_PARENT_MANIFESTS = {
    "motiongpt": REPO_ROOT / (
        "MotionGPT/experiments/robustness/"
        "motionx322_official_fresh_r2_last100k_table3_20seed_20260820/"
        "manifest.json"
    ),
    "motiongpt_satopp": REPO_ROOT / (
        "MotionGPT/experiments/robustness/"
        "motionx322_sato_plugin_pairgen1_preserve_only_bestcomposite_"
        "table3_20seed_20260823/manifest.json"
    ),
}
EXPECTED_MOTIONGPT_PARENT_SCHEMAS = {
    "motiongpt": "motiongpt-motionx322-table3-source/v1",
    "motiongpt_satopp": "motiongpt-motionx322-sato-plugin-eval-source/v1",
}
EXPECTED_MOTIONGPT_ASSETS_CONFIG = (
    REPO_ROOT / "MotionGPT/configs/assets_motionx322.yaml"
)
EXPECTED_T2M_ARTIFACTS = {
    "motionx322_t2mgpt_baseline_200k_best": {
        "vqvae": REPO_ROOT / "output_motionx322/VQVAE_MOTIONX322/net_best_fid.pth",
        "transformer": REPO_ROOT / (
            "output_motionx322/GPT_MOTIONX322_MAXTOK75_200K/net_best_fid.pth"
        ),
    },
    "motionx322_gemini31_sato100k": {
        "satopp_bundle": REPO_ROOT / (
            "ablation/motionx322_sato/"
            "max_token75_gemini31_human_recipe_100k_lr60k_bs132_seed123/"
            "sato_pp_iter_100000.pth"
        ),
    },
}
_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    path = path.resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _FILE_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[key] = value
    return value


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


def require_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def checkpoint_identity(path: Path) -> dict[str, Any]:
    path = require_file(path)
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def audit_assets(
    *,
    pair_prefix: Path,
    index_root: Path,
    source_root: Path,
    levels_meta_path: Path,
    dataset_meta_path: Path,
    t2m_spec_path: Path,
    expected_count: int = EXPECTED_COUNT,
    batch_size: int = 32,
    implementation_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    levels_meta_path = require_file(levels_meta_path)
    dataset_meta_path = require_file(dataset_meta_path)
    t2m_spec_path = require_file(t2m_spec_path)
    levels_meta = json.loads(levels_meta_path.read_text(encoding="utf-8"))
    dataset_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8"))
    overrides_resolved: dict[str, str] = {}
    if implementation_overrides:
        overrides_resolved = {
            str(Path(path).resolve()): str(hash_value)
            for path, hash_value in implementation_overrides.items()
        }

    if int(levels_meta.get("test_motions", -1)) != 7_733:
        raise ValueError("level metadata does not describe the 7,733-motion test set")
    if levels_meta.get("motions_per_level") != {
        level: expected_count for level in LEVELS
    }:
        raise ValueError("level metadata does not lock 7,679 motions per level")
    if int(levels_meta.get("skipped_total", -1)) != 54:
        raise ValueError("level metadata does not record exactly 54 incomplete motions")
    if levels_meta.get("skipped_counts") != {"incomplete_levels": 54}:
        raise ValueError("unexpected Motion-X level exclusion reasons")
    if (
        dataset_meta.get("dataset") != "motionx322_75"
        or int(dataset_meta.get("dim", -1)) != 322
        or int(dataset_meta.get("max_motion_tokens", -1)) != 75
        or int(dataset_meta.get("max_frames", -1)) != 300
        or int(dataset_meta.get("split_counts", {}).get("test", -1)) != 7_733
    ):
        raise ValueError("Motion-X-322 MaxToken75 dataset metadata mismatch")

    pair_assets: dict[str, dict[str, Any]] = {}
    source_orders: dict[str, list[tuple[str, str]]] = {}
    pair_source_projections: dict[str, list[dict[str, Any]]] = {}
    motion_projections: dict[str, list[dict[str, Any]]] = {}
    for level in LEVELS:
        lower = level.lower()
        pair_path = require_file(Path(f"{pair_prefix}_{lower}.json"))
        index_path = require_file(index_root / f"{lower}_sample_index_seed123.json")
        pairs = json.loads(pair_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        samples = index.get("samples")
        if not isinstance(pairs, list) or not isinstance(samples, list):
            raise ValueError(f"{level} pair/index payload must contain JSON lists")
        if len(pairs) != expected_count or len(samples) != expected_count:
            raise ValueError(
                f"{level} pair/index count mismatch: {len(pairs)}/{len(samples)}"
            )
        required_pair_fields = {
            "motion_id", "text_orig", "text_pair", "relation", "level",
            "pair_id", "edit_type",
        }
        for row_index, row in enumerate(pairs):
            if required_pair_fields - set(row):
                raise ValueError(f"{level} pair row {row_index} lacks required fields")
            if any(
                not str(row[field]).strip()
                for field in ("motion_id", "text_orig", "text_pair")
            ):
                raise ValueError(f"{level} pair row {row_index} contains empty text")
        if int(index.get("count", -1)) != expected_count:
            raise ValueError(f"{level} sample index count field mismatch")
        sample_hash = canonical_sha256(samples)
        if index.get("sample_index_sha256") != sample_hash:
            raise ValueError(f"{level} sample-index checksum mismatch")
        if Counter(pair_signature(row) for row in pairs) != Counter(
            sample_signature(row) for row in samples
        ):
            raise ValueError(f"{level} pair and sample-index identities differ")
        if Counter(str(row.get("level")) for row in pairs) != {
            level: expected_count
        }:
            raise ValueError(f"{level} pair file contains another level")
        if any(str(row.get("relation")) != "preserve" for row in pairs):
            raise ValueError(f"{level} contains a non-preserving pair")
        if any(
            row.get("caption_pair_exact") is not True
            or str(row.get("loader_name")) != str(row.get("motion_id"))
            or normalized(row.get("caption"))
            != normalized(row.get("pair_text_orig"))
            or row.get("length_repair", {}).get("policy") != "repair-all"
            or int(row.get("motion_length", 0)) < 1
            or int(row.get("motion_length", 0)) > 300
            for row in samples
        ):
            raise ValueError(f"{level} index violates caption-locked repair-all")
        pair_ids = [row.get("pair_id") for row in samples]
        motion_ids = [str(row.get("motion_id")) for row in samples]
        if None in pair_ids or len(set(pair_ids)) != expected_count:
            raise ValueError(f"{level} pair IDs are missing or duplicated")
        if len(set(motion_ids)) != expected_count:
            raise ValueError(f"{level} motion IDs are not unique")

        pair_source_projection = [
            {
                key: value for key, value in row.items()
                if key not in {"text_pair", "level"}
            }
            for row in pairs
        ]
        source_order = [
            (str(row["motion_id"]), normalized(row["caption"])) for row in samples
        ]
        motion_projection = [
            {
                "loader_name": str(row["loader_name"]),
                "motion_id": str(row["motion_id"]),
                "caption": normalized(row["caption"]),
                "motion_length": int(row["motion_length"]),
                "length_repair": row.get("length_repair"),
            }
            for row in samples
        ]
        source_orders[level] = source_order
        pair_source_projections[level] = pair_source_projection
        motion_projections[level] = motion_projection
        pair_assets[level] = {
            "pair_path": str(pair_path),
            "pair_file_sha256": file_sha256(pair_path),
            "pair_records_sha256": canonical_sha256(pairs),
            "index_path": str(index_path),
            "index_file_sha256": file_sha256(index_path),
            "sample_index_sha256": sample_hash,
            "protocol_fingerprint": str(index.get("protocol_fingerprint", "")),
            "count": expected_count,
            "unique_pair_ids": len(set(pair_ids)),
            "unique_motion_ids": len(set(motion_ids)),
            "source_order_sha256": canonical_sha256(source_order),
            "motion_projection_sha256": canonical_sha256(motion_projection),
            "level_counts": {level: expected_count},
        }

    first_order = source_orders["P1"]
    first_pair_projection = pair_source_projections["P1"]
    first_projection = motion_projections["P1"]
    if any(source_orders[level] != first_order for level in LEVELS[1:]):
        raise ValueError("P1/P2/P3 source motion/caption order differs")
    if any(
        pair_source_projections[level] != first_pair_projection
        for level in LEVELS[1:]
    ):
        raise ValueError(
            "P1/P2/P3 pair rows differ outside the text_pair and level fields"
        )
    if any(motion_projections[level] != first_projection for level in LEVELS[1:]):
        raise ValueError("P1/P2/P3 motion length/crop projection differs")

    source_assets: dict[str, dict[str, Any]] = {}
    checkpoint_hashes: dict[str, set[str]] = {
        model_key: set() for model_key in MOTIONGPT_MODELS
    }
    for model_key in MOTIONGPT_MODELS:
        source_assets[model_key] = {}
        config_hashes_by_seed: dict[int, set[str]] = {
            seed: set() for seed in range(123, 143)
        }
        assets_config_hashes: set[str] = set()
        implementation_fingerprints: set[str] = set()
        model_fingerprints: set[str] = set()
        for level in LEVELS:
            manifest_path = require_file(
                source_root / model_key / level.lower() / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != "motiongpt-motionx322-level-source/v1":
                raise ValueError(f"{model_key}/{level} source manifest schema mismatch")
            recorded_manifest_hash = str(manifest.get("manifest_sha256", ""))
            computed_manifest_hash = canonical_sha256({
                key: value for key, value in manifest.items()
                if key != "manifest_sha256"
            })
            if recorded_manifest_hash != computed_manifest_hash:
                raise ValueError(f"{model_key}/{level} source manifest checksum mismatch")
            parent = manifest.get("parent_manifest", {})
            expected_parent_path = EXPECTED_MOTIONGPT_PARENT_MANIFESTS[
                model_key
            ].resolve()
            parent_path = require_file(Path(str(parent.get("path", ""))))
            if (
                parent_path != expected_parent_path
                or parent.get("schema_version")
                != EXPECTED_MOTIONGPT_PARENT_SCHEMAS[model_key]
                or file_sha256(parent_path) != parent.get("sha256")
            ):
                raise ValueError(
                    f"{model_key}/{level} historical parent manifest identity mismatch"
                )
            parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
            if (
                parent.get("protocol_fingerprint")
                != parent_payload.get("protocol", {}).get("fingerprint")
            ):
                raise ValueError(
                    f"{model_key}/{level} historical parent protocol mismatch"
                )
            protocol = manifest.get("protocol", {})
            model = manifest.get("model", {})
            pairs = manifest.get("pairs", {})
            if protocol.get("fingerprint") != canonical_sha256({
                key: value for key, value in protocol.items() if key != "fingerprint"
            }):
                raise ValueError(f"{model_key}/{level} source protocol checksum mismatch")
            if (
                protocol.get("samples_per_seed") != expected_count
                or protocol.get("independent_level") != level
                or protocol.get("level_counts") != {level: expected_count}
                or protocol.get("seeds") != list(range(123, 143))
                or protocol.get("loader", {}).get("mode") != "caption-locked"
            ):
                raise ValueError(f"{model_key}/{level} source protocol mismatch")
            pair_asset = pair_assets[level]
            if (
                pairs.get("raw_sha256") != pair_asset["pair_file_sha256"]
                or pairs.get("canonical_sha256")
                != pair_asset["pair_records_sha256"]
            ):
                raise ValueError(f"{model_key}/{level} source pair hash mismatch")
            commands = manifest.get("commands")
            if not isinstance(commands, list) or len(commands) != 20:
                raise ValueError(f"{model_key}/{level} source command list mismatch")
            config_files: list[dict[str, Any]] = []
            for command in commands:
                seed = int(command.get("seed", -1))
                if seed not in range(123, 143):
                    raise ValueError(f"{model_key}/{level} source command seed mismatch")
                config_path = require_file(Path(str(command.get("config", ""))))
                config_hash = file_sha256(config_path)
                if config_hash != command.get("config_sha256"):
                    raise ValueError(
                        f"{model_key}/{level}/seed{seed} config checksum mismatch"
                    )
                config_files.append({
                    "seed": seed,
                    "path": str(config_path),
                    "sha256": config_hash,
                })
            if {row["seed"] for row in config_files} != set(range(123, 143)):
                raise ValueError(f"{model_key}/{level} source config seeds mismatch")
            for config_file in config_files:
                config_hashes_by_seed[config_file["seed"]].add(
                    config_file["sha256"]
                )
            assets_config = manifest.get("assets_config", {})
            assets_config_path = require_file(
                Path(str(assets_config.get("path", "")))
            )
            if assets_config_path != EXPECTED_MOTIONGPT_ASSETS_CONFIG.resolve():
                raise ValueError(f"{model_key}/{level} assets config path mismatch")
            assets_config_hash = file_sha256(assets_config_path)
            if assets_config_hash != assets_config.get("sha256"):
                raise ValueError(f"{model_key}/{level} assets config checksum mismatch")
            assets_config_hashes.add(assets_config_hash)
            implementation = manifest.get("implementation")
            if not isinstance(implementation, dict) or not implementation:
                raise ValueError(f"{model_key}/{level} implementation audit is empty")
            for implementation_path, expected_hash in implementation.items():
                current_path = require_file(Path(implementation_path))
                current_hash = file_sha256(current_path)
                key = str(current_path)
                if key in overrides_resolved:
                    if current_hash != overrides_resolved[key]:
                        raise ValueError(
                            f"{model_key}/{level} implementation override hash "
                            f"mismatch: {current_path}"
                        )
                elif current_hash != expected_hash:
                    raise ValueError(
                        f"{model_key}/{level} implementation checksum mismatch: "
                        f"{current_path}"
                    )
            implementation_fingerprints.add(
                canonical_sha256(dict(sorted(implementation.items())))
            )
            checkpoint = checkpoint_identity(Path(str(model.get("checkpoint", ""))))
            expected_checkpoint_path = EXPECTED_MOTIONGPT_CHECKPOINTS[model_key].resolve()
            if Path(checkpoint["path"]) != expected_checkpoint_path:
                raise ValueError(
                    f"{model_key}/{level} checkpoint path mismatch: "
                    f"{checkpoint['path']} != {expected_checkpoint_path}"
                )
            if checkpoint["sha256"] != model.get("checkpoint_sha256"):
                raise ValueError(f"{model_key}/{level} checkpoint checksum mismatch")
            if model.get("uses_t2mgpt_generator_checkpoint") is not False:
                raise ValueError(f"{model_key}/{level} permits a T2M-GPT checkpoint")
            expected_load = model.get("expected_checkpoint_load")
            if expected_load != {
                "missing_key_count": 0,
                "unexpected_key_count": 0,
            }:
                raise ValueError(f"{model_key}/{level} checkpoint-load contract mismatch")
            if model_key == "motiongpt":
                if (
                    model.get("name") != "MotionGPT"
                    or int(model.get("selection_step", -1)) != 100_000
                ):
                    raise ValueError(f"{model_key}/{level} model identity mismatch")
            elif (
                model.get("target") != "mGPT.models.mgpt_sato_pp.MotionGPTSATOPP"
                or model.get("uses_motiongpt_baseline_checkpoint") is not True
                or int(model.get("selection_step", -1)) != 40_000
            ):
                raise ValueError(f"{model_key}/{level} plug-in identity mismatch")
            model_fingerprints.add(canonical_sha256(model))
            checkpoint_hashes[model_key].add(checkpoint["sha256"])
            source_assets[model_key][level] = {
                "manifest_path": str(manifest_path),
                "manifest_file_sha256": file_sha256(manifest_path),
                "manifest_sha256": recorded_manifest_hash,
                "protocol_fingerprint": protocol.get("fingerprint"),
                "checkpoint": checkpoint,
                "expected_checkpoint_load": model.get("expected_checkpoint_load"),
                "assets_config": {
                    "path": str(assets_config_path),
                    "sha256": str(assets_config.get("sha256")),
                },
                "configs": sorted(config_files, key=lambda row: row["seed"]),
                "implementation": dict(sorted(implementation.items())),
            }
        if len(checkpoint_hashes[model_key]) != 1:
            raise ValueError(f"{model_key} checkpoint differs across levels")
        drifting_config_seeds = sorted(
            seed for seed, hashes in config_hashes_by_seed.items()
            if len(hashes) != 1
        )
        if drifting_config_seeds:
            raise ValueError(
                f"{model_key} seed configs differ across levels: "
                f"{drifting_config_seeds}"
            )
        if len(assets_config_hashes) != 1:
            raise ValueError(f"{model_key} assets config differs across levels")
        if len(implementation_fingerprints) != 1:
            raise ValueError(f"{model_key} implementation identity differs across levels")
        if len(model_fingerprints) != 1:
            raise ValueError(f"{model_key} model metadata differs across levels")

    t2m_specs = json.loads(t2m_spec_path.read_text(encoding="utf-8"))
    if not isinstance(t2m_specs, list) or {row.get("name") for row in t2m_specs} != {
        "motionx322_t2mgpt_baseline_200k_best",
        "motionx322_gemini31_sato100k",
    }:
        raise ValueError("T2M-GPT/SATO++ spec does not contain the locked models")
    t2m_models: dict[str, Any] = {}
    expected_t2m_specs = {
        "motionx322_t2mgpt_baseline_200k_best": {
            "kind": "baseline",
            "dataname": "motionx322_75",
            "resume_pth": "output_motionx322/VQVAE_MOTIONX322/net_best_fid.pth",
            "resume_trans": (
                "output_motionx322/GPT_MOTIONX322_MAXTOK75_200K/net_best_fid.pth"
            ),
            "args": {"down_t": 2, "block_size": 76},
        },
        "motionx322_gemini31_sato100k": {
            "kind": "sato_pp",
            "checkpoint": (
                "ablation/motionx322_sato/"
                "max_token75_gemini31_human_recipe_100k_lr60k_bs132_seed123/"
                "sato_pp_iter_100000.pth"
            ),
        },
    }
    for spec in t2m_specs:
        spec_name = str(spec["name"])
        expected_spec = expected_t2m_specs[spec_name]
        allowed_spec_keys = {"name", "description", *expected_spec}
        unexpected_spec_keys = sorted(set(spec) - allowed_spec_keys)
        if unexpected_spec_keys:
            raise ValueError(
                f"locked T2M-family spec contains unsupported overrides for "
                f"{spec_name}: {unexpected_spec_keys}"
            )
        if any(spec.get(key) != value for key, value in expected_spec.items()):
            raise ValueError(f"locked T2M-family spec identity mismatch: {spec_name}")
        artifacts: list[tuple[str, Path]] = []
        if spec.get("kind") == "baseline":
            artifacts.extend([
                ("vqvae", Path(spec["resume_pth"])),
                ("transformer", Path(spec["resume_trans"])),
            ])
        elif spec.get("kind") == "sato_pp":
            artifacts.append(("satopp_bundle", Path(spec["checkpoint"])))
        else:
            raise ValueError(f"unsupported T2M family spec kind: {spec.get('kind')}")
        artifact_identities = []
        expected_artifacts = EXPECTED_T2M_ARTIFACTS[spec_name]
        if {role for role, _ in artifacts} != set(expected_artifacts):
            raise ValueError(f"locked T2M-family artifact roles mismatch: {spec_name}")
        for role, path in artifacts:
            resolved = (path if path.is_absolute() else REPO_ROOT / path).resolve()
            expected_path = expected_artifacts[role].resolve()
            if resolved != expected_path:
                raise ValueError(
                    f"locked T2M-family artifact path mismatch for "
                    f"{spec_name}/{role}: {resolved} != {expected_path}"
                )
            artifact_identities.append({
                "role": role,
                **checkpoint_identity(resolved),
            })
        t2m_models[spec_name] = {
            "kind": spec["kind"],
            "spec": spec,
            "artifacts": artifact_identities,
        }

    evaluator_files = [
        checkpoint_identity(REPO_ROOT / "checkpoints/motionx322" / relative)
        for relative in (
            "Comp_v6_KLD005/opt.txt",
            "Decomp_SP001_SM001_H512/meta/mean.npy",
            "Decomp_SP001_SM001_H512/meta/std.npy",
            "Decomp_SP001_SM001_H512/model/finest.tar",
            "text_mot_match/model/finest.tar",
        )
    ]
    tail = expected_count % batch_size
    batch_sizes = [batch_size] * (expected_count // batch_size)
    if tail:
        batch_sizes.append(tail)
    if not batch_sizes or batch_sizes[-1] < 3:
        raise ValueError("final evaluator batch cannot support R@3")

    manifest = {
        "schema_version": "motionx322-independent-level-rprecision-assets/v1",
        "dataset": "motionx322_75",
        "levels": list(LEVELS),
        "expected_samples_per_level": expected_count,
        "excluded_incomplete_levels": 54,
        "batch_size": batch_size,
        "batch_count": math.ceil(expected_count / batch_size),
        "batch_sizes": batch_sizes,
        "retrieval_scope": "within each independent level-specific evaluator batch",
        "caption_locked": True,
        "length_policy": "repair-all",
        "shuffle": False,
        "drop_last": False,
        "common_source_order_sha256": canonical_sha256(first_order),
        "common_motion_projection_sha256": canonical_sha256(first_projection),
        "levels_metadata": {
            "path": str(levels_meta_path),
            "sha256": file_sha256(levels_meta_path),
        },
        "dataset_metadata": {
            "path": str(dataset_meta_path),
            "sha256": file_sha256(dataset_meta_path),
        },
        "pairs_and_indices": pair_assets,
        "motiongpt_sources": source_assets,
        "t2m_family_spec": {
            "path": str(t2m_spec_path),
            "sha256": file_sha256(t2m_spec_path),
            "models": t2m_models,
        },
        "evaluator_files": evaluator_files,
    }
    if overrides_resolved:
        manifest["implementation_overrides"] = dict(sorted(overrides_resolved.items()))
    manifest["fingerprint"] = canonical_sha256(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-prefix",
        type=Path,
        default=Path("dataset/Motion-X-322/test_perb_gemini31_max75"),
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path("ablation/motionx322_level_rprecision/index_audit"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("ablation/motionx322_level_rprecision/motiongpt_sources"),
    )
    parser.add_argument(
        "--levels-meta",
        type=Path,
        default=Path(
            "dataset/Motion-X-322/test_perb_gemini31_max75_levels.meta.json"
        ),
    )
    parser.add_argument(
        "--dataset-meta",
        type=Path,
        default=Path("dataset/Motion-X-322-MaxToken75/dataset_meta.json"),
    )
    parser.add_argument(
        "--t2m-spec",
        type=Path,
        default=Path(
            "ablation/motionx322_sato/"
            "gemini31_100k_full_test_human_protocol_cfg1p3_repeat20/"
            "baseline_vs_sato100k_spec.json"
        ),
    )
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--implementation-overrides-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON map {repo-relative path: allowed sha256} for "
            "implementation files changed by a documented protocol extension "
            "(e.g. the paraphrase-aware ruler evaluator override)."
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    implementation_overrides = None
    if args.implementation_overrides_json:
        implementation_overrides = json.loads(
            args.implementation_overrides_json.read_text(encoding="utf-8")
        )

    manifest = audit_assets(
        pair_prefix=args.pair_prefix,
        index_root=args.index_root,
        source_root=args.source_root,
        levels_meta_path=args.levels_meta,
        dataset_meta_path=args.dataset_meta,
        t2m_spec_path=args.t2m_spec,
        expected_count=args.expected_count,
        batch_size=args.batch_size,
        implementation_overrides=implementation_overrides,
    )
    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()