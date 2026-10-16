#!/usr/bin/env python3
"""Project audited MotionGPT Motion-X checkpoints onto one frozen P1/P2/P3 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


SEEDS = list(range(123, 143))
LEVELS = ("P1", "P2", "P3")
EXPECTED_SAMPLES = 7_679


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


def validate_pairs(path: Path, level: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or len(pairs) != EXPECTED_SAMPLES:
        raise ValueError(
            f"{level} pair file must contain {EXPECTED_SAMPLES} rows: {path}"
        )
    levels = Counter(str(row.get("level")) for row in pairs)
    if levels != {level: EXPECTED_SAMPLES}:
        raise ValueError(f"{path} level counts differ: {dict(levels)}")
    if any(str(row.get("relation")) != "preserve" for row in pairs):
        raise ValueError(f"{path} contains a non-preserving pair")
    source_keys = [
        (str(row.get("motion_id", "")), normalized(row.get("text_orig", "")))
        for row in pairs
    ]
    if any(not motion_id or not caption for motion_id, caption in source_keys):
        raise ValueError(f"{path} contains an empty motion/caption key")
    if len(set(source_keys)) != EXPECTED_SAMPLES:
        raise ValueError(f"{path} contains duplicate motion/caption keys")
    if len({motion_id for motion_id, _ in source_keys}) != EXPECTED_SAMPLES:
        raise ValueError(f"{path} does not contain unique motions")
    if any(not str(row.get("text_pair", "")).strip() for row in pairs):
        raise ValueError(f"{path} contains an empty perturbation")
    return pairs, {
        "path": str(path),
        "raw_sha256": file_sha256(path),
        "canonical_sha256": canonical_sha256(pairs),
        "rows": len(pairs),
        "unique_motions": len({motion_id for motion_id, _ in source_keys}),
        "source_order_sha256": canonical_sha256(source_keys),
        "level_counts": dict(sorted(levels.items())),
    }


def command_assets_config(command: list[object]) -> Path:
    values = [str(value) for value in command]
    try:
        return Path(values[values.index("--cfg_assets") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise ValueError("parent manifest command lacks --cfg_assets") from exc


def validate_parent(parent: dict[str, Any], model_key: str) -> None:
    protocol = parent.get("protocol", {})
    model = parent.get("model", {})
    commands = parent.get("commands", [])
    if protocol.get("dataset") != "motionx322":
        raise ValueError("parent manifest is not Motion-X-322")
    if protocol.get("seeds") != SEEDS:
        raise ValueError("parent manifest does not contain seeds 123..142")
    if len(commands) != len(SEEDS) or {
        int(row.get("seed", -1)) for row in commands
    } != set(SEEDS):
        raise ValueError("parent manifest command list is incomplete")
    checkpoint = Path(str(model.get("checkpoint", ""))).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if model.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise ValueError("parent checkpoint checksum mismatch")
    if model.get("uses_t2mgpt_generator_checkpoint") is not False:
        raise ValueError("parent manifest permits a T2M-GPT generator dependency")
    if model_key == "motiongpt":
        if model.get("name") != "MotionGPT" or int(model.get("selection_step", -1)) != 100_000:
            raise ValueError("MotionGPT parent is not the audited terminal 100k model")
    elif model_key == "motiongpt_satopp":
        if (
            model.get("target") != "mGPT.models.mgpt_sato_pp.MotionGPTSATOPP"
            or model.get("uses_motiongpt_baseline_checkpoint") is not True
            or int(model.get("selection_step", -1)) != 40_000
        ):
            raise ValueError(
                "MotionGPT+SATO++ parent is not the preserve-only composite 40k plug-in"
            )
    else:
        raise ValueError(f"unsupported model key: {model_key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-key",
        choices=("motiongpt", "motiongpt_satopp"),
        required=True,
    )
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--level", choices=LEVELS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    parent_path = args.parent_manifest.resolve()
    pair_path = args.pair_file.resolve()
    for required in (parent_path, pair_path):
        if not required.is_file():
            parser.error(f"missing required file: {required}")

    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    validate_parent(parent, args.model_key)
    pairs, pair_identity = validate_pairs(pair_path, args.level)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not fresh: {output_dir}")
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    parent_commands = {
        int(row["seed"]): row for row in parent["commands"]
    }
    commands = []
    assets_configs: set[Path] = set()
    for seed in SEEDS:
        parent_row = parent_commands[seed]
        source_config = Path(str(parent_row["config"])).resolve()
        if not source_config.is_file():
            parser.error(f"missing parent seed config: {source_config}")
        assets_config = command_assets_config(parent_row["command"])
        if not assets_config.is_file():
            parser.error(f"missing parent assets config: {assets_config}")
        assets_configs.add(assets_config)
        config_path = config_dir / f"{args.model_key}_motionx322_seed{seed}.yaml"
        shutil.copy2(source_config, config_path)
        commands.append({
            "seed": seed,
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "command": [
                "python", "test.py", "--cfg", str(config_path),
                "--cfg_assets", str(assets_config), "--nodebug",
            ],
        })
    if len(assets_configs) != 1:
        parser.error("parent commands use more than one assets config")

    protocol = {
        "name": (
            f"motionx322-{args.model_key}-{args.level.lower()}-"
            "caption-locked-rprecision/v1"
        ),
        "dataset": "motionx322",
        "dataname": "motionx322_75",
        "samples_per_seed": EXPECTED_SAMPLES,
        "unique_pairs_per_seed": EXPECTED_SAMPLES,
        "independent_level": args.level,
        "level_counts": {args.level: EXPECTED_SAMPLES},
        "sampling": {
            "mode": "categorical",
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
        },
        "pair_rng_mode": "sequential",
        "generation_order": "original-first",
        "prompt_mode": "official-default",
        "generation_length_mode": "generated",
        "loader": {
            "mode": "caption-locked",
            "shuffle": False,
            "drop_last": False,
            "length_policy": "repair-all",
            "pair_match": "exact normalized (motion_id, text_orig)",
        },
        "seeds": SEEDS,
    }
    protocol["fingerprint"] = canonical_sha256(protocol)

    repo_root = Path(__file__).resolve().parents[1]
    implementation_paths = (
        Path(__file__).resolve(),
        (repo_root / "MotionGPT/eval_loader_robustness.py").resolve(),
        (repo_root / "tools/aggregate_motiongpt_loader_eval.py").resolve(),
        (repo_root / "dataset/dataset_TM_eval.py").resolve(),
    )
    manifest = {
        "schema_version": "motiongpt-motionx322-level-source/v1",
        "protocol": protocol,
        "model": dict(parent["model"]),
        "pairs": pair_identity,
        "parent_manifest": {
            "path": str(parent_path),
            "sha256": file_sha256(parent_path),
            "schema_version": parent.get("schema_version"),
            "protocol_fingerprint": parent.get("protocol", {}).get("fingerprint"),
        },
        "assets_config": {
            "path": str(next(iter(assets_configs))),
            "sha256": file_sha256(next(iter(assets_configs))),
        },
        "commands": commands,
        "implementation": {
            str(path): file_sha256(path) for path in implementation_paths
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()