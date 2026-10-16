#!/usr/bin/env python3
"""Prepare a locked MotionGPT-last HumanML3D Table-1 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
import zipfile
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf


SEEDS = list(range(123, 143))


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_metadata(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        data_name = next(
            (name for name in archive.namelist() if name.endswith("/data.pkl")),
            None,
        )
        if data_name is None:
            raise RuntimeError(f"checkpoint has no data.pkl: {path}")
        operations = list(pickletools.genops(archive.read(data_name)))
    values = {}
    integer_ops = {"BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"}
    for index, (operation, argument, _) in enumerate(operations):
        if argument not in ("epoch", "global_step") or argument in values:
            continue
        for value_operation, value_argument, _ in operations[index + 1:index + 5]:
            if value_operation.name in integer_ops:
                values[str(argument)] = int(value_argument)
                break
    return {
        "global_step": int(values.get("global_step", -1)),
        "epoch": int(values.get("epoch", -1)),
    }


def normalized(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def pair_signature(row: dict, *, sample: bool) -> tuple[str, str, str, str]:
    return (
        str(row["motion_id"]),
        normalized(row["caption"] if sample else row["text_orig"]),
        str(row["perturbed_caption"] if sample else row["text_pair"]),
        str(row["level"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument(
        "--assets-config",
        type=Path,
        default=Path("MotionGPT/configs/assets_humanml3d_official_local.yaml"),
    )
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--fixed-sample-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--level",
        choices=("P1", "P2", "P3"),
        help=(
            "prepare a 4,380-sample independent single-level evaluation; "
            "omit for the existing 4,631/4,640 mixed Table-1 protocol"
        ),
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    base_config = args.base_config.resolve()
    assets_config = args.assets_config.resolve()
    pair_file = args.pair_file.resolve()
    fixed_index_path = args.fixed_sample_index.resolve()
    for required in (
        checkpoint,
        base_config,
        assets_config,
        pair_file,
        fixed_index_path,
    ):
        if not required.is_file():
            parser.error(f"missing required file: {required}")
    if checkpoint.name != "last.ckpt":
        parser.error("Table-1 terminal evaluation requires a literal last.ckpt")

    audited_selection_path = checkpoint.parent / "selected-validation.json"
    if not audited_selection_path.is_file():
        parser.error(f"missing Stage-3 dependency audit: {audited_selection_path}")
    audited = json.loads(audited_selection_path.read_text(encoding="utf-8"))
    if audited.get("stage") != "stage3" or audited.get("split") != "val":
        parser.error("Stage-3 dependency audit is not validation-only")
    if audited.get("uses_t2mgpt_generator_checkpoint") is not False:
        parser.error("Stage-3 dependency audit permits a T2M-GPT generator")

    checkpoint_meta = checkpoint_metadata(checkpoint)
    if checkpoint_meta["global_step"] <= 0:
        parser.error("last.ckpt does not contain a positive global_step")
    checkpoint_sha256 = file_sha256(checkpoint)

    pairs = json.loads(pair_file.read_text(encoding="utf-8"))
    fixed_index = json.loads(fixed_index_path.read_text(encoding="utf-8"))
    samples = fixed_index.get("samples")
    if not isinstance(pairs, list) or not isinstance(samples, list):
        parser.error("pair file and fixed sample index must contain JSON lists")
    sample_count = int(fixed_index.get("count", -1))
    supported_counts = (4380,) if args.level else (4631, 4640)
    if sample_count not in supported_counts:
        parser.error(
            f"unsupported HumanML3D Table-1 sample count for "
            f"level={args.level!r}: {sample_count}"
        )
    if len(pairs) != sample_count or len(samples) != sample_count:
        parser.error(
            "HumanML3D pair/index row mismatch: "
            f"{len(pairs)}/{len(samples)}/{sample_count}"
        )
    if fixed_index.get("sample_index_sha256") != canonical_sha256(samples):
        parser.error("fixed sample-index hash differs from its records")
    pair_counts = Counter(pair_signature(row, sample=False) for row in pairs)
    sample_counts = Counter(pair_signature(row, sample=True) for row in samples)
    if pair_counts != sample_counts:
        parser.error("pair-file multiplicity differs from the fixed sample index")
    level_counts = Counter(str(row.get("level")) for row in samples)
    if args.level:
        expected_level_counts = {args.level: 4380}
    else:
        expected_level_counts = {"P1": 2138, "P2": 1240, "P3": 1253}
        if sample_count == 4640:
            expected_level_counts["fallback"] = 9
    if level_counts != expected_level_counts:
        parser.error(f"unexpected HumanML3D P1/P2/P3 counts: {dict(level_counts)}")
    if any(row.get("caption_pair_exact") is not True for row in samples):
        parser.error("fixed sample index contains a non-exact caption match")
    unique_pair_ids = len({row.get("pair_id") for row in samples})
    expected_unique_pairs = 4380 if args.level else (4353 if sample_count == 4640 else 4344)
    if (
        unique_pair_ids != expected_unique_pairs
        or None in {row.get("pair_id") for row in samples}
    ):
        parser.error(
            "fixed sample index does not contain exactly "
            f"{expected_unique_pairs} resolved pairs"
        )

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"evaluation output directory is not fresh: {output_dir}")
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    base = OmegaConf.load(base_config)
    if base.model.target != "mGPT.models.mgpt.MotionGPT":
        parser.error("base config does not instantiate MotionGPT")
    if base.TRAIN.STAGE != "lm_instruct":
        parser.error("base config is not a Stage-3 lm_instruct config")
    token_code_path = str(base.DATASET.CODE_PATH)

    commands = []
    for seed in SEEDS:
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
        cfg.NAME = f"MotionGPT_HumanML3D_Table1_Last_{seed}"
        cfg.DEVICE = [0]
        cfg.DEBUG = False
        cfg.SEED_VALUE = seed
        cfg.TRAIN.RESUME = ""
        cfg.TRAIN.PRETRAINED = ""
        cfg.TRAIN.PRETRAINED_VAE = ""
        cfg.TEST.CHECKPOINTS = str(checkpoint)
        cfg.TEST.SPLIT = "test"
        cfg.TEST.BATCH_SIZE = 32
        cfg.DATASET.CODE_PATH = token_code_path
        cfg.CHECKPOINT_SELECTION.ENABLED = False
        config_path = config_dir / f"motiongpt_humanml3d_seed{seed}.yaml"
        OmegaConf.save(cfg, config_path)
        commands.append({
            "seed": seed,
            "config": str(config_path),
            "command": [
                "python", "test.py", "--cfg", str(config_path),
                "--cfg_assets", str(assets_config), "--nodebug",
            ],
        })

    protocol = {
        "name": (
            f"humanml3d-table1-{args.level.lower()}-caption-locked/v1"
            if args.level
            else "humanml3d-table1-sato++-caption-locked-full/v1"
        ),
        "dataset": "humanml3d",
        "samples_per_seed": sample_count,
        "unique_pairs_per_seed": expected_unique_pairs,
        "independent_level": args.level,
        "level_counts": dict(sorted(level_counts.items())),
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
    implementation_paths = (
        Path(__file__),
        Path(__file__).with_name("aggregate_motiongpt_loader_eval.py"),
        Path(__file__).parents[1] / "MotionGPT/eval_loader_robustness.py",
        Path(__file__).parents[1] / "MotionGPT/eval_loader_humanml3d_table1.py",
        Path(__file__).parents[1] / "MotionGPT/scripts/run_humanml3d_loader_eval_shard.sh",
        Path(__file__).parents[1] / "MotionGPT/scripts/run_humanml3d_table1_eval.sh",
        Path(__file__).parents[1] / "dataset/dataset_TM_eval.py",
    )
    missing = [str(path) for path in implementation_paths if not path.is_file()]
    if missing:
        parser.error(f"missing formal evaluation implementation files: {missing}")
    manifest = {
        "schema_version": (
            "motiongpt-humanml3d-table1-single-level-source/v1"
            if args.level
            else "motiongpt-humanml3d-table1-source/v1"
        ),
        "protocol": protocol,
        "model": {
            "name": "MotionGPT",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "selection": (
                "literal terminal last.ckpt at "
                f"global_step={checkpoint_meta['global_step']}"
            ),
            "selection_policy": "last",
            "selection_step": checkpoint_meta["global_step"],
            "checkpoint_metadata": checkpoint_meta,
            "audited_training_selection": str(audited_selection_path),
            "audited_training_selection_sha256": file_sha256(audited_selection_path),
            "stage1_checkpoint_sha256": audited["stage1_checkpoint_sha256"],
            "stage2_checkpoint_sha256": audited["predecessor_checkpoint_sha256"],
            "uses_t2mgpt_generator_checkpoint": False,
            "uses_released_motiongpt_generator_checkpoint": False,
            "token_code_path": token_code_path,
            "expected_checkpoint_load": {
                "missing_key_count": 0,
                "unexpected_key_count": 0,
            },
        },
        "pairs": {
            "path": str(pair_file),
            "sha256": file_sha256(pair_file),
            "rows": len(pairs),
            "unique_pairs": unique_pair_ids,
            "level_counts": dict(sorted(level_counts.items())),
        },
        "fixed_sample_index": {
            "path": str(fixed_index_path),
            "sha256": file_sha256(fixed_index_path),
            "sample_index_sha256": fixed_index["sample_index_sha256"],
            "protocol_fingerprint": fixed_index["protocol_fingerprint"],
            "count": fixed_index["count"],
        },
        "commands": commands,
        "implementation": {
            str(path.resolve()): file_sha256(path.resolve())
            for path in implementation_paths
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
