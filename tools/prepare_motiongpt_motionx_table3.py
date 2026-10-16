#!/usr/bin/env python3
"""Prepare locked MotionGPT Motion-X Table-3 seed configs and source manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
import zipfile
from pathlib import Path

from omegaconf import OmegaConf


SEEDS = list(range(123, 143))


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
        for value_operation, value_argument, _ in operations[index + 1:index + 8]:
            if value_operation.name in integer_ops:
                values[str(argument)] = int(value_argument)
                break
    return {
        "global_step": int(values.get("global_step", -1)),
        "epoch": int(values.get("epoch", -1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--selection-policy",
        choices=("validation_best", "last"),
        default="validation_best",
        help="evaluate the validation-selected checkpoint or the literal last.ckpt",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("MotionGPT/configs/config_motionx322_stage3.yaml"),
    )
    parser.add_argument(
        "--assets-config",
        type=Path,
        default=Path("MotionGPT/configs/assets_motionx322.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    base_config = args.base_config.resolve()
    assets_config = args.assets_config.resolve()
    for required in (checkpoint, base_config, assets_config):
        if not required.is_file():
            parser.error(f"missing required file: {required}")

    output_dir = args.output_dir.resolve()
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    audited_selection_path = checkpoint.parent / "selected-validation.json"
    if not audited_selection_path.is_file():
        parser.error(
            "formal MotionGPT evaluation requires selected-validation.json with "
            f"the audited dependency chain: {audited_selection_path}"
        )
    audited_selection = json.loads(
        audited_selection_path.read_text(encoding="utf-8")
    )
    if audited_selection.get("stage") != "stage3":
        parser.error("audited selection is not a Stage 3 checkpoint")
    if audited_selection.get("split") != "val":
        parser.error("audited Stage 3 checkpoint was not selected on validation data")
    if audited_selection.get("uses_t2mgpt_generator_checkpoint") is not False:
        parser.error("audited Stage 3 dependency chain is not baseline-free")
    checkpoint_sha256 = file_sha256(checkpoint)
    if args.selection_policy == "validation_best":
        selection_path = checkpoint.parent / "best-validation.json"
        if not selection_path.is_file():
            parser.error(
                "validation-best evaluation requires best-validation.json next to "
                f"the selected checkpoint: {selection_path}"
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("split") != "val":
            parser.error(
                f"checkpoint was not selected on validation data: {selection_path}"
            )
        recorded_checkpoint = Path(selection.get("checkpoint", "")).resolve()
        if recorded_checkpoint != checkpoint:
            parser.error(
                f"selection metadata points to {recorded_checkpoint}, not {checkpoint}"
            )
        audited_checkpoint = Path(
            audited_selection.get("checkpoint", "")
        ).resolve()
        if audited_checkpoint != checkpoint:
            parser.error(
                f"audited selection points to {audited_checkpoint}, not {checkpoint}"
            )
        if audited_selection.get("checkpoint_sha256") != checkpoint_sha256:
            parser.error(
                "Stage 3 checkpoint hash differs from audited selection metadata"
            )
        checkpoint_meta = {
            "global_step": int(selection["step"]),
            "epoch": int(selection.get("epoch", -1)),
        }
        selection_description = (
            "validation min FID; within 1e-4 min Matching; then max R@3"
        )
        selection_metrics = {
            "FID": float(selection["FID"]),
            "Matching": float(selection["Matching"]),
            "R@3": float(selection["R@3"]),
            "samples": int(selection["samples"]),
        }
    else:
        if checkpoint.name != "last.ckpt":
            parser.error("--selection-policy last requires a checkpoint named last.ckpt")
        selection_path = audited_selection_path
        selection = audited_selection
        checkpoint_meta = checkpoint_metadata(checkpoint)
        if checkpoint_meta["global_step"] <= 0:
            parser.error("last.ckpt does not contain a positive global_step")
        selection_description = (
            f"literal terminal last.ckpt at global_step={checkpoint_meta['global_step']}"
        )
        selection_metrics = None
    base = OmegaConf.load(base_config)
    token_code_path = str(base.DATASET.CODE_PATH)
    commands = []
    for seed in SEEDS:
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
        cfg.NAME = f"MotionGPT_MOTIONX322_Table3_{seed}"
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
        config_path = config_dir / f"motiongpt_motionx322_seed{seed}.yaml"
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
        "name": "motionx322-table3-sato++-parity/v1",
        "dataset": "motionx322",
        "samples_per_seed": 7733,
        "pair_matching": "7733/7733 exact motion ids",
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
        "seeds": SEEDS,
    }
    protocol["fingerprint"] = canonical_sha256(protocol)
    manifest = {
        "schema_version": "motiongpt-motionx322-table3-source/v1",
        "protocol": protocol,
        "model": {
            "name": "MotionGPT",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "selection": selection_description,
            "selection_policy": args.selection_policy,
            "selection_step": int(checkpoint_meta["global_step"]),
            "checkpoint_metadata": checkpoint_meta,
            "selection_metrics": selection_metrics,
            "selection_metadata": str(selection_path),
            "selection_metadata_sha256": file_sha256(selection_path),
            "audited_selection": str(audited_selection_path),
            "audited_selection_sha256": file_sha256(audited_selection_path),
            "stage1_checkpoint_sha256": audited_selection["stage1_checkpoint_sha256"],
            "stage2_checkpoint_sha256": audited_selection[
                "predecessor_checkpoint_sha256"
            ],
            "uses_t2mgpt_generator_checkpoint": False,
            "token_code_path": token_code_path,
            "expected_checkpoint_load": {
                "missing_key_count": 0,
                "unexpected_key_count": 0,
            },
        },
        "commands": commands,
        "implementation": {
            str(path.resolve()): file_sha256(path.resolve())
            for path in (
                Path(__file__),
                Path(__file__).parent / "aggregate_motiongpt_loader_eval.py",
                Path(__file__).parents[1] / "MotionGPT/eval_loader_robustness.py",
                Path(__file__).parents[1]
                / "MotionGPT/scripts/run_motionx322_table3_eval.sh",
            )
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
