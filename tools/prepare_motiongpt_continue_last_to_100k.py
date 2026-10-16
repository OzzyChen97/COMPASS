#!/usr/bin/env python3
"""Prepare MotionGPT stage-3 continuation configs from literal last.ckpt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    base_config = args.base_config.resolve()
    experiment_dir = args.experiment_dir.resolve()
    checkpoint = experiment_dir / "checkpoints" / "last.ckpt"
    if not base_config.is_file():
        parser.error(f"missing base config: {base_config}")
    if not checkpoint.is_file():
        parser.error(f"missing last checkpoint: {checkpoint}")

    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(base_config), resolve=False))
    cfg.DEVICE = list(args.devices)
    cfg.TRAIN.MAX_STEPS = int(args.max_steps)
    cfg.TRAIN.RESUME = str(experiment_dir)
    cfg.TRAIN.PRETRAINED = ""
    if "CHECKPOINT_SELECTION" in cfg and "EARLY_STOPPING" in cfg.CHECKPOINT_SELECTION:
        cfg.CHECKPOINT_SELECTION.EARLY_STOPPING.ENABLED = False
    cfg.LOGGER.VAL_EVERY_N_STEPS = int(cfg.LOGGER.get("VAL_EVERY_N_STEPS", 5000))
    cfg.DEBUG = False

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output_config)

    manifest = {
        "schema_version": "motiongpt-stage3-continue-last-to-100k/v1",
        "base_config": str(base_config),
        "base_config_sha256": sha256_file(base_config),
        "output_config": str(args.output_config.resolve()),
        "output_config_sha256": sha256_file(args.output_config),
        "experiment_dir": str(experiment_dir),
        "resume_checkpoint": str(checkpoint),
        "resume_checkpoint_sha256": sha256_file(checkpoint),
        "max_steps": int(args.max_steps),
        "devices": list(args.devices),
        "early_stopping_enabled": bool(
            cfg.get("CHECKPOINT_SELECTION", {})
            .get("EARLY_STOPPING", {})
            .get("ENABLED", True)
        ),
        "checkpoint_policy": "resume literal checkpoints/last.ckpt; evaluate terminal last.ckpt",
    }
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output_manifest)


if __name__ == "__main__":
    main()
