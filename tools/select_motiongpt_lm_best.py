#!/usr/bin/env python3
"""Audit and materialize a validation-selected MotionGPT LM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_better(candidate: dict, best: dict | None, tolerance: float) -> bool:
    if best is None:
        return True
    if candidate["FID"] < best["FID"] - tolerance:
        return True
    if abs(candidate["FID"] - best["FID"]) <= tolerance:
        if candidate["Matching"] < best["Matching"]:
            return True
        if (
            candidate["Matching"] == best["Matching"]
            and candidate["R@3"] > best["R@3"]
        ):
            return True
    return False


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_hashed_checkpoint(selection_path: Path, label: str) -> dict:
    selection = load_json(selection_path)
    checkpoint = Path(selection.get("checkpoint", "")).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{label} checkpoint is missing: {checkpoint}")
    expected = selection.get("checkpoint_sha256")
    if not expected or sha256_file(checkpoint) != expected:
        raise RuntimeError(f"{label} checkpoint hash does not match selection metadata")
    if selection.get("split") != "val":
        raise RuntimeError(f"{label} was not selected on validation data")
    return selection


def select_lm_best(
    experiment_dir: Path,
    source_manifest: Path,
    stage: str,
    stage1_selection: Path,
    predecessor_selection: Path | None,
    expected_samples: int,
    validation_interval: int,
    max_steps: int,
    patience: int,
    min_validations: int,
    fid_tolerance: float,
) -> dict:
    experiment_dir = experiment_dir.resolve()
    checkpoint_dir = experiment_dir / "checkpoints"
    history_path = checkpoint_dir / "validation_metrics.jsonl"
    summary_path = checkpoint_dir / "best-validation.json"
    checkpoint = checkpoint_dir / "best-validation.ckpt"
    for required in (history_path, summary_path, checkpoint, source_manifest):
        if not required.is_file():
            raise FileNotFoundError(f"missing LM selection artifact: {required}")

    manifest = load_json(source_manifest.resolve())
    if manifest.get("method", {}).get("uses_t2mgpt_generator_checkpoint") is not False:
        raise RuntimeError("source manifest does not prove baseline-free initialization")
    name_key = "stage2" if stage == "stage2" else "stage3"
    if manifest.get("names", {}).get(name_key) != experiment_dir.name:
        raise RuntimeError("source manifest LM experiment name mismatch")

    stage1 = require_hashed_checkpoint(stage1_selection.resolve(), "stage1")
    predecessor = None
    if stage == "stage3":
        if predecessor_selection is None:
            raise RuntimeError("stage3 selection requires a stage2 predecessor selection")
        predecessor = require_hashed_checkpoint(
            predecessor_selection.resolve(), "stage2 predecessor"
        )
    elif predecessor_selection is not None:
        raise RuntimeError("stage2 selection must not have an LM predecessor")

    rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < min_validations:
        raise RuntimeError(
            f"LM run has only {len(rows)} validations; minimum is {min_validations}"
        )

    best = None
    bad_checks = 0
    previous_step = 0
    for index, row in enumerate(rows, start=1):
        for key in ("FID", "Matching", "R@3"):
            value = float(row[key])
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite validation metric at check {index}: {key}")
            row[key] = value
        step = int(row["step"])
        if step <= previous_step or step % validation_interval != 0:
            raise RuntimeError(f"invalid validation step sequence at check {index}: {step}")
        if int(row.get("samples", -1)) != expected_samples:
            raise RuntimeError(
                f"validation check {index} covers {row.get('samples')} samples, "
                f"expected {expected_samples}"
            )
        if row.get("split") != "val":
            raise RuntimeError(f"validation check {index} is not split=val")

        improved = is_better(row, best, fid_tolerance)
        bad_checks = 0 if improved else bad_checks + 1
        expected_stop = index >= min_validations and bad_checks >= patience
        if bool(row.get("selected")) != improved:
            raise RuntimeError(f"incorrect selected flag at validation check {index}")
        if int(row.get("validation_checks", -1)) != index:
            raise RuntimeError(f"incorrect validation_checks counter at check {index}")
        if int(row.get("bad_validation_checks", -1)) != bad_checks:
            raise RuntimeError(f"incorrect bad-validation counter at check {index}")
        if bool(row.get("early_stop")) != expected_stop:
            raise RuntimeError(f"incorrect early-stop flag at validation check {index}")
        if expected_stop and index != len(rows):
            raise RuntimeError("validation history continues after early stopping")
        if improved:
            best = dict(row)
        previous_step = step

    if not bool(rows[-1].get("early_stop")) and int(rows[-1]["step"]) != max_steps:
        raise RuntimeError(
            "LM run is not terminal: neither validation early stopping nor max_steps reached"
        )

    summary = load_json(summary_path)
    if Path(summary.get("checkpoint", "")).resolve() != checkpoint.resolve():
        raise RuntimeError("best-validation summary points outside the fresh experiment")
    for key in ("step", "epoch", "samples"):
        if int(summary[key]) != int(best[key]):
            raise RuntimeError(f"best-validation summary mismatch for {key}")
    for key in ("FID", "Matching", "R@3"):
        if float(summary[key]) != float(best[key]):
            raise RuntimeError(f"best-validation summary mismatch for {key}")
    if summary.get("split") != "val":
        raise RuntimeError("best-validation summary is not validation-selected")

    config_paths = sorted(experiment_dir.glob("config_*_train.yaml"))
    if len(config_paths) != 1:
        raise RuntimeError(f"expected one resolved training config, found {len(config_paths)}")
    config = yaml.safe_load(config_paths[0].read_text(encoding="utf-8"))
    expected_vae = str(Path(stage1["checkpoint"]).resolve())
    if str(config["TRAIN"].get("PRETRAINED_VAE", "")) != expected_vae:
        raise RuntimeError("resolved LM config does not use selected fresh Stage 1")
    if stage == "stage2":
        if str(config["TRAIN"].get("PRETRAINED", "")):
            raise RuntimeError("Stage 2 reused a pretrained generator checkpoint")
        if config["TRAIN"].get("STAGE") != "lm_pretrain":
            raise RuntimeError("Stage 2 resolved config is not lm_pretrain")
        if config["lm"]["default"]["params"].get("model_path") != "./deps/flan-t5-base":
            raise RuntimeError("Stage 2 does not use the official flan-t5-base initialization")
    else:
        expected_predecessor = str(Path(predecessor["checkpoint"]).resolve())
        if str(config["TRAIN"].get("PRETRAINED", "")) != expected_predecessor:
            raise RuntimeError("Stage 3 does not use selected fresh Stage 2")
        if config["TRAIN"].get("STAGE") != "lm_instruct":
            raise RuntimeError("Stage 3 resolved config is not lm_instruct")

    payload = {
        "schema_version": "motiongpt-motionx322-lm-selection/v1",
        "stage": stage,
        "split": "val",
        "selection": "min FID; within 1e-4 min Matching; then max R@3",
        "FID": float(best["FID"]),
        "Matching": float(best["Matching"]),
        "R@3": float(best["R@3"]),
        "samples": int(best["samples"]),
        "step": int(best["step"]),
        "epoch": int(best["epoch"]),
        "validation_checks": len(rows),
        "terminal_step": int(rows[-1]["step"]),
        "early_stopped": bool(rows[-1]["early_stop"]),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "history": str(history_path.resolve()),
        "history_sha256": sha256_file(history_path),
        "resolved_config": str(config_paths[0].resolve()),
        "resolved_config_sha256": sha256_file(config_paths[0]),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest.resolve()),
        "stage1_selection": str(stage1_selection.resolve()),
        "stage1_checkpoint": stage1["checkpoint"],
        "stage1_checkpoint_sha256": stage1["checkpoint_sha256"],
        "uses_t2mgpt_generator_checkpoint": False,
    }
    if predecessor is not None:
        payload.update({
            "predecessor_selection": str(predecessor_selection.resolve()),
            "predecessor_checkpoint": predecessor["checkpoint"],
            "predecessor_checkpoint_sha256": predecessor["checkpoint_sha256"],
        })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("stage2", "stage3"), required=True)
    parser.add_argument("--stage1-selection", type=Path, required=True)
    parser.add_argument("--predecessor-selection", type=Path)
    parser.add_argument("--expected-samples", type=int, default=2552)
    parser.add_argument("--validation-interval", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--min-validations", type=int, required=True)
    parser.add_argument("--fid-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = select_lm_best(
        args.experiment_dir,
        args.source_manifest,
        args.stage,
        args.stage1_selection,
        args.predecessor_selection,
        args.expected_samples,
        args.validation_interval,
        args.max_steps,
        args.patience,
        args.min_validations,
        args.fid_tolerance,
    )
    output = args.output or (
        args.experiment_dir / "checkpoints/selected-validation.json"
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
