#!/usr/bin/env python3
"""Audit and materialize a clean MotionGPT SATO++ terminal last checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickletools
import zipfile
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if not expected or sha256_file(path) != expected:
        raise RuntimeError(f"{label} hash mismatch: {path}")


def is_better(candidate: dict, best: dict | None, tolerance: float) -> bool:
    if best is None:
        return True
    if candidate["FID"] < best["FID"] - tolerance:
        return True
    if abs(candidate["FID"] - best["FID"]) <= tolerance:
        if candidate["Matching"] < best["Matching"]:
            return True
        if candidate["Matching"] == best["Matching"] and candidate["R@3"] > best["R@3"]:
            return True
    return False


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
        "validation_callback_state": None,
    }


def select_last_clean(
    experiment_dir: Path,
    source_manifest: Path,
    expected_samples: int,
    validation_interval: int,
    max_steps: int,
    fid_tolerance: float,
) -> dict:
    experiment_dir = experiment_dir.resolve()
    source_manifest = source_manifest.resolve()
    checkpoint_dir = experiment_dir / "checkpoints"
    history_path = checkpoint_dir / "validation_metrics.jsonl"
    checkpoint = checkpoint_dir / "last.ckpt"
    for required in (history_path, checkpoint, source_manifest):
        if not required.is_file():
            raise FileNotFoundError(f"missing terminal SATO++ artifact: {required}")

    manifest = load_json(source_manifest)
    method = manifest.get("method", {})
    training = manifest.get("training", {})
    if method.get("uses_motiongpt_baseline_checkpoint") is not True:
        raise RuntimeError("source manifest lacks MotionGPT baseline initialization")
    if method.get("uses_t2mgpt_generator_checkpoint") is not False:
        raise RuntimeError("source manifest permits a T2M-GPT generator checkpoint")
    if training.get("initialization_policy") != "clean_pretrained_from_motiongpt_last100k":
        raise RuntimeError("source manifest is not a clean last-baseline initialization")
    if training.get("evaluation_checkpoint_policy") != "last":
        raise RuntimeError("source manifest does not freeze last-checkpoint evaluation")
    if training.get("early_stopping_enabled") is not False:
        raise RuntimeError("source manifest does not prove early stopping was disabled")
    if int(training.get("max_steps", -1)) != max_steps:
        raise RuntimeError("source manifest max_steps mismatch")
    if Path(training.get("experiment_dir", "")).resolve() != experiment_dir:
        raise RuntimeError("source manifest experiment directory mismatch")
    if training.get("experiment_name") != experiment_dir.name:
        raise RuntimeError("source manifest experiment name mismatch")

    repo_root = Path(manifest.get("repo_root", "")).resolve()
    for relative, expected_hash in manifest.get("source_files", {}).items():
        require_file_hash(repo_root / relative, expected_hash, f"frozen source {relative}")
    frozen_config = Path(manifest.get("config", {}).get("path", "")).resolve()
    require_file_hash(
        frozen_config,
        manifest.get("config", {}).get("sha256", ""),
        "frozen training config",
    )
    baseline_meta = manifest.get("baseline_selection", {})
    baseline_checkpoint = Path(baseline_meta.get("checkpoint", "")).resolve()
    require_file_hash(
        baseline_checkpoint,
        baseline_meta.get("checkpoint_sha256", ""),
        "MotionGPT baseline checkpoint",
    )
    if baseline_meta.get("selection_policy") != "last":
        raise RuntimeError("baseline provenance is not last-checkpoint")
    if int(baseline_meta.get("selection_step", -1)) != 100000:
        raise RuntimeError("baseline provenance is not step 100000")
    stage1_meta = manifest.get("stage1_selection", {})
    stage1_checkpoint = Path(stage1_meta.get("checkpoint", "")).resolve()
    require_file_hash(
        stage1_checkpoint,
        stage1_meta.get("checkpoint_sha256", ""),
        "MotionGPT Stage 1 checkpoint",
    )
    pair_meta = manifest.get("pairs", {})
    pair_file = Path(pair_meta.get("path", "")).resolve()
    require_file_hash(pair_file, pair_meta.get("sha256", ""), "SATO++ pair file")

    rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_steps = list(range(validation_interval, max_steps + 1, validation_interval))
    actual_steps = [int(row.get("step", -1)) for row in rows]
    if actual_steps != expected_steps:
        raise RuntimeError(
            f"terminal validation sequence mismatch: {actual_steps} != {expected_steps}"
        )
    best = None
    bad_checks = 0
    for index, row in enumerate(rows, start=1):
        for key in ("FID", "Matching", "R@3"):
            value = float(row[key])
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite validation metric at check {index}: {key}")
            row[key] = value
        if row.get("split") != "val":
            raise RuntimeError(f"validation check {index} is not split=val")
        if int(row.get("samples", -1)) != expected_samples:
            raise RuntimeError(f"validation check {index} sample count mismatch")
        improved = is_better(row, best, fid_tolerance)
        bad_checks = 0 if improved else bad_checks + 1
        if bool(row.get("selected")) != improved:
            raise RuntimeError(f"incorrect selected flag at validation check {index}")
        if int(row.get("validation_checks", -1)) != index:
            raise RuntimeError(f"incorrect validation counter at check {index}")
        if int(row.get("bad_validation_checks", -1)) != bad_checks:
            raise RuntimeError(f"incorrect bad-validation counter at check {index}")
        if bool(row.get("early_stop")):
            raise RuntimeError("early_stop must remain false when early stopping is disabled")
        if improved:
            best = dict(row)

    config_paths = sorted(experiment_dir.glob("config_*_train.yaml"))
    if len(config_paths) != 1:
        raise RuntimeError(f"expected one resolved config, found {len(config_paths)}")
    config = yaml.safe_load(config_paths[0].read_text(encoding="utf-8"))
    if config.get("NAME") != experiment_dir.name:
        raise RuntimeError("resolved config experiment name mismatch")
    if config.get("DEVICE") != [0, 1, 2, 3]:
        raise RuntimeError("resolved config did not use physical GPUs 0--3")
    if config.get("TRAIN", {}).get("STAGE") != "lm_sato_pp":
        raise RuntimeError("resolved config is not lm_sato_pp")
    if int(config["TRAIN"].get("MAX_STEPS", -1)) != max_steps:
        raise RuntimeError("resolved config max_steps mismatch")
    if str(config["TRAIN"].get("RESUME", "")).strip():
        raise RuntimeError("clean last-baseline run must not set TRAIN.RESUME")
    if Path(config["TRAIN"].get("PRETRAINED", "")).resolve() != baseline_checkpoint:
        raise RuntimeError("resolved config baseline checkpoint mismatch")
    if Path(config["TRAIN"].get("PRETRAINED_VAE", "")).resolve() != stage1_checkpoint:
        raise RuntimeError("resolved config Stage 1 checkpoint mismatch")
    if bool(
        config.get("CHECKPOINT_SELECTION", {})
        .get("EARLY_STOPPING", {})
        .get("ENABLED", True)
    ):
        raise RuntimeError("resolved config did not disable early stopping")
    if config.get("model", {}).get("target") != "mGPT.models.mgpt_sato_pp.MotionGPTSATOPP":
        raise RuntimeError("resolved config does not instantiate MotionGPTSATOPP")
    sato = config.get("SATO", {})
    if sato.get("ATTRIBUTION_CREATE_GRAPH") is not True:
        raise RuntimeError("resolved config lacks trainable Grad x Input")
    if sato.get("DISTRIBUTED_VAR_COV") is not True:
        raise RuntimeError("resolved config lacks cross-rank variance/covariance")
    if sato.get("DISTRIBUTED_RELATION_LOSSES") is not True:
        raise RuntimeError("resolved config lacks cross-rank relation losses")
    if [Path(value).resolve() for value in sato.get("PAIR_DATA_PATHS", [])] != [pair_file]:
        raise RuntimeError("resolved config pair-file mismatch")

    checkpoint_meta = checkpoint_metadata(checkpoint)
    if checkpoint_meta["global_step"] != max_steps:
        raise RuntimeError(
            f"last checkpoint global_step={checkpoint_meta['global_step']}, expected {max_steps}"
        )
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 == baseline_meta.get("checkpoint_sha256"):
        raise RuntimeError("terminal plug-in checkpoint is byte-identical to baseline")
    terminal = rows[-1]
    return {
        "schema_version": "motiongpt-motionx322-sato-plugin-selection/v2",
        "stage": "sato_plugin",
        "split": "train_terminal",
        "selection_policy": "last",
        "selection": f"terminal last.ckpt at global_step={max_steps}; early stopping disabled",
        "FID": float(terminal["FID"]),
        "Matching": float(terminal["Matching"]),
        "R@3": float(terminal["R@3"]),
        "samples": int(terminal["samples"]),
        "step": max_steps,
        "epoch": int(checkpoint_meta["epoch"]),
        "validation_checks": len(rows),
        "early_stopping_enabled": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_metadata": checkpoint_meta,
        "history": str(history_path),
        "history_sha256": sha256_file(history_path),
        "resolved_config": str(config_paths[0].resolve()),
        "resolved_config_sha256": sha256_file(config_paths[0]),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "baseline_selection": baseline_meta["path"],
        "baseline_checkpoint": str(baseline_checkpoint),
        "baseline_checkpoint_sha256": baseline_meta["checkpoint_sha256"],
        "stage1_selection": stage1_meta["path"],
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage1_checkpoint_sha256": stage1_meta["checkpoint_sha256"],
        "pair_file": str(pair_file),
        "pair_file_sha256": pair_meta["sha256"],
        "pair_rows": int(pair_meta["rows"]),
        "pair_counts": pair_meta["counts"],
        "model_target": config["model"]["target"],
        "attribution": "early motion-token log-likelihood Grad x Input",
        "attribution_k": int(sato["ATTRIBUTION_K"]),
        "attribution_create_graph": True,
        "distributed_var_cov": True,
        "distributed_relation_losses": True,
        "uses_motiongpt_baseline_checkpoint": True,
        "uses_t2mgpt_generator_checkpoint": False,
        "initialization_policy": "clean_pretrained_from_motiongpt_last100k",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=2552)
    parser.add_argument("--validation-interval", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--fid-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = select_last_clean(
        args.experiment_dir,
        args.source_manifest,
        args.expected_samples,
        args.validation_interval,
        args.max_steps,
        args.fid_tolerance,
    )
    output = args.output or (args.experiment_dir / "checkpoints/selected-last.json")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
