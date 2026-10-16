#!/usr/bin/env python3
"""Audit completion and provenance of the fresh HumanML3D MotionGPT retrain."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def require_selection(path: Path, label: str) -> dict:
    value = load_json(path)
    if value.get("split") != "val":
        raise RuntimeError(f"{label} is not validation-selected")
    checkpoint = Path(value.get("checkpoint", "")).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{label} checkpoint is missing: {checkpoint}")
    if sha256_file(checkpoint) != value.get("checkpoint_sha256"):
        raise RuntimeError(f"{label} checkpoint hash mismatch")
    if value.get("uses_t2mgpt_generator_checkpoint") is not False:
        raise RuntimeError(f"{label} does not prove baseline-free provenance")
    return value


def resolved_config(experiment: Path) -> tuple[Path, dict]:
    paths = sorted(experiment.glob("config_*_train.yaml"))
    if len(paths) != 1:
        raise RuntimeError(
            f"expected exactly one resolved config in {experiment}, found {len(paths)}"
        )
    return paths[0], yaml.safe_load(paths[0].read_text(encoding="utf-8"))


def audit(repo_root: Path, run_root: Path) -> dict:
    repo_root = repo_root.resolve()
    run_root = run_root.resolve()
    manifest_path = run_root / "source_manifest.json"
    manifest = load_json(manifest_path)
    method = manifest.get("method", {})
    for key in (
        "uses_t2mgpt_generator_checkpoint",
        "uses_released_motiongpt_generator_checkpoint",
    ):
        if method.get(key) is not False:
            raise RuntimeError(f"manifest does not prove {key}=false")
    if manifest.get("hardware", {}).get("physical_gpu_ids") != [0, 1, 2, 3]:
        raise RuntimeError("manifest does not lock physical GPUs 0,1,2,3")
    if manifest.get("early_stopping", {}).get("test_data_used_for_selection") is not False:
        raise RuntimeError("manifest permits test-data model selection")
    for relative, expected in manifest["source_files"].items():
        path = repo_root / relative
        if sha256_file(path) != expected:
            raise RuntimeError(f"tracked training source drift: {relative}")
    source_configs = {}
    for stage, item in manifest["configs"].items():
        path = Path(item["path"]).resolve()
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"generated config drift: {stage}")
        source_configs[stage] = yaml.safe_load(path.read_text(encoding="utf-8"))
    for stage in ("stage1", "stage2", "stage3"):
        cfg = source_configs[stage]
        if cfg["DEVICE"] != [0, 1, 2, 3]:
            raise RuntimeError(f"{stage} does not use four configured devices")
        if cfg["EVAL"]["SPLIT"] != "val":
            raise RuntimeError(f"{stage} selection split is not val")
        if cfg["TRAIN"].get("PRETRAINED") or cfg["TRAIN"].get("PRETRAINED_VAE"):
            raise RuntimeError(f"{stage} source config embeds a generator checkpoint")
    if source_configs["stage1"]["EARLY_STOPPING"] != {
        "ENABLED": True, "MONITOR": "total/val", "MODE": "min",
        "MIN_DELTA": 0.0001, "PATIENCE": 10,
    }:
        raise RuntimeError("Stage 1 early-stopping contract changed")
    for stage, patience in (("stage2", 5), ("stage3", 6)):
        selection = source_configs[stage]["CHECKPOINT_SELECTION"]
        if selection["SPLIT"] != "val" or selection["DATASET"] != "t2m":
            raise RuntimeError(f"{stage} external selector is not HumanML3D val")
        if selection["EXPECTED_SAMPLES"] != 1504:
            raise RuntimeError(f"{stage} validation coverage is not 1504")
        if selection["EARLY_STOPPING"] != {
            "ENABLED": True, "PATIENCE": patience, "MIN_VALIDATIONS": 8,
        }:
            raise RuntimeError(f"{stage} early-stopping contract changed")

    names = manifest["names"]
    experiment_root = repo_root / "MotionGPT/experiments/mgpt"
    experiments = {
        stage: experiment_root / names[stage] for stage in ("stage1", "stage2", "stage3")
    }
    stage1_path = experiments["stage1"] / "checkpoints/best-validation.json"
    stage2_path = experiments["stage2"] / "checkpoints/selected-validation.json"
    stage3_path = experiments["stage3"] / "checkpoints/selected-validation.json"
    stage1 = require_selection(stage1_path, "stage1")
    stage2 = require_selection(stage2_path, "stage2")
    stage3 = require_selection(stage3_path, "stage3")
    if stage1.get("monitor") != "total/val" or stage1.get("mode") != "min":
        raise RuntimeError("Stage 1 was not selected by minimum validation loss")
    for label, selection in (("stage2", stage2), ("stage3", stage3)):
        if selection.get("samples") != 1504:
            raise RuntimeError(f"{label} did not cover exactly 1504 validation examples")
        if not selection.get("early_stopped") and selection.get("terminal_step") not in (200000, 100000):
            raise RuntimeError(f"{label} is not terminal")

    resolved = {}
    for stage, experiment in experiments.items():
        path, cfg = resolved_config(experiment)
        resolved[stage] = {"path": path, "config": cfg}
    if resolved["stage1"]["config"]["TRAIN"].get("PRETRAINED") or resolved["stage1"]["config"]["TRAIN"].get("PRETRAINED_VAE"):
        raise RuntimeError("resolved Stage 1 reused a generator checkpoint")
    expected_stage1 = str(Path(stage1["checkpoint"]).resolve())
    if resolved["stage2"]["config"]["TRAIN"]["PRETRAINED_VAE"] != expected_stage1:
        raise RuntimeError("resolved Stage 2 did not use selected fresh Stage 1")
    if resolved["stage2"]["config"]["TRAIN"].get("PRETRAINED"):
        raise RuntimeError("resolved Stage 2 reused an LM generator checkpoint")
    if resolved["stage2"]["config"]["lm"]["default"]["params"]["model_path"] != "./deps/flan-t5-base":
        raise RuntimeError("Stage 2 did not use official flan-t5-base initialization")
    if resolved["stage3"]["config"]["TRAIN"]["PRETRAINED_VAE"] != expected_stage1:
        raise RuntimeError("resolved Stage 3 did not use selected fresh Stage 1")
    if resolved["stage3"]["config"]["TRAIN"]["PRETRAINED"] != str(Path(stage2["checkpoint"]).resolve()):
        raise RuntimeError("resolved Stage 3 did not use selected fresh Stage 2")

    token_audit_path = run_root / "token_audit.json"
    token_audit = load_json(token_audit_path)
    if token_audit.get("status") != "pass":
        raise RuntimeError("fresh token audit did not pass")
    if token_audit.get("expected_count") != 23726 or token_audit.get("actual_count") != 23726:
        raise RuntimeError("fresh token coverage is not exactly 23,726 motions")
    if any(token_audit.get(key) for key in ("missing_count", "extra_count", "invalid_token_count")):
        raise RuntimeError("fresh token audit contains coverage/content failures")
    if token_audit.get("stage1_checkpoint_sha256") != stage1["checkpoint_sha256"]:
        raise RuntimeError("tokens are not bound to selected fresh Stage 1")

    for stage in ("stage1", "stage2", "stage3"):
        log_path = run_root / f"{stage}_launcher.log"
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "Training ends!" not in text:
            raise RuntimeError(f"{stage} launcher log is not complete")
        if any(marker in text for marker in ("Traceback", "CUDA out of memory", "non-finite validation")):
            raise RuntimeError(f"{stage} launcher log contains a fatal marker")

    return {
        "schema_version": "motiongpt-humanml3d-official-completion/v1",
        "status": "pass",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "token_audit": str(token_audit_path),
        "token_audit_sha256": sha256_file(token_audit_path),
        "selected_checkpoints": {
            "stage1": {"path": stage1["checkpoint"], "sha256": stage1["checkpoint_sha256"]},
            "stage2": {"path": stage2["checkpoint"], "sha256": stage2["checkpoint_sha256"]},
            "stage3": {"path": stage3["checkpoint"], "sha256": stage3["checkpoint_sha256"]},
        },
        "validation": {
            "stage1_monitor": "total/val",
            "lm_samples_per_check": 1504,
            "test_used_for_selection": False,
        },
        "generator_checkpoint_reuse": False,
        "physical_gpu_ids": [0, 1, 2, 3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.repo_root, args.run_root)
    args.output.resolve().write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
