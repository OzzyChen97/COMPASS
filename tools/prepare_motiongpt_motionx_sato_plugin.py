#!/usr/bin/env python3
"""Freeze the formal MotionGPT SATO++ Motion-X plug-in experiment inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selection(path: Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(value.get("checkpoint", "")).resolve()
    if value.get("split") != "val" or not checkpoint.is_file():
        raise RuntimeError(f"{label} is not a valid validation selection")
    if sha256_file(checkpoint) != value.get("checkpoint_sha256"):
        raise RuntimeError(f"{label} checkpoint hash mismatch")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-selection", type=Path, required=True)
    parser.add_argument("--stage1-selection", type=Path, required=True)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--predecessor-source-manifest", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-max-steps", type=int, default=50000)
    parser.add_argument("--expected-patience", type=int, default=10)
    parser.add_argument(
        "--early-stopping", choices=("enabled", "disabled"), default="enabled"
    )
    parser.add_argument(
        "--checkpoint-policy", choices=("validation_best", "last"),
        default="validation_best",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source_config = args.config.resolve()
    run_root = args.run_root.resolve()
    if run_root.exists() and not args.overwrite:
        parser.error(f"fresh run root already exists: {run_root}")
    baseline = load_selection(args.baseline_selection.resolve(), "MotionGPT baseline")
    stage1 = load_selection(args.stage1_selection.resolve(), "MotionGPT Stage 1")
    pair_file = args.pair_file.resolve()
    baseline_summary = (
        args.baseline_summary.resolve()
        if args.baseline_summary is not None
        else repo_root / "MotionGPT/experiments/robustness/"
        "motionx322_official_fresh_r2_table3_20seed_20260817/combined/summary.json"
    )
    if not pair_file.is_file():
        parser.error(f"missing formal pair file: {pair_file}")
    if not baseline_summary.is_file():
        parser.error(f"missing baseline repeat20 summary: {baseline_summary}")
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))

    if config["DEVICE"] != [0, 1, 2, 3]:
        parser.error("formal plug-in config must use GPUs 0,1,2,3")
    if config["TRAIN"]["STAGE"] != "lm_sato_pp":
        parser.error("formal plug-in config is not lm_sato_pp")
    if Path(config["TRAIN"]["PRETRAINED"]).resolve() != Path(baseline["checkpoint"]).resolve():
        parser.error("plug-in does not initialize from selected MotionGPT baseline")
    if Path(config["TRAIN"]["PRETRAINED_VAE"]).resolve() != Path(stage1["checkpoint"]).resolve():
        parser.error("plug-in does not use selected MotionGPT Stage 1")
    if config["model"]["target"] != "mGPT.models.mgpt_sato_pp.MotionGPTSATOPP":
        parser.error("formal config does not instantiate the SATO++ plug-in")
    if config["CHECKPOINT_SELECTION"]["SPLIT"] != "val":
        parser.error("plug-in checkpoint selection must use val")
    if int(config["CHECKPOINT_SELECTION"].get("EXPECTED_SAMPLES", -1)) != 2552:
        parser.error("formal validation must cover exactly 2552 Motion-X samples")
    if config["SATO"].get("ATTRIBUTION_CREATE_GRAPH") is not True:
        parser.error("formal SATO++ must train through Grad x Input attribution")
    if int(config["SATO"].get("ATTRIBUTION_K", -1)) != 16:
        parser.error("formal SATO++ attribution horizon must be K=16")
    if int(config["SATO"].get("PROJ_DIM", -1)) != 512:
        parser.error("formal SATO++ projection dimension must be 512")
    if config["SATO"].get("DISTRIBUTED_VAR_COV") is not True:
        parser.error("formal SATO++ must compute variance/covariance across DDP ranks")
    if config["SATO"].get("DISTRIBUTED_RELATION_LOSSES") is not True:
        parser.error("formal SATO++ must normalize relation losses across DDP ranks")
    if int(config["TRAIN"].get("MAX_STEPS", -1)) != args.expected_max_steps:
        parser.error(
            f"formal plug-in training must use MAX_STEPS={args.expected_max_steps}"
        )
    if int(config["LOGGER"].get("VAL_EVERY_N_STEPS", -1)) != 5000:
        parser.error("formal plug-in validation interval must be 5000 steps")
    early = config["CHECKPOINT_SELECTION"]["EARLY_STOPPING"]
    early_enabled = bool(early.get("ENABLED", False))
    if args.early_stopping == "enabled":
        if not early_enabled:
            parser.error("formal early stopping must be enabled")
        if (
            int(early.get("PATIENCE", -1)) != args.expected_patience
            or int(early.get("MIN_VALIDATIONS", -1)) != 8
        ):
            parser.error(
                "formal early stopping settings do not match the requested protocol"
            )
    elif early_enabled:
        parser.error("formal continuation must disable early stopping")
    if config["TRAIN"].get("STRICT_LOAD") is not False:
        parser.error("baseline plug-in initialization requires STRICT_LOAD=false")
    configured_pairs = [Path(value).resolve() for value in config["SATO"]["PAIR_DATA_PATHS"]]
    if configured_pairs != [pair_file]:
        parser.error("formal config pair file does not match --pair-file")

    counts = Counter()
    rows = 0
    with pair_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            counts[str(row.get("relation"))] += 1
            rows += 1
    if rows != 210140 or counts != Counter({"preserve": 208939, "alter": 1201}):
        parser.error(f"unexpected formal pair inventory: rows={rows}, counts={dict(counts)}")

    continuation = None
    resume_value = str(config["TRAIN"].get("RESUME", "")).strip()
    if resume_value:
        predecessor_dir = Path(resume_value).resolve()
        predecessor_manifest_path = (
            args.predecessor_source_manifest.resolve()
            if args.predecessor_source_manifest is not None
            else None
        )
        if predecessor_manifest_path is None or not predecessor_manifest_path.is_file():
            parser.error("continuation requires --predecessor-source-manifest")
        predecessor_manifest = json.loads(
            predecessor_manifest_path.read_text(encoding="utf-8")
        )
        recorded_predecessor = Path(
            predecessor_manifest.get("training", {}).get("experiment_dir", "")
        ).resolve()
        if recorded_predecessor != predecessor_dir:
            parser.error("predecessor source manifest experiment mismatch")
        artifacts = {
            "resume_checkpoint": predecessor_dir / "checkpoints/last.ckpt",
            "validation_history": predecessor_dir / "checkpoints/validation_metrics.jsonl",
            "best_summary": predecessor_dir / "checkpoints/best-validation.json",
            "best_checkpoint": predecessor_dir / "checkpoints/best-validation.ckpt",
        }
        missing_artifacts = [str(path) for path in artifacts.values() if not path.is_file()]
        if missing_artifacts:
            parser.error(f"missing continuation artifacts: {missing_artifacts}")
        history_rows = sum(
            1 for line in artifacts["validation_history"].read_text(
                encoding="utf-8").splitlines() if line.strip()
        )
        continuation = {
            "predecessor_experiment_dir": str(predecessor_dir),
            "predecessor_source_manifest": str(predecessor_manifest_path),
            "predecessor_source_manifest_sha256": sha256_file(predecessor_manifest_path),
            "resume_checkpoint": str(artifacts["resume_checkpoint"]),
            "resume_checkpoint_sha256": sha256_file(artifacts["resume_checkpoint"]),
            "validation_history": str(artifacts["validation_history"]),
            "validation_history_sha256": sha256_file(artifacts["validation_history"]),
            "validation_history_rows": history_rows,
            "best_summary": str(artifacts["best_summary"]),
            "best_summary_sha256": sha256_file(artifacts["best_summary"]),
            "best_checkpoint": str(artifacts["best_checkpoint"]),
            "best_checkpoint_sha256": sha256_file(artifacts["best_checkpoint"]),
        }
    elif args.predecessor_source_manifest is not None:
        parser.error("--predecessor-source-manifest requires TRAIN.RESUME")

    run_root.mkdir(parents=True, exist_ok=args.overwrite)
    frozen_config = run_root / "config.yaml"
    shutil.copyfile(source_config, frozen_config)
    tracked = [
        "MotionGPT/mGPT/models/mgpt_sato_pp.py",
        "MotionGPT/mGPT/archs/mgpt_lm.py",
        "MotionGPT/mGPT/data/MotionX322.py",
        "MotionGPT/mGPT/callback.py",
        "MotionGPT/train.py",
        "MotionGPT/eval_loader_robustness.py",
        "MotionGPT/scripts/run_motionx322_table3_eval.sh",
        "tools/aggregate_motiongpt_loader_eval.py",
        "tools/prepare_motiongpt_motionx_sato_plugin.py",
        "tools/select_motiongpt_sato_plugin_best.py",
        "tools/select_motiongpt_sato_plugin_last.py",
        "tools/prepare_motiongpt_motionx_sato_plugin_eval.py",
        "tools/compare_motiongpt_motionx_sato_plugin.py",
    ]
    missing = [relative for relative in tracked if not (repo_root / relative).is_file()]
    if missing:
        parser.error(f"missing plug-in implementation files: {missing}")
    manifest = {
        "schema_version": "motiongpt-motionx322-sato-plugin-source/v1",
        "repo_root": str(repo_root),
        "method": {
            "name": "SATO++ plug-in for MotionGPT",
            "backbone": "selected MotionGPT Motion-X Stage 3",
            "attribution": "early motion-token log-likelihood Grad x Input",
            "uses_t2mgpt_generator_checkpoint": False,
            "uses_motiongpt_baseline_checkpoint": True,
        },
        "hardware": {"physical_gpu_ids": [0, 1, 2, 3]},
        "training": {
            "experiment_name": config["NAME"],
            "experiment_dir": str(
                repo_root / "MotionGPT/experiments/mgpt_sato_pp" / config["NAME"]
            ),
            "stage": "lm_sato_pp",
            "seed": int(config["SEED_VALUE"]),
            "global_batch_size": int(config["TRAIN"]["BATCH_SIZE"]) * 4,
            "max_steps": int(config["TRAIN"]["MAX_STEPS"]),
            "validation_interval": int(config["LOGGER"]["VAL_EVERY_N_STEPS"]),
            "expected_validation_samples": 2552,
            "early_stopping_enabled": early_enabled,
            "early_stopping_patience": int(early["PATIENCE"]),
            "early_stopping_min_validations": int(early["MIN_VALIDATIONS"]),
            "evaluation_checkpoint_policy": args.checkpoint_policy,
            "attribution_k": int(config["SATO"]["ATTRIBUTION_K"]),
            "attribution_create_graph": True,
            "distributed_var_cov": True,
            "distributed_relation_losses": True,
        },
        "config": {"path": str(frozen_config), "sha256": sha256_file(frozen_config)},
        "baseline_selection": {
            "path": str(args.baseline_selection.resolve()),
            "checkpoint": baseline["checkpoint"],
            "checkpoint_sha256": baseline["checkpoint_sha256"],
        },
        "stage1_selection": {
            "path": str(args.stage1_selection.resolve()),
            "checkpoint": stage1["checkpoint"],
            "checkpoint_sha256": stage1["checkpoint_sha256"],
        },
        "pairs": {
            "path": str(pair_file),
            "sha256": sha256_file(pair_file),
            "rows": rows,
            "counts": dict(sorted(counts.items())),
        },
        "source_files": {
            relative: sha256_file(repo_root / relative) for relative in tracked
        },
        "baseline_repeat20_summary": {
            "path": str(baseline_summary),
            "sha256": sha256_file(baseline_summary),
        },
    }
    if continuation is not None:
        manifest["continuation"] = continuation
    output = run_root / "source_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
