#!/usr/bin/env python3
"""Prepare a clean MotionGPT SATO++ Motion-X plug-in run from baseline last.ckpt."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--stage1-selection", type=Path, required=True)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--expected-baseline-step", type=int, default=100000)
    parser.add_argument("--expected-patience", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    template_config = args.template_config.resolve()
    baseline_checkpoint = args.baseline_checkpoint.resolve()
    baseline_manifest_path = args.baseline_manifest.resolve()
    baseline_summary = args.baseline_summary.resolve()
    stage1_selection_path = args.stage1_selection.resolve()
    pair_file = args.pair_file.resolve()
    run_root = args.run_root.resolve()
    for required in (
        repo_root,
        template_config,
        baseline_checkpoint,
        baseline_manifest_path,
        baseline_summary,
        stage1_selection_path,
        pair_file,
    ):
        if not required.exists():
            parser.error(f"missing required path: {required}")
    if run_root.exists() and any(run_root.iterdir()) and not args.overwrite:
        parser.error(f"fresh run root already exists and is non-empty: {run_root}")

    baseline_manifest = load_json(baseline_manifest_path)
    baseline_model = baseline_manifest.get("model", {})
    if baseline_model.get("name") != "MotionGPT":
        parser.error("baseline manifest is not MotionGPT")
    if baseline_model.get("selection_policy") != "last":
        parser.error("baseline manifest is not a last-checkpoint evaluation")
    if baseline_model.get("uses_t2mgpt_generator_checkpoint") is not False:
        parser.error("baseline manifest permits a T2M-GPT generator checkpoint")
    if Path(baseline_model.get("checkpoint", "")).resolve() != baseline_checkpoint:
        parser.error("baseline manifest checkpoint path differs from --baseline-checkpoint")
    baseline_sha = sha256_file(baseline_checkpoint)
    if baseline_model.get("checkpoint_sha256") != baseline_sha:
        parser.error("baseline checkpoint hash differs from baseline manifest")
    baseline_meta = checkpoint_metadata(baseline_checkpoint)
    if baseline_meta["global_step"] != args.expected_baseline_step:
        parser.error(
            f"baseline last global_step={baseline_meta['global_step']}, "
            f"expected {args.expected_baseline_step}"
        )
    if int(baseline_model.get("selection_step", -1)) != args.expected_baseline_step:
        parser.error("baseline manifest selection_step mismatch")
    if baseline_model.get("stage1_checkpoint_sha256") is None:
        parser.error("baseline manifest lacks Stage-1 checkpoint provenance")

    baseline_summary_payload = load_json(baseline_summary)
    if (
        baseline_summary_payload.get("dataset") != "motionx322"
        or int(baseline_summary_payload.get("n", -1)) != 20
        or int(baseline_summary_payload.get("samples_per_seed", -1)) != 7733
    ):
        parser.error("baseline summary is not the locked 20-seed Motion-X Table-3 run")
    if baseline_summary_payload.get("identities", {}).get("checkpoint") != baseline_sha:
        parser.error("baseline summary checkpoint identity differs from manifest")

    stage1 = load_json(stage1_selection_path)
    stage1_checkpoint = Path(stage1.get("checkpoint", "")).resolve()
    if stage1.get("split") != "val" or not stage1_checkpoint.is_file():
        parser.error("stage1 selection is not a validation-selected checkpoint")
    stage1_sha = sha256_file(stage1_checkpoint)
    if stage1.get("checkpoint_sha256") != stage1_sha:
        parser.error("stage1 checkpoint hash mismatch")
    if stage1_sha != baseline_model.get("stage1_checkpoint_sha256"):
        parser.error("stage1 checkpoint differs from baseline manifest")

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
        parser.error(f"unexpected pair inventory: rows={rows}, counts={dict(counts)}")

    config = yaml.safe_load(template_config.read_text(encoding="utf-8"))
    config["NAME"] = args.experiment_name
    config["DEVICE"] = [0, 1, 2, 3]
    config["DEBUG"] = False
    config["TRAIN"]["STAGE"] = "lm_sato_pp"
    config["TRAIN"]["MAX_STEPS"] = int(args.max_steps)
    config["TRAIN"]["RESUME"] = ""
    config["TRAIN"]["PRETRAINED"] = str(baseline_checkpoint)
    config["TRAIN"]["PRETRAINED_VAE"] = str(stage1_checkpoint)
    config["TRAIN"]["STRICT_LOAD"] = False
    config["CHECKPOINT_SELECTION"]["ENABLED"] = True
    config["CHECKPOINT_SELECTION"]["SPLIT"] = "val"
    config["CHECKPOINT_SELECTION"]["DATASET"] = "motionx322_75"
    config["CHECKPOINT_SELECTION"]["EXPECTED_SAMPLES"] = 2552
    config["CHECKPOINT_SELECTION"]["EARLY_STOPPING"]["ENABLED"] = False
    config["CHECKPOINT_SELECTION"]["EARLY_STOPPING"]["PATIENCE"] = int(
        args.expected_patience
    )
    config["CHECKPOINT_SELECTION"]["EARLY_STOPPING"]["MIN_VALIDATIONS"] = 8
    config["LOGGER"]["VAL_EVERY_N_STEPS"] = 5000
    config["SATO"]["PAIR_DATA_PATHS"] = [str(pair_file)]
    config["SATO"]["ATTRIBUTION_CREATE_GRAPH"] = True
    config["SATO"]["ATTRIBUTION_K"] = 16
    config["SATO"]["PROJ_DIM"] = 512
    config["SATO"]["DISTRIBUTED_VAR_COV"] = True
    config["SATO"]["DISTRIBUTED_RELATION_LOSSES"] = True
    if config["model"]["target"] != "mGPT.models.mgpt_sato_pp.MotionGPTSATOPP":
        parser.error("template config does not instantiate MotionGPTSATOPP")

    run_root.mkdir(parents=True, exist_ok=True)
    frozen_config = run_root / "config.yaml"
    with frozen_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    tracked = [
        "MotionGPT/mGPT/models/mgpt_sato_pp.py",
        "MotionGPT/mGPT/archs/mgpt_lm.py",
        "MotionGPT/mGPT/data/MotionX322.py",
        "MotionGPT/mGPT/callback.py",
        "MotionGPT/train.py",
        "MotionGPT/eval_loader_robustness.py",
        "MotionGPT/scripts/run_motionx322_table3_eval.sh",
        "tools/aggregate_motiongpt_loader_eval.py",
        "tools/prepare_motiongpt_motionx_sato_plugin_last_baseline.py",
        "tools/select_motiongpt_sato_plugin_last_clean.py",
        "tools/prepare_motiongpt_motionx_sato_plugin_eval.py",
        "tools/compare_motiongpt_motionx_sato_plugin.py",
    ]
    missing = [relative for relative in tracked if not (repo_root / relative).is_file()]
    if missing:
        parser.error(f"missing plug-in implementation files: {missing}")

    manifest = {
        "schema_version": "motiongpt-motionx322-sato-plugin-last-baseline-source/v1",
        "repo_root": str(repo_root),
        "method": {
            "name": "SATO++ plug-in for MotionGPT",
            "backbone": "MotionGPT Motion-X Stage 3 literal last checkpoint",
            "attribution": "early motion-token log-likelihood Grad x Input",
            "uses_t2mgpt_generator_checkpoint": False,
            "uses_motiongpt_baseline_checkpoint": True,
        },
        "hardware": {"physical_gpu_ids": [0, 1, 2, 3]},
        "training": {
            "experiment_name": args.experiment_name,
            "experiment_dir": str(
                repo_root / "MotionGPT/experiments/mgpt_sato_pp" / args.experiment_name
            ),
            "stage": "lm_sato_pp",
            "seed": int(config["SEED_VALUE"]),
            "global_batch_size": int(config["TRAIN"]["BATCH_SIZE"]) * 4,
            "max_steps": int(args.max_steps),
            "validation_interval": 5000,
            "expected_validation_samples": 2552,
            "early_stopping_enabled": False,
            "early_stopping_patience": int(args.expected_patience),
            "early_stopping_min_validations": 8,
            "evaluation_checkpoint_policy": "last",
            "initialization_policy": "clean_pretrained_from_motiongpt_last100k",
            "attribution_k": 16,
            "attribution_create_graph": True,
            "distributed_var_cov": True,
            "distributed_relation_losses": True,
        },
        "config": {"path": str(frozen_config), "sha256": sha256_file(frozen_config)},
        "baseline_selection": {
            "path": str(baseline_manifest_path),
            "checkpoint": str(baseline_checkpoint),
            "checkpoint_sha256": baseline_sha,
            "checkpoint_metadata": baseline_meta,
            "selection_policy": "last",
            "selection_step": args.expected_baseline_step,
        },
        "stage1_selection": {
            "path": str(stage1_selection_path),
            "checkpoint": str(stage1_checkpoint),
            "checkpoint_sha256": stage1_sha,
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
    manifest_path = run_root / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copystat(template_config, frozen_config)
    print(manifest_path)


if __name__ == "__main__":
    main()
