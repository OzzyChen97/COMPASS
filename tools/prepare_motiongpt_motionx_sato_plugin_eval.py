#!/usr/bin/env python3
"""Prepare locked 20-seed Motion-X evaluation for an audited SATO++ checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


SEEDS = list(range(123, 143))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plugin-selection", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument(
        "--assets-config",
        type=Path,
        default=Path("MotionGPT/configs/assets_motionx322.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    selection_path = args.plugin_selection.resolve()
    source_manifest_path = args.source_manifest.resolve()
    baseline_manifest_path = args.baseline_manifest.resolve()
    base_config = args.base_config.resolve()
    assets_config = args.assets_config.resolve()
    for required in (
        checkpoint,
        selection_path,
        source_manifest_path,
        baseline_manifest_path,
        base_config,
        assets_config,
    ):
        if not required.is_file():
            parser.error(f"missing required file: {required}")

    selection = load_json(selection_path)
    source = load_json(source_manifest_path)
    baseline_manifest = load_json(baseline_manifest_path)
    selection_policy = selection.get("selection_policy", "validation_best")
    if selection.get("stage") != "sato_plugin":
        parser.error("checkpoint metadata is not for a SATO++ plug-in")
    if selection_policy == "validation_best":
        if selection.get("split") != "val":
            parser.error("validation-best checkpoint metadata is not split=val")
    elif selection_policy == "last":
        if selection.get("split") != "train_terminal":
            parser.error("last-checkpoint metadata is not terminal training state")
        if selection.get("early_stopping_enabled") is not False:
            parser.error("last-checkpoint protocol must disable early stopping")
    else:
        parser.error(f"unsupported checkpoint selection policy: {selection_policy}")
    if Path(selection.get("checkpoint", "")).resolve() != checkpoint:
        parser.error("plug-in selection points to a different checkpoint")
    checkpoint_sha256 = file_sha256(checkpoint)
    if selection.get("checkpoint_sha256") != checkpoint_sha256:
        parser.error("plug-in checkpoint hash differs from selection metadata")
    if selection.get("model_target") != "mGPT.models.mgpt_sato_pp.MotionGPTSATOPP":
        parser.error("selected checkpoint is not a MotionGPTSATOPP plug-in")
    if selection.get("uses_motiongpt_baseline_checkpoint") is not True:
        parser.error("selection does not prove MotionGPT baseline initialization")
    if selection.get("uses_t2mgpt_generator_checkpoint") is not False:
        parser.error("selection permits a T2M-GPT generator checkpoint")
    if selection.get("attribution_create_graph") is not True:
        parser.error("selection does not prove trainable Grad x Input attribution")
    if selection.get("distributed_var_cov") is not True:
        parser.error("selection does not prove cross-rank variance/covariance training")
    if selection.get("distributed_relation_losses") is not True:
        parser.error("selection does not prove cross-rank relation-loss normalization")
    if selection.get("source_manifest_sha256") != file_sha256(source_manifest_path):
        parser.error("plug-in source manifest hash differs from selection metadata")

    method = source.get("method", {})
    if method.get("uses_motiongpt_baseline_checkpoint") is not True:
        parser.error("source manifest lacks baseline plug-in provenance")
    if source.get("config", {}).get("sha256") != file_sha256(base_config):
        parser.error("evaluation base config is not the frozen training config")
    if source.get("training", {}).get(
        "evaluation_checkpoint_policy"
    ) != selection_policy:
        parser.error("training source checkpoint policy differs from selection metadata")
    repo_root = Path(source.get("repo_root", "")).resolve()
    for relative, expected_hash in source.get("source_files", {}).items():
        path = repo_root / relative
        if not path.is_file() or file_sha256(path) != expected_hash:
            parser.error(f"frozen plug-in source changed after training: {relative}")
    if source.get("pairs", {}).get("sha256") != selection.get("pair_file_sha256"):
        parser.error("pair-file provenance differs between source and selection")
    if selection.get("baseline_checkpoint_sha256") != source.get(
        "baseline_selection", {}
    ).get("checkpoint_sha256"):
        parser.error("baseline checkpoint provenance differs between source and selection")

    protocol = copy.deepcopy(baseline_manifest.get("protocol", {}))
    if protocol.get("seeds") != SEEDS:
        parser.error("baseline manifest does not lock seeds 123--142")
    if int(protocol.get("samples_per_seed", -1)) != 7733:
        parser.error("baseline manifest does not lock 7733 Motion-X samples per seed")
    baseline_model = baseline_manifest.get("model", {})
    baseline_checkpoint_sha256 = baseline_model.get("checkpoint_sha256")
    if baseline_checkpoint_sha256 != selection.get("baseline_checkpoint_sha256"):
        parser.error("plug-in was not initialized from the evaluated MotionGPT baseline")
    if checkpoint_sha256 == baseline_checkpoint_sha256:
        parser.error("plug-in checkpoint is byte-identical to the baseline checkpoint")
    shared_suffixes = (
        "MotionGPT/eval_loader_robustness.py",
        "MotionGPT/scripts/run_motionx322_table3_eval.sh",
        "tools/aggregate_motiongpt_loader_eval.py",
    )
    baseline_implementation = baseline_manifest.get("implementation", {})
    for suffix in shared_suffixes:
        matches = [
            (Path(path_text), expected_hash)
            for path_text, expected_hash in baseline_implementation.items()
            if str(path_text).endswith(suffix)
        ]
        if len(matches) != 1:
            parser.error(f"baseline manifest lacks one frozen {suffix}")
        path, expected_hash = matches[0]
        if not path.is_file() or file_sha256(path) != expected_hash:
            parser.error(f"baseline shared evaluation implementation changed: {suffix}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"evaluation output directory is not fresh: {output_dir}")
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    base = OmegaConf.load(base_config)
    if base.model.target != "mGPT.models.mgpt_sato_pp.MotionGPTSATOPP":
        parser.error("evaluation base config does not instantiate MotionGPTSATOPP")
    if base.TRAIN.STAGE != "lm_sato_pp":
        parser.error("evaluation base config is not lm_sato_pp")
    if base.SATO.DISTRIBUTED_VAR_COV is not True:
        parser.error("evaluation base config lacks cross-rank variance/covariance provenance")
    if base.SATO.DISTRIBUTED_RELATION_LOSSES is not True:
        parser.error("evaluation base config lacks cross-rank relation-loss provenance")
    token_code_path = str(base.DATASET.CODE_PATH)
    commands = []
    for seed in SEEDS:
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
        cfg.NAME = f"MotionGPT_SATO_PP_MOTIONX322_Table3_{seed}"
        cfg.DEVICE = [0]
        cfg.DEBUG = False
        cfg.SEED_VALUE = seed
        cfg.TRAIN.RESUME = ""
        cfg.TRAIN.PRETRAINED = ""
        cfg.TRAIN.PRETRAINED_VAE = ""
        cfg.TRAIN.STRICT_LOAD = True
        cfg.TEST.CHECKPOINTS = str(checkpoint)
        cfg.TEST.SPLIT = "test"
        cfg.TEST.BATCH_SIZE = 32
        cfg.DATASET.CODE_PATH = token_code_path
        cfg.CHECKPOINT_SELECTION.ENABLED = False
        config_path = config_dir / f"motiongpt_sato_pp_motionx322_seed{seed}.yaml"
        OmegaConf.save(cfg, config_path)
        commands.append({
            "seed": seed,
            "config": str(config_path),
            "command": [
                "python", "test.py", "--cfg", str(config_path),
                "--cfg_assets", str(assets_config), "--nodebug",
            ],
        })

    current_repo_root = Path(__file__).resolve().parents[1]
    implementation_paths = (
        Path(__file__),
        current_repo_root / "tools/aggregate_motiongpt_loader_eval.py",
        current_repo_root / "tools/compare_motiongpt_motionx_sato_plugin.py",
        current_repo_root / "MotionGPT/eval_loader_robustness.py",
        current_repo_root / "MotionGPT/scripts/run_motionx322_table3_eval.sh",
        current_repo_root / "MotionGPT/mGPT/models/mgpt_sato_pp.py",
    )
    missing = [str(path) for path in implementation_paths if not path.is_file()]
    if missing:
        parser.error(f"missing formal evaluation implementation files: {missing}")
    manifest = {
        "schema_version": "motiongpt-motionx322-sato-plugin-eval-source/v1",
        "protocol": protocol,
        "model": {
            "name": "MotionGPT + SATO++ plug-in",
            "target": "mGPT.models.mgpt_sato_pp.MotionGPTSATOPP",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "selection": selection["selection"],
            "selection_policy": selection_policy,
            "selection_step": int(selection["step"]),
            "selection_metrics": {
                "FID": float(selection["FID"]),
                "Matching": float(selection["Matching"]),
                "R@3": float(selection["R@3"]),
                "samples": int(selection["samples"]),
            },
            "selection_metadata": str(selection_path),
            "selection_metadata_sha256": file_sha256(selection_path),
            "training_source_manifest": str(source_manifest_path),
            "training_source_manifest_sha256": file_sha256(source_manifest_path),
            "baseline_checkpoint_sha256": baseline_checkpoint_sha256,
            "stage1_checkpoint_sha256": selection["stage1_checkpoint_sha256"],
            "pair_file_sha256": selection["pair_file_sha256"],
            "uses_motiongpt_baseline_checkpoint": True,
            "uses_t2mgpt_generator_checkpoint": False,
            "attribution": selection["attribution"],
            "attribution_k": int(selection["attribution_k"]),
            "attribution_create_graph": True,
            "distributed_var_cov": True,
            "distributed_relation_losses": True,
            "early_stopping_enabled": selection.get("early_stopping_enabled"),
            "token_code_path": token_code_path,
            "expected_checkpoint_load": {
                "missing_key_count": 0,
                "unexpected_key_count": 0,
            },
        },
        "baseline": {
            "manifest": str(baseline_manifest_path),
            "manifest_sha256": file_sha256(baseline_manifest_path),
            "checkpoint_sha256": baseline_checkpoint_sha256,
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
