#!/usr/bin/env python3
"""Audit and materialize the validation-selected MotionGPT SATO++ checkpoint."""

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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if not expected or sha256_file(path) != expected:
        raise RuntimeError(f"{label} hash mismatch: {path}")


def select_sato_plugin_best(
    experiment_dir: Path,
    source_manifest: Path,
    expected_samples: int = 2552,
    validation_interval: int = 5000,
    max_steps: int = 50000,
    patience: int = 10,
    min_validations: int = 8,
    fid_tolerance: float = 1e-4,
) -> dict:
    experiment_dir = experiment_dir.resolve()
    source_manifest = source_manifest.resolve()
    checkpoint_dir = experiment_dir / "checkpoints"
    history_path = checkpoint_dir / "validation_metrics.jsonl"
    summary_path = checkpoint_dir / "best-validation.json"
    checkpoint = checkpoint_dir / "best-validation.ckpt"
    for required in (history_path, summary_path, checkpoint, source_manifest):
        if not required.is_file():
            raise FileNotFoundError(f"missing SATO++ selection artifact: {required}")

    manifest = load_json(source_manifest)
    method = manifest.get("method", {})
    if method.get("uses_motiongpt_baseline_checkpoint") is not True:
        raise RuntimeError("source manifest does not prove MotionGPT baseline initialization")
    if method.get("uses_t2mgpt_generator_checkpoint") is not False:
        raise RuntimeError("source manifest permits a T2M-GPT generator checkpoint")
    training = manifest.get("training", {})
    if training.get("distributed_var_cov") is not True:
        raise RuntimeError("source manifest lacks cross-rank variance/covariance")
    if training.get("distributed_relation_losses") is not True:
        raise RuntimeError("source manifest lacks cross-rank relation-loss normalization")
    expected_experiment = Path(training.get("experiment_dir", "")).resolve()
    if expected_experiment != experiment_dir:
        raise RuntimeError("source manifest experiment directory mismatch")
    if training.get("experiment_name") != experiment_dir.name:
        raise RuntimeError("source manifest experiment name mismatch")

    for relative, expected_hash in manifest.get("source_files", {}).items():
        require_file_hash(
            Path(manifest["repo_root"]) / relative,
            expected_hash,
            f"frozen source {relative}",
        )
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
    continuation = manifest.get("continuation")
    if continuation is not None:
        require_file_hash(
            Path(continuation.get("predecessor_source_manifest", "")).resolve(),
            continuation.get("predecessor_source_manifest_sha256", ""),
            "predecessor source manifest",
        )
        require_file_hash(
            Path(continuation.get("resume_checkpoint", "")).resolve(),
            continuation.get("resume_checkpoint_sha256", ""),
            "continuation checkpoint",
        )

    rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < min_validations:
        raise RuntimeError(
            f"SATO++ run has only {len(rows)} validations; minimum is {min_validations}"
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
            "SATO++ run is not terminal: neither validation early stopping nor max_steps reached"
        )

    summary = load_json(summary_path)
    if Path(summary.get("checkpoint", "")).resolve() != checkpoint.resolve():
        raise RuntimeError("best-validation summary points outside the SATO++ experiment")
    if summary.get("split") != "val":
        raise RuntimeError("best-validation summary is not validation-selected")
    for key in ("step", "epoch", "samples"):
        if int(summary[key]) != int(best[key]):
            raise RuntimeError(f"best-validation summary mismatch for {key}")
    for key in ("FID", "Matching", "R@3"):
        if float(summary[key]) != float(best[key]):
            raise RuntimeError(f"best-validation summary mismatch for {key}")

    config_paths = sorted(experiment_dir.glob("config_*_train.yaml"))
    if len(config_paths) != 1:
        raise RuntimeError(f"expected one resolved training config, found {len(config_paths)}")
    config = yaml.safe_load(config_paths[0].read_text(encoding="utf-8"))
    if config.get("NAME") != experiment_dir.name:
        raise RuntimeError("resolved config experiment name mismatch")
    if config.get("DEVICE") != [0, 1, 2, 3]:
        raise RuntimeError("resolved SATO++ config did not use physical GPUs 0--3")
    if config.get("TRAIN", {}).get("STAGE") != "lm_sato_pp":
        raise RuntimeError("resolved config is not lm_sato_pp")
    if config.get("model", {}).get("target") != "mGPT.models.mgpt_sato_pp.MotionGPTSATOPP":
        raise RuntimeError("resolved config does not instantiate MotionGPTSATOPP")
    if continuation is None:
        if Path(config["TRAIN"].get("PRETRAINED", "")).resolve() != baseline_checkpoint:
            raise RuntimeError("resolved config does not initialize from the frozen baseline")
    else:
        predecessor_dir = Path(
            continuation.get("predecessor_experiment_dir", "")
        ).resolve()
        resume_checkpoint = Path(continuation.get("resume_checkpoint", "")).resolve()
        if Path(config["TRAIN"].get("RESUME", "")).resolve() != predecessor_dir:
            raise RuntimeError("resolved config continuation directory mismatch")
        if Path(config["TRAIN"].get("PRETRAINED", "")).resolve() != resume_checkpoint:
            raise RuntimeError("resolved config continuation checkpoint mismatch")
    if Path(config["TRAIN"].get("PRETRAINED_VAE", "")).resolve() != stage1_checkpoint:
        raise RuntimeError("resolved config does not use the frozen Stage 1 VAE")
    if config["TRAIN"].get("STRICT_LOAD") is not False:
        raise RuntimeError("baseline-to-plugin initialization must use STRICT_LOAD=false")
    if int(config["TRAIN"].get("MAX_STEPS", -1)) != max_steps:
        raise RuntimeError("resolved config MAX_STEPS mismatch")
    if int(config.get("LOGGER", {}).get("VAL_EVERY_N_STEPS", -1)) != validation_interval:
        raise RuntimeError("resolved config validation interval mismatch")
    selection_cfg = config.get("CHECKPOINT_SELECTION", {})
    if selection_cfg.get("SPLIT") != "val":
        raise RuntimeError("resolved config checkpoint selection is not validation-only")
    early_cfg = selection_cfg.get("EARLY_STOPPING", {})
    if int(early_cfg.get("PATIENCE", -1)) != patience:
        raise RuntimeError("resolved config early-stopping patience mismatch")
    if int(early_cfg.get("MIN_VALIDATIONS", -1)) != min_validations:
        raise RuntimeError("resolved config minimum validations mismatch")
    sato = config.get("SATO", {})
    if sato.get("ATTRIBUTION_CREATE_GRAPH") is not True:
        raise RuntimeError("resolved config does not train through Grad x Input attribution")
    if sato.get("DISTRIBUTED_VAR_COV") is not True:
        raise RuntimeError("resolved config does not use cross-rank variance/covariance")
    if sato.get("DISTRIBUTED_RELATION_LOSSES") is not True:
        raise RuntimeError("resolved config does not normalize relation losses across ranks")
    if int(sato.get("ATTRIBUTION_K", -1)) != int(training.get("attribution_k", -2)):
        raise RuntimeError("resolved config attribution horizon mismatch")
    pair_paths = [Path(value).resolve() for value in sato.get("PAIR_DATA_PATHS", [])]
    if pair_paths != [pair_file]:
        raise RuntimeError("resolved config pair file mismatch")

    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 == baseline_meta.get("checkpoint_sha256"):
        raise RuntimeError("selected plug-in checkpoint is byte-identical to the baseline")
    result = {
        "schema_version": "motiongpt-motionx322-sato-plugin-selection/v1",
        "stage": "sato_plugin",
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
        "checkpoint_sha256": checkpoint_sha256,
        "history": str(history_path.resolve()),
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
    }
    if continuation is not None:
        result["continuation"] = {
            "predecessor_experiment_dir": continuation["predecessor_experiment_dir"],
            "predecessor_source_manifest": continuation["predecessor_source_manifest"],
            "predecessor_source_manifest_sha256": continuation[
                "predecessor_source_manifest_sha256"
            ],
            "resume_checkpoint": continuation["resume_checkpoint"],
            "resume_checkpoint_sha256": continuation["resume_checkpoint_sha256"],
            "inherited_validation_checks": int(
                continuation.get("validation_history_rows", 0)
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=2552)
    parser.add_argument("--validation-interval", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-validations", type=int, default=8)
    parser.add_argument("--fid-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = select_sato_plugin_best(
        args.experiment_dir,
        args.source_manifest,
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
