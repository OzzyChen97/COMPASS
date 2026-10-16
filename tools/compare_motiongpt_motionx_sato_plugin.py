#!/usr/bin/env python3
"""Audit and compare MotionGPT baseline and SATO++ 20-seed Motion-X runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = ("FID", "FID_P", "FID_D", "Matching", "R@1", "R@2", "R@3", "Diversity")
SEEDS = list(range(123, 143))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> dict[int, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {int(row["seed"]): row for row in rows}
    if sorted(result) != SEEDS or len(rows) != len(SEEDS):
        raise RuntimeError(f"repeat20 CSV does not contain exactly seeds 123--142: {path}")
    return result


def require_summary(summary: dict, label: str) -> None:
    if summary.get("dataset") != "motionx322":
        raise RuntimeError(f"{label} summary is not Motion-X-322")
    if int(summary.get("n", -1)) != 20:
        raise RuntimeError(f"{label} summary is not a 20-seed evaluation")
    if int(summary.get("samples_per_seed", -1)) != 7733:
        raise RuntimeError(f"{label} summary does not use 7733 samples per seed")
    if set(summary.get("summary", {})) != set(METRICS):
        raise RuntimeError(f"{label} summary metric set mismatch")


def audit_seed_runs(
    run_dir: Path,
    expected_checkpoint_sha256: str,
    label: str,
) -> dict[int, dict]:
    rows = {}
    for seed in SEEDS:
        path = run_dir / f"seed{seed}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing {label} seed result: {path}")
        row = load_json(path)
        if row.get("dataset") != "motionx322" or int(row.get("seed", -1)) != seed:
            raise RuntimeError(f"{label} seed identity mismatch: {path}")
        for key in ("matched_count", "total_count"):
            if int(row.get(key, -1)) != 7733:
                raise RuntimeError(f"{label} seed {seed} is not 7733/7733")
        if int(row.get("unmatched_count", -1)) != 0:
            raise RuntimeError(f"{label} seed {seed} has unmatched samples")
        if int(row.get("runtime_fallback_count", -1)) != 0:
            raise RuntimeError(f"{label} seed {seed} used runtime fallback")
        checkpoint_sha256 = row.get("artifacts", {}).get("checkpoint", {}).get("sha256")
        if checkpoint_sha256 != expected_checkpoint_sha256:
            raise RuntimeError(f"{label} seed {seed} checkpoint identity mismatch")
        load_audit = row.get("checkpoint_load_audit", {})
        if int(load_audit.get("missing_key_count", -1)) != 0:
            raise RuntimeError(f"{label} seed {seed} checkpoint has missing model keys")
        if int(load_audit.get("unexpected_key_count", -1)) != 0:
            raise RuntimeError(f"{label} seed {seed} checkpoint has unexpected model keys")
        if set(row.get("metrics", {})) != set(METRICS):
            raise RuntimeError(f"{label} seed {seed} metric set mismatch")
        rows[seed] = row
    return rows


def compare(
    baseline_summary_path: Path,
    plugin_summary_path: Path,
    baseline_manifest_path: Path,
    plugin_manifest_path: Path,
    baseline_run_dir: Path,
    plugin_run_dir: Path,
) -> dict:
    paths = (
        baseline_summary_path,
        plugin_summary_path,
        baseline_manifest_path,
        plugin_manifest_path,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline_summary = load_json(baseline_summary_path)
    plugin_summary = load_json(plugin_summary_path)
    require_summary(baseline_summary, "baseline")
    require_summary(plugin_summary, "SATO++")
    baseline_manifest = load_json(baseline_manifest_path)
    plugin_manifest = load_json(plugin_manifest_path)
    for path_text, expected_hash in plugin_manifest.get("implementation", {}).items():
        path = Path(path_text)
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise RuntimeError(f"SATO++ evaluation implementation hash mismatch: {path}")
    if baseline_manifest.get("protocol") != plugin_manifest.get("protocol"):
        raise RuntimeError("baseline and SATO++ source protocols differ")
    plugin_model = plugin_manifest.get("model", {})
    if plugin_model.get("target") != "mGPT.models.mgpt_sato_pp.MotionGPTSATOPP":
        raise RuntimeError("SATO++ manifest does not instantiate MotionGPTSATOPP")
    if plugin_model.get("uses_motiongpt_baseline_checkpoint") is not True:
        raise RuntimeError("SATO++ manifest lacks MotionGPT baseline provenance")
    if plugin_model.get("uses_t2mgpt_generator_checkpoint") is not False:
        raise RuntimeError("SATO++ manifest permits a T2M-GPT generator checkpoint")
    if plugin_model.get("attribution_create_graph") is not True:
        raise RuntimeError("SATO++ manifest does not use trainable Grad x Input")
    if plugin_model.get("distributed_var_cov") is not True:
        raise RuntimeError("SATO++ manifest does not use cross-rank variance/covariance")
    if plugin_model.get("distributed_relation_losses") is not True:
        raise RuntimeError("SATO++ manifest does not normalize relation losses across ranks")
    shared_suffixes = (
        "MotionGPT/eval_loader_robustness.py",
        "MotionGPT/scripts/run_motionx322_table3_eval.sh",
        "tools/aggregate_motiongpt_loader_eval.py",
    )
    baseline_implementation = baseline_manifest.get("implementation", {})
    plugin_implementation = plugin_manifest.get("implementation", {})
    for suffix in shared_suffixes:
        baseline_hashes = {
            value for key, value in baseline_implementation.items()
            if str(key).endswith(suffix)
        }
        plugin_hashes = {
            value for key, value in plugin_implementation.items()
            if str(key).endswith(suffix)
        }
        if len(baseline_hashes) != 1 or baseline_hashes != plugin_hashes:
            raise RuntimeError(
                f"baseline and SATO++ shared implementation differ: {suffix}"
            )
    baseline_checkpoint = baseline_manifest.get("model", {}).get("checkpoint_sha256")
    plugin_checkpoint = plugin_model.get("checkpoint_sha256")
    if not baseline_checkpoint or not plugin_checkpoint:
        raise RuntimeError("missing checkpoint identity in source manifests")
    if baseline_checkpoint == plugin_checkpoint:
        raise RuntimeError("baseline and SATO++ checkpoint hashes are identical")
    if plugin_model.get("baseline_checkpoint_sha256") != baseline_checkpoint:
        raise RuntimeError("SATO++ provenance does not point to this baseline")
    selection_path = Path(plugin_model.get("selection_metadata", "")).resolve()
    if not selection_path.is_file():
        raise FileNotFoundError("SATO++ validation-selection metadata is missing")
    if file_sha256(selection_path) != plugin_model.get("selection_metadata_sha256"):
        raise RuntimeError("SATO++ selection metadata hash mismatch")
    selection = load_json(selection_path)
    selection_policy = plugin_model.get("selection_policy", "validation_best")
    if selection.get("stage") != "sato_plugin":
        raise RuntimeError("SATO++ checkpoint metadata has the wrong stage")
    if selection_policy == "validation_best":
        if selection.get("split") != "val":
            raise RuntimeError("SATO++ validation-best checkpoint is not split=val")
    elif selection_policy == "last":
        if selection.get("split") != "train_terminal":
            raise RuntimeError("SATO++ last checkpoint is not terminal training state")
        if selection.get("early_stopping_enabled") is not False:
            raise RuntimeError("SATO++ last-checkpoint run used early stopping")
    else:
        raise RuntimeError(f"unsupported SATO++ selection policy: {selection_policy}")
    if selection.get("checkpoint_sha256") != plugin_checkpoint:
        raise RuntimeError("SATO++ selection checkpoint hash mismatch")

    baseline_identity = baseline_summary.get("identities", {})
    plugin_identity = plugin_summary.get("identities", {})
    if baseline_identity.get("checkpoint") != baseline_checkpoint:
        raise RuntimeError("baseline summary checkpoint differs from its manifest")
    if plugin_identity.get("checkpoint") != plugin_checkpoint:
        raise RuntimeError("SATO++ summary checkpoint differs from its manifest")
    for key in ("evaluator", "protocol"):
        if baseline_identity.get(key) != plugin_identity.get(key):
            raise RuntimeError(f"baseline and SATO++ {key} identities differ")

    baseline_runs = audit_seed_runs(baseline_run_dir, baseline_checkpoint, "baseline")
    plugin_runs = audit_seed_runs(plugin_run_dir, plugin_checkpoint, "SATO++")
    baseline_csv = load_csv(baseline_run_dir / "combined/all_runs.csv")
    plugin_csv = load_csv(plugin_run_dir / "combined/all_runs.csv")
    paired_seeds = []
    paired_values = {metric: [] for metric in METRICS}
    for seed in SEEDS:
        baseline_row = baseline_runs[seed]
        plugin_row = plugin_runs[seed]
        if baseline_row.get("sample_index_sha256") != plugin_row.get("sample_index_sha256"):
            raise RuntimeError(f"baseline and SATO++ sample index differ for seed {seed}")
        if baseline_row.get("protocol_fingerprint") != plugin_row.get("protocol_fingerprint"):
            raise RuntimeError(f"baseline and SATO++ protocol differ for seed {seed}")
        baseline_eval = baseline_row.get("artifacts", {}).get("evaluator_checkpoint", {}).get("sha256")
        plugin_eval = plugin_row.get("artifacts", {}).get("evaluator_checkpoint", {}).get("sha256")
        if baseline_eval != plugin_eval:
            raise RuntimeError(f"baseline and SATO++ evaluator differ for seed {seed}")
        deltas = {}
        for metric in METRICS:
            direct_baseline = float(baseline_row["metrics"][metric])
            direct_plugin = float(plugin_row["metrics"][metric])
            if direct_baseline != float(baseline_csv[seed][metric]):
                raise RuntimeError(f"baseline CSV mismatch for seed {seed} {metric}")
            if direct_plugin != float(plugin_csv[seed][metric]):
                raise RuntimeError(f"SATO++ CSV mismatch for seed {seed} {metric}")
            delta = direct_plugin - direct_baseline
            paired_values[metric].append(delta)
            deltas[metric] = delta
        paired_seeds.append({
            "seed": seed,
            "sample_index_sha256": baseline_row["sample_index_sha256"],
            "delta_sato_minus_baseline": deltas,
        })

    metrics = {}
    for metric in METRICS:
        baseline_value = baseline_summary["summary"][metric]
        plugin_value = plugin_summary["summary"][metric]
        deltas = np.asarray(paired_values[metric], dtype=np.float64)
        metrics[metric] = {
            "baseline": {
                "mean": float(baseline_value["mean"]),
                "std": float(baseline_value["std"]),
            },
            "sato_plugin": {
                "mean": float(plugin_value["mean"]),
                "std": float(plugin_value["std"]),
            },
            "delta_sato_minus_baseline": float(
                plugin_value["mean"] - baseline_value["mean"]
            ),
            "paired_delta": {
                "mean": float(np.mean(deltas)),
                "std": float(np.std(deltas, ddof=1)),
            },
        }
    return {
        "schema_version": "motiongpt-motionx322-sato-plugin-comparison/v1",
        "dataset": "motionx322",
        "seeds": SEEDS,
        "samples_per_seed": 7733,
        "paired": True,
        "checkpoint_sha256": {
            "baseline": baseline_checkpoint,
            "sato_plugin": plugin_checkpoint,
        },
        "sato_plugin_selection_policy": selection_policy,
        "evaluator_sha256": baseline_identity["evaluator"],
        "protocol_fingerprint": baseline_identity["protocol"],
        "source_manifests": {
            "baseline": {
                "path": str(baseline_manifest_path.resolve()),
                "sha256": file_sha256(baseline_manifest_path),
            },
            "sato_plugin": {
                "path": str(plugin_manifest_path.resolve()),
                "sha256": file_sha256(plugin_manifest_path),
            },
        },
        "metrics": metrics,
        "paired_seeds": paired_seeds,
    }


def markdown_report(payload: dict) -> str:
    lines = [
        "# MotionGPT vs MotionGPT + SATO++ on Motion-X-322",
        "",
        "Paired 20-seed evaluation (seeds 123--142), 7,733 samples per seed.",
        "",
        "| Metric | MotionGPT mean ± std | SATO++ mean ± std | Δ (SATO++ − baseline) |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        row = payload["metrics"][metric]
        baseline = row["baseline"]
        plugin = row["sato_plugin"]
        lines.append(
            f"| {metric} | {baseline['mean']:.6f} ± {baseline['std']:.6f} | "
            f"{plugin['mean']:.6f} ± {plugin['std']:.6f} | "
            f"{row['delta_sato_minus_baseline']:+.6f} |"
        )
    lines.extend([
        "",
        f"Baseline checkpoint: `{payload['checkpoint_sha256']['baseline']}`",
        "",
        f"SATO++ checkpoint: `{payload['checkpoint_sha256']['sato_plugin']}`",
        "",
        f"Evaluator: `{payload['evaluator_sha256']}`",
        "",
        f"Protocol: `{payload['protocol_fingerprint']}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--plugin-summary", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--plugin-manifest", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--plugin-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    payload = compare(
        args.baseline_summary.resolve(),
        args.plugin_summary.resolve(),
        args.baseline_manifest.resolve(),
        args.plugin_manifest.resolve(),
        args.baseline_run_dir.resolve(),
        args.plugin_run_dir.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_output = args.markdown_output or args.output.with_suffix(".md")
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(markdown_report(payload), encoding="utf-8")
    print(args.output.resolve())
    print(markdown_output.resolve())


if __name__ == "__main__":
    main()
