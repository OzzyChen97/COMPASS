#!/usr/bin/env python3
"""Strict cross-run audit for the three MotionGPT-last paper evaluations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


SEEDS = list(range(123, 143))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_module():
    module_path = Path(__file__).with_name("aggregate_motiongpt_loader_eval.py")
    spec = importlib.util.spec_from_file_location("last_table_aggregate", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def audit_implementation(manifest: dict) -> list[dict]:
    compatible_updates = []
    for path_text, expected in manifest.get("implementation", {}).items():
        path = Path(path_text)
        if not path.is_file():
            raise RuntimeError(f"formal evaluation implementation drift: {path}")
        actual = file_sha256(path)
        if actual == expected:
            continue
        if path.name == "aggregate_motiongpt_loader_eval.py":
            compatible_updates.append({
                "path": str(path),
                "frozen_sha256": expected,
                "current_sha256": actual,
                "reason": (
                    "post-generation extension adding explicit HumanML3D fallback-level "
                    "validation; Motion-X aggregation is independently recomputed and "
                    "must remain byte-equivalent"
                ),
            })
            continue
        raise RuntimeError(f"formal evaluation implementation drift: {path}")
    return compatible_updates


def audit_result_loads(run_dir: Path, expected_checkpoint: str) -> dict:
    protocols = set()
    evaluators = set()
    sample_hashes = {}
    for seed in SEEDS:
        result = load_json(run_dir / f"seed{seed}.json")
        checkpoint = result.get("artifacts", {}).get("checkpoint", {}).get("sha256")
        if checkpoint != expected_checkpoint:
            raise RuntimeError(f"seed {seed} checkpoint mismatch in {run_dir}")
        load_audit = result.get("checkpoint_load_audit", {})
        if int(load_audit.get("missing_key_count", -1)) != 0:
            raise RuntimeError(f"seed {seed} has missing checkpoint keys in {run_dir}")
        if int(load_audit.get("unexpected_key_count", -1)) != 0:
            raise RuntimeError(f"seed {seed} has unexpected checkpoint keys in {run_dir}")
        protocols.add(result.get("protocol_fingerprint"))
        evaluators.add(
            result.get("artifacts", {}).get("evaluator_checkpoint", {}).get("sha256")
        )
        sample_hashes[seed] = result.get("sample_index_sha256")
    if len(protocols) != 1 or len(evaluators) != 1:
        raise RuntimeError(f"protocol/evaluator differs across seeds in {run_dir}")
    return {
        "protocol_fingerprint": next(iter(protocols)),
        "evaluator_sha256": next(iter(evaluators)),
        "sample_index_sha256": sample_hashes,
    }


def checked_stored_summary(run_dir: Path, recomputed: dict) -> Path:
    summary_path = run_dir / "combined/summary.json"
    if load_json(summary_path) != recomputed:
        raise RuntimeError(f"stored aggregate differs from strict recomputation: {run_dir}")
    if not (run_dir / "complete.txt").is_file():
        raise RuntimeError(f"missing completion marker: {run_dir}")
    return summary_path


def audit(human_dir: Path, motionx_dir: Path, plugin_dir: Path) -> dict:
    human_dir = human_dir.resolve()
    motionx_dir = motionx_dir.resolve()
    plugin_dir = plugin_dir.resolve()
    aggregator = aggregate_module()

    manifests = {
        "humanml3d_motiongpt_last": load_json(human_dir / "manifest.json"),
        "motionx_motiongpt_last": load_json(motionx_dir / "manifest.json"),
        "motionx_plugin_last": load_json(plugin_dir / "manifest.json"),
    }
    compatible_implementation_updates = {}
    for name, manifest in manifests.items():
        model = manifest.get("model", {})
        if model.get("selection_policy") != "last":
            raise RuntimeError(f"{name} does not select last.ckpt")
        if Path(model.get("checkpoint", "")).name != "last.ckpt":
            raise RuntimeError(f"{name} checkpoint is not named last.ckpt")
        checkpoint = Path(model["checkpoint"])
        if file_sha256(checkpoint) != model.get("checkpoint_sha256"):
            raise RuntimeError(f"{name} checkpoint hash mismatch")
        if model.get("uses_t2mgpt_generator_checkpoint") is not False:
            raise RuntimeError(f"{name} permits a T2M-GPT generator checkpoint")
        compatible_implementation_updates[name] = audit_implementation(manifest)

    human_rows, human_summary = aggregator.aggregate(
        human_dir,
        "humanml3d",
        expected_samples=4640,
        require_caption_locked=True,
        expected_unique_pairs=4353,
        expected_levels=["P1", "P2", "P3", "fallback"],
        source_manifest=human_dir / "manifest.json",
    )
    motionx_rows, motionx_summary = aggregator.aggregate(
        motionx_dir,
        "motionx322",
        expected_samples=7733,
        source_manifest=motionx_dir / "manifest.json",
    )
    plugin_rows, plugin_summary = aggregator.aggregate(
        plugin_dir,
        "motionx322",
        expected_samples=7733,
        source_manifest=plugin_dir / "manifest.json",
    )
    summaries = {
        "humanml3d_motiongpt_last": human_summary,
        "motionx_motiongpt_last": motionx_summary,
        "motionx_plugin_last": plugin_summary,
    }
    summary_paths = {
        "humanml3d_motiongpt_last": checked_stored_summary(human_dir, human_summary),
        "motionx_motiongpt_last": checked_stored_summary(motionx_dir, motionx_summary),
        "motionx_plugin_last": checked_stored_summary(plugin_dir, plugin_summary),
    }

    runtime = {
        "humanml3d_motiongpt_last": audit_result_loads(
            human_dir, manifests["humanml3d_motiongpt_last"]["model"]["checkpoint_sha256"]
        ),
        "motionx_motiongpt_last": audit_result_loads(
            motionx_dir, manifests["motionx_motiongpt_last"]["model"]["checkpoint_sha256"]
        ),
        "motionx_plugin_last": audit_result_loads(
            plugin_dir, manifests["motionx_plugin_last"]["model"]["checkpoint_sha256"]
        ),
    }
    baseline_runtime = runtime["motionx_motiongpt_last"]
    plugin_runtime = runtime["motionx_plugin_last"]
    if baseline_runtime["protocol_fingerprint"] != plugin_runtime["protocol_fingerprint"]:
        raise RuntimeError("Motion-X baseline and plug-in protocol fingerprints differ")
    if baseline_runtime["evaluator_sha256"] != plugin_runtime["evaluator_sha256"]:
        raise RuntimeError("Motion-X baseline and plug-in evaluator checkpoints differ")
    for seed in SEEDS:
        baseline_index = load_json(motionx_dir / f"sample_indices/seed{seed}.json")
        plugin_index = load_json(plugin_dir / f"sample_indices/seed{seed}.json")
        if baseline_index != plugin_index:
            raise RuntimeError(f"Motion-X sample index differs at seed {seed}")

    human_manifest = manifests["humanml3d_motiongpt_last"]
    frozen_index = load_json(Path(human_manifest["fixed_sample_index"]["path"]))
    for seed in SEEDS:
        if load_json(human_dir / f"sample_indices/seed{seed}.json") != frozen_index:
            raise RuntimeError(f"HumanML3D frozen sample index differs at seed {seed}")

    return {
        "schema_version": "motiongpt-last-paper-table-audit/v1",
        "status": "pass",
        "seeds": SEEDS,
        "runs": {
            name: {
                "run_dir": str(run_dir),
                "manifest": str(run_dir / "manifest.json"),
                "manifest_sha256": file_sha256(run_dir / "manifest.json"),
                "checkpoint": manifest["model"]["checkpoint"],
                "checkpoint_sha256": manifest["model"]["checkpoint_sha256"],
                "selection_step": manifest["model"]["selection_step"],
                "summary": str(summary_paths[name]),
                "summary_sha256": file_sha256(summary_paths[name]),
                "runtime": runtime[name],
                "metrics": summaries[name]["summary"],
            }
            for name, run_dir, manifest in (
                (
                    "humanml3d_motiongpt_last",
                    human_dir,
                    manifests["humanml3d_motiongpt_last"],
                ),
                (
                    "motionx_motiongpt_last",
                    motionx_dir,
                    manifests["motionx_motiongpt_last"],
                ),
                (
                    "motionx_plugin_last",
                    plugin_dir,
                    manifests["motionx_plugin_last"],
                ),
            )
        },
        "control_variables": {
            "motionx_protocol_identical": True,
            "motionx_evaluator_identical": True,
            "motionx_per_seed_sample_indices_identical": True,
            "humanml3d_caption_locked": True,
            "humanml3d_frozen_index_identical_across_seeds": True,
            "runtime_fallback_count": 0,
            "checkpoint_missing_keys": 0,
            "checkpoint_unexpected_keys": 0,
        },
        "compatible_implementation_updates": compatible_implementation_updates,
        "row_counts": {
            "humanml3d_motiongpt_last": len(human_rows),
            "motionx_motiongpt_last": len(motionx_rows),
            "motionx_plugin_last": len(plugin_rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-run", type=Path, required=True)
    parser.add_argument("--motionx-run", type=Path, required=True)
    parser.add_argument("--plugin-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.human_run, args.motionx_run, args.plugin_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
