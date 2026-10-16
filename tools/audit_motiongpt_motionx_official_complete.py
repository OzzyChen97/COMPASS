#!/usr/bin/env python3
"""Final strict audit for the fresh official MotionGPT Motion-X pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


EXPECTED_SEEDS = list(range(123, 143))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def checked_selection(path: Path, stage: str) -> dict:
    payload = load_json(path)
    checkpoint = Path(payload.get("checkpoint", "")).resolve()
    if payload.get("split") != "val":
        raise RuntimeError(f"{stage} selection is not validation-only")
    if payload.get("uses_t2mgpt_generator_checkpoint") is not False:
        raise RuntimeError(f"{stage} selection is not baseline-free")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{stage} checkpoint is missing: {checkpoint}")
    actual = sha256_file(checkpoint)
    if payload.get("checkpoint_sha256") != actual:
        raise RuntimeError(f"{stage} checkpoint SHA-256 mismatch")
    if stage != "stage1" and payload.get("stage") != stage:
        raise RuntimeError(f"selection stage mismatch for {stage}")
    return payload


def strict_aggregate(eval_root: Path, source_manifest: Path) -> tuple[list, dict]:
    module_path = Path(__file__).with_name("aggregate_motiongpt_loader_eval.py")
    spec = importlib.util.spec_from_file_location("motiongpt_final_aggregate", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.aggregate(
        eval_root,
        "motionx322",
        expected_samples=7733,
        source_manifest=source_manifest,
    )


def audit(run_root: Path, eval_root: Path) -> dict:
    run_root = run_root.resolve()
    eval_root = eval_root.resolve()
    repo_root = run_root.parents[2]
    mgpt_root = run_root.parents[1]

    source_manifest_path = run_root / "source_manifest.json"
    source = load_json(source_manifest_path)
    if source.get("method", {}).get("uses_t2mgpt_generator_checkpoint") is not False:
        raise RuntimeError("source manifest does not prove baseline-free initialization")
    if source.get("dataset", {}).get("split_counts") != {
        "train": 41159, "val": 2552, "test": 7733
    }:
        raise RuntimeError("source manifest split counts differ from Motion-X contract")

    names = source["names"]
    stage1_dir = mgpt_root / "experiments/mgpt" / names["stage1"]
    stage2_dir = mgpt_root / "experiments/mgpt" / names["stage2"]
    stage3_dir = mgpt_root / "experiments/mgpt" / names["stage3"]
    stage1_path = stage1_dir / "checkpoints/best-validation.json"
    stage2_path = stage2_dir / "checkpoints/selected-validation.json"
    stage3_path = stage3_dir / "checkpoints/selected-validation.json"
    stage1 = checked_selection(stage1_path, "stage1")
    stage2 = checked_selection(stage2_path, "stage2")
    stage3 = checked_selection(stage3_path, "stage3")
    if stage2.get("stage1_checkpoint_sha256") != stage1["checkpoint_sha256"]:
        raise RuntimeError("Stage 2 does not depend on selected Stage 1")
    if stage3.get("stage1_checkpoint_sha256") != stage1["checkpoint_sha256"]:
        raise RuntimeError("Stage 3 Stage 1 dependency mismatch")
    if stage3.get("predecessor_checkpoint_sha256") != stage2["checkpoint_sha256"]:
        raise RuntimeError("Stage 3 does not depend on selected Stage 2")

    token_audit_path = run_root / "token_audit.json"
    tokens = load_json(token_audit_path)
    if tokens.get("status") != "pass":
        raise RuntimeError("fresh token audit did not pass")
    required_token_counts = {
        "split_count": 41159,
        "eligible_count": 41108,
        "token_file_count": 41108,
        "missing_count": 0,
        "extra_count": 0,
        "invalid_token_count": 0,
    }
    for key, expected in required_token_counts.items():
        if int(tokens.get(key, -1)) != expected:
            raise RuntimeError(f"token audit mismatch for {key}")
    if tokens.get("stage1_checkpoint_sha256") != stage1["checkpoint_sha256"]:
        raise RuntimeError("token inventory was not produced by selected Stage 1")

    table_manifest_path = eval_root / "manifest.json"
    table = load_json(table_manifest_path)
    protocol = table.get("protocol", {})
    if protocol.get("seeds") != EXPECTED_SEEDS:
        raise RuntimeError("Table-3 manifest does not contain exactly seeds 123--142")
    if protocol.get("samples_per_seed") != 7733:
        raise RuntimeError("Table-3 manifest does not require 7733 samples per seed")
    expected_protocol = {
        "pair_rng_mode": "sequential",
        "generation_order": "original-first",
        "prompt_mode": "official-default",
        "generation_length_mode": "generated",
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise RuntimeError(f"Table-3 protocol mismatch for {key}")
    if protocol.get("sampling") != {
        "mode": "categorical", "temperature": 1.0, "top_k": 0, "top_p": 1.0
    }:
        raise RuntimeError("Table-3 sampling protocol mismatch")
    model = table.get("model", {})
    if model.get("uses_t2mgpt_generator_checkpoint") is not False:
        raise RuntimeError("Table-3 manifest does not prove baseline-free provenance")
    if model.get("checkpoint_sha256") != stage3["checkpoint_sha256"]:
        raise RuntimeError("Table-3 manifest does not use selected Stage 3")
    if model.get("stage1_checkpoint_sha256") != stage1["checkpoint_sha256"]:
        raise RuntimeError("Table-3 Stage 1 dependency mismatch")
    if model.get("stage2_checkpoint_sha256") != stage2["checkpoint_sha256"]:
        raise RuntimeError("Table-3 Stage 2 dependency mismatch")

    expected_root_results = {f"seed{seed}.json" for seed in EXPECTED_SEEDS}
    actual_root_results = {path.name for path in eval_root.glob("seed*.json")}
    if actual_root_results != expected_root_results:
        raise RuntimeError("final result files are not exactly seeds 123--142")
    for subdir in ("sample_indices", "generation_audits"):
        actual = {path.name for path in (eval_root / subdir).glob("seed*.json")}
        if actual != expected_root_results:
            raise RuntimeError(f"{subdir} files are not exactly seeds 123--142")
    if not (eval_root / "complete.txt").is_file():
        raise RuntimeError("20-seed evaluation lacks completion marker")

    rows, recomputed = strict_aggregate(eval_root, table_manifest_path)
    if len(rows) != 20 or recomputed.get("n") != 20:
        raise RuntimeError("strict aggregate does not contain exactly 20 seeds")
    if recomputed.get("samples_per_seed") != 7733:
        raise RuntimeError("strict aggregate sample count mismatch")
    if recomputed.get("identities", {}).get("checkpoint") != stage3["checkpoint_sha256"]:
        raise RuntimeError("strict aggregate checkpoint identity mismatch")
    stored_summary_path = eval_root / "combined/summary.json"
    stored_summary = load_json(stored_summary_path)
    if stored_summary != recomputed:
        raise RuntimeError("stored aggregate differs from independently recomputed aggregate")

    return {
        "schema_version": "motiongpt-motionx322-official-completion/v1",
        "status": "pass",
        "repository_root": str(repo_root),
        "run_root": str(run_root),
        "evaluation_root": str(eval_root),
        "official_motiongpt_repository": source["method"]["repository"],
        "official_motiongpt_git_head": source["method"]["git_head"],
        "uses_t2mgpt_generator_checkpoint": False,
        "stage1": {
            "checkpoint": stage1["checkpoint"],
            "checkpoint_sha256": stage1["checkpoint_sha256"],
            "step": stage1["global_step"],
            "score": stage1["score"],
        },
        "tokens": {
            "inventory_sha256": tokens["inventory_sha256"],
            **required_token_counts,
        },
        "stage2": {
            "checkpoint": stage2["checkpoint"],
            "checkpoint_sha256": stage2["checkpoint_sha256"],
            "step": stage2["step"],
            "FID": stage2["FID"],
            "validation_checks": stage2["validation_checks"],
            "early_stopped": stage2["early_stopped"],
        },
        "stage3": {
            "checkpoint": stage3["checkpoint"],
            "checkpoint_sha256": stage3["checkpoint_sha256"],
            "step": stage3["step"],
            "FID": stage3["FID"],
            "validation_checks": stage3["validation_checks"],
            "early_stopped": stage3["early_stopped"],
        },
        "evaluation": {
            "seeds": EXPECTED_SEEDS,
            "seed_count": 20,
            "samples_per_seed": 7733,
            "total_original_generations": 20 * 7733,
            "total_perturbed_generations": 20 * 7733,
            "checkpoint_sha256": stage3["checkpoint_sha256"],
            "protocol_fingerprint": recomputed["identities"]["protocol"],
            "evaluator_sha256": recomputed["identities"]["evaluator"],
            "summary": recomputed["summary"],
        },
        "artifacts": {
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "token_audit": str(token_audit_path),
            "token_audit_sha256": sha256_file(token_audit_path),
            "table_manifest": str(table_manifest_path),
            "table_manifest_sha256": sha256_file(table_manifest_path),
            "aggregate_summary": str(stored_summary_path),
            "aggregate_summary_sha256": sha256_file(stored_summary_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.run_root, args.eval_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
