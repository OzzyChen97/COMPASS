#!/usr/bin/env python
"""Build a reproducibility manifest for Motion-X Table 3 7733-sample evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(body).hexdigest()


def file_identity(path: Path) -> dict:
    value = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        value.update({"size": path.stat().st_size, "sha256": sha256_file(path)})
    return value


def load_pair_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            return json.load(handle)
        return [json.loads(line) for line in handle if line.strip()]


def build_manifest(*, pair_file: Path, spec_json: Path, summary_json: Path,
                   expected_seeds, expected_samples: int,
                   evaluator_root: Path = Path("checkpoints/motionx322"),
                   default_cfg_scale: float = 1.0,
                   default_cfg_uncond: str = "zero") -> dict:
    pair_rows = load_pair_rows(pair_file)
    level_counts = Counter(row.get("level") for row in pair_rows)
    if len(pair_rows) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} pairs in {pair_file}, got {len(pair_rows)}"
        )

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    rows = [row for row in summary.get("runs", []) if row.get("kind") in {"baseline", "sato_pp"}]
    if not rows:
        raise ValueError(f"No evaluator rows found in {summary_json}")

    expected_seed_set = set(expected_seeds)
    model_rows = {}
    for row in rows:
        model_rows.setdefault(row["model"], []).append(row)

    spec = json.loads(spec_json.read_text(encoding="utf-8"))
    spec_by_model = {spec_entry["name"]: spec_entry for spec_entry in spec}
    manifest_models = {}
    for model_name, values in model_rows.items():
        seen_seeds = {int(row["seed"]) for row in values}
        if seen_seeds != expected_seed_set:
            raise ValueError(
                f"Seed mismatch for model {model_name}: expected {sorted(expected_seed_set)}, "
                f"got {sorted(seen_seeds)}"
            )
        if any(int(row["matched_pairs"]) != expected_samples for row in values):
            raise ValueError(f"matched_pairs mismatch found for model {model_name}")
        if any(int(row["total_pairs"]) != expected_samples for row in values):
            raise ValueError(f"total_pairs mismatch found for model {model_name}")

        spec_entry = spec_by_model.get(model_name, {})
        checkpoint_paths = []
        for key in ("checkpoint", "resume_pth", "resume_trans", "clip_pth"):
            value = spec_entry.get(key)
            if value:
                identity = file_identity(Path(value))
                if not identity["exists"]:
                    raise FileNotFoundError(identity["path"])
                checkpoint_paths.append(identity)
        manifest_models[model_name] = {
            "kind": values[0]["kind"],
            "checkpoint": values[0]["checkpoint"],
            "cfg_scale": float(spec_entry.get(
                "cfg_scale", values[0].get("cfg_scale", default_cfg_scale))),
            "cfg_uncond": spec_entry.get(
                "cfg_uncond", values[0].get("cfg_uncond", default_cfg_uncond)),
            "seed_count": len(values),
            "expected_pairs_per_seed": expected_samples,
            "matched_pairs_contract": f"{expected_samples}/{expected_samples}",
            "iter": values[0]["iter"],
            "path": values[0]["path"],
            "checkpoint_files": checkpoint_paths,
        }

    evaluator_files = [
        file_identity(evaluator_root / relative)
        for relative in (
            "Comp_v6_KLD005/opt.txt",
            "Decomp_SP001_SM001_H512/meta/mean.npy",
            "Decomp_SP001_SM001_H512/meta/std.npy",
            "Decomp_SP001_SM001_H512/model/finest.tar",
            "text_mot_match/model/finest.tar",
        )
    ]
    missing_evaluator = [item["path"] for item in evaluator_files if not item["exists"]]
    if missing_evaluator:
        raise FileNotFoundError(
            "Missing Motion-X evaluator files: " + ", ".join(missing_evaluator)
        )

    manifest = {
        "table": "Motion-X Table 3",
        "expected_samples": expected_samples,
        "expected_seed_list": list(expected_seeds),
        "loader_protocol": {
            "sampling": {
                "mode": "categorical",
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
            },
            "pair_file": {
                "path": str(pair_file),
                "raw_sha256": sha256_file(pair_file),
                "canonical_sha256": canonical_sha256(pair_rows),
            },
            "pair_level_counts": {
                key: value for key, value in sorted(level_counts.items()) if key is not None
            },
            "pair_matching_contract": f"{expected_samples}/{expected_samples}",
            "std_ddof": 1,
        },
        "spec": {
            "path": str(spec_json),
            "raw_sha256": sha256_file(spec_json),
            "entries": len(spec),
        },
        "summary_path": str(summary_json),
        "evaluator_files": evaluator_files,
        "models": manifest_models,
    }

    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--spec-json", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=list(range(123, 143)))
    parser.add_argument("--expected-samples", type=int, default=7733)
    parser.add_argument("--evaluator-root", type=Path, default=Path("checkpoints/motionx322"))
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--cfg-uncond", default="zero")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_manifest(
        pair_file=args.pair_file,
        spec_json=args.spec_json,
        summary_json=args.summary_json,
        expected_seeds=args.expected_seeds,
        expected_samples=args.expected_samples,
        evaluator_root=args.evaluator_root,
        default_cfg_scale=args.cfg_scale,
        default_cfg_uncond=args.cfg_uncond,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
