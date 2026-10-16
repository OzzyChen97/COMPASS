#!/usr/bin/env python3
"""Strictly audit and aggregate MotionGPT HumanML3D repeat-20 results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


METRICS = ("FID", "FID_P", "FID_D", "Matching", "R@1", "R@2", "R@3", "Diversity")
EXPECTED_SEEDS = list(range(123, 143))
EXPECTED_SAMPLING = {
    "mode": "categorical",
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "kwargs_mode": "explicit",
    "implementation": "explicit Hugging Face generate kwargs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def metric_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary = {}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        population_std = float(np.std(values, ddof=0))
        summary[metric] = {
            "mean": float(np.mean(values)),
            "population_std_ddof0": population_std,
            "sample_std_ddof1": float(np.std(values, ddof=1)),
            "paper_style_ci95": float(1.96 * population_std / math.sqrt(len(values))),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def load_sato_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("model") == "SATO-50k"]
    if [int(row["seed"]) for row in rows] != EXPECTED_SEEDS:
        raise ValueError("SATO-50k comparison rows are not exactly seeds 123--142")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sato-runs", type=Path)
    args = parser.parse_args()

    results = []
    rows = []
    identities = {
        "checkpoint_sha256": set(),
        "evaluator_sha256": set(),
        "raw_protocol_fingerprint": set(),
        "normalized_protocol_fingerprint": set(),
        "source_manifest_fingerprint": set(),
    }
    for seed in EXPECTED_SEEDS:
        result_path = args.run_dir / f"seed{seed}.json"
        index_path = args.run_dir / f"seed{seed}_sample_index.json"
        diagnostics_path = args.run_dir / f"seed{seed}_diagnostics.json"
        for path in (result_path, index_path, diagnostics_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty artifact: {path}")

        result = json.loads(result_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if result.get("dataset") != "humanml3d" or int(result.get("seed", -1)) != seed:
            raise ValueError(f"dataset/seed mismatch in {result_path}")
        if int(index.get("seed", -1)) != seed:
            raise ValueError(f"seed mismatch in {index_path}")
        if result.get("matched_count") != 4640 or result.get("total_count") != 4640:
            raise ValueError(f"seed {seed} is not 4640/4640")
        if result.get("formal_sample_count") != 4640:
            raise ValueError(f"seed {seed} formal sample count is not 4640")
        if result.get("unmatched_count") != 0 or result.get("runtime_fallback_count") != 0:
            raise ValueError(f"seed {seed} contains unmatched samples or runtime fallbacks")
        if result.get("sample_index_sha256") != index.get("sample_index_sha256"):
            raise ValueError(f"seed {seed} sample-index hash mismatch")

        protocol = result["protocol"]
        if protocol.get("generation_length_mode") != "reference":
            raise ValueError(f"seed {seed} did not use reference-length evaluation")
        evaluator_length_mode = protocol.get(
            "evaluator_length_mode", "min(generated_length,reference_length)"
        )
        if evaluator_length_mode != "min(generated_length,reference_length)":
            raise ValueError(f"seed {seed} evaluator length mode is incorrect")
        if protocol.get("pair_rng_mode") != "sequential":
            raise ValueError(f"seed {seed} did not use sequential pair RNG")
        if protocol.get("sampling") != EXPECTED_SAMPLING:
            raise ValueError(f"seed {seed} did not use explicit full-categorical sampling")
        loader = protocol.get("loader", {})
        expected_loader = {"batch_size": 32, "shuffle": True, "drop_last": True, "expected_samples": 4640}
        if any(loader.get(key) != value for key, value in expected_loader.items()):
            raise ValueError(f"seed {seed} loader settings differ from the legacy 4,640 protocol")
        if len(diagnostics.get("batches", [])) != 145:
            raise ValueError(f"seed {seed} diagnostics do not contain 145 batches")
        if sum(int(batch.get("count", 0)) for batch in diagnostics["batches"]) != 4640:
            raise ValueError(f"seed {seed} diagnostics do not cover 4,640 samples")

        identities["checkpoint_sha256"].add(result["artifacts"]["checkpoint"]["sha256"])
        identities["evaluator_sha256"].add(result["artifacts"]["evaluator_checkpoint"]["sha256"])
        identities["raw_protocol_fingerprint"].add(result["protocol_fingerprint"])
        normalized_protocol = dict(protocol)
        normalized_protocol["evaluator_length_mode"] = evaluator_length_mode
        normalized_payload = json.dumps(
            normalized_protocol, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        identities["normalized_protocol_fingerprint"].add(
            hashlib.sha256(normalized_payload).hexdigest()
        )
        identities["source_manifest_fingerprint"].add(result["source_manifest_fingerprint"])
        results.append(result)
        rows.append({
            "seed": seed,
            **{metric: float(result["metrics"][metric]) for metric in METRICS},
            "matched_count": 4640,
            "total_count": 4640,
            "sample_index_sha256": result["sample_index_sha256"],
        })

    singleton_identities = (
        "checkpoint_sha256",
        "evaluator_sha256",
        "normalized_protocol_fingerprint",
        "source_manifest_fingerprint",
    )
    if any(len(identities[key]) != 1 for key in singleton_identities):
        raise ValueError("checkpoint, evaluator, normalized protocol, or source manifest differs across seeds")

    motiongpt_summary = metric_summary(rows)
    comparison = {
        "MotionGPT_corrected_reference_length": motiongpt_summary,
        "MotionGPT_paper": {"FID": {"mean": 0.160, "reported_ci95": 0.008}},
    }
    if args.sato_runs:
        comparison["SATO-50k"] = metric_summary(load_sato_rows(args.sato_runs))

    first = results[0]
    protocol_manifest = {
        "name": "humanml3d-legacy-loader-reference-length-repeat20/v1",
        "dataset": "HumanML3D",
        "seeds": EXPECTED_SEEDS,
        "replications": 20,
        "samples_per_seed": 4640,
        "batches_per_seed": 145,
        "pair_matching_per_seed": "4640/4640",
        "generation_length_mode": "min(generated_length, ground_truth_length)",
        "pair_rng_mode": "sequential",
        "sampling": EXPECTED_SAMPLING,
        "loader": first["protocol"]["loader"],
        "metric_scope": {
            "FID": "ground truth vs original-caption generation",
            "FID_P": "ground truth vs perturbed-caption generation",
            "FID_D": "original-caption vs perturbed-caption generation",
            "Matching/R-Precision": "original loader captions only",
        },
        "statistics": {
            "paper_style_ci95": "1.96 * np.std(values, ddof=0) / sqrt(20)",
            "sample_std": "np.std(values, ddof=1)",
        },
        "identities": {
            "checkpoint_sha256": next(iter(identities["checkpoint_sha256"])),
            "evaluator_sha256": next(iter(identities["evaluator_sha256"])),
            "source_manifest_fingerprint": next(iter(identities["source_manifest_fingerprint"])),
            "normalized_protocol_fingerprint": next(
                iter(identities["normalized_protocol_fingerprint"])
            ),
            "raw_protocol_fingerprints": sorted(identities["raw_protocol_fingerprint"]),
            "raw_protocol_fingerprint_note": (
                "Two raw values exist because evaluator_length_mode metadata was added during the run; "
                "normalization fills the missing field with the implemented reference-length behavior."
            ),
        },
        "artifacts": first["artifacts"],
        "pair_file": first["protocol"]["pair_file"],
        "run_dir": str(args.run_dir.resolve()),
        "run_artifact_counts": {"results": 20, "sample_indices": 20, "diagnostics": 20},
        "limitations": [
            "Loader captions are not locked to pair-file text_orig.",
            "P1/P2/P3 perturbation levels are mixed, not reported separately.",
            "Perturbed-caption Matching and R-Precision are not computed.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["seed", *METRICS, "matched_count", "total_count", "sample_index_sha256"]
    with (args.output_dir / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"motiongpt": {"n": 20, "summary": motiongpt_summary}, "comparison": comparison}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "protocol_manifest.json").write_text(
        json.dumps(protocol_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# MotionGPT HumanML3D corrected repeat-20",
        "",
        "Seeds 123--142; 4,640/4,640 matched samples per seed; reference-length evaluation.",
        "",
        "| Metric | Mean | Paper-style 95% CI | Sample std |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        item = motiongpt_summary[metric]
        lines.append(f"| {metric} | {item['mean']:.4f} | ± {item['paper_style_ci95']:.4f} | {item['sample_std_ddof1']:.4f} |")
    lines.extend([
        "",
        f"Corrected MotionGPT FID: **{motiongpt_summary['FID']['mean']:.4f} ± {motiongpt_summary['FID']['paper_style_ci95']:.4f}** (95% CI).",
        "",
        "MotionGPT paper FID: **0.160 ± 0.008**.",
        "",
        "Previous free-length evaluation artifacts were invalid and have been removed.",
    ])
    if "SATO-50k" in comparison:
        sato_fid = comparison["SATO-50k"]["FID"]
        lines.extend(["", f"SATO-50k FID under the same 20-seed loader protocol: **{sato_fid['mean']:.4f} ± {sato_fid['paper_style_ci95']:.4f}**."])
    lines.extend(["", "The ± column follows MotionGPT `test.py`: population std (`ddof=0`) converted to a normal-approximation 95% confidence interval.", ""])
    (args.output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
