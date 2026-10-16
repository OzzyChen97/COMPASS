#!/usr/bin/env python3
"""Aggregate the legacy HumanML fast-protocol SATO-50k/MotionGPT rerun."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


METRICS = ("FID", "FID_P", "FID_D", "Matching", "R@1", "R@2", "R@3", "Diversity")
SATO_KEYS = {
    "FID": "fid",
    "FID_P": "fid_p",
    "FID_D": "fid_d",
    "Matching": "matching",
    "R@1": "top1",
    "R@2": "top2",
    "R@3": "top3",
    "Diversity": "diversity",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(np.mean([float(row[metric]) for row in rows])),
            "std": float(np.std([float(row[metric]) for row in rows], ddof=1)),
        }
        for metric in METRICS
    }


def paired_comparison(
    left: list[dict[str, object]], right: list[dict[str, object]]
) -> dict[str, dict[str, float | int]]:
    left_by_seed = {int(row["seed"]): row for row in left}
    right_by_seed = {int(row["seed"]): row for row in right}
    comparison = {}
    for metric in METRICS:
        differences = np.asarray(
            [
                float(left_by_seed[seed][metric])
                - float(right_by_seed[seed][metric])
                for seed in range(123, 143)
            ],
            dtype=np.float64,
        )
        std = float(np.std(differences, ddof=1))
        comparison[metric] = {
            "mean_difference": float(np.mean(differences)),
            "sample_std": std,
            "normal_ci95": float(1.96 * std / math.sqrt(len(differences))),
            "left_lower_count": int(np.sum(differences < 0)),
            "right_lower_count": int(np.sum(differences > 0)),
            "tie_count": int(np.sum(differences == 0)),
        }
    return comparison


def load_sato(shard_root: Path) -> list[dict[str, object]]:
    paths = sorted(shard_root.glob("shard_gpu*/*/runs.csv"))
    if len(paths) != 4:
        raise SystemExit(f"expected four SATO shard CSVs, found {len(paths)}")
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "model": "SATO-50k",
                        "seed": int(row["seed"]),
                        **{metric: float(row[key]) for metric, key in SATO_KEYS.items()},
                        "matched_count": int(row["matched_pairs"]),
                        "total_count": int(row["total_pairs"]),
                    }
                )
    return sorted(rows, key=lambda row: int(row["seed"]))


def load_sato_pp(runs_path: Path) -> list[dict[str, object]]:
    rows = []
    with runs_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "model": "SATO++",
                    "seed": int(row["seed"]),
                    **{metric: float(row[key]) for metric, key in SATO_KEYS.items()},
                    "matched_count": int(row["matched_pairs"]),
                    "total_count": int(row["total_pairs"]),
                }
            )
    return sorted(rows, key=lambda row: int(row["seed"]))


def load_motiongpt(run_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    raw = []
    for seed in range(123, 143):
        result = json.loads((run_dir / f"seed{seed}.json").read_text())
        index = json.loads((run_dir / f"seed{seed}_sample_index.json").read_text())
        if result["seed"] != seed or index["seed"] != seed:
            raise SystemExit(f"MotionGPT seed mismatch for {seed}")
        if result["sample_index_sha256"] != index["sample_index_sha256"]:
            raise SystemExit(f"MotionGPT sample-index mismatch for {seed}")
        if result["protocol"].get("pair_rng_mode") != "sequential":
            raise SystemExit(f"MotionGPT seed {seed} is not sequential RNG")
        if result["protocol"].get("prompt", {}).get("mode") != "official-default":
            raise SystemExit(f"MotionGPT seed {seed} is not using the official default prompt")
        if result["protocol"].get("generation_order") != "original-first":
            raise SystemExit(f"MotionGPT seed {seed} generation order is not original-first")
        if result["protocol"].get("generation_length_mode") != "generated":
            raise SystemExit(f"MotionGPT seed {seed} is not using SATO++ generated lengths")
        if (
            result["protocol"].get("evaluator_length_mode")
            != "generated_length_capped_to_batch_seq"
        ):
            raise SystemExit(f"MotionGPT seed {seed} evaluator length contract mismatch")
        sampling = result["protocol"].get("sampling", {})
        expected_sampling = {
            "mode": "categorical",
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "kwargs_mode": "explicit",
            "implementation": "explicit Hugging Face generate kwargs",
        }
        if sampling != expected_sampling:
            raise SystemExit(
                f"MotionGPT seed {seed} sampling mismatch: {sampling}"
            )
        raw.append(result)
        rows.append(
            {
                "model": "MotionGPT",
                "seed": seed,
                **{metric: float(result["metrics"][metric]) for metric in METRICS},
                "matched_count": int(result["matched_count"]),
                "total_count": int(result["total_count"]),
            }
        )
    return rows, raw


def validate(rows: list[dict[str, object]], label: str) -> None:
    seeds = [int(row["seed"]) for row in rows]
    if seeds != list(range(123, 143)):
        raise SystemExit(f"{label} seeds mismatch: {seeds}")
    if any(row["matched_count"] != 4640 or row["total_count"] != 4640 for row in rows):
        raise SystemExit(f"{label} contains a run that is not 4640/4640")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sato-shard-root", type=Path, required=True)
    parser.add_argument("--sato-pp-runs", type=Path)
    parser.add_argument("--motiongpt-run-dir", type=Path, required=True)
    parser.add_argument("--sato-checkpoint", type=Path, required=True)
    parser.add_argument("--sato-clip-checkpoint", type=Path, required=True)
    parser.add_argument("--motiongpt-checkpoint", type=Path, required=True)
    parser.add_argument("--evaluator-checkpoint", type=Path, required=True)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sato = load_sato(args.sato_shard_root)
    sato_pp = load_sato_pp(args.sato_pp_runs) if args.sato_pp_runs else []
    motiongpt, motiongpt_raw = load_motiongpt(args.motiongpt_run_dir)
    validate(sato, "SATO-50k")
    if sato_pp:
        validate(sato_pp, "SATO++")
    validate(motiongpt, "MotionGPT")
    fingerprints = {row["protocol_fingerprint"] for row in motiongpt_raw}
    if len(fingerprints) != 1:
        raise SystemExit("MotionGPT protocol fingerprints differ")

    model_rows = {"SATO-50k": sato, "MotionGPT": motiongpt}
    if sato_pp:
        model_rows["SATO++"] = sato_pp
    summaries = {model: summarize(rows) for model, rows in model_rows.items()}
    paired_vs_sato_pp = (
        paired_comparison(motiongpt, sato_pp) if sato_pp else None
    )
    protocol = {
        "name": "legacy-humanml-fast-mixed-caption/v1",
        "dataset": "HumanML3D",
        "seeds": list(range(123, 143)),
        "loader": {
            "implementation": "dataset.dataset_TM_eval.DATALoader",
            "batch_size": 32,
            "shuffle": True,
            "drop_last": True,
            "caption_locked": False,
            "samples_per_seed": 4640,
        },
        "pair_matching_per_seed": "4640/4640",
        "pair_lookup": "motion_id",
        "pair_rng_mode": "sequential",
        "sampling": {
            "SATO-50k": {
                "mode": "categorical",
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "cfg_scale": 1.0,
            },
            "MotionGPT": {
                "mode": "categorical",
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "implementation": "explicit_hf_generate_kwargs",
                "cfg": "not_applicable",
            },
            **({
                "SATO++": {
                    "mode": "categorical",
                    "temperature": 1.0,
                    "top_k": 0,
                    "top_p": 1.0,
                    "cfg_scale": 1.0,
                }
            } if sato_pp else {}),
        },
        "metric_scope": {
            "FID": "ground-truth vs original-caption generated distribution",
            "FID_P": "ground-truth vs perturbed-caption generated distribution",
            "FID_D": "original-caption vs perturbed-caption generated distribution",
            "Matching/R-Precision": "original loader captions only",
        },
        "limitations": [
            "Original captions are randomly selected by the loader and are not locked to pair-file text_orig.",
            "P1/P2/P3 levels are mixed and not reported separately.",
            "Perturbed-caption Matching and R-Precision are not computed.",
        ],
        "std_ddof": 1,
        "motiongpt_protocol_fingerprint": next(iter(fingerprints)),
        "artifacts": {
            "pair_file": {"path": str(args.pair_file), "sha256": sha256(args.pair_file)},
            "test_split": {"path": str(args.test_split), "sha256": sha256(args.test_split)},
            "SATO-50k": {
                "checkpoint": {"path": str(args.sato_checkpoint), "sha256": sha256(args.sato_checkpoint)},
                "clip_checkpoint": {"path": str(args.sato_clip_checkpoint), "sha256": sha256(args.sato_clip_checkpoint)},
            },
            "MotionGPT": {
                "checkpoint": {"path": str(args.motiongpt_checkpoint), "sha256": sha256(args.motiongpt_checkpoint)},
            },
            "evaluator": {"path": str(args.evaluator_checkpoint), "sha256": sha256(args.evaluator_checkpoint)},
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = sorted(
        [row for rows in model_rows.values() for row in rows],
        key=lambda row: (str(row["model"]), int(row["seed"])),
    )
    fields = ["model", "seed", *METRICS, "matched_count", "total_count"]
    with (args.output_dir / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(combined)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "summaries": summaries,
                "paired_motiongpt_minus_sato_pp": paired_vs_sato_pp,
                "protocol": protocol,
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    (args.output_dir / "protocol_manifest.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# HumanML3D legacy-standard rerun",
        "",
        "All models use the legacy shuffled 4,640-sample HumanML fast protocol over seeds 123--142.",
        "",
        "| Metric | SATO-50k |" + (" SATO++ |" if sato_pp else "") + " MotionGPT |",
        "| --- | ---: |" + (" ---: |" if sato_pp else "") + " ---: |",
    ]
    for metric in METRICS:
        left = summaries["SATO-50k"][metric]
        right = summaries["MotionGPT"][metric]
        middle = summaries.get("SATO++", {}).get(metric)
        lines.append(
            f"| {metric} | {left['mean']:.4f} +/- {left['std']:.4f} |"
            + (f" {middle['mean']:.4f} +/- {middle['std']:.4f} |" if middle else "")
            + " "
            f"{right['mean']:.4f} +/- {right['std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Standard deviations use ddof=1.",
            "",
            "This is a legacy mixed-caption distribution evaluation, not a caption-locked or per-level P1/P2/P3 test.",
            "",
        ]
    )
    if paired_vs_sato_pp:
        lines.extend([
            "## Paired MotionGPT - SATO++",
            "",
            "Negative differences favor MotionGPT for lower-is-better FID metrics.",
            "",
            "| Metric | Mean difference | 95% CI | MotionGPT lower | SATO++ lower |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for metric in ("FID", "FID_P", "FID_D"):
            item = paired_vs_sato_pp[metric]
            lines.append(
                f"| {metric} | {item['mean_difference']:+.4f} | "
                f"+/- {item['normal_ci95']:.4f} | "
                f"{item['left_lower_count']} | {item['right_lower_count']} |"
            )
        lines.append("")
    (args.output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
