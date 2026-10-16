#!/usr/bin/env python3
"""Validate and aggregate HumanML3D SATO shards against SATO++ repeat20."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

METRICS = ("fid", "fid_p", "fid_d", "matching", "top1", "top2", "top3", "diversity")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows):
    return {
        metric: {
            "mean": float(np.mean([float(row[metric]) for row in rows])),
            "std": float(np.std([float(row[metric]) for row in rows], ddof=1)),
        }
        for metric in METRICS
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--sato-pp-runs", type=Path, required=True)
    parser.add_argument("--sato-100k-runs", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    shard_files = sorted(args.shard_root.glob("shard_gpu*/*/runs.csv"))
    if len(shard_files) != 4:
        raise SystemExit(f"expected 4 shard CSVs, found {len(shard_files)}")
    sato = [row for path in shard_files for row in read_rows(path)]
    sato_pp = read_rows(args.sato_pp_runs)
    sato_100k = read_rows(args.sato_100k_runs) if args.sato_100k_runs else None
    expected_seeds = list(range(123, 143))
    for label, rows in (("SATO", sato), ("SATO++", sato_pp)):
        seeds = sorted(int(row["seed"]) for row in rows)
        if seeds != expected_seeds:
            raise SystemExit(f"{label} seeds mismatch: {seeds}")
        if any(int(row["matched_pairs"]) != 4640 or int(row["total_pairs"]) != 4640 for row in rows):
            raise SystemExit(f"{label} contains a run that is not 4640/4640")
    if sato_100k is not None:
        seeds = sorted(int(row["seed"]) for row in sato_100k)
        if seeds != expected_seeds or any(
            int(row["matched_pairs"]) != 4640 or int(row["total_pairs"]) != 4640
            for row in sato_100k
        ):
            raise SystemExit("SATO-100k reference is not a complete 4640/4640 repeat20")
    if len({row["path"] for row in sato}) != 1:
        raise SystemExit("SATO shards do not share one checkpoint path")

    sato = sorted(sato, key=lambda row: int(row["seed"]))
    sato_pp = sorted(sato_pp, key=lambda row: int(row["seed"]))
    sato_summary, pp_summary = summarize(sato), summarize(sato_pp)
    sato_100k_summary = summarize(sato_100k) if sato_100k is not None else None
    comparison = {
        metric: {
            "sato_mean": sato_summary[metric]["mean"],
            "sato_std": sato_summary[metric]["std"],
            "sato_pp_mean": pp_summary[metric]["mean"],
            "sato_pp_std": pp_summary[metric]["std"],
            "sato_minus_sato_pp": sato_summary[metric]["mean"] - pp_summary[metric]["mean"],
        }
        for metric in METRICS
    }
    protocol = {
        "dataset": "HumanML3D",
        "seeds": expected_seeds,
        "loader": "dataset.dataset_TM_eval.DATALoader",
        "batch_size": 32,
        "shuffle": True,
        "drop_last": True,
        "samples_per_seed": 4640,
        "pair_matching_per_seed": "4640/4640",
        "fast_batch": True,
        "cfg_scale": 1.0,
        "cfg_uncond": "zero",
        "sampling": {"mode": "categorical", "temperature": 1.0, "top_k": 0, "top_p": 1.0},
        "std_ddof": 1,
        "test_split": {"path": str(args.test_split), "sha256": file_hash(args.test_split)},
        "pair_file": {"path": str(args.pair_file), "sha256": file_hash(args.pair_file)},
        "sato_checkpoint": {"path": str(args.checkpoint), "sha256": file_hash(args.checkpoint)},
        "sato_clip_checkpoint": {"path": str(args.clip_checkpoint), "sha256": file_hash(args.clip_checkpoint)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(sato[0].keys())
    with (args.output_dir / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(sato)
    payload = {
        "protocol": protocol,
        "sato": {"n": len(sato), "summary": sato_summary},
        "sato_pp": {"n": len(sato_pp), "source_runs": str(args.sato_pp_runs), "summary": pp_summary},
        "comparison": comparison,
    }
    if sato_100k_summary is not None:
        payload["sato_100k"] = {
            "n": len(sato_100k),
            "source_runs": str(args.sato_100k_runs),
            "summary": sato_100k_summary,
        }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "protocol_manifest.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    labels = {"fid": "FID", "fid_p": "FID_P", "fid_d": "FID_D", "matching": "Matching", "top1": "R@1", "top2": "R@2", "top3": "R@3", "diversity": "Diversity"}
    lines = [
        "# HumanML3D SATO (reproduced) vs SATO++",
        "",
        "Both models use the same HumanML3D test split, perturbation file, seeds 123--142, 4,640 loader samples per seed, fast evaluator, categorical sampling, and CFG=1.0.",
        "",
        ("| Metric | SATO-50k | SATO-100k | SATO++ |" if sato_100k_summary is not None
         else "| Metric | SATO (reproduced) | SATO++ | SATO - SATO++ |"),
        ("| --- | ---: | ---: | ---: |" if sato_100k_summary is not None
         else "| --- | ---: | ---: | ---: |"),
    ]
    for metric in METRICS:
        row = comparison[metric]
        if sato_100k_summary is not None:
            mid = sato_100k_summary[metric]
            lines.append(f"| {labels[metric]} | {row['sato_mean']:.4f} ± {row['sato_std']:.4f} | {mid['mean']:.4f} ± {mid['std']:.4f} | {row['sato_pp_mean']:.4f} ± {row['sato_pp_std']:.4f} |")
        else:
            lines.append(f"| {labels[metric]} | {row['sato_mean']:.4f} ± {row['sato_std']:.4f} | {row['sato_pp_mean']:.4f} ± {row['sato_pp_std']:.4f} | {row['sato_minus_sato_pp']:+.4f} |")
    lines.extend(["", "Standard deviations are sample standard deviations (`ddof=1`).", ""])
    (args.output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
