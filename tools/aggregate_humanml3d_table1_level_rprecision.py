#!/usr/bin/env python3
"""Strictly audit and aggregate independent HumanML3D P1/P2/P3 R-Precision."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.aggregate_motiongpt_loader_eval import (  # noqa: E402
    aggregate as aggregate_motiongpt,
    canonical_sha256,
    file_sha256,
    retrieval_metric_failures,
)


LEVELS = ("P1", "P2", "P3")
MODEL_ORDER = ("t2mgpt", "motiongpt", "satopp")
MODEL_LABELS = {
    "t2mgpt": "T2M-GPT",
    "motiongpt": "MotionGPT",
    "satopp": "SATO++",
}
T2M_INTERNAL_MODELS = {
    "t2mgpt": "baseline_t2mgpt_original",
    "satopp": "sato_pp_allloss_l1_iter130k",
}
METRICS = ("top1_pert", "top2_pert", "top3_pert")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_monotonic(metrics: dict[str, float], identity: str) -> None:
    values = [float(metrics[name]) for name in METRICS]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{identity} contains a non-finite R-Precision value")
    if not (0.0 <= values[0] <= values[1] <= values[2] <= 1.0):
        raise ValueError(f"{identity} violates 0 <= R@1 <= R@2 <= R@3 <= 1")


def audit_t2m_seed_diagnostics(
    diagnostic_path: Path,
    sample_index_path: Path,
    row: dict[str, Any],
    *,
    expected_samples: int,
    batch_size: int,
) -> dict[str, Any]:
    sample_index = json.loads(sample_index_path.read_text(encoding="utf-8"))
    samples = sample_index.get("samples")
    if not isinstance(samples, list) or len(samples) != expected_samples:
        raise ValueError(f"sample index count mismatch: {sample_index_path}")
    sample_hash = canonical_sha256(samples)
    if (
        sample_index.get("sample_index_sha256") != sample_hash
        or row.get("sample_index_sha256") != sample_hash
    ):
        raise ValueError(f"sample index checksum mismatch: {sample_index_path}")

    with np.load(diagnostic_path, allow_pickle=False) as dump:
        required = {
            "perturbed_text_embedding", "perturbed_embedding", "loader_name",
            "motion_id", "caption", "perturbed_caption", "level",
        }
        missing = sorted(required - set(dump.files))
        if missing:
            raise ValueError(f"{diagnostic_path} lacks {missing}")
        text = np.asarray(dump["perturbed_text_embedding"])
        motion = np.asarray(dump["perturbed_embedding"])
        arrays = (
            np.asarray(dump["loader_name"]).astype(str).tolist(),
            np.asarray(dump["motion_id"]).astype(str).tolist(),
            np.asarray(dump["caption"]).astype(str).tolist(),
            np.asarray(dump["perturbed_caption"]).astype(str).tolist(),
            np.asarray(dump["level"]).astype(str).tolist(),
        )
    if text.ndim != 2 or motion.ndim != 2 or text.shape != motion.shape:
        raise ValueError(f"embedding shape mismatch: {diagnostic_path}")
    if text.shape[0] != expected_samples or any(
        len(values) != expected_samples for values in arrays
    ):
        raise ValueError(f"diagnostic sample count mismatch: {diagnostic_path}")
    for offset, sample in enumerate(samples):
        expected = (
            str(sample["loader_name"]), str(sample["motion_id"]),
            str(sample["caption"]), str(sample["perturbed_caption"]),
            str(sample["level"]),
        )
        if tuple(values[offset] for values in arrays) != expected:
            raise ValueError(
                f"diagnostic/sample identity mismatch at seed {row['seed']} "
                f"offset {offset}"
            )

    counts = np.zeros(3, dtype=np.int64)
    matching_sum = 0.0
    batch_sizes = []
    for start in range(0, expected_samples, batch_size):
        stop = min(start + batch_size, expected_samples)
        current = stop - start
        if current < 3:
            raise ValueError("retrieval batch is too small for R@3")
        batch_sizes.append(current)
        text_batch = text[start:stop]
        motion_batch = motion[start:stop]
        squared = (
            -2 * np.dot(text_batch, motion_batch.T)
            + np.sum(np.square(text_batch), axis=1, keepdims=True)
            + np.sum(np.square(motion_batch), axis=1)
        )
        distances = np.sqrt(np.maximum(squared, 0.0))
        ranking = np.argsort(distances, axis=1)
        targets = np.arange(current)
        correct = np.zeros(current, dtype=bool)
        for top_index in range(3):
            correct |= ranking[:, top_index] == targets
            counts[top_index] += int(correct.sum())
        matching_sum += float(np.trace(distances))
    recomputed = {
        "top1_pert": float(counts[0] / expected_samples),
        "top2_pert": float(counts[1] / expected_samples),
        "top3_pert": float(counts[2] / expected_samples),
        "matching_pert": float(matching_sum / expected_samples),
    }
    errors = {
        metric: abs(float(row[metric]) - value)
        for metric, value in recomputed.items()
    }
    metric_failures = retrieval_metric_failures(errors)
    if metric_failures:
        raise ValueError(
            f"seed {row['seed']} embedding recomputation mismatch: {errors}"
        )
    return {
        "seed": int(row["seed"]),
        "diagnostic_path": str(diagnostic_path.resolve()),
        "diagnostic_sha256": file_sha256(diagnostic_path),
        "sample_index_sha256": sample_hash,
        "sample_count": expected_samples,
        "batch_count": len(batch_sizes),
        "batch_sizes": batch_sizes,
        "counts": counts.tolist(),
        "max_abs_metric_error": max(errors.values()),
    }


def collect_t2m_family(
    run_root: Path,
    level: str,
    model_key: str,
    seeds: list[int],
    expected_samples: int,
    expected_index_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    internal = T2M_INTERNAL_MODELS[model_key]
    group_root = run_root / "t2mgpt_satopp" / level.lower() / internal
    model_root = group_root / internal
    rows_path = model_root / "runs.csv"
    rows_by_seed = {
        int(row["seed"]): row for row in read_csv(rows_path)
        if int(row["seed"]) in seeds
    }
    if set(rows_by_seed) != set(seeds):
        missing = sorted(set(seeds) - set(rows_by_seed))
        raise ValueError(f"{model_key}/{level} lacks seeds {missing}")

    protocol = json.loads(
        (group_root / "protocol_manifest.json").read_text(encoding="utf-8")
    )
    if protocol.get("sample_count") != expected_samples:
        raise ValueError(f"{model_key}/{level} protocol sample count mismatch")
    if protocol.get("level_counts") != {level: expected_samples}:
        raise ValueError(f"{model_key}/{level} protocol level count mismatch")
    if protocol.get("sample_index_sha256") != expected_index_hash:
        raise ValueError(f"{model_key}/{level} protocol frozen-index mismatch")
    if protocol.get("sampling") != {
        "mode": "categorical", "temperature": 1.0, "top_k": 0, "top_p": 1.0,
    }:
        raise ValueError(f"{model_key}/{level} sampling protocol mismatch")
    if protocol.get("rng") != "sequential":
        raise ValueError(f"{model_key}/{level} pair RNG is not sequential")
    if protocol.get("generation_order") != "original-first":
        raise ValueError(f"{model_key}/{level} generation order mismatch")
    if protocol.get("evaluator_length") != "generated_length":
        raise ValueError(f"{model_key}/{level} evaluator length mismatch")

    collected = []
    audits = []
    for seed in seeds:
        row = rows_by_seed[seed]
        identity = f"{model_key}/{level}/seed{seed}"
        for key in ("matched_pairs", "total_pairs", "perturbed_retrieval_samples"):
            if int(row[key]) != expected_samples:
                raise ValueError(f"{identity} {key} is not {expected_samples}")
        if row.get("sample_index_sha256") != expected_index_hash:
            raise ValueError(f"{identity} frozen-index hash mismatch")
        metrics = {name: float(row[name]) for name in METRICS}
        finite_monotonic(metrics, identity)
        audit = audit_t2m_seed_diagnostics(
            group_root / "diagnostics" / internal / f"seed{seed}.npz",
            model_root / "sample_indices" / f"seed_{seed}.json",
            row,
            expected_samples=expected_samples,
            batch_size=32,
        )
        if audit["batch_sizes"][-1] != expected_samples % 32:
            raise ValueError(f"{identity} retrieval tail batch mismatch")
        audits.append(audit)
        collected.append({
            "model": model_key,
            "method": MODEL_LABELS[model_key],
            "level": level,
            "seed": seed,
            "samples": expected_samples,
            **metrics,
            "sample_index_sha256": expected_index_hash,
            "online_offline_max_abs_error": float(audit["max_abs_metric_error"]),
        })
    return collected, {
        "protocol_fingerprint": protocol.get("fingerprint"),
        "verified_seed_count": len(audits),
        "max_abs_metric_error": max(
            float(audit["max_abs_metric_error"]) for audit in audits
        ),
        "diagnostics": audits,
    }


def collect_motiongpt(
    run_root: Path,
    source_root: Path,
    level: str,
    seeds: list[int],
    expected_samples: int,
    expected_index_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = run_root / "motiongpt" / level.lower()
    source_manifest = source_root / level.lower() / "manifest.json"
    rows, summary = aggregate_motiongpt(
        run_dir,
        "humanml3d",
        expected_samples=expected_samples,
        require_caption_locked=True,
        expected_unique_pairs=expected_samples,
        expected_levels=[level],
        source_manifest=source_manifest,
        require_retrieval_embeddings=True,
        start_seed=seeds[0],
        num_repeats=len(seeds),
    )
    if summary["identities"]["sample_index"] != expected_index_hash:
        raise ValueError(f"motiongpt/{level} frozen-index hash mismatch")
    collected = []
    for row in rows:
        metrics = {name: float(row[name]) for name in METRICS}
        identity = f"motiongpt/{level}/seed{row['seed']}"
        finite_monotonic(metrics, identity)
        collected.append({
            "model": "motiongpt",
            "method": MODEL_LABELS["motiongpt"],
            "level": level,
            "seed": int(row["seed"]),
            "samples": expected_samples,
            **metrics,
            "sample_index_sha256": row["sample_index_sha256"],
            "online_offline_max_abs_error": float(
                summary["retrieval_embedding_audit"]["max_abs_metric_error"]
            ),
        })
    return collected, summary


def summarize_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["level"]), str(row["model"]))].append(row)

    output = []
    for level in LEVELS:
        for model in MODEL_ORDER:
            values = grouped[(level, model)]
            if not values:
                raise ValueError(f"no runs collected for {model}/{level}")
            row: dict[str, Any] = {
                "level": level,
                "model": model,
                "method": MODEL_LABELS[model],
                "n": len(values),
                "samples_per_seed": int(values[0]["samples"]),
                "sample_index_sha256": values[0]["sample_index_sha256"],
                "online_offline_max_abs_error": max(
                    float(value["online_offline_max_abs_error"]) for value in values
                ),
            }
            for metric in METRICS:
                array = np.asarray(
                    [float(value[metric]) for value in values], dtype=np.float64
                )
                mean = float(array.mean())
                population_std = float(array.std(ddof=0))
                sample_std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
                ci95 = float(1.96 * population_std / math.sqrt(len(array)))
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sample_std"] = sample_std
                row[f"{metric}_population_std"] = population_std
                row[f"{metric}_ci95"] = ci95
                row[f"{metric}_ci95_low"] = mean - ci95
                row[f"{metric}_ci95_high"] = mean + ci95
            output.append(row)
    return output


def format_metric(row: dict[str, Any], metric: str) -> str:
    return f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_sample_std']:.4f}"


def format_latex_metric(row: dict[str, Any], metric: str) -> str:
    return format_metric(row, metric).replace(" ± ", r" $\pm$ ")


def write_outputs(
    output_dir: Path,
    all_runs: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_fields = [
        "level", "model", "method", "seed", "samples",
        *METRICS, "sample_index_sha256", "online_offline_max_abs_error",
    ]
    with (output_dir / "all_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(all_runs)

    summary_fields = list(summary_rows[0])
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    json_payload = {
        "schema_version": "humanml3d-table1-independent-level-rprecision/v1",
        "manifest": manifest,
        "summary": summary_rows,
        "all_runs": all_runs,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    markdown = [
        "# HumanML3D Independent-Level Perturbed R-Precision",
        "",
        "Each P1/P2/P3 result uses an independent caption-locked candidate pool ",
        "with 4,380 samples, batch size 32, no shuffle, and no dropped tail.",
        "Values are mean ± sample standard deviation across evaluation seeds.",
        "",
        "| Level | Method | n | R@1_P | R@2_P | R@3_P |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        markdown.append(
            f"| {row['level']} | {row['method']} | {row['n']} | "
            f"{format_metric(row, 'top1_pert')} | "
            f"{format_metric(row, 'top2_pert')} | "
            f"{format_metric(row, 'top3_pert')} |"
        )
    (output_dir / "table.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    latex = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Level & Method & R@1$_P$ $\uparrow$ & R@2$_P$ $\uparrow$ & R@3$_P$ $\uparrow$ \\",
        r"\midrule",
    ]
    for index, row in enumerate(summary_rows):
        if index and row["level"] != summary_rows[index - 1]["level"]:
            latex.append(r"\midrule")
        latex.append(
            f"{row['level']} & {row['method']} & "
            f"{format_latex_metric(row, 'top1_pert')} & "
            f"{format_latex_metric(row, 'top2_pert')} & "
            f"{format_latex_metric(row, 'top3_pert')} \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "table.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )
    (output_dir / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("ablation/humanml3d_table1_levels/runs"),
    )
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        default=Path("ablation/humanml3d_table1_levels/assets_manifest.json"),
    )
    parser.add_argument(
        "--motiongpt-source-root",
        type=Path,
        default=Path("ablation/humanml3d_table1_levels/motiongpt_sources"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ablation/humanml3d_table1_levels/final"),
    )
    parser.add_argument("--start-seed", type=int, default=123)
    parser.add_argument("--num-repeats", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_repeats < 1:
        raise ValueError("num_repeats must be positive")
    seeds = list(range(args.start_seed, args.start_seed + args.num_repeats))
    if min(seeds) < 123 or max(seeds) > 142:
        raise ValueError("requested seeds must be an ordered subset of 123..142")

    assets = json.loads(args.asset_manifest.read_text(encoding="utf-8"))
    expected_samples = int(assets["expected_samples_per_level"])
    all_runs: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for level in LEVELS:
        expected_hash = assets["assets"][level]["sample_index_sha256"]
        for model in ("t2mgpt", "satopp"):
            rows, audit = collect_t2m_family(
                args.run_root,
                level,
                model,
                seeds,
                expected_samples,
                expected_hash,
            )
            all_runs.extend(rows)
            audits[f"{model}/{level}"] = audit
        rows, audit = collect_motiongpt(
            args.run_root,
            args.motiongpt_source_root,
            level,
            seeds,
            expected_samples,
            expected_hash,
        )
        all_runs.extend(rows)
        audits[f"motiongpt/{level}"] = audit

    all_runs.sort(
        key=lambda row: (
            LEVELS.index(str(row["level"])),
            MODEL_ORDER.index(str(row["model"])),
            int(row["seed"]),
        )
    )
    summary_rows = summarize_runs(all_runs)

    checkpoint_paths = {
        "t2mgpt_vq": REPO_ROOT / "pretrained/VQVAE/net_best_fid.pth",
        "t2mgpt_transformer": (
            REPO_ROOT / "pretrained/VQTransformer_corruption05/net_best_fid.pth"
        ),
        "satopp": REPO_ROOT / (
            "output_GPT_Final/"
            "sato_pp_allloss_gpu0_teachl1_05_eval5k_resume104k_200k_single/"
            "sato_pp_iter_130000.pth"
        ),
        "motiongpt": REPO_ROOT / (
            "MotionGPT/experiments/mgpt/"
            "Instruct_H3D_OFFICIAL_FRESH_SEED123_20260818/checkpoints/last.ckpt"
        ),
        "tm2t_evaluator": (
            REPO_ROOT / "checkpoints/t2m/text_mot_match/model/finest.tar"
        ),
    }
    checkpoint_hashes = {
        name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for name, path in checkpoint_paths.items()
    }
    motiongpt_evaluator_hashes = {
        audit["identities"]["evaluator"]
        for name, audit in audits.items() if name.startswith("motiongpt/")
    }
    if motiongpt_evaluator_hashes != {checkpoint_hashes["tm2t_evaluator"]["sha256"]}:
        raise ValueError("MotionGPT and T2M-GPT-family evaluator hashes differ")

    manifest = {
        "schema_version": "humanml3d-table1-independent-level-evaluation/v1",
        "asset_manifest": str(args.asset_manifest.resolve()),
        "asset_manifest_fingerprint": assets["fingerprint"],
        "run_root": str(args.run_root.resolve()),
        "levels": list(LEVELS),
        "models": [MODEL_LABELS[model] for model in MODEL_ORDER],
        "seeds": seeds,
        "samples_per_level_per_seed": expected_samples,
        "batch_size": 32,
        "retrieval_scope": "within each independent level-specific evaluator batch",
        "metric_names": ["R@1_P", "R@2_P", "R@3_P"],
        "aggregation": {
            "reported_std": "sample standard deviation",
            "reported_std_ddof": 1,
            "ci95": "1.96 * population_std / sqrt(n)",
        },
        "checkpoints": checkpoint_hashes,
        "audits": audits,
    }
    write_outputs(args.output_dir, all_runs, summary_rows, manifest)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()