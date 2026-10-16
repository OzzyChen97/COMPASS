#!/usr/bin/env python3
"""Measure association between SATO++ attribution and TM2T motion spaces.

The analysis treats Grad x Input and learned attribution pooling as text-side,
motion-conditioned representations and the frozen HumanML3D TM2T evaluator
embedding as the motion latent. It measures association, not causal influence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("P1", "P2", "P3")
MODEL_INTERNAL = {
    "baseline": "baseline_t2mgpt_original",
    "satopp": "sato_pp_allloss_l1_iter130k",
}
PRIMARY_REPRESENTATIONS = {
    "baseline": ("grad_input_weighted",),
    "satopp": ("grad_input_weighted", "sato_learned_pool"),
}


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    stops = np.r_[starts[1:], len(values)]
    for start, stop in zip(starts, stops):
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    y = np.asarray(y, dtype=np.float64) - np.mean(y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 1e-15 else math.nan


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centered kernel alignment without materializing sample Gram matrices."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("CKA inputs must be aligned two-dimensional arrays")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    numerator = np.sum(cross * cross)
    x_norm = np.linalg.norm(x.T @ x)
    y_norm = np.linalg.norm(y.T @ y)
    denominator = x_norm * y_norm
    return float(numerator / denominator) if denominator > 1e-15 else math.nan


def cosine_distance_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", x, y)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    return np.clip(1.0 - numerator / np.maximum(denominator, 1e-12), 0.0, 2.0)


def sample_pair_indices(n: int, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("RSA needs at least two samples")
    rng = np.random.default_rng(seed)
    first, second = [], []
    needed = max(1, count)
    while len(first) < needed:
        draw = max(needed - len(first), 1024)
        a = rng.integers(0, n, size=draw)
        b = rng.integers(0, n, size=draw)
        keep = a != b
        first.extend(a[keep].tolist())
        second.extend(b[keep].tolist())
    return np.asarray(first[:needed]), np.asarray(second[:needed])


def rsa_spearman(
    x: np.ndarray, y: np.ndarray, pair_indices: tuple[np.ndarray, np.ndarray]
) -> float:
    first, second = pair_indices
    return spearman(
        cosine_distance_rows(x[first], x[second]),
        cosine_distance_rows(y[first], y[second]),
    )


def pair_displacement(
    attr_orig: np.ndarray, attr_pair: np.ndarray,
    motion_orig: np.ndarray, motion_pair: np.ndarray,
) -> dict[str, object]:
    attr_distance = cosine_distance_rows(attr_orig, attr_pair)
    motion_distance = cosine_distance_rows(motion_orig, motion_pair)
    return {
        "correlation": spearman(attr_distance, motion_distance),
        "mean_attr_distance": float(attr_distance.mean()),
        "mean_motion_distance": float(motion_distance.mean()),
        "attr_distance": attr_distance,
        "motion_distance": motion_distance,
    }


def effective_rank(x: np.ndarray) -> float:
    centered = np.asarray(x, dtype=np.float64) - np.mean(x, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    spectrum = singular * singular
    if spectrum.sum() <= 1e-20:
        return 1.0
    probabilities = spectrum / spectrum.sum()
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-20)))
    return float(np.exp(entropy))


def grouped_folds(groups: Sequence[str], folds: int, seed: int) -> list[np.ndarray]:
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    if folds < 2 or len(unique) < folds:
        raise ValueError("grouped folds require at least one group per fold")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    assignments = {group: index % folds for index, group in enumerate(unique)}
    return [np.asarray([assignments[group] == fold for group in groups]) for fold in range(folds)]


def pca_project(
    train: np.ndarray, test: np.ndarray, components: int,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if components <= 0 or components >= min(train.shape):
        return train, test
    _, _, vh = np.linalg.svd(train, full_matrices=False)
    basis = vh[:components].T
    return train @ basis, test @ basis


def ridge_probe(
    x: np.ndarray, y: np.ndarray, groups: Sequence[str], folds: int = 5,
    alpha: float = 100.0, seed: int = 123, pca_components: int = 128,
) -> dict[str, float]:
    predictions = np.zeros_like(y, dtype=np.float64)
    fold_masks = grouped_folds(groups, folds, seed)
    for test in fold_masks:
        train = ~test
        x_mean, y_mean = x[train].mean(0), y[train].mean(0)
        x_train = np.asarray(x[train], dtype=np.float64) - x_mean
        x_test = np.asarray(x[test], dtype=np.float64) - x_mean
        x_train, x_test = pca_project(x_train, x_test, pca_components)
        y_train = np.asarray(y[train], dtype=np.float64) - y_mean
        coefficients = np.linalg.solve(
            x_train.T @ x_train + alpha * np.eye(x_train.shape[1]),
            x_train.T @ y_train,
        )
        predictions[test] = x_test @ coefficients + y_mean
    residual = np.sum((y - predictions) ** 2)
    total = np.sum((y - np.mean(y, axis=0, keepdims=True)) ** 2)
    cosine = 1.0 - cosine_distance_rows(y, predictions)
    return {
        "r2": float(1.0 - residual / max(total, 1e-20)),
        "cosine": float(np.mean(cosine)),
    }


def paired_sign_permutation(
    candidate: Sequence[float], reference: Sequence[float], repeats: int, seed: int
) -> dict[str, float]:
    differences = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    if len(differences) == 1:
        p_value = math.nan
    else:
        exceedances = 0
        for _ in range(repeats):
            permuted = differences * rng.choice((-1.0, 1.0), size=len(differences))
            exceedances += abs(permuted.mean()) >= abs(observed) - 1e-15
        p_value = float((exceedances + 1) / (repeats + 1))
    return {
        "n_pairs": int(len(differences)),
        "mean_difference": observed,
        "p_value": p_value,
    }


def bootstrap_paired_ci(
    candidate: Sequence[float], reference: Sequence[float], repeats: int, seed: int
) -> list[float]:
    differences = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    rng = np.random.default_rng(seed)
    samples = [
        float(rng.choice(differences, size=len(differences), replace=True).mean())
        for _ in range(repeats)
    ]
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p_values))
    order = finite[np.argsort(p_values[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def load_attribution(path: Path | str) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as dump:
        required = {
            "schema_version", "model", "sample_index_hashes_json", "level",
            "motion_id", "pair_id", "prompt_role", "text",
            "attribution_entropy", "target_token_sha256", "clip_pooled",
            "grad_input_weighted",
        }
        missing = required - set(dump.files)
        if missing:
            raise ValueError(f"attribution dump {path} lacks {sorted(missing)}")
        if str(dump["schema_version"].item()) != "attribution-motion-alignment/v1":
            raise ValueError(f"attribution dump {path} has unsupported schema")
        result = {key: np.asarray(dump[key]) for key in dump.files}
    result["model"] = str(result["model"].item())
    result["sample_index_hashes"] = json.loads(
        str(result["sample_index_hashes_json"].item())
    )
    return result


def attribution_level_role(
    dump: Mapping[str, object], level: str, role: str, representation: str,
    expected_motion_ids: Sequence[str], expected_texts: Sequence[str],
) -> np.ndarray:
    if representation not in dump:
        raise ValueError(f"attribution representation {representation!r} is unavailable")
    mask = (dump["level"].astype(str) == level) & (
        dump["prompt_role"].astype(str) == role
    )
    indices = np.flatnonzero(mask)
    if len(indices) != len(expected_motion_ids):
        raise ValueError(f"{level}/{role} attribution count mismatch")
    if dump["motion_id"][indices].astype(str).tolist() != list(expected_motion_ids):
        raise ValueError(f"{level}/{role} attribution motion order mismatch")
    if dump["text"][indices].astype(str).tolist() != list(expected_texts):
        raise ValueError(f"{level}/{role} attribution caption order mismatch")
    return np.asarray(dump[representation][indices], dtype=np.float64)


def load_diagnostic(path: Path | str, expected_level: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as dump:
        required = {
            "gt_embedding", "original_embedding", "perturbed_embedding",
            "motion_id", "caption", "perturbed_caption", "level",
        }
        missing = required - set(dump.files)
        if missing:
            raise ValueError(f"diagnostic {path} lacks {sorted(missing)}")
        result = {key: np.asarray(dump[key]) for key in required}
    count = len(result["motion_id"])
    for name in ("gt_embedding", "original_embedding", "perturbed_embedding"):
        if result[name].shape != (count, 512):
            raise ValueError(f"diagnostic {path} {name} has shape {result[name].shape}")
    if set(result["level"].astype(str)) != {expected_level}:
        raise ValueError(f"diagnostic {path} level mismatch")
    return result


def diagnostic_path(run_root: Path, model: str, level: str, seed: int) -> Path:
    internal = MODEL_INTERNAL[model]
    return (
        run_root / level.lower() / internal / "diagnostics" / internal /
        f"seed{seed}.npz"
    )


def _metric_rows_for_seed(
    model: str, level: str, seed: int, representation: str,
    attr_orig: np.ndarray, attr_pair: np.ndarray, diagnostic: Mapping[str, np.ndarray],
    rsa_pairs: tuple[np.ndarray, np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for role, attr, motion_name in (
        ("orig", attr_orig, "original_embedding"),
        ("pair", attr_pair, "perturbed_embedding"),
        ("orig", attr_orig, "gt_embedding"),
        ("pair", attr_pair, "gt_embedding"),
    ):
        motion = diagnostic[motion_name]
        target = motion_name.replace("_embedding", "")
        rows.extend([
            {"model": model, "level": level, "seed": seed,
             "representation": representation, "role": role, "target": target,
             "metric": "cka", "value": linear_cka(attr, motion)},
            {"model": model, "level": level, "seed": seed,
             "representation": representation, "role": role, "target": target,
             "metric": "rsa", "value": rsa_spearman(attr, motion, rsa_pairs)},
        ])
    displacement = pair_displacement(
        attr_orig, attr_pair,
        diagnostic["original_embedding"], diagnostic["perturbed_embedding"],
    )
    rows.extend([
        {"model": model, "level": level, "seed": seed,
         "representation": representation, "role": "delta", "target": "generated",
         "metric": "delta_spearman", "value": displacement["correlation"]},
        {"model": model, "level": level, "seed": seed,
         "representation": representation, "role": "delta", "target": "attribution",
         "metric": "mean_distance", "value": displacement["mean_attr_distance"]},
        {"model": model, "level": level, "seed": seed,
         "representation": representation, "role": "delta", "target": "generated",
         "metric": "mean_distance", "value": displacement["mean_motion_distance"]},
    ])
    return rows, displacement


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_forest(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    selected = [
        row for row in rows
        if row["metric"] in {"cka", "rsa"} and row["target"] == "perturbed"
        and row["role"] == "pair"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=False)
    for ax, metric in zip(axes, ("cka", "rsa")):
        metric_rows = [row for row in selected if row["metric"] == metric]
        labels, means, errors = [], [], []
        for level in LEVELS:
            for model, representation, label in (
                ("baseline", "grad_input_weighted", "T2M-GPT Grad×Input"),
                ("satopp", "grad_input_weighted", "SATO++ Grad×Input"),
                ("satopp", "sato_learned_pool", "SATO++ learned pool"),
            ):
                values = [float(row["value"]) for row in metric_rows
                          if row["level"] == level and row["model"] == model
                          and row["representation"] == representation]
                labels.append(f"{level} {label}")
                means.append(float(np.mean(values)))
                errors.append(
                    float(np.std(values, ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1 else 0.0
                )
        y = np.arange(len(labels))
        ax.errorbar(means, y, xerr=np.asarray(errors) * 1.96, fmt="o", capsize=2)
        ax.set_yticks(y, labels if metric == "cka" else [])
        ax.invert_yaxis()
        ax.set_title(metric.upper())
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Attribution–generated-motion representational association (95% CI over seeds)")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_displacement(scatter: Mapping[str, np.ndarray], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, level in zip(axes, LEVELS):
        values = scatter[level]
        if len(values) > 2000:
            rng = np.random.default_rng(123 + LEVELS.index(level))
            values = values[rng.choice(len(values), 2000, replace=False)]
        ax.scatter(values[:, 0], values[:, 1], s=6, alpha=0.28)
        ax.set_title(level)
        ax.set_xlabel("Attribution cosine displacement")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Generated-motion latent displacement")
    fig.suptitle("SATO++ learned-pool displacement coupling (seed 123)")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def json_safe(value):
    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def analyze(args: argparse.Namespace) -> dict[str, object]:
    baseline_attr = load_attribution(args.baseline_attribution)
    satopp_attr = load_attribution(args.satopp_attribution)
    if "sato_learned_pool" not in satopp_attr:
        raise ValueError("SATO++ attribution dump lacks learned pooling")
    if baseline_attr["sample_index_hashes"] != satopp_attr["sample_index_hashes"]:
        raise ValueError("baseline/SATO++ sample-index hashes differ")
    for identity_key in ("level", "motion_id", "pair_id", "prompt_role", "text"):
        if not np.array_equal(baseline_attr[identity_key], satopp_attr[identity_key]):
            raise ValueError(f"baseline/SATO++ attribution {identity_key} order differs")
    if not np.array_equal(
        baseline_attr["target_token_sha256"], satopp_attr["target_token_sha256"]
    ):
        raise ValueError("baseline/SATO++ attribution targets differ")
    baseline_clip = np.asarray(baseline_attr["clip_pooled"], dtype=np.float32)
    satopp_clip = np.asarray(satopp_attr["clip_pooled"], dtype=np.float32)
    clip_max_abs_difference = float(np.max(np.abs(baseline_clip - satopp_clip)))
    if not np.allclose(baseline_clip, satopp_clip, rtol=1e-5, atol=1e-6):
        raise ValueError(
            "frozen CLIP control differs between baseline and SATO++: "
            f"max_abs={clip_max_abs_difference}"
        )

    run_rows: list[dict[str, object]] = []
    diagnostics_manifest = []
    scatter = {}
    probe_rows = []
    collapse_rows = []
    attr_cache = {}
    for level_index, level in enumerate(LEVELS):
        first_path = diagnostic_path(args.run_root, "baseline", level, args.start_seed)
        first = load_diagnostic(first_path, level)
        available = int(np.sum(
            (baseline_attr["level"].astype(str) == level)
            & (baseline_attr["prompt_role"].astype(str) == "orig")
        ))
        if available < 1:
            raise ValueError(f"attribution dump has no {level} rows")
        if available < len(first["motion_id"]):
            first = {key: value[:available] for key, value in first.items()}
        motion_ids = first["motion_id"].astype(str).tolist()
        captions = first["caption"].astype(str).tolist()
        perturbed = first["perturbed_caption"].astype(str).tolist()
        rsa_pairs = sample_pair_indices(len(motion_ids), args.rsa_pairs, args.seed + level_index)

        for model, attr_dump in (("baseline", baseline_attr), ("satopp", satopp_attr)):
            for representation in PRIMARY_REPRESENTATIONS[model]:
                attr_orig = attribution_level_role(
                    attr_dump, level, "orig", representation, motion_ids, captions
                )
                attr_pair = attribution_level_role(
                    attr_dump, level, "pair", representation, motion_ids, perturbed
                )
                attr_cache[(model, level, representation)] = (attr_orig, attr_pair)
                indices = np.flatnonzero(
                    (attr_dump["level"].astype(str) == level)
                    & (attr_dump["prompt_role"].astype(str) == "pair")
                )
                collapse_rows.append({
                    "model": model, "level": level, "representation": representation,
                    "mean_attribution_entropy": float(np.mean(
                        attr_dump["attribution_entropy"][indices]
                    )),
                    "effective_rank": effective_rank(attr_pair),
                })

        seed_motion_means = defaultdict(list)
        for seed in range(args.start_seed, args.start_seed + args.num_seeds):
            for model in ("baseline", "satopp"):
                path = diagnostic_path(args.run_root, model, level, seed)
                diagnostic = load_diagnostic(path, level)
                if available < len(diagnostic["motion_id"]):
                    diagnostic = {
                        key: value[:available] for key, value in diagnostic.items()
                    }
                for key, expected in (
                    ("motion_id", motion_ids), ("caption", captions),
                    ("perturbed_caption", perturbed),
                ):
                    if diagnostic[key].astype(str).tolist() != expected:
                        raise ValueError(f"diagnostic identity mismatch: {path} ({key})")
                diagnostics_manifest.append({
                    "model": model, "level": level, "seed": seed,
                    "path": str(path.resolve()), "sample_count": len(motion_ids),
                    "motion_order_sha256": canonical_sha256(motion_ids),
                })
                for representation in PRIMARY_REPRESENTATIONS[model]:
                    attr_orig, attr_pair = attr_cache[(model, level, representation)]
                    rows, displacement = _metric_rows_for_seed(
                        model, level, seed, representation, attr_orig, attr_pair,
                        diagnostic, rsa_pairs,
                    )
                    run_rows.extend(rows)
                    seed_motion_means[(model, representation, "orig")].append(
                        diagnostic["original_embedding"]
                    )
                    seed_motion_means[(model, representation, "pair")].append(
                        diagnostic["perturbed_embedding"]
                    )
                    if model == "satopp" and representation == "sato_learned_pool" and seed == args.start_seed:
                        scatter[level] = np.column_stack([
                            displacement["attr_distance"], displacement["motion_distance"]
                        ])

        # Secondary grouped probes use the reference target and seed-mean generation.
        for model in ("baseline", "satopp"):
            for representation in PRIMARY_REPRESENTATIONS[model]:
                attr_orig, attr_pair = attr_cache[(model, level, representation)]
                targets = {
                    "reference": first["gt_embedding"],
                    "generated_orig_mean": np.mean(
                        seed_motion_means[(model, representation, "orig")], axis=0
                    ),
                    "generated_pair_mean": np.mean(
                        seed_motion_means[(model, representation, "pair")], axis=0
                    ),
                }
                for target_name, target in targets.items():
                    role_attr = attr_pair if target_name == "generated_pair_mean" else attr_orig
                    values = ridge_probe(
                        role_attr, target, motion_ids, folds=5,
                        alpha=args.ridge_alpha, seed=args.seed + level_index,
                        pca_components=args.ridge_pca_components,
                    )
                    probe_rows.append({
                        "model": model, "level": level,
                        "representation": representation, "target": target_name,
                        **values,
                    })

    comparisons = []
    comparison_specs = (
        ("satopp", "grad_input_weighted", "baseline", "grad_input_weighted"),
        ("satopp", "sato_learned_pool", "baseline", "grad_input_weighted"),
    )
    grouped = defaultdict(list)
    for row in run_rows:
        key = (
            row["model"], row["level"], row["representation"], row["role"],
            row["target"], row["metric"],
        )
        grouped[key].append(float(row["value"]))
    for spec_index, (candidate_model, candidate_rep, reference_model, reference_rep) in enumerate(comparison_specs):
        for level in LEVELS:
            metric_keys = sorted({
                (row["role"], row["target"], row["metric"])
                for row in run_rows if row["level"] == level
            })
            for role, target, metric in metric_keys:
                candidate = grouped[(candidate_model, level, candidate_rep, role, target, metric)]
                reference = grouped[(reference_model, level, reference_rep, role, target, metric)]
                if not candidate or len(candidate) != len(reference):
                    continue
                test = paired_sign_permutation(
                    candidate, reference, args.permutations,
                    args.seed + 1000 * spec_index + 10 * LEVELS.index(level),
                )
                ci = bootstrap_paired_ci(
                    candidate, reference, args.bootstrap,
                    args.seed + 5000 + 1000 * spec_index,
                )
                comparisons.append({
                    "candidate": f"{candidate_model}/{candidate_rep}",
                    "reference": f"{reference_model}/{reference_rep}",
                    "level": level, "role": role, "target": target, "metric": metric,
                    **test,
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                })
    adjusted = holm_adjust([row["p_value"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["p_value_holm"] = value

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "per_seed_metrics.csv", run_rows,
        ("model", "level", "seed", "representation", "role", "target", "metric", "value"),
    )
    write_csv(
        args.output_dir / "paired_comparisons.csv", comparisons,
        ("candidate", "reference", "level", "role", "target", "metric", "n_pairs",
         "mean_difference", "ci95_low", "ci95_high", "p_value", "p_value_holm"),
    )
    write_csv(
        args.output_dir / "ridge_probes.csv", probe_rows,
        ("model", "level", "representation", "target", "r2", "cosine"),
    )
    write_csv(
        args.output_dir / "collapse_diagnostics.csv", collapse_rows,
        ("model", "level", "representation", "mean_attribution_entropy", "effective_rank"),
    )
    plot_forest(run_rows, args.output_dir / "association_forest.png")
    plot_displacement(scatter, args.output_dir / "displacement_coupling.png")

    summary = {
        "schema_version": 1,
        "claim_scope": "representational association; not causal attribution",
        "sample_index_hashes": baseline_attr["sample_index_hashes"],
        "clip_control_max_abs_difference": clip_max_abs_difference,
        "seeds": list(range(args.start_seed, args.start_seed + args.num_seeds)),
        "rsa_sampled_pairs": args.rsa_pairs,
        "primary_metrics": ["linear_cka", "rsa_spearman", "delta_spearman"],
        "diagnostics": diagnostics_manifest,
        "collapse_diagnostics": collapse_rows,
        "ridge_probes": probe_rows,
        "paired_comparisons": comparisons,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-attribution", type=Path, required=True)
    parser.add_argument("--satopp-attribution", type=Path, required=True)
    parser.add_argument(
        "--run-root", type=Path,
        default=REPO_ROOT / "ablation/humanml3d_table1_levels/runs/t2mgpt_satopp",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=123)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--rsa-pairs", type=int, default=100_000)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--ridge-pca-components", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    if args.num_seeds < 1 or args.rsa_pairs < 1 or args.bootstrap < 1 or args.permutations < 1:
        parser.error("seed count, RSA pairs, bootstrap, and permutations must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = analyze(args)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "comparisons": len(summary["paired_comparisons"]),
        "diagnostics": len(summary["diagnostics"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())