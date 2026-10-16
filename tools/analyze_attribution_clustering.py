#!/usr/bin/env python3
"""Quantify and visualize semantic clustering in attribution embeddings.

The input is JSON/JSONL with one embedding per record.  Required fields are
``motion_id``, ``variant``, ``embedding`` and either ``semantic_class`` or
``relation``.  The implementation deliberately keeps the core metrics NumPy
only so that exported attribution records can be analyzed in lightweight
environments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_DIRECTIONS = {
    "within_class_cosine_distance": "lower",
    "between_centroid_cosine_distance": "higher",
    "fisher_ratio": "higher",
    "silhouette": "higher",
    "loo_knn_accuracy": "higher",
    "nearest_centroid_accuracy": "higher",
    "spectral_entropy": "diagnostic",
    "normalized_spectral_entropy": "diagnostic",
    "effective_rank": "diagnostic",
    "mean_attribution_entropy": "diagnostic",
}

BOOTSTRAP_METRICS = {
    "within_class_cosine_distance",
    "between_centroid_cosine_distance",
    "fisher_ratio",
}


def read_records(path: Path | str) -> List[dict]:
    """Read records from a JSON array/object or newline-delimited JSON."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping):
            payload = payload.get("records", payload.get("data"))
        if not isinstance(payload, list):
            raise ValueError("JSON input must be a list or contain a 'records' list")
        raw = payload
    else:
        raw = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        raw.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSON on line {line_no}: {exc}") from exc
    return validate_records(raw)


def validate_records(records: Iterable[Mapping]) -> List[dict]:
    validated: List[dict] = []
    dimension = None
    for index, record in enumerate(records):
        missing = [key for key in ("motion_id", "embedding") if key not in record]
        semantic_class = record.get("semantic_class", record.get("relation"))
        if missing or semantic_class is None:
            names = missing + ([] if semantic_class is not None else ["semantic_class/relation"])
            raise ValueError(f"record {index} missing required fields: {', '.join(names)}")
        embedding = np.asarray(record["embedding"], dtype=np.float64).reshape(-1)
        if embedding.size == 0 or not np.isfinite(embedding).all():
            raise ValueError(f"record {index} has an empty or non-finite embedding")
        if dimension is None:
            dimension = embedding.size
        elif embedding.size != dimension:
            raise ValueError(
                f"record {index} embedding dimension {embedding.size} != expected {dimension}"
            )
        validated.append(
            {
                **dict(record),
                "motion_id": str(record["motion_id"]),
                "semantic_class": str(semantic_class),
                "variant": str(record.get("variant", "default")),
                "embedding": embedding,
            }
        )
    if not validated:
        raise ValueError("input contains no records")
    return validated


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _cosine_distance_matrix(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    xn = _normalize_rows(np.asarray(x, dtype=np.float64))
    yn = xn if y is None else _normalize_rows(np.asarray(y, dtype=np.float64))
    return np.clip(1.0 - xn @ yn.T, 0.0, 2.0)


def _mean_pairwise_distance(x: np.ndarray) -> float:
    if len(x) < 2:
        return math.nan
    distances = _cosine_distance_matrix(x)
    upper = distances[np.triu_indices(len(x), 1)]
    return float(upper.mean())


def _classification_metrics(x: np.ndarray, labels: np.ndarray, knn_k: int) -> Tuple[float, float]:
    n = len(x)
    if n < 2 or len(np.unique(labels)) < 2:
        return math.nan, math.nan
    distances = _cosine_distance_matrix(x)
    np.fill_diagonal(distances, np.inf)
    predictions = []
    for i in range(n):
        k = min(max(1, knn_k), n - 1)
        neighbors = np.argsort(distances[i], kind="stable")[:k]
        votes = defaultdict(list)
        for neighbor in neighbors:
            votes[labels[neighbor]].append(distances[i, neighbor])
        # Resolve ties by mean neighbor distance and then lexical class name.
        predictions.append(
            min(votes, key=lambda label: (-len(votes[label]), np.mean(votes[label]), str(label)))
        )
    knn_accuracy = float(np.mean(np.asarray(predictions) == labels))

    centroid_predictions = []
    classes = np.unique(labels)
    for i in range(n):
        candidates = []
        for label in classes:
            indices = np.flatnonzero(labels == label)
            indices = indices[indices != i]
            if len(indices):
                centroid = x[indices].mean(axis=0, keepdims=True)
                distance = float(_cosine_distance_matrix(x[i : i + 1], centroid)[0, 0])
                candidates.append((distance, str(label), label))
        centroid_predictions.append(min(candidates)[2] if candidates else None)
    centroid_accuracy = float(np.mean(np.asarray(centroid_predictions) == labels))
    return knn_accuracy, centroid_accuracy


def _silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2 or len(x) <= len(counts) or np.any(counts < 2):
        return math.nan
    try:
        from sklearn.metrics import silhouette_score

        return float(silhouette_score(x, labels, metric="cosine"))
    except ImportError:
        # Exact NumPy fallback, including the standard singleton convention.
        distances = _cosine_distance_matrix(x)
        values = []
        for i, label in enumerate(labels):
            own = np.flatnonzero(labels == label)
            own = own[own != i]
            if not len(own):
                values.append(0.0)
                continue
            a = distances[i, own].mean()
            b = min(distances[i, labels == other].mean() for other in np.unique(labels) if other != label)
            values.append((b - a) / max(a, b, 1e-12))
        return float(np.mean(values))


def compute_metrics(x: np.ndarray, labels: Sequence[str], knn_k: int = 5) -> Dict[str, float]:
    """Compute clustering, retrieval, and representation-collapse diagnostics."""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    classes = np.unique(labels)
    within_values = [_mean_pairwise_distance(x[labels == label]) for label in classes]
    within_values = [value for value in within_values if np.isfinite(value)]
    within = float(np.mean(within_values)) if within_values else math.nan

    centroids = np.stack([x[labels == label].mean(axis=0) for label in classes])
    between = _mean_pairwise_distance(centroids)
    fisher = between / max(within, 1e-12) if np.isfinite(between) and np.isfinite(within) else math.nan
    knn, nearest_centroid = _classification_metrics(x, labels, knn_k)

    centered = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    spectrum = singular_values**2
    probabilities = spectrum / spectrum.sum() if spectrum.sum() > 1e-20 else np.array([1.0])
    entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-20))))
    max_entropy = math.log(max(1, min(x.shape[0] - 1, x.shape[1])))

    return {
        "within_class_cosine_distance": within,
        "between_centroid_cosine_distance": between,
        "fisher_ratio": fisher,
        "silhouette": _silhouette(x, labels),
        "loo_knn_accuracy": knn,
        "nearest_centroid_accuracy": nearest_centroid,
        "spectral_entropy": entropy,
        "normalized_spectral_entropy": entropy / max_entropy if max_entropy > 0 else 0.0,
        "effective_rank": float(math.exp(entropy)),
    }


def _variant_arrays(records: Sequence[Mapping], variant: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [record for record in records if record["variant"] == variant]
    return (
        np.stack([record["embedding"] for record in selected]),
        np.asarray([record["semantic_class"] for record in selected], dtype=object),
        np.asarray([record["motion_id"] for record in selected], dtype=object),
    )


def bootstrap_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    motion_ids: np.ndarray,
    repeats: int,
    seed: int,
    knn_k: int,
) -> Dict[str, dict]:
    """Motion-cluster bootstrap: resample motions, retaining all their records."""
    point = compute_metrics(x, labels, knn_k)
    if repeats <= 0:
        return {name: {"value": value, "ci95": [math.nan, math.nan]} for name, value in point.items()}
    rng = np.random.default_rng(seed)
    unique_motions = np.unique(motion_ids)
    indices_by_motion = {motion: np.flatnonzero(motion_ids == motion) for motion in unique_motions}
    samples = defaultdict(list)
    for _ in range(repeats):
        drawn = rng.choice(unique_motions, size=len(unique_motions), replace=True)
        indices = np.concatenate([indices_by_motion[motion] for motion in drawn])
        # Bootstrap the claim-bearing compactness/separation metrics only.
        # Recomputing silhouette, retrieval and a full SVD thousands of times
        # is prohibitively expensive and does not change their point estimate.
        sampled_x, sampled_labels = x[indices], labels[indices]
        classes = np.unique(sampled_labels)
        within_values = [_mean_pairwise_distance(sampled_x[sampled_labels == label]) for label in classes]
        within_values = [value for value in within_values if np.isfinite(value)]
        within = float(np.mean(within_values)) if within_values else math.nan
        centroids = np.stack([sampled_x[sampled_labels == label].mean(axis=0) for label in classes])
        between = _mean_pairwise_distance(centroids)
        values = {
            "within_class_cosine_distance": within,
            "between_centroid_cosine_distance": between,
            "fisher_ratio": between / max(within, 1e-12),
        }
        for name, value in values.items():
            if np.isfinite(value):
                samples[name].append(value)
    output = {}
    for name, value in point.items():
        ci = np.percentile(samples[name], [2.5, 97.5]).tolist() if name in BOOTSTRAP_METRICS and samples[name] else [math.nan, math.nan]
        output[name] = {"value": value, "ci95": [float(ci[0]), float(ci[1])]}
    return output


def _motion_compactness(records: Sequence[Mapping], variant: str) -> Dict[str, float]:
    selected = [record for record in records if record["variant"] == variant]
    class_centroids = {}
    for label in sorted({record["semantic_class"] for record in selected}):
        class_x = np.stack([record["embedding"] for record in selected if record["semantic_class"] == label])
        class_centroids[label] = class_x.mean(axis=0, keepdims=True)
    per_motion = defaultdict(list)
    for record in selected:
        distance = _cosine_distance_matrix(
            record["embedding"][None], class_centroids[record["semantic_class"]]
        )[0, 0]
        per_motion[record["motion_id"]].append(float(distance))
    return {motion: float(np.mean(values)) for motion, values in per_motion.items()}


def paired_permutation_test(
    records: Sequence[Mapping], reference: str, candidate: str, repeats: int, seed: int
) -> dict:
    """Paired sign-permutation test on per-motion semantic compactness."""
    reference_scores = _motion_compactness(records, reference)
    candidate_scores = _motion_compactness(records, candidate)
    motions = sorted(set(reference_scores) & set(candidate_scores))
    if not motions:
        return {"n_pairs": 0, "mean_difference": math.nan, "p_value": math.nan}
    differences = np.asarray([candidate_scores[m] - reference_scores[m] for m in motions])
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    if repeats <= 0:
        p_value = math.nan
    else:
        exceedances = 0
        for _ in range(repeats):
            permuted = differences * rng.choice((-1.0, 1.0), size=len(differences))
            exceedances += abs(permuted.mean()) >= abs(observed) - 1e-15
        p_value = (exceedances + 1.0) / (repeats + 1.0)
    return {
        "n_pairs": len(motions),
        "mean_difference": observed,
        "p_value": float(p_value),
        "interpretation": "negative favors candidate (lower within-class compactness distance)",
    }


def generate_synthetic_records(seed: int = 7, motions: int = 24, dimension: int = 8) -> List[dict]:
    """Generate deterministic baseline/SATO-like records for smoke testing."""
    rng = np.random.default_rng(seed)
    classes = ("preserve", "direction", "speed", "body_part", "action")
    centers = _normalize_rows(rng.normal(size=(len(classes), dimension)))
    records = []
    for motion_index in range(motions):
        label_index = motion_index % len(classes)
        for variant, noise in (("baseline", 0.55), ("sato_pp", 0.16)):
            embedding = centers[label_index] + rng.normal(scale=noise, size=dimension)
            records.append(
                {
                    "motion_id": f"synthetic_{motion_index:04d}",
                    "semantic_class": classes[label_index],
                    "variant": variant,
                    "embedding": embedding,
                }
            )
    return validate_records(records)


def _json_safe(value):
    """Convert non-finite NumPy/Python scalars to strict-JSON-compatible values."""
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _project_embeddings(x: np.ndarray, method: str, seed: int) -> Tuple[np.ndarray, str]:
    if method in {"auto", "umap"}:
        try:
            import umap

            neighbors = min(15, max(2, len(x) - 1))
            return umap.UMAP(n_components=2, n_neighbors=neighbors, random_state=seed).fit_transform(x), "UMAP"
        except ImportError:
            if method == "umap":
                raise RuntimeError("--plot-method umap requested but umap-learn is not installed")
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = centered @ vh[: min(2, len(vh))].T
    if components.shape[1] == 1:
        components = np.column_stack([components[:, 0], np.zeros(len(components))])
    return components, "PCA"


def write_plot(records: Sequence[Mapping], output: Path, method: str, seed: int) -> str:
    classes = sorted({record["semantic_class"] for record in records})
    variants = sorted({record["variant"] for record in records})
    colors = plt.get_cmap("tab10", len(classes))
    cols = min(3, len(variants))
    rows = int(math.ceil(len(variants) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.2 * rows), squeeze=False)
    used_methods = []
    for variant_index, variant in enumerate(variants):
        ax = axes.flat[variant_index]
        selected = [record for record in records if record["variant"] == variant]
        x = np.stack([record["embedding"] for record in selected])
        projected, used_method = _project_embeddings(x, method, seed + variant_index)
        used_methods.append(used_method)
        for class_index, label in enumerate(classes):
            mask = np.asarray([record["semantic_class"] == label for record in selected])
            if mask.any():
                ax.scatter(
                    projected[mask, 0], projected[mask, 1], s=28, alpha=0.72,
                    c=[colors(class_index)], marker="o", label=label,
                )
        ax.set_title(variant)
        ax.set_xlabel(f"{used_method} 1")
        ax.set_ylabel(f"{used_method} 2")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, frameon=False)
    for index in range(len(variants), rows * cols):
        axes.flat[index].axis("off")
    figure_method = used_methods[0] if len(set(used_methods)) == 1 else "mixed"
    fig.suptitle(f"Attribution embedding semantic space ({figure_method}, per representation)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return figure_method


def analyze(
    records: Sequence[Mapping], output_dir: Path | str, bootstrap: int = 1000,
    permutations: int = 10000, seed: int = 7, knn_k: int = 5,
    reference_variant: str | None = None, plot_method: str = "auto",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = sorted({record["variant"] for record in records})
    reference_variant = reference_variant or variants[0]
    if reference_variant not in variants:
        raise ValueError(f"reference variant {reference_variant!r} not found; choices: {variants}")

    variant_results = {}
    for variant_index, variant in enumerate(variants):
        x, labels, motion_ids = _variant_arrays(records, variant)
        selected_records = [record for record in records if record["variant"] == variant]
        entropy_values = [
            float(record["attribution_entropy"])
            for record in selected_records
            if record.get("attribution_entropy") is not None
            and np.isfinite(float(record["attribution_entropy"]))
        ]
        metrics = bootstrap_metrics(x, labels, motion_ids, bootstrap, seed + variant_index, knn_k)
        metrics["mean_attribution_entropy"] = {
            "value": float(np.mean(entropy_values)) if entropy_values else math.nan,
            "ci95": [math.nan, math.nan],
        }
        variant_results[variant] = {
            "n_records": len(x),
            "n_motions": len(np.unique(motion_ids)),
            "n_classes": len(np.unique(labels)),
            "metrics": metrics,
        }
    comparisons = {
        candidate: paired_permutation_test(
            records, reference_variant, candidate, permutations, seed + 1000 + index
        )
        for index, candidate in enumerate(variants)
        if candidate != reference_variant
    }
    plot_file = output_dir / "attribution_semantic_space.png"
    used_method = write_plot(records, plot_file, plot_method, seed)
    summary = {
        "schema_version": 1,
        "reference_variant": reference_variant,
        "bootstrap_repeats": bootstrap,
        "permutation_repeats": permutations,
        "seed": seed,
        "metric_directions": METRIC_DIRECTIONS,
        "variants": variant_results,
        "paired_permutation_tests": comparisons,
        "plot": {"path": plot_file.name, "method": used_method},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("variant", "metric", "value", "ci95_low", "ci95_high", "direction"))
        writer.writeheader()
        for variant, result in variant_results.items():
            for metric, values in result["metrics"].items():
                writer.writerow(
                    {"variant": variant, "metric": metric, "value": values["value"],
                     "ci95_low": values["ci95"][0], "ci95_high": values["ci95"][1],
                     "direction": METRIC_DIRECTIONS[metric]}
                )
    with (output_dir / "comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("reference", "candidate", "n_pairs", "mean_difference", "p_value"))
        writer.writeheader()
        for candidate, result in comparisons.items():
            writer.writerow({"reference": reference_variant, "candidate": candidate, **{key: result[key] for key in ("n_pairs", "mean_difference", "p_value")}})
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+",
                        help="one or more JSON/JSONL attribution embedding record files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic-smoke", action="store_true", help="analyze deterministic synthetic records")
    parser.add_argument("--synthetic-motions", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--reference-variant")
    parser.add_argument("--plot-method", choices=("auto", "pca", "umap"), default="auto")
    args = parser.parse_args(argv)
    if args.synthetic_smoke == (args.input is not None):
        parser.error("provide exactly one of --input or --synthetic-smoke")
    if args.bootstrap < 0 or args.permutations < 0 or args.knn_k < 1:
        parser.error("--bootstrap/--permutations must be nonnegative and --knn-k positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = (
        generate_synthetic_records(args.seed, args.synthetic_motions)
        if args.synthetic_smoke
        else [record for path in args.input for record in read_records(path)]
    )
    summary = analyze(
        records, args.output_dir, args.bootstrap, args.permutations, args.seed,
        args.knn_k, args.reference_variant, args.plot_method,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "variants": list(summary["variants"]), "plot_method": summary["plot"]["method"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
