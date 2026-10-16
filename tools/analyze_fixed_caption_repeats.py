#!/usr/bin/env python3
"""Compare repeated fixed-caption generations from MotionGPT and SATO++."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np


ACTION_PATTERNS = (
    ("walking", r"\b(walk|walks|walking|stroll|march|pace)\b"),
    ("running", r"\b(run|runs|running|jog|jogs|jogging|sprint)\b"),
    ("jumping", r"\b(jump|jumps|jumping|hop|hops|hopping|leap)\b"),
    ("turning", r"\b(turn|turns|turning|rotate|rotates|spins?|pivot)\b"),
    ("interaction", r"\b(pick|picks|grab|grabs|hold|holds|carry|carries|throw|throws|catch|push|pull|kick|punch|object|chair|ball|box)\b"),
    ("gestures", r"\b(wave|waves|clap|claps|point|points|gesture|salute)\b"),
    ("sitting", r"\b(sit|sits|sitting|stand|stands|standing|kneel|kneels)\b"),
    ("dancing", r"\b(dance|dances|dancing)\b"),
)


def summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return 1.0 - np.sum(left * right, axis=-1) / np.maximum(denom, 1e-12)


def correct_ranks(text: np.ndarray, motion: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(text[:, None, :] - motion[None, :, :], axis=-1)
    ranking = np.argsort(distances, axis=1)
    return np.argmax(ranking == np.arange(len(text))[:, None], axis=1)


def action_category(caption: str) -> str:
    lowered = caption.lower()
    for name, pattern in ACTION_PATTERNS:
        if re.search(pattern, lowered):
            return name
    return "other"


def load_model(name: str, pattern: str) -> dict[str, np.ndarray]:
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if len(paths) < 2:
        raise ValueError(f"{name}: need at least two repeat files, found {len(paths)}")
    runs = []
    for path in paths:
        with np.load(path) as data:
            motion_key = "original_generated" if "original_generated" in data else "original_embedding"
            required = {motion_key, "text_embedding", "original_motion", "caption", "loader_name", "original_length"}
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"{path}: missing {missing}")
            runs.append({key: data[key].copy() for key in required})

    reference = runs[0]
    count = len(reference["caption"])
    for path, run in zip(paths[1:], runs[1:]):
        if len(run["caption"]) != count:
            raise ValueError(f"{path}: sample count changed")
        for key in ("caption", "loader_name"):
            if not np.array_equal(run[key].astype(str), reference[key].astype(str)):
                raise ValueError(f"{path}: fixed sample identity changed for {key}")
        if not np.allclose(run["text_embedding"], reference["text_embedding"], atol=1e-6, rtol=0):
            raise ValueError(f"{path}: TM2T text embeddings changed across repeats")

    motion_key = "original_generated" if "original_generated" in reference else "original_embedding"
    return {
        "name": name,
        "paths": np.asarray([str(path) for path in paths]),
        "caption": reference["caption"].astype(str),
        "loader_name": reference["loader_name"].astype(str),
        "text": reference["text_embedding"],
        "embedding": np.stack([run[motion_key] for run in runs]),
        "motion": np.stack([run["original_motion"] for run in runs]),
        "length": np.stack([run["original_length"] for run in runs]),
    }


def per_caption_pairwise(model: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings = model["embedding"]
    motions = model["motion"]
    lengths = model["length"]
    repeat_count, caption_count = embeddings.shape[:2]
    embedding_l2 = np.zeros(caption_count)
    embedding_cosine = np.zeros(caption_count)
    motion_rms = np.zeros(caption_count)
    pairs = list(combinations(range(repeat_count), 2))
    for caption_index in range(caption_count):
        l2_values = []
        cosine_values = []
        motion_values = []
        for left, right in pairs:
            l2_values.append(np.linalg.norm(embeddings[left, caption_index] - embeddings[right, caption_index]))
            cosine_values.append(cosine_distance(
                embeddings[left, caption_index][None], embeddings[right, caption_index][None]
            )[0])
            common = int(min(lengths[left, caption_index], lengths[right, caption_index]))
            delta = motions[left, caption_index, :common] - motions[right, caption_index, :common]
            motion_values.append(np.sqrt(np.mean(np.square(delta))))
        embedding_l2[caption_index] = np.mean(l2_values)
        embedding_cosine[caption_index] = np.mean(cosine_values)
        motion_rms[caption_index] = np.mean(motion_values)
    return embedding_l2, embedding_cosine, motion_rms


def analyze(model: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    embedding_l2, embedding_cosine, motion_rms = per_caption_pairwise(model)
    text = model["text"]
    embeddings = model["embedding"]
    l2_alignment = np.linalg.norm(embeddings - text[None], axis=-1)
    cosine_alignment = cosine_distance(embeddings, text[None])
    ranks = np.stack([correct_ranks(text, run) for run in embeddings])
    r1 = (ranks == 0).mean(axis=0)
    mean_rank = ranks.mean(axis=0)
    categories = np.asarray([action_category(value) for value in model["caption"]])
    centroids = embeddings.mean(axis=0)
    all_zero = np.all(model["motion"] == 0, axis=(2, 3))
    unique_per_caption = np.asarray([
        len({embeddings[repeat, index].tobytes() for repeat in range(len(embeddings))})
        for index in range(len(model["caption"]))
    ])
    exact_duplicate_pairs = np.asarray([
        sum(
            embeddings[left, index].tobytes() == embeddings[right, index].tobytes()
            for left, right in combinations(range(len(embeddings)), 2)
        )
        for index in range(len(model["caption"]))
    ])
    repeat_pair_count = len(list(combinations(range(len(embeddings)), 2)))
    all_hashes: dict[bytes, list[tuple[int, int]]] = {}
    for repeat in range(embeddings.shape[0]):
        for index in range(embeddings.shape[1]):
            all_hashes.setdefault(embeddings[repeat, index].tobytes(), []).append((repeat, index))
    cross_caption_duplicate_groups = sum(
        len({index for _, index in occurrences}) > 1
        for occurrences in all_hashes.values()
    )

    rows = []
    for index, caption in enumerate(model["caption"]):
        rows.append({
            "model": model["name"],
            "sample_index": index,
            "loader_name": model["loader_name"][index],
            "category": categories[index],
            "caption": caption,
            "r1_rate": float(r1[index]),
            "mean_correct_rank": float(mean_rank[index]),
            "embedding_pairwise_l2": float(embedding_l2[index]),
            "embedding_pairwise_cosine_distance": float(embedding_cosine[index]),
            "motion_pairwise_rms": float(motion_rms[index]),
            "text_motion_l2": float(l2_alignment[:, index].mean()),
            "text_motion_cosine_distance": float(cosine_alignment[:, index].mean()),
            "unique_outputs": int(unique_per_caption[index]),
            "exact_duplicate_pair_rate": float(exact_duplicate_pairs[index] / repeat_pair_count),
            "all_zero_runs": int(all_zero[:, index].sum()),
            "mean_generated_length": float(model["length"][:, index].mean()),
        })

    grouped = {}
    for category in sorted(set(categories)):
        mask = categories == category
        grouped[category] = {
            "count": int(mask.sum()),
            "r1": float(r1[mask].mean()),
            "mean_correct_rank": float(mean_rank[mask].mean()),
            "text_motion_l2": float(l2_alignment[:, mask].mean()),
            "embedding_pairwise_l2": float(embedding_l2[mask].mean()),
            "motion_pairwise_rms": float(motion_rms[mask].mean()),
        }

    result = {
        "repeat_count": int(embeddings.shape[0]),
        "caption_count": int(embeddings.shape[1]),
        "fixed_identity": True,
        "r1_over_global_100_pool": float((ranks == 0).mean()),
        "correct_rank": summary(ranks),
        "text_motion_l2": summary(l2_alignment),
        "text_motion_cosine_distance": summary(cosine_alignment),
        "embedding_pairwise_l2": summary(embedding_l2),
        "embedding_pairwise_cosine_distance": summary(embedding_cosine),
        "motion_pairwise_rms_common_frames": summary(motion_rms),
        "within_caption_embedding_variance": float(np.mean(np.var(embeddings, axis=0))),
        "across_caption_centroid_variance": float(np.mean(np.var(centroids, axis=0))),
        "unique_embeddings": int(len(all_hashes)),
        "total_generations": int(embeddings.shape[0] * embeddings.shape[1]),
        "captions_all_repeats_unique": int((unique_per_caption == embeddings.shape[0]).sum()),
        "captions_fully_deterministic": int((unique_per_caption == 1).sum()),
        "exact_duplicate_pair_rate": float(
            exact_duplicate_pairs.sum() / (repeat_pair_count * len(unique_per_caption))
        ),
        "cross_caption_exact_duplicate_groups": int(cross_caption_duplicate_groups),
        "all_zero_generations": int(all_zero.sum()),
        "captions_with_all_zero_generation": int(np.any(all_zero, axis=0).sum()),
        "generated_length": summary(model["length"]),
        "category": grouped,
    }
    return result, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="NAME=glob for repeat NPZ files")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    models = []
    for value in args.model:
        if "=" not in value:
            parser.error("--model must be NAME=glob")
        name, pattern = value.split("=", 1)
        models.append(load_model(name, pattern))

    reference = models[0]
    for model in models[1:]:
        if not np.array_equal(model["caption"], reference["caption"]):
            raise ValueError(f"{model['name']}: captions differ from {reference['name']}")
        if not np.array_equal(model["loader_name"], reference["loader_name"]):
            raise ValueError(f"{model['name']}: loader identities differ from {reference['name']}")
        if not np.allclose(model["text"], reference["text"], atol=1e-6, rtol=0):
            raise ValueError(f"{model['name']}: TM2T text embeddings differ from {reference['name']}")

    results = {}
    rows = []
    for model in models:
        results[model["name"]], model_rows = analyze(model)
        rows.extend(model_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_caption.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Fixed-caption repeated-generation diagnostic",
        "",
        "All models use the same 100 loader identities, captions, and TM2T text embeddings.",
        "R@1 is computed per repeat against the same global 100-motion candidate pool.",
        "",
        "| Model | repeats | R@1 | embed pair L2 | motion pair RMS | text-motion L2 | within var | across-centroid var |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, value in results.items():
        lines.append(
            f"| {name} | {value['repeat_count']} | {value['r1_over_global_100_pool']:.4f} | "
            f"{value['embedding_pairwise_l2']['mean']:.4f} | "
            f"{value['motion_pairwise_rms_common_frames']['mean']:.4f} | "
            f"{value['text_motion_l2']['mean']:.4f} | "
            f"{value['within_caption_embedding_variance']:.6f} | "
            f"{value['across_caption_centroid_variance']:.6f} |"
        )
    lines.extend([
        "## Exact-output audit", "",
        "| Model | unique/total | captions 10/10 unique | fully deterministic | duplicate pair rate | cross-caption duplicate groups | all-zero | length min/mean/max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name, value in results.items():
        length = value["generated_length"]
        lines.append(
            f"| {name} | {value['unique_embeddings']}/{value['total_generations']} | "
            f"{value['captions_all_repeats_unique']}/100 | {value['captions_fully_deterministic']} | "
            f"{value['exact_duplicate_pair_rate']:.4f} | "
            f"{value['cross_caption_exact_duplicate_groups']} | {value['all_zero_generations']} | "
            f"{length['min']:.0f}/{length['mean']:.1f}/{length['max']:.0f} |"
        )
    lines.extend(["", "## Action groups", ""])
    for name, value in results.items():
        lines.extend([
            f"### {name}", "",
            "| Category | n | R@1 | mean rank | text-motion L2 | embed pair L2 | motion pair RMS |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for category, group in value["category"].items():
            lines.append(
                f"| {category} | {group['count']} | {group['r1']:.4f} | "
                f"{group['mean_correct_rank']:.2f} | {group['text_motion_l2']:.4f} | "
                f"{group['embedding_pairwise_l2']:.4f} | {group['motion_pairwise_rms']:.4f} |"
            )
        lines.append("")
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
