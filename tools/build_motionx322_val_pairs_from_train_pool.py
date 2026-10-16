#!/usr/bin/env python
"""Build an independent Motion-X validation perturbation set from train-only pairs.

The builder never reads a test perturbation file. It reuses a historical
Gemini-generated paraphrase only when a validation caption is an exact match
for a caption in the train-only pair pool. This yields a smaller but strictly
split-separated validation subset suitable for post-training checkpoint
selection when the historical pair-generation service is unavailable.
"""

import argparse
import codecs as cs
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


LEVELS = ("P1", "P2", "P3")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/Motion-X-322-MaxToken75"),
    )
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument(
        "--source-dataset-root",
        type=Path,
        default=Path("dataset/Motion-X-322"),
    )
    parser.add_argument(
        "--source-pool",
        type=Path,
        default=Path(
            "dataset/Motion-X-322/"
            "preserving_pairs_gemini31_human_recipe_filtered.jsonl"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--weights",
        nargs=3,
        type=float,
        metavar=("P1", "P2", "P3"),
        default=(0.46, 0.27, 0.27),
    )
    return parser.parse_args()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_digest(seed, *parts):
    payload = ":".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def basic_filter(original, candidate):
    if not isinstance(candidate, str) or not candidate.strip():
        return False
    original = original.strip()
    candidate = candidate.strip()
    if candidate.lower().rstrip(".") == original.lower().rstrip("."):
        return False
    original_len = len(original.split())
    candidate_len = len(candidate.split())
    return original_len * 0.3 <= candidate_len <= original_len * 3.0


def load_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            return json.load(handle)
        return [json.loads(line) for line in handle if line.strip()]


def parse_tag(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return 0.0 if math.isnan(value) else value


def load_annotations(path):
    rows = []
    with cs.open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("#")
            if not fields or not fields[0].strip():
                continue
            from_tag = parse_tag(fields[2]) if len(fields) >= 4 else 0.0
            to_tag = parse_tag(fields[3]) if len(fields) >= 4 else 0.0
            rows.append(
                {
                    "caption": fields[0].strip(),
                    "full_motion": from_tag == 0.0 and to_tag == 0.0,
                }
            )
    return rows


def read_split(root, split):
    path = root / f"{split}.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def allocate_level_counts(total, weights):
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("--weights must have a positive sum")
    counts = [round(total * weight / weight_sum) for weight in weights]
    counts[-1] += total - sum(counts)
    return counts


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    source_root = args.source_dataset_root.resolve()
    source_pool = args.source_pool.resolve()
    output = args.output.resolve()

    target_splits = {
        split: read_split(dataset_root, split) for split in ("train", "val", "test")
    }
    target_sets = {key: set(value) for key, value in target_splits.items()}
    split_intersections = {
        "train_val": len(target_sets["train"] & target_sets["val"]),
        "train_test": len(target_sets["train"] & target_sets["test"]),
        "val_test": len(target_sets["val"] & target_sets["test"]),
    }
    if any(split_intersections.values()):
        raise ValueError(f"Target dataset splits overlap: {split_intersections}")

    source_train = set(read_split(source_root, "train"))
    pool_rows = load_rows(source_pool)
    pool_motion_ids = {
        str(row.get("motion_id", "")).strip()
        for row in pool_rows
        if str(row.get("motion_id", "")).strip()
    }
    outside_source_train = sorted(pool_motion_ids - source_train)
    if outside_source_train:
        raise ValueError(
            f"Source pair pool contains {len(outside_source_train)} non-train motions"
        )

    by_caption = defaultdict(lambda: defaultdict(list))
    accepted_pool_rows = 0
    for row in pool_rows:
        if row.get("relation") != "preserve":
            continue
        level = str(row.get("level", ""))
        if level not in LEVELS:
            continue
        original = str(row.get("text_orig", "")).strip()
        candidate = str(row.get("text_pair", "")).strip()
        if not original or not basic_filter(original, candidate):
            continue
        by_caption[original][level].append(
            {
                "text_pair": candidate,
                "source_motion_id": str(row.get("motion_id", "")).strip(),
                "source_pair_id": row.get("pair_id"),
            }
        )
        accepted_pool_rows += 1

    eligible = []
    exclusion_counts = Counter()
    for motion_id in target_splits[args.split]:
        text_path = dataset_root / "texts" / f"{motion_id}.txt"
        if not text_path.is_file():
            exclusion_counts["missing_text_file"] += 1
            continue
        annotations = load_annotations(text_path)
        complete = [
            row
            for row in annotations
            if all(by_caption.get(row["caption"], {}).get(level) for level in LEVELS)
        ]
        if not complete:
            exclusion_counts["no_exact_train_caption_with_p1_p2_p3"] += 1
            continue
        chosen = next((row for row in complete if row["full_motion"]), complete[0])
        if not chosen["full_motion"]:
            exclusion_counts["segment_caption_only"] += 1
            continue
        eligible.append(
            {
                "motion_id": motion_id,
                "text_orig": chosen["caption"],
            }
        )

    eligible.sort(
        key=lambda row: deterministic_digest(args.seed, "level", row["motion_id"])
    )
    level_counts = allocate_level_counts(len(eligible), args.weights)
    preferred_levels = []
    for level, count in zip(LEVELS, level_counts):
        preferred_levels.extend([level] * count)

    selected = []
    source_motion_counts = Counter()
    for item, level in zip(eligible, preferred_levels):
        candidates = sorted(
            by_caption[item["text_orig"]][level],
            key=lambda row: deterministic_digest(
                args.seed,
                "candidate",
                item["motion_id"],
                level,
                item["text_orig"],
                row["text_pair"],
                row["source_motion_id"],
                row["source_pair_id"],
            ),
        )
        source = candidates[0]
        source_motion_counts[source["source_motion_id"]] += 1
        selected.append(
            {
                "motion_id": item["motion_id"],
                "text_orig": item["text_orig"],
                "text_pair": source["text_pair"],
                "relation": "preserve",
                "edit_type": "",
                "level": level,
                "source_split": "train",
                "source_motion_id": source["source_motion_id"],
                "source_pair_id": source["source_pair_id"],
            }
        )

    selected.sort(
        key=lambda row: deterministic_digest(args.seed, "output", row["motion_id"])
    )
    for pair_id, row in enumerate(selected, start=1):
        row["pair_id"] = pair_id

    if len({row["motion_id"] for row in selected}) != len(selected):
        raise RuntimeError("Validation output must contain one row per motion")
    if {row["motion_id"] for row in selected} & target_sets["test"]:
        raise RuntimeError("Validation output unexpectedly contains test motion IDs")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(selected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    generator_meta_path = source_pool.with_suffix(source_pool.suffix + ".meta.json")
    generator_meta = None
    if generator_meta_path.is_file():
        generator_meta = json.loads(generator_meta_path.read_text(encoding="utf-8"))

    metadata = {
        "schema_version": 1,
        "builder": str(Path(__file__).resolve()),
        "construction": (
            "exact-caption reuse from a train-only historical perturbation pool; "
            "no test perturbation file is read"
        ),
        "reason": (
            "The historical Gemini endpoint/key is unavailable. Exact-caption "
            "reuse preserves its recorded perturbation recipe without changing "
            "the generator model or consulting the test perturbation protocol."
        ),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "split_counts": {key: len(value) for key, value in target_splits.items()},
        "split_intersections": split_intersections,
        "split_id_sha256": {
            key: canonical_sha256(value) for key, value in target_splits.items()
        },
        "source_dataset_root": str(source_root),
        "source_pool": str(source_pool),
        "source_pool_sha256": sha256_file(source_pool),
        "source_pool_metadata": generator_meta,
        "source_pool_rows": len(pool_rows),
        "accepted_source_preserve_rows": accepted_pool_rows,
        "source_pool_motion_ids": len(pool_motion_ids),
        "source_pool_non_train_motion_ids": len(outside_source_train),
        "selection_seed": args.seed,
        "level_weights": dict(zip(LEVELS, args.weights)),
        "target_split_motions": len(target_splits[args.split]),
        "eligible_motions": len(eligible),
        "coverage_fraction": (
            len(eligible) / len(target_splits[args.split])
            if target_splits[args.split]
            else 0.0
        ),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "output": str(output),
        "output_pairs": len(selected),
        "output_level_counts": dict(sorted(Counter(row["level"] for row in selected).items())),
        "output_motion_id_sha256": canonical_sha256(
            [row["motion_id"] for row in selected]
        ),
        "output_rows_sha256": canonical_sha256(selected),
        "output_file_sha256": sha256_file(output),
        "unique_source_motions_reused": len(source_motion_counts),
        "test_pair_files_used": [],
    }
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()