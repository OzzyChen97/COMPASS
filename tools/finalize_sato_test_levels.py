#!/usr/bin/env python
"""Finalize deterministic P1/P2/P3 test perturbations from pair progress."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


LEVELS = ("P1", "P2", "P3")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select one deterministic lexical, syntactic, and narrative "
            "perturbation per motion from build_sato_pairs progress data."
        )
    )
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--mixed-ratios",
        nargs=3,
        type=float,
        metavar=("P1", "P2", "P3"),
        default=(0.46, 0.27, 0.27),
        help="Deterministic one-perturbation-per-motion P1/P2/P3 mixture.",
    )
    return parser.parse_args()


def normalize_caption(caption):
    return re.sub(r"\s+", " ", str(caption).strip())


def basic_filter(original, candidate):
    if not isinstance(candidate, str):
        return False
    candidate = normalize_caption(candidate)
    if not candidate or candidate.lower().rstrip(".") == original.lower().rstrip("."):
        return False
    original_len = len(original.split())
    candidate_len = len(candidate.split())
    return max(1, int(original_len * 0.3)) <= candidate_len <= max(
        3, int(original_len * 3.0)
    )


def result_by_index(results):
    indexed = {}
    for row in results:
        try:
            indexed[int(row.get("index")) - 1] = row
        except (AttributeError, TypeError, ValueError):
            continue
    return indexed


def deterministic_choice(values, motion_id, level, seed):
    digest = hashlib.sha256(
        f"{seed}:{motion_id}:{level}".encode("utf-8")
    ).digest()
    return values[int.from_bytes(digest[:8], "big") % len(values)]


def load_rows(progress_path, seed):
    rows_by_level = {level: [] for level in LEVELS}
    fallback_counts = Counter()
    missing_rows = []
    latest = {}

    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            bundle = json.loads(line)
            preserve = result_by_index(bundle.get("preserve", []))
            for local_index, item in enumerate(bundle.get("items", [])):
                motion_id = str(item["motion_id"])
                original = normalize_caption(item["text_orig"])
                latest[motion_id] = (original, preserve.get(local_index, {}))

    for motion_id, (original, result) in latest.items():
        for level in LEVELS:
            values = result.get(level, []) if isinstance(result, dict) else []
            if isinstance(values, str):
                values = [values]
            candidates = [
                normalize_caption(candidate)
                for candidate in values
                if basic_filter(original, candidate)
            ]
            if candidates:
                text_pair = deterministic_choice(
                    candidates, motion_id, level, seed
                )
            else:
                missing_rows.append({
                    "motion_id": motion_id,
                    "text_orig": original,
                    "level": level,
                })
                continue

            rows_by_level[level].append(
                {
                    "motion_id": motion_id,
                    "text_orig": original,
                    "text_pair": text_pair,
                    "relation": "preserve",
                    "edit_type": "",
                    "level": level,
                }
            )

    if missing_rows:
        raise RuntimeError(
            f"Missing valid generated candidates for {len(missing_rows)} "
            "motion/level rows; fallback text is disabled"
        )
    return rows_by_level, fallback_counts


def add_pair_ids(rows):
    return [dict(row, pair_id=index) for index, row in enumerate(rows, start=1)]


def write_json(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(add_pair_ids(rows), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def choose_mixed_level(motion_id, ratios, seed):
    total = sum(ratios)
    if total <= 0:
        raise ValueError("--mixed-ratios must have positive total weight")
    value = int.from_bytes(
        hashlib.sha256(f"mixed:{seed}:{motion_id}".encode("utf-8")).digest()[:8],
        "big",
    ) / float(2**64)
    cumulative = 0.0
    for level, ratio in zip(LEVELS, ratios):
        cumulative += ratio / total
        if value < cumulative:
            return level
    return LEVELS[-1]


def main():
    args = parse_args()
    if not args.progress.is_file():
        raise FileNotFoundError(args.progress)

    rows_by_level, fallback_counts = load_rows(args.progress, args.seed)
    motion_counts = {level: len(rows) for level, rows in rows_by_level.items()}
    if len(set(motion_counts.values())) != 1:
        raise ValueError(f"Level counts do not match: {motion_counts}")

    all_rows = []
    by_motion = defaultdict(dict)
    for level in LEVELS:
        level_path = Path(f"{args.output_prefix}_{level.lower()}.json")
        write_json(level_path, rows_by_level[level])
        for row in rows_by_level[level]:
            by_motion[row["motion_id"]][level] = row

    for motion_id in rows_by_level["P1"]:
        current_id = motion_id["motion_id"]
        for level in LEVELS:
            all_rows.append(by_motion[current_id][level])

    combined_path = Path(f"{args.output_prefix}_p123.json")
    write_json(combined_path, all_rows)

    mixed_rows = []
    mixed_counts = Counter()
    for motion_id in sorted(by_motion):
        level = choose_mixed_level(motion_id, args.mixed_ratios, args.seed)
        mixed_rows.append(by_motion[motion_id][level])
        mixed_counts[level] += 1
    mixed_path = Path(f"{args.output_prefix}_mixed_human_ratio.json")
    write_json(mixed_path, mixed_rows)

    metadata = {
        "progress": str(args.progress.resolve()),
        "output_prefix": str(args.output_prefix.resolve()),
        "seed": args.seed,
        "levels": list(LEVELS),
        "motions_per_level": motion_counts,
        "combined_pairs": len(all_rows),
        "unique_motions": len(by_motion),
        "fallback_counts": dict(fallback_counts),
        "files": {
            level: str(Path(f"{args.output_prefix}_{level.lower()}.json").resolve())
            for level in LEVELS
        },
        "combined_file": str(combined_path.resolve()),
        "mixed_ratios": dict(zip(LEVELS, args.mixed_ratios)),
        "mixed_counts": dict(mixed_counts),
        "mixed_file": str(mixed_path.resolve()),
    }
    meta_path = Path(f"{args.output_prefix}_p123.meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
