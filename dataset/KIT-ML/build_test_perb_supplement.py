#!/usr/bin/env python
"""Supplement missing P1/P2/P3 candidates for every KIT-ML test motion.

The base generator deduplicates captions globally. This pass keeps the base
pool intact, generates only genuinely missing levels, and writes a merged pool
that can be finalized into one pair per test motion.
"""

import argparse
import codecs as cs
import json
import os
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import build_test_perb as generator


LEVELS = ("P1", "P2", "P3")
ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=ROOT / "test_perb.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "test_perb_full_motion_pool.json"
    )
    parser.add_argument(
        "--seed-supplement",
        type=Path,
        default=ROOT / "test_perb_supplement_seed.json",
    )
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_motion_captions():
    with (ROOT / "test.txt").open("r", encoding="utf-8") as handle:
        motion_ids = [line.strip() for line in handle if line.strip()]

    per_motion = OrderedDict()
    for motion_id in motion_ids:
        path = ROOT / "texts" / f"{motion_id}.txt"
        if not path.is_file():
            continue
        captions = []
        with cs.open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split("#")
                caption = parts[0].strip() if parts else ""
                if caption:
                    captions.append(caption)
        if captions:
            per_motion[motion_id] = captions
    return motion_ids, per_motion


def index_pool(rows):
    indexed = defaultdict(lambda: defaultdict(list))
    for row in rows:
        level = row.get("level")
        original = str(row.get("text_orig", "")).strip()
        candidate = str(row.get("text_pair", "")).strip()
        if (
            row.get("relation") == "preserve"
            and level in LEVELS
            and generator.basic_filter(original, candidate)
        ):
            indexed[original][level].append(candidate)
    return indexed


def find_requirements(per_motion, indexed):
    requirements = OrderedDict()
    affected = defaultdict(list)
    for motion_id, captions in per_motion.items():
        if any(all(indexed[caption][level] for level in LEVELS) for caption in captions):
            continue
        chosen = max(
            captions,
            key=lambda caption: sum(bool(indexed[caption][level]) for level in LEVELS),
        )
        missing = [level for level in LEVELS if not indexed[chosen][level]]
        requirements.setdefault(chosen, set()).update(missing)
        affected[chosen].append(motion_id)
    return requirements, affected


def generate_round(requirements, batch_size):
    captions = list(requirements)
    generated = []
    for offset in range(0, len(captions), batch_size):
        batch = captions[offset:offset + batch_size]
        results = generator.call_llm(batch)
        for index, caption in enumerate(batch):
            result = results[index] if index < len(results) and isinstance(results[index], dict) else {}
            for level in sorted(requirements[caption]):
                for candidate in result.get(level, []):
                    candidate = str(candidate).strip()
                    if generator.basic_filter(caption, candidate):
                        generated.append(
                            {
                                "motion_id": "supplement",
                                "text_orig": caption,
                                "text_pair": candidate,
                                "relation": "preserve",
                                "edit_type": "supplement",
                                "level": level,
                            }
                        )
    return generated


def main():
    args = parse_args()
    base_rows = json.loads(args.pool.read_text(encoding="utf-8"))
    motion_ids, per_motion = collect_motion_captions()
    seed_rows = []
    if args.seed_supplement.is_file():
        seed_rows = json.loads(args.seed_supplement.read_text(encoding="utf-8"))
    merged = list(base_rows) + seed_rows

    initial_index = index_pool(merged)
    requirements, affected = find_requirements(per_motion, initial_index)
    initial_missing = sum(len(levels) for levels in requirements.values())
    print(
        json.dumps(
            {
                "test_motions": len(motion_ids),
                "motions_with_captions": len(per_motion),
                "captions_requiring_supplement": len(requirements),
                "missing_caption_levels": initial_missing,
                "affected_motions": sum(len(values) for values in affected.values()),
                "requirements": {
                    caption: {
                        "levels": sorted(requirements[caption]),
                        "motion_ids": affected[caption],
                    }
                    for caption in requirements
                },
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    generated_count = 0
    for round_index in range(1, args.max_rounds + 1):
        indexed = index_pool(merged)
        requirements, _ = find_requirements(per_motion, indexed)
        if not requirements:
            break
        generated = generate_round(requirements, args.batch_size)
        merged.extend(generated)
        generated_count += len(generated)
        print(
            f"Supplement round {round_index}: added {len(generated)} candidates",
            flush=True,
        )

    final_index = index_pool(merged)
    remaining, _ = find_requirements(per_motion, final_index)
    if remaining:
        raise RuntimeError(
            "Supplement did not produce complete P1/P2/P3 candidates for: "
            + ", ".join(remaining)
        )

    for pair_id, row in enumerate(merged, start=1):
        row["pair_id"] = pair_id
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "input": str(args.pool.resolve()),
        "output": str(args.output.resolve()),
        "base_pairs": len(base_rows),
        "seed_supplement_pairs": len(seed_rows),
        "supplement_pairs": generated_count,
        "total_pairs": len(merged),
        "test_motions": len(motion_ids),
        "complete_test_motions": len(per_motion),
        "supplement_level_counts": dict(
            Counter(row["level"] for row in merged[len(base_rows):])
        ),
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
