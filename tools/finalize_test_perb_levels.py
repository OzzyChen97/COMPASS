#!/usr/bin/env python
"""
Finalize deterministic P1/P2/P3 test perturbation sets for a SATO dataset.

Follows the HumanML3D convention (see dataset/HumanML3D/build_test_perb.py and
build_test_perb_supplement.py, and tools/finalize_sato_test_levels.py):

  * build_test_perb.py deduplicates captions globally, so a motion whose
    captions all appear on an earlier motion gets no perturbation. HumanML3D
    repairs this with a supplement pass so the eval set covers the whole test
    split. Here the repair is exact rather than generative: those motions share
    a caption verbatim with a covered motion, so the caption's existing
    paraphrases apply unchanged.
  * One deterministic pair per motion per level, chosen with the same
    sha256(seed:motion_id:level) rule as tools/finalize_sato_test_levels.py.
  * The same basic_filter validity rule as build_test_perb.py.
  * The mixed set uses HumanML3D's realized level ratio, 0.46/0.27/0.27
    (its shipped test_perb_deterministic.json is P1 1988 / P2 1178 / P3 1178).

Outputs, given --output-prefix PREFIX:
  PREFIX_p1.json / _p2.json / _p3.json   pure single-level sets, identical motions
  PREFIX_p123.json                       all three levels per motion
  PREFIX_p2p3.json                       deterministic P2/P3 mix
  PREFIX_mixed_full.json                 P1/P2/P3 mix at --mixed-ratios
  PREFIX_levels.meta.json                provenance and counts

Example:
  python tools/finalize_test_perb_levels.py \
    --dataset-root dataset/Motion-X-322 \
    --output-prefix dataset/Motion-X-322/test_perb_gemini31 \
    --seed 123
"""

import argparse
import codecs as cs
import hashlib
import json
import os
from collections import Counter, OrderedDict, defaultdict
from os.path import join as pjoin

LEVELS = ("P1", "P2", "P3")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset directory holding test.txt and texts/.",
    )
    parser.add_argument(
        "--pool",
        default=None,
        help="Candidate pool from build_test_perb.py "
        "(default: <dataset-root>/test_perb.json).",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output path prefix (default: <dataset-root>/test_perb_gemini31).",
    )
    parser.add_argument(
        "--mixed-ratios",
        nargs=3,
        type=float,
        metavar=("P1", "P2", "P3"),
        default=(0.46, 0.27, 0.27),
        help="Level mixture for PREFIX_mixed_full.json; HumanML3D's ratio.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--single-level-only",
        action="store_true",
        help=(
            "Write only the P1/P2/P3 files and metadata. Use this when the "
            "dataset has test motions with no available perturbation and a "
            "full-coverage mixed set is neither required nor valid."
        ),
    )
    args = parser.parse_args()
    if args.pool is None:
        args.pool = pjoin(args.dataset_root, "test_perb.json")
    if args.output_prefix is None:
        args.output_prefix = pjoin(args.dataset_root, "test_perb_gemini31")
    return args


def basic_filter(orig, para):
    """Identical to build_test_perb.py."""
    if not para or not para.strip():
        return False
    o = orig.lower().rstrip(".")
    p = para.lower().rstrip(".")
    if o == p:
        return False
    ol, pl = len(orig.split()), len(para.split())
    return ol * 0.3 <= pl <= ol * 3.0


def deterministic_choice(values, motion_id, level, seed):
    """Identical to tools/finalize_sato_test_levels.py."""
    digest = hashlib.sha256(
        f"{seed}:{motion_id}:{level}".encode("utf-8")
    ).digest()
    return values[int.from_bytes(digest[:8], "big") % len(values)]


def deterministic_key(motion_id, seed, salt=None):
    """Matches tools/build_mixed_perturbation_set.py (no salt) and
    tools/build_sato_level_mix.py (salt='P2P3')."""
    payload = f"{seed}:{motion_id}" if salt is None else f"{seed}:{motion_id}:{salt}"
    return hashlib.sha256(payload.encode("utf-8")).digest()


def collect_test_captions(dataset_root):
    """Captions per test motion, in file order."""
    test_file = pjoin(dataset_root, "test.txt")
    text_dir = pjoin(dataset_root, "texts")
    with open(test_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    per_motion = OrderedDict()
    for mid in ids:
        path = pjoin(text_dir, mid + ".txt")
        if not os.path.exists(path):
            continue
        caps = []
        with cs.open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("#")
                if not parts:
                    continue
                cap = parts[0].strip()
                if cap:
                    caps.append(cap)
        if caps:
            per_motion[mid] = caps
    return ids, per_motion


def index_pool_by_caption(pool_rows):
    """caption -> level -> sorted unique candidate paraphrases."""
    raw = defaultdict(lambda: defaultdict(list))
    for row in pool_rows:
        if row.get("relation") != "preserve":
            continue
        level = row.get("level")
        if level not in LEVELS:
            continue
        orig = row.get("text_orig", "")
        para = str(row.get("text_pair", "")).strip()
        if basic_filter(orig, para):
            raw[orig][level].append(para)

    return {
        caption: {level: sorted(set(values)) for level, values in by_level.items()}
        for caption, by_level in raw.items()
    }


def build_full_coverage_candidates(ids, per_motion, by_caption, seed):
    candidates = {}
    for mid in ids:
        by_level = {}
        for level in LEVELS:
            caption = next(
                (
                    value
                    for value in per_motion.get(mid, [])
                    if by_caption.get(value, {}).get(level)
                ),
                None,
            )
            if caption is None:
                continue
            by_level[level] = {
                "motion_id": mid,
                "text_orig": caption,
                "text_pair": deterministic_choice(
                    by_caption[caption][level], mid, level, seed),
                "relation": "preserve",
                "edit_type": "",
                "level": level,
            }
        candidates[mid] = by_level
    return candidates

def build_full_coverage_mixed(ids, by_motion, mixed_ratios, seed):
    """Select one deterministic available perturbation for every motion."""
    ordered = sorted(ids, key=lambda mid: deterministic_key(mid, seed))
    weight_sum = sum(mixed_ratios)
    block = [round(len(ordered) * weight / weight_sum) for weight in mixed_ratios]
    block[-1] += len(ordered) - sum(block)

    preferred_levels = []
    for level, size in zip(LEVELS, block):
        preferred_levels.extend([level] * size)

    fallback_counts = Counter()
    mixed = []
    for mid, preferred in zip(ordered, preferred_levels):
        available = by_motion.get(mid, {})
        if not available:
            raise ValueError(f"No perturbation available for test motion: {mid}")
        if preferred in available:
            selected = preferred
        else:
            selected = min(
                available,
                key=lambda level: (
                    abs(LEVELS.index(level) - LEVELS.index(preferred)),
                    LEVELS.index(level),
                ),
            )
            fallback_counts[f"{preferred}->{selected}"] += 1
        mixed.append(available[selected])

    mixed.sort(key=lambda row: deterministic_key(row["motion_id"], seed + 1))
    return mixed, dict(sorted(fallback_counts.items()))

def write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [dict(row, pair_id=i) for i, row in enumerate(rows, start=1)],
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    args = parse_args()

    with open(args.pool, "r", encoding="utf-8") as f:
        pool_rows = json.load(f)
    by_caption = index_pool_by_caption(pool_rows)
    pool_motions = {str(row["motion_id"]) for row in pool_rows}

    ids, per_motion = collect_test_captions(args.dataset_root)

    full_coverage_candidates = build_full_coverage_candidates(
        ids, per_motion, by_caption, args.seed)
    rows_by_level = {level: [] for level in LEVELS}
    reused = 0
    skipped = []

    for mid in ids:
        captions = per_motion.get(mid)
        if not captions:
            skipped.append({"motion_id": mid, "reason": "no_caption"})
            continue

        # Prefer a caption with candidates at every level so the three level
        # files stay on an identical motion set.
        chosen_caption = next(
            (
                cap
                for cap in captions
                if all(by_caption.get(cap, {}).get(level) for level in LEVELS)
            ),
            None,
        )
        if chosen_caption is None:
            skipped.append({"motion_id": mid, "reason": "incomplete_levels"})
            continue
        # Motions whose captions were globally deduped reuse the identical
        # caption's paraphrases rather than a fresh LLM call.
        if mid not in pool_motions:
            reused += 1

        for level in LEVELS:
            rows_by_level[level].append(
                {
                    "motion_id": mid,
                    "text_orig": chosen_caption,
                    "text_pair": deterministic_choice(
                        by_caption[chosen_caption][level], mid, level, args.seed
                    ),
                    "relation": "preserve",
                    "edit_type": "",
                    "level": level,
                }
            )

    counts = {level: len(rows) for level, rows in rows_by_level.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Level counts do not match: {counts}")

    for level in LEVELS:
        write_rows(f"{args.output_prefix}_{level.lower()}.json", rows_by_level[level])

    by_motion = defaultdict(dict)
    for level in LEVELS:
        for row in rows_by_level[level]:
            by_motion[row["motion_id"]][level] = row

    combined = [
        by_motion[row["motion_id"]][level]
        for row in rows_by_level["P1"]
        for level in LEVELS
    ]
    write_rows(f"{args.output_prefix}_p123.json", combined)

    # Deterministic P2/P3 mix, matching tools/build_sato_level_mix.py.
    p2p3 = []
    for mid in sorted(by_motion):
        level = "P2" if deterministic_key(mid, args.seed, "P2P3")[0] < 128 else "P3"
        p2p3.append(by_motion[mid][level])
    write_rows(f"{args.output_prefix}_p2p3.json", p2p3)

    # P1/P2/P3 mix at HumanML3D's ratio, covering every test motion. Motions
    # without the preferred level use their nearest available level. A pure
    # per-level evaluation intentionally skips this artifact because all three
    # files must share the same complete-level motion/caption source set.
    mixed = []
    mixed_fallback_counts = {}
    if not args.single_level_only:
        mixed, mixed_fallback_counts = build_full_coverage_mixed(
            ids, full_coverage_candidates, args.mixed_ratios, args.seed)
        write_rows(f"{args.output_prefix}_mixed_full.json", mixed)

    metadata = {
        "dataset_root": os.path.abspath(args.dataset_root),
        "pool": os.path.abspath(args.pool),
        "output_prefix": os.path.abspath(args.output_prefix),
        "seed": args.seed,
        "test_motions": len(ids),
        "motions_per_level": counts,
        "motions_reusing_deduped_caption": reused,
        "combined_pairs": len(combined),
        "mixed_ratios": dict(zip(LEVELS, args.mixed_ratios)),
        "mixed_counts": dict(Counter(row["level"] for row in mixed)),
        "mixed_fallback_counts": mixed_fallback_counts,
        "mixed_full_coverage": (
            None if args.single_level_only else len(mixed) == len(ids)
        ),
        "single_level_only": bool(args.single_level_only),
        "p2p3_counts": dict(Counter(row["level"] for row in p2p3)),
        "skipped_counts": dict(Counter(row["reason"] for row in skipped)),
        "skipped": skipped[:50],
        "skipped_total": len(skipped),
        "one_pair_per_motion": all(
            len({row["motion_id"] for row in rows}) == len(rows)
            for rows in rows_by_level.values()
        ),
    }
    with open(f"{args.output_prefix}_levels.meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    summary = dict(metadata)
    summary.pop("skipped", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
