"""
Finalize deterministic P1/P2/P3 KIT-ML test perturbation sets.

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

Output:
  test_perb_gemini31_p1.json / _p2.json / _p3.json   pure single-level sets
  test_perb_gemini31_p123.json                       all three levels per motion
  test_perb_gemini31_levels.meta.json                provenance and counts
"""

import argparse
import codecs as cs
import hashlib
import json
import os
from collections import Counter, OrderedDict, defaultdict
from os.path import join as pjoin

LEVELS = ("P1", "P2", "P3")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = pjoin(ROOT_DIR, "texts")
TEST_FILE = pjoin(ROOT_DIR, "test.txt")
POOL_JSON = pjoin(ROOT_DIR, "test_perb.json")
OUTPUT_PREFIX = pjoin(ROOT_DIR, "test_perb_gemini31")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=POOL_JSON)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


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


def collect_test_captions():
    """Captions per test motion, in file order."""
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    per_motion = OrderedDict()
    for mid in ids:
        path = pjoin(TEXT_DIR, mid + ".txt")
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

    indexed = {}
    for caption, by_level in raw.items():
        indexed[caption] = {
            level: sorted(set(values)) for level, values in by_level.items()
        }
    return indexed


def main():
    args = parse_args()

    with open(args.pool, "r", encoding="utf-8") as f:
        pool_rows = json.load(f)
    by_caption = index_pool_by_caption(pool_rows)
    pool_motions = {str(row["motion_id"]) for row in pool_rows}

    ids, per_motion = collect_test_captions()

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
            candidates = by_caption[chosen_caption][level]
            rows_by_level[level].append(
                {
                    "motion_id": mid,
                    "text_orig": chosen_caption,
                    "text_pair": deterministic_choice(
                        candidates, mid, level, args.seed
                    ),
                    "relation": "preserve",
                    "edit_type": "",
                    "level": level,
                }
            )

    counts = {level: len(rows) for level, rows in rows_by_level.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Level counts do not match: {counts}")

    def write(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [dict(row, pair_id=i) for i, row in enumerate(rows, start=1)],
                f,
                ensure_ascii=False,
                indent=2,
            )

    for level in LEVELS:
        write(f"{args.output_prefix}_{level.lower()}.json", rows_by_level[level])

    by_motion = defaultdict(dict)
    for level in LEVELS:
        for row in rows_by_level[level]:
            by_motion[row["motion_id"]][level] = row
    combined = [
        by_motion[row["motion_id"]][level]
        for row in rows_by_level["P1"]
        for level in LEVELS
    ]
    write(f"{args.output_prefix}_p123.json", combined)

    metadata = {
        "pool": os.path.abspath(args.pool),
        "output_prefix": os.path.abspath(args.output_prefix),
        "seed": args.seed,
        "test_motions": len(ids),
        "motions_per_level": counts,
        "motions_reusing_deduped_caption": reused,
        "combined_pairs": len(combined),
        "skipped": skipped,
        "skipped_counts": dict(Counter(row["reason"] for row in skipped)),
        "one_pair_per_motion": all(
            len({row["motion_id"] for row in rows}) == len(rows)
            for rows in rows_by_level.values()
        ),
    }
    meta_path = f"{args.output_prefix}_levels.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
