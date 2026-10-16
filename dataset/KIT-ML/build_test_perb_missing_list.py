"""
List KIT-ML test motions that build_test_perb.py never covered.

build_test_perb.py deduplicates captions globally (`if cap not in captions`),
so a motion whose captions all appear on an earlier motion never gets a
perturbation. HumanML3D handles this with test_perb_missing.json plus
build_test_perb_supplement.py; this script produces the KIT equivalent.

Output:
  test_perb_missing.json   [{"motion_id": ..., "text_orig": ...}, ...]
"""

import codecs as cs
import json
import os
from collections import OrderedDict
from os.path import join as pjoin

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = pjoin(ROOT_DIR, "texts")
TEST_FILE = pjoin(ROOT_DIR, "test.txt")
POOL_JSON = pjoin(ROOT_DIR, "test_perb.json")
MISSING_JSON = pjoin(ROOT_DIR, "test_perb_missing.json")


def collect_test_captions():
    """Mirror build_test_perb.py: global caption dedup, first motion wins."""
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    captions = OrderedDict()
    per_motion = OrderedDict()
    for mid in ids:
        path = pjoin(TEXT_DIR, mid + ".txt")
        if not os.path.exists(path):
            continue
        motion_caps = []
        with cs.open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("#")
                if not parts:
                    continue
                cap = parts[0].strip()
                if not cap:
                    continue
                motion_caps.append(cap)
                if cap not in captions:
                    captions[cap] = mid
        if motion_caps:
            per_motion[mid] = motion_caps
    return ids, captions, per_motion


def main():
    ids, captions, per_motion = collect_test_captions()

    with open(POOL_JSON, "r", encoding="utf-8") as f:
        covered = {str(row["motion_id"]) for row in json.load(f)}

    missing = [
        {"motion_id": mid, "text_orig": per_motion[mid][0]}
        for mid in ids
        if mid in per_motion and mid not in covered
    ]

    with open(MISSING_JSON, "w", encoding="utf-8") as f:
        json.dump(missing, f, ensure_ascii=False, indent=2)

    print(f"Test motions: {len(ids)}")
    print(f"Motions with captions: {len(per_motion)}")
    print(f"Covered by {os.path.basename(POOL_JSON)}: {len(covered)}")
    print(f"Missing: {len(missing)} -> {MISSING_JSON}")


if __name__ == "__main__":
    main()
