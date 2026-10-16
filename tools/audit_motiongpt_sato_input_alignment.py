#!/usr/bin/env python3
"""Compare MotionGPT's frozen input index with the SATO++ loader path."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import dataset_TM_eval
from utils.sato_pp_eval import (
    _extract_motion_id,
    build_pair_index,
    sample_perturbed_texts,
)
from utils.word_vectorizer import WordVectorizer


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--motiongpt-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    pair_data = json.loads(args.pair_file.read_text(encoding="utf-8"))
    pair_index = build_pair_index(pair_data)
    level_index = {
        str(row["motion_id"]): str(row.get("level", "unknown"))
        for row in pair_data
    }
    expected_payload = json.loads(args.motiongpt_index.read_text(encoding="utf-8"))
    expected = expected_payload["samples"]

    reset_seed(args.seed)
    vectorizer = WordVectorizer(str(REPO_ROOT / "glove"), "our_vab")
    loader = dataset_TM_eval.DATALoader(
        "t2m", True, 32, vectorizer, num_workers=args.num_workers
    )

    actual = []
    matched = 0
    for batch in loader:
        _, _, captions, _, _, motion_lengths, _, names = batch
        perturbed, batch_matched = sample_perturbed_texts(names, pair_index)
        matched += batch_matched
        perturbed = [
            paired if paired is not None else original
            for paired, original in zip(perturbed, captions)
        ]
        actual.extend(
            {
                "loader_name": str(name),
                "motion_id": _extract_motion_id(name),
                "caption": str(caption),
                "perturbed_caption": str(paired),
                "level": level_index.get(_extract_motion_id(name)),
                "motion_length": int(length),
            }
            for name, caption, paired, length in zip(
                names, captions, perturbed, motion_lengths
            )
        )

    mismatches = [
        {"offset": index, "sato": left, "motiongpt": right}
        for index, (left, right) in enumerate(zip(actual, expected))
        if left != right
    ]
    result = {
        "schema_version": "motiongpt-sato-input-alignment/v1",
        "seed": args.seed,
        "sato_count": len(actual),
        "motiongpt_count": len(expected),
        "matched_pairs": matched,
        "exact_equal": actual == expected,
        "mismatch_count": len(mismatches) + abs(len(actual) - len(expected)),
        "first_mismatches": mismatches[:10],
        "sato_sample_index_sha256": canonical_sha256(actual),
        "motiongpt_sample_index_sha256": canonical_sha256(expected),
        "motiongpt_recorded_sha256": expected_payload.get("sample_index_sha256"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not result["exact_equal"]:
        raise SystemExit("SATO++ and MotionGPT input records differ")


if __name__ == "__main__":
    main()
