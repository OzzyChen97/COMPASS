#!/usr/bin/env python3
"""Strictly audit fresh MotionGPT Motion-X tokenization coverage and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


MOTION_DIM = 322
UNIT_LENGTH = 4
CODEBOOK_SIZE = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_tokens(data_root: Path, token_dir: Path, selection_json: Path) -> dict:
    data_root = data_root.resolve()
    token_dir = token_dir.resolve()
    selection_json = selection_json.resolve()
    selection = json.loads(selection_json.read_text(encoding="utf-8"))
    if selection.get("split") != "val" or selection.get("monitor") != "total/val":
        raise RuntimeError("tokenization requires a validation-selected stage-1 checkpoint")
    if selection.get("uses_t2mgpt_generator_checkpoint") is not False:
        raise RuntimeError("tokenization provenance does not prove baseline-free stage 1")
    checkpoint = Path(selection["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"selected stage-1 checkpoint is missing: {checkpoint}")
    if sha256_file(checkpoint) != selection.get("checkpoint_sha256"):
        raise RuntimeError("selected stage-1 checkpoint hash no longer matches selection metadata")

    split_path = data_root / "train.txt"
    names = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise RuntimeError("Motion-X train split contains duplicate IDs")
    expected = {}
    invalid = {}
    for name in names:
        motion_path = data_root / "new_joint_vecs" / f"{name}.npy"
        if not motion_path.is_file():
            invalid[name] = "missing_motion"
            continue
        try:
            motion = np.load(motion_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError):
            invalid[name] = "unreadable_motion"
            continue
        if motion.ndim != 2 or motion.shape[1] != MOTION_DIM:
            invalid[name] = f"invalid_shape:{tuple(motion.shape)}"
            continue
        usable_frames = (int(motion.shape[0]) // UNIT_LENGTH) * UNIT_LENGTH
        if usable_frames < UNIT_LENGTH:
            invalid[name] = f"too_short:{int(motion.shape[0])}"
            continue
        expected[name] = usable_frames // UNIT_LENGTH

    actual = {}
    for path in token_dir.rglob("*.npy"):
        name = path.relative_to(token_dir).with_suffix("").as_posix()
        if name in actual:
            raise RuntimeError(f"duplicate token ID after path normalization: {name}")
        actual[name] = path
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    errors = []
    inventory = hashlib.sha256()
    for name in sorted(set(expected) & set(actual)):
        path = actual[name]
        try:
            tokens = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            errors.append({"motion_id": name, "error": f"unreadable_token:{exc}"})
            continue
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            errors.append({"motion_id": name, "error": f"invalid_shape:{tuple(tokens.shape)}"})
            continue
        if int(tokens.shape[1]) != expected[name]:
            errors.append({
                "motion_id": name,
                "error": f"length:{int(tokens.shape[1])}!={expected[name]}",
            })
            continue
        if not np.issubdtype(tokens.dtype, np.integer):
            errors.append({"motion_id": name, "error": f"non_integer_dtype:{tokens.dtype}"})
            continue
        if tokens.size and (int(tokens.min()) < 0 or int(tokens.max()) >= CODEBOOK_SIZE):
            errors.append({
                "motion_id": name,
                "error": f"code_range:{int(tokens.min())}:{int(tokens.max())}",
            })
            continue
        digest = sha256_file(path)
        inventory.update(
            f"{name}\0{tokens.shape[1]}\0{tokens.dtype}\0{digest}\n".encode("utf-8")
        )

    status = "pass" if not missing and not extra and not errors else "fail"
    payload = {
        "schema_version": "motiongpt-motionx322-token-audit/v1",
        "status": status,
        "data_root": str(data_root),
        "token_dir": str(token_dir),
        "split": "train",
        "split_count": len(names),
        "eligible_count": len(expected),
        "ineligible_count": len(invalid),
        "ineligible_reasons": {
            reason: sum(value == reason for value in invalid.values())
            for reason in sorted(set(invalid.values()))
        },
        "token_file_count": len(actual),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "invalid_token_count": len(errors),
        "missing_ids": missing[:100],
        "extra_ids": extra[:100],
        "token_errors": errors[:100],
        "inventory_sha256": inventory.hexdigest(),
        "unit_length": UNIT_LENGTH,
        "codebook_size": CODEBOOK_SIZE,
        "stage1_selection": str(selection_json),
        "stage1_checkpoint": str(checkpoint),
        "stage1_checkpoint_sha256": selection["checkpoint_sha256"],
    }
    if status != "pass":
        raise RuntimeError(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--stage1-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_tokens(args.data_root, args.token_dir, args.stage1_selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
