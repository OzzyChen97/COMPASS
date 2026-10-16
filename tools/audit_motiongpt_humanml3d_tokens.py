#!/usr/bin/env python3
"""Fail-closed audit of fresh MotionGPT HumanML3D motion tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_tokens(data_root: Path) -> dict[str, int]:
    expected = {}
    for line in (data_root / "train.txt").read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        motion_path = data_root / "new_joint_vecs" / f"{name}.npy"
        if not motion_path.is_file():
            continue
        frames = len(np.load(motion_path, mmap_mode="r"))
        if 40 <= frames < 200:
            expected[name] = frames // 4
    return expected


def audit(
    data_root: Path,
    token_dir: Path,
    stage1_selection_path: Path,
    source_manifest_path: Path,
) -> dict:
    data_root = data_root.resolve()
    token_dir = token_dir.resolve()
    stage1_selection_path = stage1_selection_path.resolve()
    source_manifest_path = source_manifest_path.resolve()
    if not token_dir.is_dir():
        raise FileNotFoundError(f"fresh token directory is missing: {token_dir}")
    selection = json.loads(stage1_selection_path.read_text(encoding="utf-8"))
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if selection.get("split") != "val" or selection.get("monitor") != "total/val":
        raise RuntimeError("Stage 1 token source was not selected by validation total/val")
    if selection.get("uses_t2mgpt_generator_checkpoint") is not False:
        raise RuntimeError("Stage 1 selection does not prove baseline-free provenance")
    if manifest.get("method", {}).get("uses_released_motiongpt_generator_checkpoint") is not False:
        raise RuntimeError("source manifest permits a released MotionGPT generator")
    expected_dir_name = manifest.get("dataset", {}).get("token_code_path")
    if token_dir.name != expected_dir_name:
        raise RuntimeError(
            f"token directory does not match manifest: {token_dir.name!r} != {expected_dir_name!r}"
        )
    checkpoint = Path(selection.get("checkpoint", "")).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"selected Stage 1 checkpoint is missing: {checkpoint}")
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != selection.get("checkpoint_sha256"):
        raise RuntimeError("selected Stage 1 checkpoint hash mismatch")

    expected = expected_tokens(data_root)
    actual_paths = {path.stem: path for path in token_dir.glob("*.npy")}
    missing = sorted(set(expected) - set(actual_paths))
    extra = sorted(set(actual_paths) - set(expected))
    invalid = []
    token_lengths = []
    for name in sorted(set(expected) & set(actual_paths)):
        path = actual_paths[name]
        try:
            values = np.load(path, allow_pickle=False)
        except Exception as exc:
            invalid.append({"name": name, "reason": f"load failed: {exc}"})
            continue
        reason = None
        if values.ndim != 2 or values.shape[0] != 1:
            reason = f"shape {values.shape}, expected (1, {expected[name]})"
        elif values.shape[1] != expected[name]:
            reason = f"token length {values.shape[1]}, expected {expected[name]}"
        elif not np.issubdtype(values.dtype, np.integer):
            reason = f"non-integer dtype {values.dtype}"
        elif values.size and (int(values.min()) < 0 or int(values.max()) >= 512):
            reason = f"token values outside [0, 511]: [{values.min()}, {values.max()}]"
        if reason:
            invalid.append({"name": name, "reason": reason})
        else:
            token_lengths.append(int(values.shape[1]))

    status = "pass" if not missing and not extra and not invalid else "fail"
    payload = {
        "schema_version": "motiongpt-humanml3d-token-audit/v1",
        "status": status,
        "data_root": str(data_root),
        "token_dir": str(token_dir),
        "expected_count": len(expected),
        "actual_count": len(actual_paths),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "invalid_token_count": len(invalid),
        "missing": missing[:100],
        "extra": extra[:100],
        "invalid": invalid[:100],
        "token_length_min": min(token_lengths) if token_lengths else None,
        "token_length_max": max(token_lengths) if token_lengths else None,
        "stage1_selection": str(stage1_selection_path),
        "stage1_selection_sha256": sha256_file(stage1_selection_path),
        "stage1_checkpoint": str(checkpoint),
        "stage1_checkpoint_sha256": checkpoint_hash,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "rules": {
            "coverage": "exactly train IDs with motion length in [40, 200)",
            "token_length": "floor(original_frames / 4)",
            "shape": "one row per motion",
            "dtype": "integer",
            "value_range": "[0, 511]",
        },
    }
    if status != "pass":
        raise RuntimeError(json.dumps(payload, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--stage1-selection", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(
        args.data_root,
        args.token_dir,
        args.stage1_selection,
        args.source_manifest,
    )
    args.output.resolve().write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
