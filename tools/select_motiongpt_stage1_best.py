#!/usr/bin/env python3
"""Audit and materialize the validation-selected MotionGPT stage-1 checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_stage1_best(experiment_dir: Path, source_manifest: Path) -> dict:
    experiment_dir = experiment_dir.resolve()
    source_manifest = source_manifest.resolve()
    checkpoint_dir = experiment_dir / "checkpoints"
    last_path = checkpoint_dir / "last.ckpt"
    if not last_path.is_file():
        raise FileNotFoundError(f"missing completed stage-1 checkpoint: {last_path}")
    if not source_manifest.is_file():
        raise FileNotFoundError(f"missing fresh-run source manifest: {source_manifest}")

    last = torch.load(last_path, map_location="cpu")
    matches = []
    for callback_key, state in last.get("callbacks", {}).items():
        if not isinstance(state, dict):
            continue
        callback_text = str(callback_key)
        if "ModelCheckpoint" not in callback_text:
            continue
        if "'monitor': 'total/val'" not in callback_text:
            continue
        if "'mode': 'min'" not in callback_text:
            continue
        matches.append(state)
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one min-total/val ModelCheckpoint callback, "
            f"found {len(matches)}"
        )
    del last
    gc.collect()

    state = matches[0]
    score_value = state.get("best_model_score")
    if score_value is None:
        raise RuntimeError("stage-1 callback has no best_model_score")
    score = float(score_value.item() if hasattr(score_value, "item") else score_value)
    if not math.isfinite(score):
        raise RuntimeError(f"non-finite stage-1 best validation loss: {score}")

    stored_path = str(state.get("best_model_path", "")).strip()
    if not stored_path:
        raise RuntimeError("stage-1 callback has no best_model_path")
    # Only permit a checkpoint physically inside this fresh experiment.  The
    # callback may store a MotionGPT-root-relative path, so use its basename
    # after rejecting directory traversal and unexpected suffixes.
    stored = Path(stored_path)
    normalized_parts = Path(stored_path.replace("\\", "/")).parts
    expected_tail = (
        "experiments", "mgpt", experiment_dir.name, "checkpoints", stored.name
    )
    if tuple(normalized_parts[-5:]) != expected_tail:
        raise RuntimeError(f"invalid stored checkpoint path: {stored_path}")
    if stored.suffix != ".ckpt":
        raise RuntimeError(f"stage-1 best path is not a checkpoint: {stored_path}")
    best_path = checkpoint_dir / stored.name
    if not best_path.is_file():
        raise FileNotFoundError(
            f"selected stage-1 checkpoint is absent from fresh experiment: {best_path}"
        )

    best = torch.load(best_path, map_location="cpu")
    global_step = int(best.get("global_step", -1))
    epoch = int(best.get("epoch", -1))
    if global_step <= 0 or epoch < 0:
        raise RuntimeError(
            f"invalid selected checkpoint counters: step={global_step}, epoch={epoch}"
        )
    del best
    gc.collect()
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    expected_name = manifest.get("names", {}).get("stage1")
    if expected_name != experiment_dir.name:
        raise RuntimeError(
            "source manifest stage-1 name does not match experiment: "
            f"{expected_name!r} != {experiment_dir.name!r}"
        )
    if manifest.get("method", {}).get("uses_t2mgpt_generator_checkpoint") is not False:
        raise RuntimeError("source manifest does not prove baseline-free initialization")

    return {
        "schema_version": "motiongpt-motionx322-stage1-selection/v1",
        "stage": "vae",
        "split": "val",
        "monitor": "total/val",
        "mode": "min",
        "score": score,
        "global_step": global_step,
        "epoch": epoch,
        "checkpoint": str(best_path.resolve()),
        "checkpoint_sha256": sha256_file(best_path),
        "selected_from": str(last_path.resolve()),
        "selected_from_sha256": sha256_file(last_path),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "uses_t2mgpt_generator_checkpoint": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = select_stage1_best(args.experiment_dir, args.source_manifest)
    output = args.output or (args.experiment_dir / "checkpoints/best-validation.json")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
