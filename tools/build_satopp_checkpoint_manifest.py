#!/usr/bin/env python
"""Create a reproducible, generation-weight-deduplicated checkpoint manifest."""

import argparse
import gc
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch


PERIODIC_RE = re.compile(r"^sato_pp_iter_(\d+)\.pth$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--best-name", default="sato_pp_best_fid.pth")
    return parser.parse_args()


def sha256_file(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def update_tensor_hash(digest, key, tensor):
    tensor = tensor.detach().cpu().contiguous()
    metadata = json.dumps(
        {"key": key, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    byte_view = tensor.reshape(-1).view(torch.uint8).numpy()
    digest.update(memoryview(byte_view))
    return tensor.numel(), tensor.element_size() * tensor.numel()


def generation_fingerprint(checkpoint):
    digest = hashlib.sha256()
    tensor_count = 0
    parameter_count = 0
    parameter_bytes = 0
    for component in ("trans", "clip"):
        state = checkpoint.get(component)
        if not isinstance(state, dict):
            raise ValueError(f"Checkpoint lacks a {component!r} state dict")
        for key in sorted(state):
            value = state[key]
            if not torch.is_tensor(value):
                raise TypeError(f"Non-tensor value in {component}.{key}: {type(value)}")
            count, size = update_tensor_hash(digest, f"{component}.{key}", value)
            tensor_count += 1
            parameter_count += count
            parameter_bytes += size
    return {
        "sha256": digest.hexdigest(),
        "components": ["trans", "clip"],
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
    }


def finite_or_none(value):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return value if math.isfinite(numeric) else None


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()

    paths = []
    for path in run_dir.glob("sato_pp_iter_*.pth"):
        match = PERIODIC_RE.match(path.name)
        if match:
            paths.append((0, int(match.group(1)), path))
    best_path = run_dir / args.best_name
    if best_path.is_file():
        paths.append((1, 0, best_path))
    if not paths:
        raise FileNotFoundError(f"No periodic/best checkpoints found in {run_dir}")
    paths.sort(key=lambda item: (item[0], item[1], item[2].name))

    aliases = []
    for source_order, filename_iter, path in paths:
        print(f"Hashing and loading {path}", flush=True)
        file_digest = sha256_file(path)
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint_iter = int(checkpoint.get("iter", -1))
        fingerprint = generation_fingerprint(checkpoint)
        checkpoint_args = checkpoint.get("args", {})
        aliases.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "source_kind": "periodic" if source_order == 0 else "best_clean_fid",
                "filename_iter": filename_iter if source_order == 0 else None,
                "checkpoint_iter": checkpoint_iter,
                "file_size_bytes": path.stat().st_size,
                "file_sha256": file_digest,
                "generation_fingerprint": fingerprint,
                "dataname": checkpoint_args.get("dataname"),
                "recorded_best_fid": finite_or_none(checkpoint.get("best_fid")),
                "recorded_best_iter": checkpoint.get("best_iter"),
                "recorded_best_robust_score": finite_or_none(
                    checkpoint.get("best_robust_score")
                ),
            }
        )
        del checkpoint
        gc.collect()

    groups = defaultdict(list)
    for alias in aliases:
        groups[alias["generation_fingerprint"]["sha256"]].append(alias)

    candidates = []
    for fingerprint_sha, group in groups.items():
        ordered = sorted(
            group,
            key=lambda row: (
                row["checkpoint_iter"],
                0 if row["source_kind"] == "periodic" else 1,
                row["path"],
            ),
        )
        canonical = ordered[0]
        candidates.append(
            {
                "candidate_id": f"iter{canonical['checkpoint_iter']}_{fingerprint_sha[:12]}",
                "checkpoint_iter": canonical["checkpoint_iter"],
                "canonical_path": canonical["path"],
                "generation_sha256": fingerprint_sha,
                "generation_fingerprint": canonical["generation_fingerprint"],
                "aliases": [
                    {
                        key: row[key]
                        for key in (
                            "path",
                            "name",
                            "source_kind",
                            "filename_iter",
                            "checkpoint_iter",
                            "file_size_bytes",
                            "file_sha256",
                            "recorded_best_fid",
                            "recorded_best_iter",
                            "recorded_best_robust_score",
                        )
                    }
                    for row in ordered
                ],
            }
        )
    candidates.sort(
        key=lambda row: (row["checkpoint_iter"], row["generation_sha256"])
    )

    payload = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "candidate_definition": (
            "all sato_pp_iter_*.pth periodic checkpoints plus "
            f"{args.best_name}, deduplicated by exact trans+clip tensor content"
        ),
        "evaluation_components": ["trans", "clip"],
        "alias_count": len(aliases),
        "unique_candidate_count": len(candidates),
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()