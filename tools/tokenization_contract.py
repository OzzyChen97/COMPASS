#!/usr/bin/env python
"""Record and verify that motion tokens came from the selected VQ checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

try:
    from tools.stage_checkpoint import dataset_hashes, sha256
except ModuleNotFoundError:
    from stage_checkpoint import dataset_hashes, sha256


def token_digest(dataset_root, vq_name):
    digest = hashlib.sha256()
    token_dir = dataset_root / vq_name
    motion_ids = [
        line.strip()
        for line in (dataset_root / "train.txt").read_text().splitlines()
        if line.strip()
    ]
    for motion_id in motion_ids:
        path = token_dir / f"{motion_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(motion_id.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return len(motion_ids), digest.hexdigest()


def build_payload(args):
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    count, digest = token_digest(args.dataset_root, args.vq_name)
    return {
        "stage": "tokenize",
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_hashes": dataset_hashes(args.dataset_root),
        "vq_name": args.vq_name,
        "vq_checkpoint": str(checkpoint),
        "vq_checkpoint_sha256": sha256(checkpoint),
        "token_count": count,
        "token_digest": digest,
    }


def record(args):
    payload = build_payload(args)
    args.contract.parent.mkdir(parents=True, exist_ok=True)
    args.contract.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def verify(args):
    expected = json.loads(args.contract.read_text(encoding="utf-8"))
    current = build_payload(args)
    failures = [
        key for key in (
            "dataset_root",
            "dataset_hashes",
            "vq_name",
            "vq_checkpoint",
            "vq_checkpoint_sha256",
            "token_count",
            "token_digest",
        )
        if current[key] != expected.get(key)
    ]
    if failures:
        raise RuntimeError(
            "Tokenization contract mismatch: " + ", ".join(failures))
    print(json.dumps({"verified": True, "stage": "tokenize"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("record", "verify"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vq-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    record(args) if args.command == "record" else verify(args)


if __name__ == "__main__":
    main()
