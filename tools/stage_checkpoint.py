#!/usr/bin/env python
"""Record and verify immutable best-checkpoint stage dependencies."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_upstreams(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {"path": str(path.resolve()), "sha256": sha256(path)}
    return result


def dataset_hashes(root):
    hashes = {}
    for name in (
        "manifest.jsonl",
        "dataset_meta.json",
        "Mean.npy",
        "Std.npy",
        "train.txt",
        "val.txt",
        "test.txt",
    ):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = sha256(path)
    return hashes


def record(args):
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = {
        "stage": args.stage,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_hashes": dataset_hashes(args.dataset_root),
        "upstream": parse_upstreams(args.upstream),
    }
    args.contract.parent.mkdir(parents=True, exist_ok=True)
    args.contract.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def verify(args):
    payload = json.loads(args.contract.read_text(encoding="utf-8"))
    checkpoint = Path(payload["checkpoint"])
    failures = []
    if not checkpoint.is_file() or sha256(checkpoint) != payload["checkpoint_sha256"]:
        failures.append("checkpoint")
    root = Path(payload["dataset_root"])
    for name, expected in payload["dataset_hashes"].items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"dataset:{name}")
    for name, item in payload.get("upstream", {}).items():
        path = Path(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            failures.append(f"upstream:{name}")
    if failures:
        raise RuntimeError("Checkpoint contract mismatch: " + ", ".join(failures))
    print(json.dumps({"verified": True, "stage": payload["stage"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--stage", required=True)
    record_parser.add_argument("--checkpoint", type=Path, required=True)
    record_parser.add_argument("--dataset-root", type=Path, required=True)
    record_parser.add_argument("--contract", type=Path, required=True)
    record_parser.add_argument("--upstream", action="append", default=[])
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    record(args) if args.command == "record" else verify(args)


if __name__ == "__main__":
    main()
