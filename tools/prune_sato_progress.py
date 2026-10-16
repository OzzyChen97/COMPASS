#!/usr/bin/env python
"""Prune a SATO pair progress.jsonl so it only retains records whose
(motion_id, text_orig) still matches a current, valid source caption.

Records tied to captions that were sanitized (old polluted text) or to motions
dropped from the split are removed, so a subsequent build_sato_pairs run
regenerates exactly those motions and finalize emits only clean pairs.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_sato_pairs", str(Path(__file__).with_name("build_sato_pairs.py")))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "val"], required=True)
    parser.add_argument("--protocol", choices=["train", "test"], required=True)
    parser.add_argument("--progress", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    bsp = load_builder()
    rows = bsp.read_motion_captions(
        args.dataset_root, args.split,
        one_per_motion=args.protocol == "test")
    valid = {(row["motion_id"], row["text_orig"]) for row in rows}

    latest = bsp.progress_records(args.progress)
    kept = dropped = 0
    tmp = args.progress.with_suffix(args.progress.suffix + ".pruned")
    with tmp.open("w", encoding="utf-8") as handle:
        for key, record in latest.items():
            if key not in valid:
                dropped += 1
                continue
            preserve = [dict(record["preserve"], index=1)] if record["preserve"] else []
            alter = [dict(record["alter"], index=1)] if record["alter"] else []
            handle.write(json.dumps(
                {"items": [record["item"]], "preserve": preserve, "alter": alter},
                ensure_ascii=True) + "\n")
            kept += 1

    backup = args.progress.with_suffix(args.progress.suffix + ".prebackup")
    if args.progress.exists():
        args.progress.replace(backup)
    tmp.replace(args.progress)
    print(json.dumps({
        "split": args.split, "protocol": args.protocol,
        "valid_keys": len(valid), "kept": kept, "dropped": dropped,
        "backup": str(backup),
    }, indent=2))


if __name__ == "__main__":
    main()
