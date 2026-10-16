#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.config import dataset_max_token_length


def main():
    parser = argparse.ArgumentParser(description="Write a baseline/full SATO++ eval spec.")
    parser.add_argument(
        "--dataname",
        choices=["t2m", "kit", "motionx", "motionx322", "motionx322_75"],
        required=True,
    )
    parser.add_argument("--vq-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--sato-checkpoint", required=True)
    parser.add_argument("--baseline-name", default=None)
    parser.add_argument("--sato-name", default=None)
    parser.add_argument("--down-t", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    block_size = args.block_size or dataset_max_token_length(
        args.dataname, 2 ** args.down_t)

    specs = [
        {
            "name": args.baseline_name or f"{args.dataname}_t2mgpt_baseline",
            "kind": "baseline",
            "dataname": args.dataname,
            "resume_pth": args.vq_checkpoint,
            "resume_trans": args.baseline_checkpoint,
            "args": {
                "down_t": args.down_t,
                "block_size": block_size,
            },
            "description": f"T2M-GPT baseline on {args.dataname}",
        },
        {
            "name": args.sato_name or f"{args.dataname}_t2mgpt_sato_pp",
            "kind": "sato_pp",
            "checkpoint": args.sato_checkpoint,
            "description": f"SATO++ checkpoint on {args.dataname}",
        },
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(specs, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
