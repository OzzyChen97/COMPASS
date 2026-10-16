#!/usr/bin/env python
"""Write a shared-evaluator spec for an original SATO checkpoint."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", default="motionx322_75")
    parser.add_argument("--name", default="motionx322_sato_official_50k_best_fid")
    parser.add_argument(
        "--description",
        default="Original SATO reproduced on Motion-X with a 50k budget",
    )
    parser.add_argument("--vq-checkpoint", required=True)
    parser.add_argument("--transformer-checkpoint", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--block-size", type=int, default=76)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # The fast evaluator's baseline layout is also the correct representation
    # for original SATO: VQ, transformer, and adapted CLIP are separate files.
    spec = [{
        "name": args.name,
        "kind": "baseline",
        "dataname": args.dataname,
        "resume_pth": args.vq_checkpoint,
        "resume_trans": args.transformer_checkpoint,
        "clip_pth": args.clip_checkpoint,
        "args": {"down_t": 2, "block_size": args.block_size},
        "description": args.description,
    }]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
