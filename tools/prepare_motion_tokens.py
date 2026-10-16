#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models.vqvae as vqvae
from dataset import dataset_tokenize
from dataset.config import get_dataset_spec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode a dataset training split with a pretrained T2M-GPT VQ-VAE."
    )
    parser.add_argument(
        "--dataname", choices=["t2m", "kit", "motionx", "motionx322"],
        required=True,
    )
    parser.add_argument("--resume-pth", required=True)
    parser.add_argument("--vq-name", default="VQVAE")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--nb-code", type=int, default=512)
    parser.add_argument("--code-dim", type=int, default=512)
    parser.add_argument("--output-emb-width", type=int, default=512)
    parser.add_argument("--down-t", type=int, default=2)
    parser.add_argument("--stride-t", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation-growth-rate", type=int, default=3)
    parser.add_argument("--quantizer", default="ema_reset")
    parser.add_argument("--mu", type=float, default=0.99)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = get_dataset_spec(args.dataname).root
    output_dir = data_root / args.vq_name
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.resume_pth, map_location="cpu")
    state = checkpoint.get("net", checkpoint)
    net = vqvae.HumanVQVAE(
        args,
        args.nb_code,
        args.code_dim,
        args.output_emb_width,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
    )
    net.load_state_dict(state, strict=True)
    net.eval().to(device)

    loader = dataset_tokenize.DATALoader(
        args.dataname,
        batch_size=1,
        num_workers=0,
        unit_length=2 ** args.down_t,
        shuffle=False,
    )
    saved = 0
    skipped = 0
    token_min = None
    token_max = None
    lengths = []

    with torch.no_grad():
        for index, (pose, names) in enumerate(loader):
            if index % args.num_shards != args.shard_index:
                continue
            if args.limit is not None and index >= args.limit:
                break
            output_path = output_dir / f"{names[0]}.npy"
            if output_path.exists() and not args.overwrite:
                skipped += 1
                continue

            tokens = net.encode(pose.to(device).float()).cpu().numpy()
            np.save(output_path, tokens)
            saved += 1
            token_min = int(tokens.min()) if token_min is None else min(token_min, int(tokens.min()))
            token_max = int(tokens.max()) if token_max is None else max(token_max, int(tokens.max()))
            lengths.extend(int(row.shape[0]) for row in tokens)

    train_ids = [line.strip() for line in (data_root / "train.txt").read_text().splitlines() if line.strip()]
    min_motion_len = get_dataset_spec(args.dataname).min_motion_len
    eligible_ids = []
    for motion_id in train_ids:
        motion_path = data_root / "new_joint_vecs" / f"{motion_id}.npy"
        if not motion_path.exists():
            continue
        motion_len = int(np.load(motion_path, mmap_mode="r").shape[0])
        spec = get_dataset_spec(args.dataname)
        if motion_len >= min_motion_len and (spec.variable_length or motion_len < 200):
            eligible_ids.append(motion_id)
    available = sum((output_dir / f"{motion_id}.npy").exists() for motion_id in eligible_ids)
    summary = {
        "dataname": args.dataname,
        "checkpoint": str(Path(args.resume_pth).resolve()),
        "output_dir": str(output_dir.resolve()),
        "saved": saved,
        "skipped": skipped,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "train_ids": len(train_ids),
        "eligible_train_ids": len(eligible_ids),
        "available_tokens": available,
        "coverage": available / len(eligible_ids) if eligible_ids else 0.0,
        "token_min": token_min,
        "token_max": token_max,
        "min_length": min(lengths) if lengths else None,
        "max_length": max(lengths) if lengths else None,
    }
    summary_name = (
        "tokenization_summary.json" if args.num_shards == 1
        else f"tokenization_summary_shard{args.shard_index:02d}.json"
    )
    (output_dir / summary_name).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

    if not args.allow_partial and available != len(eligible_ids):
        raise RuntimeError(
            f"Incomplete motion-token coverage: {available}/{len(eligible_ids)}"
        )
    if token_min is not None and (token_min < 0 or token_max >= args.nb_code):
        raise RuntimeError(
            f"Token ids outside codebook range [0, {args.nb_code}): {token_min}..{token_max}"
        )


if __name__ == "__main__":
    main()
