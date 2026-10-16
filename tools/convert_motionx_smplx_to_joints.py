#!/usr/bin/env python
"""Convert a deterministic shard of Motion-X++ SMPL-X 322D files to 22 joints."""

import argparse
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from tqdm import tqdm
import smplx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("dataset/Motion-X/raw/bypy/motion/motion_generation/smplx322"))
    parser.add_argument("--output", type=Path, default=Path("dataset/Motion-X/raw/joint"))
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/mnt/workspace/Wilson/ck/umrm/GENMO/inputs/checkpoints/body_models"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--frame-batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def safe_member_path(member):
    parts = [part for part in PurePosixPath(member).parts if part not in {"", ".", ".."}]
    if not parts:
        raise ValueError(f"Invalid archive member {member!r}")
    return Path(*parts).with_suffix(".npy")


def build_model(model_root, device):
    return smplx.create(
        str(model_root), model_type="smplx", gender="neutral", ext="npz",
        use_pca=False, num_betas=10, num_expression_coeffs=10,
        flat_hand_mean=False,
    ).to(device).eval()


@torch.no_grad()
def convert_sequence(model, motion, device, frame_batch_size):
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 322:
        raise ValueError(f"Expected [T, 322], got {motion.shape}")
    outputs = []
    for start in range(0, len(motion), frame_batch_size):
        values = torch.from_numpy(motion[start:start + frame_batch_size]).to(device)
        eye_pose = torch.zeros((len(values), 3), device=device, dtype=values.dtype)
        result = model(
            global_orient=values[:, :3],
            body_pose=values[:, 3:66],
            left_hand_pose=values[:, 66:111],
            right_hand_pose=values[:, 111:156],
            jaw_pose=values[:, 156:159],
            leye_pose=eye_pose,
            reye_pose=eye_pose,
            expression=values[:, 159:169],
            transl=values[:, 309:312],
            betas=values[:, 312:322],
            return_verts=False,
        )
        outputs.append(result.joints[:, :22].detach().cpu().numpy())
    joints = np.concatenate(outputs, axis=0).astype(np.float32)
    if joints.shape != (len(motion), 22, 3) or not np.isfinite(joints).all():
        raise ValueError(f"Invalid converted joints {joints.shape}")
    return joints


def main():
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    archives = sorted(args.input.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(args.input)
    tasks = []
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            members = sorted(name for name in handle.namelist() if name.lower().endswith(".npy"))
        for member in members:
            tasks.append((archive, member))
    tasks = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        tasks = tasks[:args.limit]

    device = torch.device(args.device)
    model = build_model(args.model_root, device)
    args.output.mkdir(parents=True, exist_ok=True)
    converted = skipped = 0
    errors = []
    for archive, member in tqdm(tasks, desc=f"SMPL-X shard {args.shard_index}"):
        category = archive.stem.lower()
        relative = safe_member_path(member)
        parts = list(relative.parts)
        while parts and parts[0].lower() in {category, "smplx322", "motion_generation", "motion"}:
            parts.pop(0)
        relative = Path(*parts) if parts else Path(re.sub(r"\W+", "_", member) + ".npy")
        output_path = args.output / category / relative
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            with zipfile.ZipFile(archive) as handle:
                motion = np.load(io.BytesIO(handle.read(member)), allow_pickle=False)
            joints = convert_sequence(model, motion, device, args.frame_batch_size)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, joints)
            converted += 1
        except Exception as exc:
            errors.append({"archive": str(archive), "member": member, "error": repr(exc)})

    report = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "tasks": len(tasks),
        "converted": converted,
        "skipped": skipped,
        "errors": len(errors),
    }
    report_path = args.output / f"conversion_shard{args.shard_index:02d}.json"
    report_path.write_text(json.dumps({**report, "error_details": errors}, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise RuntimeError(f"SMPL-X conversion failed for {len(errors)} files")


if __name__ == "__main__":
    main()
