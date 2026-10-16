import argparse
import json
import os
import random
import re
import sys
from os.path import join as pjoin

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import clip
import imageio
import matplotlib
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from textwrap import wrap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import models.t2m_trans as trans
import models.vqvae as vqvae
from options import option_transformer as option_trans
from utils.motion_process import recover_from_ric

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PROMPTS = [
    "a person walks forward slowly",
    "a person walks backward slowly",
    "a person waves their left arm",
    "a person waves their right arm",
]


def slugify(text, max_len=70):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return text[:max_len] or "prompt"


def iter_pair_records(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            for row in json.load(f):
                yield row
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_prompts(args):
    prompts = []
    if args.prompt:
        prompts.extend(args.prompt)

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts.extend([line.strip() for line in f if line.strip()])

    if args.pair_data_path:
        rows = list(iter_pair_records(args.pair_data_path))
        if args.relation:
            rows = [r for r in rows if r.get("relation") == args.relation]
        if args.edit_type:
            rows = [r for r in rows if r.get("edit_type") == args.edit_type]
        if args.max_pairs > 0:
            random.Random(args.seed).shuffle(rows)
            rows = rows[: args.max_pairs]
        for row in rows:
            orig = row.get("text_orig")
            pair = row.get("text_pair")
            if orig:
                prompts.append(orig)
            if pair:
                prompts.append(pair)

    if not prompts:
        prompts = DEFAULT_PROMPTS

    deduped = []
    seen = set()
    for prompt in prompts:
        if prompt not in seen:
            deduped.append(prompt)
            seen.add(prompt)
    return deduped


def build_model(args, device):
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
    trans_encoder = trans.Text2Motion_Transformer(
        num_vq=args.nb_code,
        embed_dim=args.embed_dim_gpt,
        clip_dim=args.clip_dim,
        block_size=args.block_size,
        num_layers=args.num_layers,
        n_head=args.n_head_gpt,
        drop_out_rate=args.drop_out_rate,
        fc_rate=args.ff_rate,
    )

    print(f"Loading VQVAE checkpoint: {args.resume_pth}")
    ckpt = torch.load(args.resume_pth, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.eval().to(device)

    print(f"Loading transformer checkpoint: {args.resume_trans}")
    ckpt = torch.load(args.resume_trans, map_location="cpu")
    trans_encoder.load_state_dict(ckpt["trans"], strict=True)
    trans_encoder.eval().to(device)
    return net, trans_encoder


@torch.no_grad()
def generate_one(prompt, clip_model, net, trans_encoder, mean, std, args, device):
    text = clip.tokenize([prompt], truncate=True).to(device)
    feat_clip_text = clip_model.encode_text(text).float()
    index_motion = trans_encoder.sample(feat_clip_text, args.sample_categorical)
    pred_pose = net.forward_decoder(index_motion)
    pred_denorm = pred_pose.detach().cpu().numpy() * std + mean
    joints = recover_from_ric(
        torch.from_numpy(pred_denorm).float().to(device),
        21 if args.dataname == "kit" else 22,
    )
    return index_motion.detach().cpu().numpy(), pred_pose.detach().cpu().numpy(), joints[0].cpu().numpy()


def render_motion_frames(joints, title=None):
    """Renderer based on visualization/plot_3d_global.py, with modern 3D axes."""
    data = joints.copy().reshape(len(joints), -1, 3)
    nb_joints = data.shape[1]
    kinematic_chain = (
        [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]
        if nb_joints == 21
        else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    )
    colors = [
        "red", "blue", "black", "red", "blue",
        "darkblue", "darkblue", "darkblue", "darkblue", "darkblue",
        "darkred", "darkred", "darkred", "darkred", "darkred",
    ]
    limits = 1000 if nb_joints == 21 else 2
    mins = data.min(axis=0).min(axis=0)
    maxs = data.max(axis=0).max(axis=0)

    data[:, :, 1] -= mins[1]
    trajec = data[:, 0, [0, 2]].copy()
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    frames = []
    for index in range(data.shape[0]):
        fig = plt.figure(figsize=(480 / 96.0, 320 / 96.0), dpi=96) if nb_joints == 21 else plt.figure(figsize=(6, 6), dpi=96)
        ax = fig.add_subplot(111, projection="3d")
        if title is not None:
            fig.suptitle("\n".join(wrap(title, 40)), fontsize=12)

        ax.set_xlim(-limits, limits)
        ax.set_ylim(-limits, limits)
        ax.set_zlim(0, limits)
        ax.grid(False)
        ax.view_init(elev=110, azim=-90)
        try:
            ax.dist = 7.5
        except Exception:
            pass

        verts = [[
            [mins[0] - trajec[index, 0], 0, mins[2] - trajec[index, 1]],
            [mins[0] - trajec[index, 0], 0, maxs[2] - trajec[index, 1]],
            [maxs[0] - trajec[index, 0], 0, maxs[2] - trajec[index, 1]],
            [maxs[0] - trajec[index, 0], 0, mins[2] - trajec[index, 1]],
        ]]
        xz_plane = Poly3DCollection(verts)
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.35))
        ax.add_collection3d(xz_plane)

        if index > 1:
            ax.plot3D(
                trajec[:index, 0] - trajec[index, 0],
                np.zeros_like(trajec[:index, 0]),
                trajec[:index, 1] - trajec[index, 1],
                linewidth=1.0,
                color="blue",
            )

        for chain_id, (chain, color) in enumerate(zip(kinematic_chain, colors)):
            linewidth = 4.0 if chain_id < 5 else 2.0
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=linewidth,
                color=color,
            )

        plt.axis("off")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(fig)
    return frames


def save_gif(joints, out_path, title):
    frames = render_motion_frames(joints, title)
    imageio.mimsave(out_path, frames, fps=20)


def main():
    base_parser = option_trans.get_parser()
    for action in list(base_parser._actions):
        if action.dest == "help":
            base_parser._remove_action(action)
            for option_string in action.option_strings:
                base_parser._option_string_actions.pop(option_string, None)
    parser = argparse.ArgumentParser(
        parents=[base_parser],
        description="Generate T2M-GPT skeleton GIFs for VLM-as-judge inspection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(
        dataname="t2m",
        resume_pth="pretrained/VQVAE/net_best_fid.pth",
        resume_trans="pretrained/VQTransformer_corruption05/net_best_fid.pth",
        out_dir="ablation/t2mgpt_vlm_visuals",
        vq_name="VQVAE",
        down_t=2,
        block_size=51,
        embed_dim_gpt=1024,
        num_layers=9,
        n_head_gpt=16,
        pkeep=0.5,
    )
    parser.add_argument("--prompt", action="append", default=[], help="Prompt to visualize; can be repeated.")
    parser.add_argument("--prompt-file", type=str, default=None, help="One prompt per line.")
    parser.add_argument("--relation", choices=["preserve", "alter"], default=None)
    parser.add_argument("--edit-type", type=str, default=None)
    parser.add_argument("--max-pairs", type=int, default=0, help="Max pair rows to sample from pair-data-path; 0 means all.")
    parser.add_argument("--sample-categorical", action="store_true", help="Use categorical token sampling instead of argmax.")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--meta-dir", type=str, default="pretrained/meta")
    cli_args = parser.parse_args()

    random.seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    device = torch.device(f"cuda:{cli_args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(cli_args.gpu_id)

    os.makedirs(cli_args.out_dir, exist_ok=True)
    mean = np.load(pjoin(cli_args.meta_dir, "mean.npy"))
    std = np.load(pjoin(cli_args.meta_dir, "std.npy"))

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip.model.convert_weights(clip_model)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    net, trans_encoder = build_model(cli_args, device)
    prompts = load_prompts(cli_args)
    manifest_path = pjoin(cli_args.out_dir, "manifest.jsonl")

    with open(manifest_path, "w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts):
            stem = f"{idx:03d}_{slugify(prompt)}"
            gif_path = pjoin(cli_args.out_dir, stem + ".gif")
            joints_path = pjoin(cli_args.out_dir, stem + "_joints.npy")
            codes_path = pjoin(cli_args.out_dir, stem + "_codes.npy")
            motion_path = pjoin(cli_args.out_dir, stem + "_motion.npy")

            codes, motion, joints = generate_one(
                prompt, clip_model, net, trans_encoder, mean, std, cli_args, device
            )
            np.save(joints_path, joints.astype(np.float32))
            np.save(codes_path, codes)
            np.save(motion_path, motion.astype(np.float32))
            save_gif(joints, gif_path, prompt)

            row = {
                "index": idx,
                "prompt": prompt,
                "gif": gif_path,
                "joints_npy": joints_path,
                "codes_npy": codes_path,
                "motion_npy": motion_path,
                "num_frames": int(joints.shape[0]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{idx + 1}/{len(prompts)}] {gif_path}")

    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
