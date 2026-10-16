import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import clip
import imageio
import matplotlib
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from textwrap import wrap

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models.t2m_trans as trans
import models.vqvae as vqvae
from options import option_transformer as option_trans
from utils.motion_process import recover_from_ric

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_SPECS = {
    "t2mgpt": {
        "kind": "trans",
        "trans": "pretrained/VQTransformer_corruption05/net_best_fid.pth",
    },
    "t2mgpt_aug": {
        "kind": "trans",
        "trans": "output_GPT_Final/baseline_aug_syn/net_best_fid.pth",
    },
    "sato_pp": {
        "kind": "sato_pp",
        "ckpt": "output_GPT_Final/sato_pp_allloss_gpu0_teachl1_05_eval5k_resume104k_200k_single/sato_pp_iter_130000.pth",
    },
}


def slugify(text, max_len=80):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return text[:max_len] or "item"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_base_args():
    parser = option_trans.get_parser()
    args = parser.parse_args([])
    args.dataname = "t2m"
    args.resume_pth = "pretrained/VQVAE/net_best_fid.pth"
    args.vq_name = "VQVAE"
    args.down_t = 2
    args.block_size = 51
    args.embed_dim_gpt = 1024
    args.num_layers = 9
    args.n_head_gpt = 16
    args.pkeep = 0.5
    return args


def load_vq(args, device):
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
    ckpt = torch.load(args.resume_pth, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.eval().to(device)
    return net


def build_trans(args, device):
    model = trans.Text2Motion_Transformer(
        num_vq=args.nb_code,
        embed_dim=args.embed_dim_gpt,
        clip_dim=args.clip_dim,
        block_size=args.block_size,
        num_layers=args.num_layers,
        n_head=args.n_head_gpt,
        drop_out_rate=args.drop_out_rate,
        fc_rate=args.ff_rate,
    )
    model.eval().to(device)
    return model


def load_model_spec(name, args, device):
    spec = MODEL_SPECS[name]
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip.model.convert_weights(clip_model)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    trans_encoder = build_trans(args, device)
    if spec["kind"] == "trans":
        ckpt = torch.load(spec["trans"], map_location="cpu")
        trans_encoder.load_state_dict(ckpt["trans"], strict=True)
    else:
        ckpt = torch.load(spec["ckpt"], map_location="cpu")
        trans_encoder.load_state_dict(ckpt["trans"], strict=True)
        if "clip" in ckpt:
            clip_model.load_state_dict(ckpt["clip"], strict=False)
    trans_encoder.eval()
    return {"clip": clip_model, "trans": trans_encoder}


@torch.no_grad()
def generate(prompt, model_pack, net, mean, std, args, device, sample_categorical):
    text = clip.tokenize([prompt], truncate=True).to(device)
    feat_clip_text = model_pack["clip"].encode_text(text).float()
    codes = model_pack["trans"].sample(feat_clip_text, sample_categorical)
    pred_pose = net.forward_decoder(codes)
    pred_denorm = pred_pose.detach().cpu().numpy() * std + mean
    joints = recover_from_ric(torch.from_numpy(pred_denorm).float().to(device), 22)
    return codes.detach().cpu().numpy(), pred_pose.detach().cpu().numpy(), joints[0].cpu().numpy()


def render_motion_frames(joints, show_title=False, title=None):
    data = joints.copy().reshape(len(joints), -1, 3)
    kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    colors = ["red", "blue", "black", "red", "blue", "darkblue", "darkblue", "darkblue", "darkblue", "darkblue"]
    limits = 2
    mins = data.min(axis=0).min(axis=0)
    maxs = data.max(axis=0).max(axis=0)
    data[:, :, 1] -= mins[1]
    trajec = data[:, 0, [0, 2]].copy()
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    frames = []
    for index in range(data.shape[0]):
        fig = plt.figure(figsize=(6, 6), dpi=96)
        ax = fig.add_subplot(111, projection="3d")
        if show_title and title:
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
        plane = Poly3DCollection(verts)
        plane.set_facecolor((0.5, 0.5, 0.5, 0.35))
        ax.add_collection3d(plane)
        if index > 1:
            ax.plot3D(
                trajec[:index, 0] - trajec[index, 0],
                np.zeros_like(trajec[:index, 0]),
                trajec[:index, 1] - trajec[index, 1],
                linewidth=1.0,
                color="blue",
            )
        for chain_id, (chain, color) in enumerate(zip(kinematic_chain, colors)):
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=4.0 if chain_id < 5 else 2.0,
                color=color,
            )
        plt.axis("off")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    return frames


def save_outputs(out_dir, stem, prompt, joints, motion, codes, show_title):
    gif_path = out_dir / f"{stem}.gif"
    joints_path = out_dir / f"{stem}_joints.npy"
    motion_path = out_dir / f"{stem}_motion.npy"
    codes_path = out_dir / f"{stem}_codes.npy"
    np.save(joints_path, joints.astype(np.float32))
    np.save(motion_path, motion.astype(np.float32))
    np.save(codes_path, codes)
    imageio.mimsave(gif_path, render_motion_frames(joints, show_title=show_title, title=prompt), fps=20)
    return gif_path, joints_path, motion_path, codes_path


def load_completed(manifest_path):
    completed = set()
    if not manifest_path.exists():
        return completed
    for row in read_jsonl(manifest_path):
        required_paths = [row.get("gif"), row.get("joints_npy"), row.get("motion_npy"), row.get("codes_npy")]
        if not all(path and Path(path).exists() for path in required_paths):
            continue
        completed.add((row.get("model"), row.get("edit_type"), row.get("pair_uid"), row.get("side")))
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", type=Path, default=Path("ablation/vlm_judge/pairs_20_each.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("ablation/vlm_judge/visuals_20_each"))
    parser.add_argument("--models", nargs="+", default=["t2mgpt", "t2mgpt_aug", "sato_pp"], choices=sorted(MODEL_SPECS))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--meta-dir", type=Path, default=Path("pretrained/meta"))
    parser.add_argument("--sample-categorical", action="store_true")
    parser.add_argument("--show-title", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu_id)
    base_args = build_base_args()
    net = load_vq(base_args, device)
    mean = np.load(args.meta_dir / "mean.npy")
    std = np.load(args.meta_dir / "std.npy")
    pairs = list(read_jsonl(args.pair_manifest))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / "manifest.jsonl"
    completed = load_completed(manifest_path) if args.resume else set()
    manifest_mode = "a" if args.resume and manifest_path.exists() else "w"
    with open(manifest_path, manifest_mode, encoding="utf-8") as f:
        for model_name in args.models:
            model_pack = load_model_spec(model_name, base_args, device)
            for row in pairs:
                for side, prompt_key in [("orig", "text_orig"), ("edit", "text_pair")]:
                    done_key = (model_name, row["edit_type"], row["pair_uid"], side)
                    if done_key in completed:
                        print(f"skip {model_name} {row['edit_type']} {row['pair_uid']} {side}", flush=True)
                        continue
                    prompt = row[prompt_key]
                    stem = f"{model_name}/{row['edit_type']}/{row['pair_uid']}_{side}_{slugify(prompt)}"
                    out_subdir = args.out_dir / model_name / row["edit_type"]
                    out_subdir.mkdir(parents=True, exist_ok=True)
                    file_stem = f"{row['pair_uid']}_{side}_{slugify(prompt)}"
                    codes, motion, joints = generate(
                        prompt, model_pack, net, mean, std, base_args, device, args.sample_categorical
                    )
                    gif, joints_npy, motion_npy, codes_npy = save_outputs(
                        out_subdir, file_stem, prompt, joints, motion, codes, args.show_title
                    )
                    out = dict(row)
                    out.update(
                        {
                            "model": model_name,
                            "side": side,
                            "prompt": prompt,
                            "gif": str(gif),
                            "joints_npy": str(joints_npy),
                            "motion_npy": str(motion_npy),
                            "codes_npy": str(codes_npy),
                            "num_frames": int(joints.shape[0]),
                        }
                    )
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"{model_name} {row['edit_type']} {row['pair_uid']} {side}: {gif}", flush=True)
            del model_pack
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(manifest_path)


if __name__ == "__main__":
    main()
