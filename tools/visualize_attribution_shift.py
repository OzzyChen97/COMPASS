import argparse
import csv
import json
import math
import os
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import clip
import models.t2m_trans as trans
from clip_local.simple_tokenizer import SimpleTokenizer
from models.sato_pp_modules import AttributionPooling


TOKENIZER = SimpleTokenizer()


MODEL_NAMES = {
    "sato_pp": "SATO++",
    "t2mgpt": "T2M-GPT",
}


def read_json_records(path):
    path = Path(path)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            yield from json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def clean_token(token_id):
    text = TOKENIZER.decode([int(token_id)])
    text = text.replace("</w>", "").replace("\n", "").strip()
    return text if text else " "


def visible_token_indices(token_ids):
    ids = token_ids.detach().cpu().tolist()
    nonzero = [i for i, tid in enumerate(ids) if tid != 0]
    if len(nonzero) <= 2:
        return []
    return nonzero[1:-1]


def token_labels(token_ids):
    indices = visible_token_indices(token_ids)
    return [clean_token(token_ids[i]) for i in indices], indices


def pad_rows(rows, fill=np.nan):
    width = max(len(r) for r in rows)
    out = np.full((len(rows), width), fill, dtype=float)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def safe_cos_dist(a, b):
    a = a.detach().float()
    b = b.detach().float()
    return float(1.0 - F.cosine_similarity(a[None], b[None], dim=-1).item())


def normalize_visible(values):
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 0)
    total = values.sum()
    if total <= 1e-12:
        return np.ones_like(values) / max(len(values), 1)
    return values / total


def js_divergence(p, q):
    p = normalize_visible(p)
    q = normalize_visible(q)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def kl_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return float(np.sum(p[mask] * np.log((p[mask] + 1e-12) / (q[mask] + 1e-12))))


def slot_for_token(token, mode):
    t = token.lower().strip(" .,;:!?")
    if mode == "preserve":
        if t in {"walk", "walks", "walking", "walked", "move", "moves", "moving", "stroll", "strolled"}:
            return "action"
        if t in {"forward", "forwards", "ahead", "front"}:
            return "direction"
        if t in {"slow", "slowly", "pace"}:
            return "speed"
        return "other"
    if mode == "alter":
        if t in {"turn", "turns", "turning"}:
            return "action"
        if t in {"left", "right", "forward", "backward", "forwards", "backwards"}:
            return "edited"
        return "other"
    if t in {"left", "right", "forward", "backward", "forwards", "backwards"}:
        return "edited"
    return "other"


def slot_distribution(labels, values, mode):
    slots = ["action", "direction", "speed", "edited", "other"]
    dist = {slot: 0.0 for slot in slots}
    for label, value in zip(labels, values):
        dist[slot_for_token(label, mode)] += float(value)
    active = ["action", "direction", "speed", "other"] if mode == "preserve" else ["action", "edited", "other"]
    total = sum(dist[s] for s in active)
    if total <= 1e-12:
        return {s: 1.0 / len(active) for s in active}
    return {s: dist[s] / total for s in active}


def aligned_diff_fraction(labels_a, values_a, labels_b, values_b, edited_words):
    width = max(len(values_a), len(values_b))
    va = np.zeros(width)
    vb = np.zeros(width)
    va[: len(values_a)] = values_a
    vb[: len(values_b)] = values_b
    diff = np.abs(va - vb)
    edited = np.zeros(width, dtype=bool)
    for idx in range(width):
        la = labels_a[idx].lower() if idx < len(labels_a) else ""
        lb = labels_b[idx].lower() if idx < len(labels_b) else ""
        if la in edited_words or lb in edited_words:
            edited[idx] = True
    denom = diff.sum()
    return float(diff[edited].sum() / denom) if denom > 1e-12 else 0.0


def build_trans_encoder(args, device):
    model = trans.Text2Motion_Transformer(
        num_vq=args["nb_code"],
        embed_dim=args["embed_dim_gpt"],
        clip_dim=args["clip_dim"],
        block_size=args["block_size"],
        num_layers=args["num_layers"],
        n_head=args["n_head_gpt"],
        drop_out_rate=args["drop_out_rate"],
        fc_rate=args["ff_rate"],
    )
    return model.to(device).eval()


def load_models(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.float()
    clip_model.load_state_dict(ckpt["clip"], strict=False)
    clip_model.eval()

    trans_encoder = build_trans_encoder(args, device)
    trans_encoder.load_state_dict(ckpt["trans"], strict=True)
    trans_encoder.eval()

    attr_pool = AttributionPooling(
        clip_dim=args.get("clip_dim_attr", 512),
        proj_dim=args.get("proj_dim", 512),
        proj_type=args.get("proj_type", "linear"),
    ).to(device)
    if "attr_pooling" in ckpt:
        attr_pool.load_state_dict(ckpt["attr_pooling"], strict=True)
    attr_pool.eval()

    for module in [clip_model, trans_encoder, attr_pool]:
        for p in module.parameters():
            p.requires_grad = False
    return clip_model, trans_encoder, attr_pool, args


def motion_tokens_from_vq(data_root, motion_id, nb_code, block_size):
    path = Path(data_root) / "VQVAE" / f"{motion_id}.npy"
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[0]
    arr = arr.astype(np.int64)
    if len(arr) > block_size - 1:
        arr = arr[: block_size - 1]
    valid_len = min(len(arr) + 1, block_size)
    tokens = np.concatenate([arr, np.array([nb_code], dtype=np.int64)])
    if len(tokens) < block_size:
        pad = np.ones(block_size - len(tokens), dtype=np.int64) * (nb_code + 1)
        tokens = np.concatenate([tokens, pad])
    return tokens, valid_len


def motion_tokens_from_codes(path, nb_code, block_size):
    arr = np.load(path).reshape(-1).astype(np.int64)
    if len(arr) > block_size - 1:
        arr = arr[: block_size - 1]
    valid_len = min(len(arr) + 1, block_size)
    tokens = np.concatenate([arr, np.array([nb_code], dtype=np.int64)])
    if len(tokens) < block_size:
        tokens = np.concatenate([tokens, np.ones(block_size - len(tokens), dtype=np.int64) * (nb_code + 1)])
    return tokens, valid_len


def compute_grad_input_batched(clip_model, trans_encoder, text_tokens, target_tokens, valid_len, k):
    dtype = clip_model.dtype
    h = clip_model.token_embedding(text_tokens).type(dtype)
    h = h.detach().requires_grad_(True)

    x = h + clip_model.positional_embedding.type(dtype)
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x).type(dtype)

    eos_idx = text_tokens.argmax(dim=-1)
    pooled = x[torch.arange(x.shape[0], device=x.device), eos_idx]
    if clip_model.text_projection is not None:
        pooled = pooled @ clip_model.text_projection

    input_index = target_tokens[:, :-1]
    cls_pred = trans_encoder(input_index, pooled.float())
    max_steps = min(k, input_index.shape[1])
    log_probs = F.log_softmax(cls_pred[:, :max_steps, :], dim=-1)
    target_slice = target_tokens[:, :max_steps].clamp(min=0, max=cls_pred.shape[-1] - 1)
    token_log_probs = log_probs.gather(2, target_slice.unsqueeze(-1)).squeeze(-1)

    valid_steps = valid_len.to(token_log_probs.device).long().clamp(min=1, max=max_steps)
    step_ids = torch.arange(max_steps, device=token_log_probs.device)
    valid_mask = step_ids.unsqueeze(0) < valid_steps.unsqueeze(1)
    score = (token_log_probs * valid_mask.float()).sum(dim=1) / valid_steps.float()

    grad_h, = torch.autograd.grad(score.sum(), h, retain_graph=False, create_graph=False)
    return (grad_h.float() * h.detach().float()).abs().sum(dim=-1), pooled.float(), x.float()


def compute_pair(clip_model, trans_encoder, attr_pool, args, text_a, text_b, target_tokens, valid_len, mode, device):
    text_tokens = clip.tokenize([text_a, text_b], truncate=True).to(device)
    target = torch.tensor(target_tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
    valid = torch.tensor([valid_len, valid_len], dtype=torch.long, device=device)
    raw_attr, pooled, token_features = compute_grad_input_batched(
        clip_model, trans_encoder, text_tokens, target, valid, args.get("attribution_K", 16)
    )
    alpha = F.softmax(raw_attr / float(args.get("tau", 1.0)), dim=-1)
    with torch.no_grad():
        u = attr_pool(token_features, alpha)
        raw_h = clip_model.token_embedding(text_tokens).float()

    labels_a, idx_a = token_labels(text_tokens[0])
    labels_b, idx_b = token_labels(text_tokens[1])
    va = normalize_visible(raw_attr[0, idx_a].detach().cpu().numpy())
    vb = normalize_visible(raw_attr[1, idx_b].detach().cpu().numpy())
    sa = slot_distribution(labels_a, va, mode)
    sb = slot_distribution(labels_b, vb, mode)
    active_slots = list(sa.keys())
    slot_jsd = js_divergence([sa[s] for s in active_slots], [sb[s] for s in active_slots])

    def mean_visible_embedding(i, indices):
        if not indices:
            return raw_h[i].mean(dim=0)
        return raw_h[i, indices].mean(dim=0)

    return {
        "text_a": text_a,
        "text_b": text_b,
        "labels_a": labels_a,
        "labels_b": labels_b,
        "attr_a": va.tolist(),
        "attr_b": vb.tolist(),
        "slot_a": sa,
        "slot_b": sb,
        "slot_jsd": slot_jsd,
        "raw_token_emb_dist": safe_cos_dist(mean_visible_embedding(0, idx_a), mean_visible_embedding(1, idx_b)),
        "clip_pooled_dist": safe_cos_dist(pooled[0], pooled[1]),
        "attr_pooled_dist": safe_cos_dist(u[0], u[1]),
        "alpha_l1_77": float(torch.abs(alpha[0] - alpha[1]).sum().item()),
    }


def sample_rows(path, relation, limit, data_root, seed, edit_type=None):
    rng = random.Random(seed)
    rows = []
    for row in read_json_records(path):
        if row.get("relation") != relation:
            continue
        if edit_type and row.get("edit_type") != edit_type:
            continue
        mid = str(row.get("motion_id", ""))
        if not (Path(data_root) / "VQVAE" / f"{mid}.npy").exists():
            continue
        if not row.get("text_orig") or not row.get("text_pair"):
            continue
        rows.append(row)
    rng.shuffle(rows)
    return rows[:limit]


def compute_stats(clip_model, trans_encoder, attr_pool, args, data_root, preserve_path, alter_path, out_json, limit, device):
    out_json = Path(out_json)
    if out_json.exists():
        return json.loads(out_json.read_text(encoding="utf-8"))
    rows = []
    preserve = sample_rows(preserve_path, "preserve", limit, data_root, 31)
    alter = sample_rows(alter_path, "alter", limit, data_root, 37, edit_type="direction")
    for relation, selected in [("preserve", preserve), ("alter", alter)]:
        mode = "preserve" if relation == "preserve" else "alter"
        for i, row in enumerate(selected):
            target, valid_len = motion_tokens_from_vq(data_root, str(row["motion_id"]), args["nb_code"], args["block_size"])
            metrics = compute_pair(
                clip_model,
                trans_encoder,
                attr_pool,
                args,
                row["text_orig"],
                row["text_pair"],
                target,
                valid_len,
                mode,
                device,
            )
            rows.append(
                {
                    "relation": relation,
                    "motion_id": str(row["motion_id"]),
                    "text_orig": row["text_orig"],
                    "text_pair": row["text_pair"],
                    "raw_token_emb_dist": metrics["raw_token_emb_dist"],
                    "clip_pooled_dist": metrics["clip_pooled_dist"],
                    "attr_pooled_dist": metrics["attr_pooled_dist"],
                    "slot_jsd": metrics["slot_jsd"],
                    "alpha_l1_77": metrics["alpha_l1_77"],
                }
            )
            if (i + 1) % 10 == 0:
                print(f"stats {relation}: {i + 1}/{len(selected)}", flush=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def draw_token_heatmap(ax, labels_a, attr_a, labels_b, attr_b, title, mode, edited_words=None):
    edited_words = edited_words or set()
    labels = [labels_a, labels_b]
    values = [np.asarray(attr_a), np.asarray(attr_b)]
    mat = pad_rows(values)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#f7f7f7")
    ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=max(np.nanmax(mat), 1e-4))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["prompt A", "prompt B"], fontsize=7)
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=4)
    ax.tick_params(length=0)
    max_len = mat.shape[1]
    ax.set_xticks(range(max_len))
    xticklabels = []
    for i in range(max_len):
        top = labels_a[i] if i < len(labels_a) else ""
        bot = labels_b[i] if i < len(labels_b) else ""
        xticklabels.append(top if top == bot else f"{top}\n{bot}".strip())
    ax.set_xticklabels(xticklabels, fontsize=6, rotation=0)
    for i in range(max_len + 1):
        ax.axvline(i - 0.5, color="white", lw=0.6)
    ax.axhline(0.5, color="white", lw=1.0)
    for i in range(max_len):
        la = labels_a[i].lower() if i < len(labels_a) else ""
        lb = labels_b[i].lower() if i < len(labels_b) else ""
        slot = slot_for_token(la or lb, mode)
        if la in edited_words or lb in edited_words or slot == "edited":
            ax.add_patch(Rectangle((i - 0.5, -0.5), 1, 2, fill=False, edgecolor="#2c7fb8", lw=1.2))
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_slot_bars(ax, case, title):
    slots = list(case["slot_a"].keys())
    a = np.array([case["slot_a"][s] for s in slots])
    b = np.array([case["slot_b"][s] for s in slots])
    x = np.arange(len(slots))
    ax.bar(x - 0.18, a, width=0.35, color="#7aa6c2", label="prompt A")
    ax.bar(x + 0.18, b, width=0.35, color="#d9915d", label="prompt B")
    ax.set_ylim(0, max(0.55, float(max(a.max(), b.max()) * 1.2)))
    ax.set_xticks(x)
    ax.set_xticklabels(slots, rotation=25, ha="right", fontsize=6)
    ax.set_ylabel("slot attribution", fontsize=7)
    ax.set_title(title, loc="left", fontsize=7.5, pad=4)
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(fontsize=6, loc="upper right")


def draw_distance_scatter(ax, stats):
    colors = {"preserve": "#4677a3", "alter": "#c66a3d"}
    all_x = []
    all_y = []
    for relation in ["preserve", "alter"]:
        sub = [r for r in stats if r["relation"] == relation]
        xs = [r["raw_token_emb_dist"] for r in sub]
        ys = [r["slot_jsd"] for r in sub]
        all_x.extend(xs)
        all_y.extend(ys)
        ax.scatter(
            xs,
            ys,
            s=16,
            alpha=0.75,
            color=colors[relation],
            label=relation,
            edgecolor="white",
            linewidth=0.3,
        )
    if all_x and all_y:
        x_pad = max(0.03, (max(all_x) - min(all_x)) * 0.06)
        y_pad = max(0.02, (max(all_y) - min(all_y)) * 0.08)
        ax.set_xlim(max(0, min(all_x) - x_pad), max(all_x) + x_pad)
        ax.set_ylim(0, max(max(all_y) + y_pad, 0.12))
    ax.set_xlabel("raw CLIP token-embedding distance", fontsize=7)
    ax.set_ylabel("slot-attribution JSD", fontsize=7)
    ax.set_title("D Many pairs: embedding shift vs attribution shift", loc="left", fontsize=8, fontweight="bold", pad=4)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc="upper left")

    summaries = []
    for relation in ["preserve", "alter"]:
        sub = [r for r in stats if r["relation"] == relation]
        if not sub:
            continue
        raw_mean = np.mean([r["raw_token_emb_dist"] for r in sub])
        jsd_mean = np.mean([r["slot_jsd"] for r in sub])
        summaries.append(f"{relation}: raw {raw_mean:.2f}, attr-JSD {jsd_mean:.3f}")
    if summaries:
        ax.text(
            0.98,
            0.03,
            "\n".join(summaries),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            bbox={"facecolor": "white", "edgecolor": "#d9d9d9", "linewidth": 0.5, "pad": 2.0},
        )


def crop_nonwhite(frame, pad=14, threshold=245):
    rgb = np.asarray(frame)[..., :3]
    mask = np.any(rgb < threshold, axis=-1)
    if not mask.any():
        return rgb
    ys, xs = np.where(mask)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(rgb.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(rgb.shape[1], int(xs.max()) + pad + 1)
    crop = rgb[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    side = max(h, w, 120)
    canvas = np.ones((side, side, 3), dtype=np.uint8) * 255
    yy = (side - h) // 2
    xx = (side - w) // 2
    canvas[yy : yy + h, xx : xx + w] = crop
    return canvas


def draw_failure(ax_traj, ax_strip, failure):
    j0 = np.load(failure["orig_joints"])
    j1 = np.load(failure["edit_joints"])
    r0 = j0[:, 0, [0, 2]]
    r1 = j1[:, 0, [0, 2]]
    r0 = r0 - r0[0]
    r1 = r1 - r1[0]
    ax_traj.plot(r0[:, 0], r0[:, 1], color="#4677a3", lw=1.5, label="forward prompt")
    ax_traj.plot(r1[:, 0], r1[:, 1], color="#c66a3d", lw=1.5, label="backward prompt")
    ax_traj.scatter([r0[0, 0], r1[0, 0]], [r0[0, 1], r1[0, 1]], s=12, c="black", marker="o")
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.set_title("E Motion does not follow the direction edit", loc="left", fontsize=7.5, fontweight="bold", pad=4)
    ax_traj.tick_params(labelsize=6)
    ax_traj.legend(fontsize=6, loc="best")
    disp_cos = float(np.dot(r0[-1], r1[-1]) / (np.linalg.norm(r0[-1]) * np.linalg.norm(r1[-1]) + 1e-8))
    ax_traj.text(
        0.02,
        0.02,
        f"root displacement cosine = {disp_cos:.2f}",
        transform=ax_traj.transAxes,
        fontsize=6,
        bbox={"facecolor": "white", "edgecolor": "#d9d9d9", "linewidth": 0.4, "pad": 1.5},
    )

    frames = []
    for gif in [failure["orig_gif"], failure["edit_gif"]]:
        all_frames = imageio.mimread(gif)
        idx = np.linspace(0, len(all_frames) - 1, 3, dtype=int)
        frames.extend([crop_nonwhite(np.asarray(all_frames[i])) for i in idx])
    thumbs = []
    for fr in frames:
        thumbs.append(fr)
    height = max(t.shape[0] for t in thumbs)
    gap = 10
    width = sum(t.shape[1] for t in thumbs) + gap
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    x = 0
    for i, t in enumerate(thumbs):
        if i == 3:
            x += gap
        canvas[: t.shape[0], x: x + t.shape[1]] = t
        x += t.shape[1]
    ax_strip.imshow(canvas)
    ax_strip.axis("off")
    ax_strip.set_title("F Sampled generated frames", loc="left", fontsize=7.5, fontweight="bold", pad=4)


def save_csv(stats, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["relation", "motion_id", "raw_token_emb_dist", "clip_pooled_dist", "attr_pooled_dist", "slot_jsd", "alpha_l1_77", "text_orig", "text_pair"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in stats:
            writer.writerow({k: row.get(k, "") for k in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="output_GPT_Final/sato_pp_allloss_gpu0_teachl1_05_eval5k_resume104k_200k_single/sato_pp_iter_130000.pth")
    parser.add_argument("--data-root", default="dataset/HumanML3D")
    parser.add_argument("--out-dir", default="ablation/attribution_visuals")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--stats-limit", type=int, default=64)
    parser.add_argument("--preserve-path", default="dataset/HumanML3D/preserving_pairs_filtered.jsonl")
    parser.add_argument("--alter-path", default="dataset/HumanML3D/train_alter_v2.json")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu_id)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_model, trans_encoder, attr_pool, ckpt_args = load_models(args.checkpoint, device)
    print(f"loaded checkpoint: {args.checkpoint}", flush=True)

    preserve_tokens, preserve_len = motion_tokens_from_vq(args.data_root, "013865", ckpt_args["nb_code"], ckpt_args["block_size"])
    alter_tokens, alter_len = motion_tokens_from_vq(args.data_root, "M002131", ckpt_args["nb_code"], ckpt_args["block_size"])

    preserve_case = compute_pair(
        clip_model,
        trans_encoder,
        attr_pool,
        ckpt_args,
        "walk forward slowly",
        "moves ahead at a slow pace",
        preserve_tokens,
        preserve_len,
        "preserve",
        device,
    )
    alter_case = compute_pair(
        clip_model,
        trans_encoder,
        attr_pool,
        ckpt_args,
        "this person walks forward turns left to walk back",
        "this person walks forward turns right to walk back",
        alter_tokens,
        alter_len,
        "alter",
        device,
    )
    failure_tokens, failure_len = motion_tokens_from_codes(
        "ablation/vlm_judge/visuals_20_each/t2mgpt/direction/direction_006_orig_a_person_makes_stiff_staggering_motions_forward_codes.npy",
        ckpt_args["nb_code"],
        ckpt_args["block_size"],
    )
    failure_case = compute_pair(
        clip_model,
        trans_encoder,
        attr_pool,
        ckpt_args,
        "a person makes stiff staggering motions forward",
        "a person makes stiff staggering motions backward",
        failure_tokens,
        failure_len,
        "alter",
        device,
    )
    failure = {
        "orig_gif": "ablation/vlm_judge/visuals_20_each/t2mgpt/direction/direction_006_orig_a_person_makes_stiff_staggering_motions_forward.gif",
        "edit_gif": "ablation/vlm_judge/visuals_20_each/t2mgpt/direction/direction_006_edit_a_person_makes_stiff_staggering_motions_backward.gif",
        "orig_joints": "ablation/vlm_judge/visuals_20_each/t2mgpt/direction/direction_006_orig_a_person_makes_stiff_staggering_motions_forward_joints.npy",
        "edit_joints": "ablation/vlm_judge/visuals_20_each/t2mgpt/direction/direction_006_edit_a_person_makes_stiff_staggering_motions_backward_joints.npy",
    }

    stats = compute_stats(
        clip_model,
        trans_encoder,
        attr_pool,
        ckpt_args,
        args.data_root,
        args.preserve_path,
        args.alter_path,
        out_dir / f"stats_limit{args.stats_limit}.json",
        args.stats_limit,
        device,
    )
    save_csv(stats, out_dir / "attribution_distance_stats.csv")

    alter_case["edited_shift_fraction"] = aligned_diff_fraction(
        alter_case["labels_a"],
        alter_case["attr_a"],
        alter_case["labels_b"],
        alter_case["attr_b"],
        {"left", "right"},
    )
    failure_case["edited_shift_fraction"] = aligned_diff_fraction(
        failure_case["labels_a"],
        failure_case["attr_a"],
        failure_case["labels_b"],
        failure_case["attr_b"],
        {"forward", "backward"},
    )

    cases = {"preserve": preserve_case, "alter": alter_case, "failure_attr": failure_case, "failure_motion": failure}
    (out_dir / "main_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    fig = plt.figure(figsize=(12.4, 7.6), dpi=180)
    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[1.55, 0.95, 1.4],
        height_ratios=[1.0, 1.0, 1.15],
        wspace=0.55,
        hspace=0.95,
    )

    ax = fig.add_subplot(gs[0, 0])
    draw_token_heatmap(
        ax,
        preserve_case["labels_a"],
        preserve_case["attr_a"],
        preserve_case["labels_b"],
        preserve_case["attr_b"],
        "A Preserve: equivalent semantics",
        "preserve",
    )
    ax = fig.add_subplot(gs[0, 1])
    draw_slot_bars(ax, preserve_case, "Preserve slot mass")
    ax.text(
        0.02,
        0.94,
        f"raw emb. {preserve_case['raw_token_emb_dist']:.2f}\nattr-pool {preserve_case['attr_pooled_dist']:.2f}\nslot JSD {preserve_case['slot_jsd']:.3f}",
        transform=ax.transAxes,
        fontsize=6,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#d9d9d9", "linewidth": 0.4, "pad": 1.5},
    )

    ax = fig.add_subplot(gs[1, 0])
    draw_token_heatmap(
        ax,
        alter_case["labels_a"],
        alter_case["attr_a"],
        alter_case["labels_b"],
        alter_case["attr_b"],
        "B Alter: localized left/right shift",
        "alter",
        edited_words={"left", "right"},
    )
    ax = fig.add_subplot(gs[1, 1])
    draw_slot_bars(ax, alter_case, "Alter slot mass")
    ax.text(
        0.02,
        0.94,
        f"edited-token\nshift frac. {alter_case['edited_shift_fraction']:.2f}\nattr-pool {alter_case['attr_pooled_dist']:.2f}",
        transform=ax.transAxes,
        fontsize=6,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#d9d9d9", "linewidth": 0.4, "pad": 1.5},
    )

    ax = fig.add_subplot(gs[:2, 2])
    draw_distance_scatter(ax, stats)

    ax = fig.add_subplot(gs[2, 0])
    draw_token_heatmap(
        ax,
        failure_case["labels_a"],
        failure_case["attr_a"],
        failure_case["labels_b"],
        failure_case["attr_b"],
        "C Failure: correct text shift, weak motion edit",
        "alter",
        edited_words={"forward", "backward"},
    )
    ax.text(
        0.02,
        0.94,
        f"edited-token shift frac. {failure_case['edited_shift_fraction']:.2f}",
        transform=ax.transAxes,
        fontsize=6,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#d9d9d9", "linewidth": 0.4, "pad": 1.5},
    )
    ax_traj = fig.add_subplot(gs[2, 1])
    ax_strip = fig.add_subplot(gs[2, 2])
    draw_failure(ax_traj, ax_strip, failure)

    fig.suptitle(
        "Attribution-level selective robustness: semantic invariance for paraphrases, localized shift for edits",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.045, right=0.99, top=0.92, bottom=0.075)
    fig.savefig(out_dir / "attribution_shift_figure.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "attribution_shift_figure.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "attribution_shift_figure.svg", bbox_inches="tight")
    plt.close(fig)

    print(out_dir / "attribution_shift_figure.png")
    print(out_dir / "main_cases.json")
    print(out_dir / "attribution_distance_stats.csv")


if __name__ == "__main__":
    main()
