import json
import os
import random
import sys
import time
from argparse import Namespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

import clip
from dataset import dataset_TM_eval
from models.evaluator_wrapper import EvaluatorModelWrapper
import models.t2m_trans as trans
import models.vqvae as vqvae
from options.get_eval_option import get_opt
from utils.word_vectorizer import WordVectorizer
import utils.sato_pp_eval as sato_pp_eval


CKPT_PATH = "ablation/loss_no_attribute_l1_150k/sato_pp_last.pth"


def load_models():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    args = Namespace(**ckpt["args"])

    clip_model, _ = clip.load("ViT-B/32", device=torch.device("cuda"), jit=False)
    clip.model.convert_weights(clip_model)
    clip_model.float()
    clip_model.load_state_dict(ckpt["clip"], strict=True)
    clip_model.eval()

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
    vq_ckpt = torch.load(args.resume_pth, map_location="cpu")
    net.load_state_dict(vq_ckpt["net"], strict=False)
    net.cuda()
    net.eval()

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
    trans_encoder.load_state_dict(ckpt["trans"], strict=True)
    trans_encoder.cuda()
    trans_encoder.eval()

    return args, clip_model, net, trans_encoder


def get_batch(args):
    w_vectorizer = WordVectorizer("./glove", "our_vab")
    loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)
    return next(iter(loader))


def sample_loop_tokens(trans_encoder, feat_clip, sample_categorical):
    tokens = []
    for i in range(feat_clip.shape[0]):
        tokens.append(trans_encoder.sample(feat_clip[i : i + 1], sample_categorical))
    return tokens


def pad_tokens(tokens):
    max_len = max(t.shape[1] for t in tokens)
    padded = torch.zeros((len(tokens), max_len), dtype=torch.long, device=tokens[0].device)
    lens = []
    for i, t in enumerate(tokens):
        cur_len = t.shape[1]
        padded[i, :cur_len] = t[0]
        lens.append(cur_len)
    return padded, torch.tensor(lens, dtype=torch.long, device=tokens[0].device)


def main():
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args, clip_model, net, trans_encoder = load_models()
    batch = get_batch(args)
    _, _, clip_text, _, pose, _, _, _ = batch
    seq, dim_pose = pose.shape[1], pose.shape[2]

    text = clip.tokenize(clip_text, truncate=True).cuda()
    with torch.no_grad():
        feat_clip = clip_model.encode_text(text).float()

        loop_tokens = sample_loop_tokens(trans_encoder, feat_clip, sample_categorical=False)
        loop_code, loop_lens = pad_tokens(loop_tokens)
        fast_code, fast_lens = sato_pp_eval._sample_motion_indices_batch(
            trans_encoder, feat_clip, sample_categorical=False
        )

        token_equal = (
            loop_lens.cpu().tolist() == fast_lens.cpu().tolist()
            and loop_code.shape == fast_code.shape
            and torch.equal(loop_code, fast_code)
        )

        start = time.time()
        loop_pose, loop_pred_len = sato_pp_eval._generate_motion_batch_loop(
            net, trans_encoder, feat_clip, seq, dim_pose, sample_categorical=False
        )
        loop_sec = time.time() - start

        start = time.time()
        fast_pose, fast_pred_len = sato_pp_eval.generate_motion_batch(
            net, trans_encoder, feat_clip, seq, dim_pose,
            sample_categorical=False,
            use_fast_batch=True,
        )
        fast_sec = time.time() - start

        pose_diff = (loop_pose - fast_pose).abs()
        valid_mask = torch.zeros_like(pose_diff[..., :1], dtype=torch.bool)
        for i, length in enumerate(loop_pred_len.tolist()):
            valid_mask[i, :length] = True
        valid_diff = pose_diff[valid_mask.expand_as(pose_diff)]
        pad_diff = pose_diff[~valid_mask.expand_as(pose_diff)]
        flat_idx = int(pose_diff.argmax().item())
        sample_idx = flat_idx // (pose_diff.shape[1] * pose_diff.shape[2])
        rem = flat_idx % (pose_diff.shape[1] * pose_diff.shape[2])
        frame_idx = rem // pose_diff.shape[2]
        dim_idx = rem % pose_diff.shape[2]
        result = {
            "checkpoint": CKPT_PATH,
            "batch_size": int(feat_clip.shape[0]),
            "greedy_token_equal": bool(token_equal),
            "loop_token_lens": loop_lens.cpu().tolist(),
            "fast_token_lens": fast_lens.cpu().tolist(),
            "loop_pred_lens": loop_pred_len.cpu().tolist(),
            "fast_pred_lens": fast_pred_len.cpu().tolist(),
            "max_abs_pose_diff": float(pose_diff.max().item()),
            "mean_abs_pose_diff": float(pose_diff.mean().item()),
            "valid_max_abs_pose_diff": float(valid_diff.max().item()),
            "valid_mean_abs_pose_diff": float(valid_diff.mean().item()),
            "pad_max_abs_pose_diff": float(pad_diff.max().item()) if pad_diff.numel() else 0.0,
            "max_diff_location": {
                "sample": sample_idx,
                "frame": int(frame_idx),
                "dim": int(dim_idx),
                "pred_len": int(loop_pred_len[sample_idx].item()),
                "token_len": int(loop_lens[sample_idx].item()),
            },
            "loop_sec": loop_sec,
            "fast_sec": fast_sec,
            "speedup": loop_sec / max(fast_sec, 1e-8),
        }

        torch.manual_seed(seed)
        start = time.time()
        cat_fast_pose, cat_fast_pred_len = sato_pp_eval.generate_motion_batch(
            net, trans_encoder, feat_clip, seq, dim_pose,
            sample_categorical=True,
            use_fast_batch=True,
        )
        cat_fast_sec = time.time() - start
        result["categorical_fast_sec"] = cat_fast_sec
        result["categorical_fast_pred_lens"] = cat_fast_pred_len.cpu().tolist()
        result["categorical_fast_finite"] = bool(torch.isfinite(cat_fast_pose).all().item())
        result["categorical_fast_abs_mean"] = float(cat_fast_pose.abs().mean().item())

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
