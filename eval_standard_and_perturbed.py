#!/usr/bin/env python
"""
SATO++ Unified Evaluation Script
Evaluates both standard T2M metrics (FID, R-Precision, Diversity) and
perturbed metrics (FID_P, FID_D) for a given SATO++ checkpoint.

Usage:
    python eval_standard_and_perturbed.py \
        --checkpoint output_GPT_Final/sato_v3a.../sato_pp_best_fid.pth \
        --gpu 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clip
from dataset import dataset_TM_eval
from models.evaluator_wrapper import EvaluatorModelWrapper
import models.t2m_trans as trans
import models.vqvae as vqvae
from options.get_eval_option import get_opt
import utils.eval_trans as eval_trans
import utils.sato_pp_eval as sato_pp_eval
import utils.utils_model as utils_model
from utils.word_vectorizer import WordVectorizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to SATO++ checkpoint")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--pair-data-path", default="dataset/HumanML3D/test_perb.json",
                        help="Perturbed test pairs for FID_P/FID_D")
    parser.add_argument("--skip-standard", action="store_true",
                        help="Skip standard T2M evaluation")
    parser.add_argument("--skip-perturbed", action="store_true",
                        help="Skip perturbed FID_P/FID_D evaluation")
    parser.add_argument("--out-json", default=None,
                        help="Path to save JSON results")
    return parser.parse_args()


def resolve_repo_path(p):
    """Resolve path relative to the repo root."""
    if os.path.isabs(p):
        return p
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, p)


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = argparse.Namespace(**ckpt["args"])
    dataname = ckpt_args.dataname

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = {
        "checkpoint": os.path.basename(args.checkpoint),
        "iter": ckpt.get("iter", -1),
        "dataname": dataname,
    }

    # ======= Standard T2M Evaluation =======
    if not args.skip_standard:
        print(f"\n{'='*60}")
        print("Standard T2M Evaluation")
        print(f"{'='*60}")

        data_root = "./dataset/KIT-ML" if dataname == "kit" else "./dataset/HumanML3D"

        # VQ-VAE
        net = vqvae.HumanVQVAE(
            ckpt_args, ckpt_args.nb_code, ckpt_args.code_dim,
            ckpt_args.output_emb_width, ckpt_args.down_t, ckpt_args.stride_t,
            ckpt_args.width, ckpt_args.depth, ckpt_args.dilation_growth_rate)
        vq_ckpt = torch.load(resolve_repo_path(ckpt_args.resume_pth), map_location="cpu")
        net.load_state_dict(vq_ckpt["net"], strict=False)
        net.eval()
        net.to(device)

        # Transformer (load from SATO++ checkpoint)
        trans_encoder = trans.Text2Motion_Transformer(
            num_vq=ckpt_args.nb_code, embed_dim=ckpt_args.embed_dim_gpt,
            clip_dim=ckpt_args.clip_dim, block_size=ckpt_args.block_size,
            num_layers=ckpt_args.num_layers, n_head=ckpt_args.n_head_gpt,
            drop_out_rate=ckpt_args.drop_out_rate, fc_rate=ckpt_args.ff_rate)
        trans_encoder.load_state_dict(ckpt["trans"], strict=True)
        trans_encoder.eval()
        trans_encoder.to(device)

        # CLIP
        clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        if "clip" in ckpt:
            clip_model.load_state_dict(ckpt["clip"], strict=False)

        # Eval wrapper
        dataset_opt_path = ("checkpoints/kit/Comp_v6_KLD005/opt.txt"
                            if dataname == "kit"
                            else "checkpoints/t2m/Comp_v6_KLD005/opt.txt")
        wrapper_opt = get_opt(dataset_opt_path, device)
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

        w_vectorizer = WordVectorizer("./glove", "our_vab")
        val_loader = dataset_TM_eval.DATALoader(dataname, False, 32, w_vectorizer)

        os.makedirs("/tmp/sato_eval_standard", exist_ok=True)
        logger = utils_model.get_logger("/tmp/sato_eval_standard")
        writer = SummaryWriter("/tmp/sato_eval_standard")

        t0 = time.time()
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, \
            best_matching, writer, logger = eval_trans.evaluation_transformer(
                "/tmp/sato_eval_standard", val_loader, net, trans_encoder,
                logger, writer, ckpt.get("iter", 0),
                best_fid=1000, best_iter=0, best_div=100,
                best_top1=0, best_top2=0, best_top3=0, best_matching=100,
                clip_model=clip_model, eval_wrapper=eval_wrapper,
                draw=False, save=False)
        elapsed = time.time() - t0

        results["standard"] = {
            "FID": best_fid,
            "Diversity": best_div,
            "R_precision_top1": best_top1,
            "R_precision_top2": best_top2,
            "R_precision_top3": best_top3,
            "Matching_score": best_matching,
            "elapsed_sec": elapsed,
        }
        print(f"Standard: FID={best_fid:.5f}, Div={best_div:.4f}, "
              f"Top1={best_top1:.4f}, Top2={best_top2:.4f}, Top3={best_top3:.4f}")

    # ======= Perturbed Evaluation =======
    if not args.skip_perturbed:
        print(f"\n{'='*60}")
        print("Perturbed Evaluation (FID_P, FID_D)")
        print(f"{'='*60}")

        fid_p, fid_d, n_matched, n_total = sato_pp_eval.evaluate_perturbed_fid(
            ckpt_path=args.checkpoint,
            pair_data_path=args.pair_data_path,
            seed=args.seed,
        )

        results["perturbed"] = {
            "FID_P": fid_p,
            "FID_D": fid_d,
            "matched_pairs": n_matched,
            "total_pairs": n_total,
        }
        print(f"Perturbed: FID_P={fid_p:.5f}, FID_D={fid_d:.5f}, "
              f"matched={n_matched}/{n_total}")

    # ======= Save results =======
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(json.dumps(results, indent=2))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.out_json}")

    return results


if __name__ == "__main__":
    main()
