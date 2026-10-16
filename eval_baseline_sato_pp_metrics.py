import argparse
import csv
import json
import os
import random
import time
from argparse import Namespace

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import clip
from dataset import dataset_TM_eval
from models.evaluator_wrapper import EvaluatorModelWrapper
import models.t2m_trans as trans
import models.vqvae as vqvae
from options.get_eval_option import get_opt
import utils.sato_pp_eval as sato_pp_eval
import utils.utils_model as utils_model
from utils.word_vectorizer import WordVectorizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline T2M-GPT with FID/FID_P/FID_D."
    )
    parser.add_argument("--resume-pth", default="pretrained/VQVAE/net_best_fid.pth")
    parser.add_argument(
        "--resume-trans",
        default="pretrained/VQTransformer_corruption05/net_best_fid.pth",
    )
    parser.add_argument("--pair-data-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataname", default="t2m")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--nb-code", type=int, default=512)
    parser.add_argument("--code-dim", type=int, default=512)
    parser.add_argument("--output-emb-width", type=int, default=512)
    parser.add_argument("--down-t", type=int, default=2)
    parser.add_argument("--stride-t", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation-growth-rate", type=int, default=3)
    parser.add_argument("--quantizer", default="ema_reset", choices=["ema", "orig", "ema_reset", "reset"])
    parser.add_argument("--mu", type=float, default=0.99)
    parser.add_argument("--block-size", type=int, default=51)
    parser.add_argument("--embed-dim-gpt", type=int, default=1024)
    parser.add_argument("--clip-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--n-head-gpt", type=int, default=16)
    parser.add_argument("--drop-out-rate", type=float, default=0.1)
    parser.add_argument("--ff-rate", type=int, default=4)
    parser.add_argument(
        "--fast-batch",
        action="store_true",
        help="Use fast batched autoregressive sampling with per-sample VQ decoding.",
    )
    return parser.parse_args()


def load_pair_data(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    w_vectorizer = WordVectorizer("./glove", "our_vab")
    val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)

    dataset_opt_path = (
        "checkpoints/kit/Comp_v6_KLD005/opt.txt"
        if args.dataname == "kit"
        else "checkpoints/t2m/Comp_v6_KLD005/opt.txt"
    )
    wrapper_opt = get_opt(dataset_opt_path, torch.device("cuda"))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    clip_model, _ = clip.load("ViT-B/32", device=torch.device("cuda"), jit=False)
    clip.model.convert_weights(clip_model)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    net = vqvae.HumanVQVAE(
        Namespace(**vars(args)),
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
    net.load_state_dict(vq_ckpt["net"], strict=True)
    net.eval()
    net.cuda()

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
    trans_ckpt = torch.load(args.resume_trans, map_location="cpu")
    trans_encoder.load_state_dict(trans_ckpt["trans"], strict=True)
    trans_encoder.cuda()
    trans_encoder.train()

    pair_data = load_pair_data(args.pair_data_path)
    start = time.time()
    metrics, _, _, writer, logger = sato_pp_eval.evaluation_sato_pp(
        args.out_dir,
        val_loader,
        net,
        trans_encoder,
        clip_model,
        eval_wrapper,
        logger,
        writer,
        0,
        pair_data=pair_data,
        sample_categorical=True,
        use_fast_batch=args.fast_batch,
    )
    elapsed = time.time() - start

    row = {
        "checkpoint": os.path.basename(args.resume_trans),
        "iter": 0,
        "fid": metrics["fid"],
        "fid_p": metrics["fid_p"],
        "fid_d": metrics["fid_d"],
        "diversity_real": metrics["diversity_real"],
        "diversity": metrics["diversity"],
        "top1": metrics["top1"],
        "top2": metrics["top2"],
        "top3": metrics["top3"],
        "matching": metrics["matching"],
        "matched_pairs": metrics["matched_pairs"],
        "total_pairs": metrics["total_pairs"],
        "elapsed_sec": elapsed,
        "path": args.resume_trans,
    }

    summary_json = os.path.join(args.out_dir, "summary.json")
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    with open(summary_json, "w", encoding="utf-8") as f_json:
        json.dump([row], f_json, indent=2)
    with open(summary_csv, "w", encoding="utf-8", newline="") as f_csv:
        writer_csv = csv.DictWriter(f_csv, fieldnames=list(row.keys()))
        writer_csv.writeheader()
        writer_csv.writerow(row)

    msg_final = (
        f"FID. {metrics['fid']:.4f}, FID_P. {metrics['fid_p']:.4f}, "
        f"FID_D. {metrics['fid_d']:.4f}, "
        f"PerturbedMatched. {metrics['matched_pairs']}/{metrics['total_pairs']}"
    )
    print("final result:")
    print(msg_final)
    logger.info(msg_final)
    writer.close()


if __name__ == "__main__":
    main()
