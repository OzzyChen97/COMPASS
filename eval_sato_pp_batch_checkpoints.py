import argparse
import csv
import glob
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
        description="Batch-evaluate SATO++ checkpoints on test FID/FID_P/FID_D."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--pair-data-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pattern", default="sato_pp*.pth")
    parser.add_argument(
        "--checkpoint-names",
        nargs="+",
        default=None,
        help="Optional checkpoint basenames to evaluate, e.g. sato_pp_iter_10000.pth.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--fast-batch",
        action="store_true",
        help="Use vectorized sampling. Faster, but not numerically comparable to the original sampler.",
    )
    return parser.parse_args()


def load_pair_data(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def checkpoint_sort_key(path):
    name = os.path.basename(path)
    if "_iter_" in name:
        try:
            return (0, int(name.rsplit("_iter_", 1)[1].split(".", 1)[0]), name)
        except ValueError:
            pass
    if "best_fid" in name:
        return (1, 0, name)
    if "last" in name:
        return (2, 0, name)
    return (3, 0, name)


def resolve_repo_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(path)


def main():
    cli_args = parse_args()
    os.makedirs(cli_args.out_dir, exist_ok=True)

    ckpt_paths = sorted(
        glob.glob(os.path.join(cli_args.checkpoint_dir, cli_args.pattern)),
        key=checkpoint_sort_key,
    )
    if cli_args.checkpoint_names:
        wanted = set(cli_args.checkpoint_names)
        ckpt_paths = [p for p in ckpt_paths if os.path.basename(p) in wanted]
        found = {os.path.basename(p) for p in ckpt_paths}
        missing = sorted(wanted - found)
        if missing:
            raise FileNotFoundError(f"Missing requested checkpoints: {missing}")
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoints matched {cli_args.pattern} in {cli_args.checkpoint_dir}"
        )

    logger = utils_model.get_logger(cli_args.out_dir)
    writer = SummaryWriter(cli_args.out_dir)
    logger.info(
        json.dumps(
            {
                "checkpoint_dir": cli_args.checkpoint_dir,
                "pair_data_path": cli_args.pair_data_path,
                "out_dir": cli_args.out_dir,
                "seed": cli_args.seed,
                "checkpoints": [os.path.basename(p) for p in ckpt_paths],
            },
            indent=4,
            sort_keys=True,
        )
    )

    first_ckpt = torch.load(ckpt_paths[0], map_location="cpu")
    args = Namespace(**first_ckpt.get("args", {}))
    dataname = args.dataname

    w_vectorizer = WordVectorizer("./glove", "our_vab")
    val_loader = dataset_TM_eval.DATALoader(dataname, True, 32, w_vectorizer)

    dataset_opt_path = (
        "checkpoints/kit/Comp_v6_KLD005/opt.txt"
        if dataname == "kit"
        else "checkpoints/t2m/Comp_v6_KLD005/opt.txt"
    )
    wrapper_opt = get_opt(dataset_opt_path, torch.device("cuda"))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    clip_model, _ = clip.load("ViT-B/32", device=torch.device("cuda"), jit=False)
    clip.model.convert_weights(clip_model)
    clip_model.float()
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

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
    vq_ckpt = torch.load(resolve_repo_path(args.resume_pth), map_location="cpu")
    net.load_state_dict(vq_ckpt["net"], strict=False)
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
    trans_encoder.cuda()

    pair_data = load_pair_data(cli_args.pair_data_path)
    summary_csv = os.path.join(cli_args.out_dir, "summary.csv")
    summary_json = os.path.join(cli_args.out_dir, "summary.json")
    results = []

    with open(summary_csv, "w", encoding="utf-8", newline="") as f_csv:
        fieldnames = [
            "checkpoint",
            "iter",
            "fid",
            "fid_p",
            "fid_d",
            "diversity_real",
            "diversity",
            "top1",
            "top2",
            "top3",
            "matching",
            "matched_pairs",
            "total_pairs",
            "elapsed_sec",
            "path",
        ]
        writer_csv = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer_csv.writeheader()
        f_csv.flush()

        for i, ckpt_path in enumerate(ckpt_paths, start=1):
            name = os.path.basename(ckpt_path)
            logger.info("BatchEval start %d/%d: %s", i, len(ckpt_paths), name)
            print(f"BatchEval start {i}/{len(ckpt_paths)}: {name}", flush=True)

            torch.manual_seed(cli_args.seed)
            np.random.seed(cli_args.seed)
            random.seed(cli_args.seed)

            ckpt = first_ckpt if ckpt_path == ckpt_paths[0] else torch.load(ckpt_path, map_location="cpu")
            clip_model.load_state_dict(ckpt["clip"], strict=True)
            clip_model.eval()
            trans_encoder.load_state_dict(ckpt["trans"], strict=True)
            trans_encoder.train()

            start = time.time()
            metrics, _, _, writer, logger = sato_pp_eval.evaluation_sato_pp(
                cli_args.out_dir,
                val_loader,
                net,
                trans_encoder,
                clip_model,
                eval_wrapper,
                logger,
                writer,
                ckpt.get("iter", 0),
                pair_data=pair_data,
                sample_categorical=True,
                use_fast_batch=cli_args.fast_batch,
            )
            elapsed = time.time() - start

            row = {
                "checkpoint": name,
                "iter": int(ckpt.get("iter", 0)),
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
                "path": ckpt_path,
            }
            results.append(row)
            writer_csv.writerow(row)
            f_csv.flush()

            with open(summary_json, "w", encoding="utf-8") as f_json:
                json.dump(results, f_json, indent=2)

            logger.info(
                "BatchEval done %s: FID %.4f, FID_P %.4f, FID_D %.4f, "
                "PerturbedMatched %s/%s, elapsed %.1fs",
                name,
                row["fid"],
                row["fid_p"],
                row["fid_d"],
                row["matched_pairs"],
                row["total_pairs"],
                elapsed,
            )
            print(
                f"BatchEval done {name}: FID {row['fid']:.4f}, "
                f"FID_P {row['fid_p']:.4f}, FID_D {row['fid_d']:.4f}",
                flush=True,
            )

    best = min(results, key=lambda x: x["fid"])
    with open(os.path.join(cli_args.out_dir, "best_fid.json"), "w", encoding="utf-8") as f_best:
        json.dump(best, f_best, indent=2)

    msg = (
        f"Best FID: {best['checkpoint']} | iter {best['iter']} | "
        f"FID {best['fid']:.4f} | FID_P {best['fid_p']:.4f} | "
        f"FID_D {best['fid_d']:.4f}"
    )
    logger.info(msg)
    print(msg, flush=True)
    writer.close()


if __name__ == "__main__":
    main()
