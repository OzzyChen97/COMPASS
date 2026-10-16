import argparse
import json
import os
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
import utils.eval_trans as eval_trans
import utils.sato_pp_eval as sato_pp_eval
import utils.utils_model as utils_model
from utils.word_vectorizer import WordVectorizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a full SATO++ checkpoint with its CLIP and transformer weights."
    )
    parser.add_argument("--sato-pp-pth", required=True, help="Full SATO++ checkpoint path")
    parser.add_argument("--out-dir", default=None, help="Evaluation output directory")
    parser.add_argument("--repeat-time", type=int, default=20, help="Number of repeated eval runs")
    parser.add_argument("--save-npy", action="store_true", help="Save generated/GT npy files for the first run")
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "sato_pp"],
        default="multi",
        help=(
            "single: one-sample test eval; multi: GPT_eval_multi-style 30-sample "
            "multimodality eval; sato_pp: one-sample FID/FID_P/FID_D eval"
        ),
    )
    parser.add_argument(
        "--pair-data-path",
        default=None,
        help="jsonl pair data for sato_pp mode; defaults to checkpoint args.pair_data_path",
    )
    return parser.parse_args()


def load_pair_data(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)

        pair_data = []
        for line in f:
            line = line.strip()
            if line:
                pair_data.append(json.loads(line))
        return pair_data


def main():
    cli_args = parse_args()
    ckpt = torch.load(cli_args.sato_pp_pth, map_location="cpu")
    train_args = ckpt.get("args", {})
    args = Namespace(**train_args)
    torch.manual_seed(getattr(args, "seed", 123))

    out_dir = cli_args.out_dir
    if out_dir is None:
        stem = os.path.splitext(os.path.basename(cli_args.sato_pp_pth))[0]
        exp_name = getattr(args, "exp_name", "sato_pp")
        out_dir = os.path.join("output_GPT_Final", f"{exp_name}_eval_{stem}")
    os.makedirs(out_dir, exist_ok=True)

    logger = utils_model.get_logger(out_dir)
    writer = SummaryWriter(out_dir)
    logger.info(json.dumps({"checkpoint": cli_args.sato_pp_pth, "repeat_time": cli_args.repeat_time, **train_args},
                           indent=4, sort_keys=True))

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
    clip_model.float()
    clip_model.load_state_dict(ckpt["clip"], strict=True)
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
    vq_ckpt = torch.load(args.resume_pth, map_location="cpu")
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
    trans_encoder.load_state_dict(ckpt["trans"], strict=True)
    trans_encoder.cuda()
    trans_encoder.train()

    if cli_args.mode == "sato_pp":
        pair_data_path = cli_args.pair_data_path or getattr(args, "pair_data_path", None)
        if not pair_data_path:
            raise ValueError("sato_pp mode requires --pair-data-path or args.pair_data_path in checkpoint")

        pair_data = load_pair_data(pair_data_path)

        metrics, _, _, writer, logger = sato_pp_eval.evaluation_sato_pp(
            out_dir,
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
        )
        msg_final = (
            f"FID. {metrics['fid']:.4f}, FID_P. {metrics['fid_p']:.4f}, "
            f"FID_D. {metrics['fid_d']:.4f}, "
            f"Diversity Real. {metrics['diversity_real']:.4f}, Diversity. {metrics['diversity']:.4f}, "
            f"TOP1. {metrics['top1']:.4f}, TOP2. {metrics['top2']:.4f}, TOP3. {metrics['top3']:.4f}, "
            f"Matching. {metrics['matching']:.4f}, "
            f"PerturbedMatched. {metrics['matched_pairs']}/{metrics['total_pairs']}"
        )
        print("final result:")
        print(msg_final)
        logger.info(msg_final)
        writer.close()
        return

    if cli_args.mode == "single":
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = (
            eval_trans.evaluation_transformer(
                out_dir,
                val_loader,
                net,
                trans_encoder,
                logger,
                writer,
                ckpt.get("iter", 0),
                best_fid=1000,
                best_iter=0,
                best_div=100,
                best_top1=0,
                best_top2=0,
                best_top3=0,
                best_matching=100,
                clip_model=clip_model,
                eval_wrapper=eval_wrapper,
                draw=False,
                save=False,
                savegif=False,
            )
        )
        msg_final = (
            f"FID. {best_fid:.4f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, "
            f"TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}, Matching. {best_matching:.4f}"
        )
        print("final result:")
        print(msg_final)
        logger.info(msg_final)
        writer.close()
        return

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    multi = []

    for i in range(cli_args.repeat_time):
        result = eval_trans.evaluation_transformer_test(
            out_dir,
            val_loader,
            net,
            trans_encoder,
            logger,
            writer,
            ckpt.get("iter", 0),
            best_fid=1000,
            best_iter=0,
            best_div=100,
            best_top1=0,
            best_top2=0,
            best_top3=0,
            best_matching=100,
            best_multi=0,
            clip_model=clip_model,
            eval_wrapper=eval_wrapper,
            draw=False,
            savegif=False,
            save=False,
            savenpy=(cli_args.save_npy and i == 0),
        )
        best_fid, _, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger = result
        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)
        multi.append(best_multi)
        logger.info(
            f"Repeat {i + 1}/{cli_args.repeat_time}: FID {best_fid:.4f}, "
            f"Diversity {best_div:.4f}, TOP1 {best_top1:.4f}, TOP2 {best_top2:.4f}, "
            f"TOP3 {best_top3:.4f}, Matching {best_matching:.4f}, Multi {best_multi:.4f}"
        )

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    multi = np.array(multi)

    msg_final = (
        f"FID. {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(cli_args.repeat_time):.3f}, "
        f"Diversity. {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(cli_args.repeat_time):.3f}, "
        f"TOP1. {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(cli_args.repeat_time):.3f}, "
        f"TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(cli_args.repeat_time):.3f}, "
        f"TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(cli_args.repeat_time):.3f}, "
        f"Matching. {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(cli_args.repeat_time):.3f}, "
        f"Multi. {np.mean(multi):.3f}, conf. {np.std(multi) * 1.96 / np.sqrt(cli_args.repeat_time):.3f}"
    )
    print("final result:")
    print(msg_final)
    logger.info(msg_final)
    writer.close()


if __name__ == "__main__":
    main()
