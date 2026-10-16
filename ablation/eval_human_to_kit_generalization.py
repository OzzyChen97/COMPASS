#!/usr/bin/env python
"""Evaluate HumanML3D-trained generators on KIT-ML preserving captions.

This protocol intentionally does not compare generated HumanML3D 263D motion
against native KIT-ML 251D motion. Instead, it measures the two quantities that
remain well-defined without skeleton retargeting:

1. text-to-motion alignment for KIT captions in the HumanML3D evaluator space;
2. paired generation stability between semantically equivalent KIT captions.

The baseline and SATO++ models use identical KIT pairs, sampling seeds, VQ-VAE,
and evaluator. Lower FID-D/embedding distance and higher token agreement/cosine
indicate better cross-dataset robustness.
"""

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import clip
import numpy as np
import torch
import torch.nn.functional as F

from ablation.repeat20_perturb_fast_batch import load_model_bundle
from dataset import dataset_TM_eval
from dataset.config import configure_evaluator_source, evaluator_opt_path
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
import utils.sato_pp_eval as sato_eval
from utils.word_vectorizer import WordVectorizer


METRICS = (
    "fid_d",
    "embedding_cosine",
    "embedding_l2",
    "first_token_agreement",
    "exact_sequence_agreement",
    "token_position_agreement",
    "prefix_agreement",
    "text_clip_cosine",
    "top1_orig",
    "top2_orig",
    "top3_orig",
    "top1_pert",
    "top2_pert",
    "top3_pert",
    "matching_orig",
    "matching_pert",
    "diversity_orig",
    "diversity_pert",
    "motion_frames_orig",
    "motion_frames_pert",
    "elapsed_sec",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-data-path",
        default="dataset/KIT-ML/test_perb_gemini31_mixed_full.json",
    )
    parser.add_argument("--baseline-vq", default="pretrained/VQVAE/net_best_fid.pth")
    parser.add_argument(
        "--baseline-transformer",
        default="pretrained/VQTransformer_corruption05/net_best_fid.pth",
    )
    parser.add_argument(
        "--sato-checkpoint",
        default=(
            "output_GPT_Final/"
            "sato_pp_allloss_gpu0_teachl1_05_eval5k_resume104k_200k_single/"
            "sato_pp_iter_130000.pth"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ablation/human_to_kit_generalization_repeat20"),
    )
    parser.add_argument("--start-seed", type=int, default=123)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model", action="append", choices=("baseline", "sato_pp"))
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--cfg-uncond", choices=("zero", "empty"), default="zero")
    parser.add_argument("--sample-mode", choices=("categorical", "greedy"), default="categorical")
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_pairs(path):
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        rows = json.load(handle) if first == "[" else [
            json.loads(line) for line in handle if line.strip()
        ]
    invalid = [row for row in rows if row.get("relation") != "preserve"]
    if invalid:
        raise ValueError(f"Cross-dataset protocol requires preserving pairs; found {len(invalid)} invalid rows")
    return rows


def normalize_caption(text):
    return " ".join(str(text).strip().lower().split())


def build_pair_index(rows):
    index = {}
    for row in rows:
        key = (str(row["motion_id"]).strip(), normalize_caption(row["text_orig"]))
        if key in index:
            raise ValueError(f"Duplicate caption-locked pair: {key}")
        index[key] = str(row["text_pair"]).strip()
    return index


def make_logger(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("human_to_kit_generalization")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_specs(args):
    specs = {
        "baseline": {
            "name": "human_t2mgpt_baseline_on_kit_text",
            "kind": "baseline",
            "dataname": "t2m",
            "resume_pth": str(resolve_path(args.baseline_vq)),
            "resume_trans": str(resolve_path(args.baseline_transformer)),
            "description": "HumanML3D T2M-GPT baseline evaluated on KIT captions",
        },
        "sato_pp": {
            "name": "human_sato_pp_on_kit_text",
            "kind": "sato_pp",
            "checkpoint": str(resolve_path(args.sato_checkpoint)),
            "description": "HumanML3D SATO++ evaluated on KIT captions",
        },
    }
    selected = args.model or ["baseline", "sato_pp"]
    return [specs[name] for name in selected]


def reset_sampling_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_and_decode(bundle, text_features, args, batch_seed):
    reset_sampling_seed(batch_seed)
    uncond = sato_eval._make_cfg_uncond_feature(
        bundle["clip_model"],
        text_features.shape[0],
        text_features,
        cfg_scale=args.cfg_scale,
        cfg_uncond=args.cfg_uncond,
    )
    code_idx, code_lens = sato_eval._sample_motion_indices_batch(
        bundle["trans_encoder"],
        text_features,
        sample_categorical=args.sample_mode == "categorical",
        cfg_scale=args.cfg_scale,
        uncond_feat=uncond,
        sample_temperature=args.sample_temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    motions, motion_lens = sato_eval._decode_motion_indices_loop(
        bundle["net"], code_idx, code_lens, seq=196, dim_pose=263
    )
    return code_idx, code_lens, motions, motion_lens


def token_stability(code_a, lens_a, code_b, lens_b):
    first = exact = same_positions = prefix_total = denom_total = 0
    count = len(lens_a)
    for row in range(count):
        len_a = int(lens_a[row].item())
        len_b = int(lens_b[row].item())
        max_len = max(len_a, len_b, 1)
        min_len = min(len_a, len_b)
        seq_a = code_a[row, :len_a]
        seq_b = code_b[row, :len_b]
        first += int(len_a > 0 and len_b > 0 and seq_a[0].item() == seq_b[0].item())
        equal_prefix = 0
        for pos in range(min_len):
            if seq_a[pos].item() != seq_b[pos].item():
                break
            equal_prefix += 1
        prefix_total += equal_prefix
        if min_len:
            same_positions += int((seq_a[:min_len] == seq_b[:min_len]).sum().item())
        denom_total += max_len
        exact += int(len_a == len_b and min_len > 0 and torch.equal(seq_a, seq_b))
    return {
        "first": first,
        "exact": exact,
        "same_positions": same_positions,
        "prefix": prefix_total,
        "denom": denom_total,
        "count": count,
    }


@torch.no_grad()
def evaluate_seed(bundle, loader, pair_index, eval_wrapper, args, seed, logger):
    set_seed(seed)
    bundle["trans_encoder"].eval()
    bundle["clip_model"].eval()
    bundle["net"].eval()

    orig_embeddings = []
    pert_embeddings = []
    r_orig = np.zeros(3, dtype=np.float64)
    r_pert = np.zeros(3, dtype=np.float64)
    matching_orig = matching_pert = 0.0
    embedding_cosine = embedding_l2 = text_cosine = 0.0
    frame_orig = frame_pert = 0
    token_counts = {key: 0 for key in ("first", "exact", "same_positions", "prefix", "denom", "count")}
    samples = matched = 0
    started = time.time()

    for batch_idx, batch in enumerate(loader):
        word_embeddings, pos_one_hots, captions, sent_lens, _, _, _, motion_ids = batch
        perturbed = []
        for motion_id, caption in zip(motion_ids, captions):
            key = (str(motion_id), normalize_caption(caption))
            pair = pair_index.get(key)
            if pair is None:
                raise KeyError(f"No preserving caption pair for {key}")
            perturbed.append(pair)
        matched += len(perturbed)

        device = torch.device("cuda")
        orig_tokens = clip.tokenize(list(captions), truncate=True).to(device)
        pert_tokens = clip.tokenize(perturbed, truncate=True).to(device)
        feat_orig = bundle["clip_model"].encode_text(orig_tokens).float()
        feat_pert = bundle["clip_model"].encode_text(pert_tokens).float()
        text_cosine += F.cosine_similarity(feat_orig, feat_pert, dim=-1).sum().item()

        batch_seed = seed + batch_idx * 1_000_003
        codes_orig, code_lens_orig, motions_orig, motion_lens_orig = sample_and_decode(
            bundle, feat_orig, args, batch_seed
        )
        codes_pert, code_lens_pert, motions_pert, motion_lens_pert = sample_and_decode(
            bundle, feat_pert, args, batch_seed
        )

        text_embedding, motion_orig_embedding = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_lens,
            motions_orig, motion_lens_orig,
        )
        _, motion_pert_embedding = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_lens,
            motions_pert, motion_lens_pert,
        )

        text_np = text_embedding.cpu().numpy()
        orig_np = motion_orig_embedding.cpu().numpy()
        pert_np = motion_pert_embedding.cpu().numpy()
        temp_r, temp_matching = sato_eval.calculate_R_precision(
            text_np, orig_np, top_k=3, sum_all=True
        )
        r_orig += temp_r
        matching_orig += temp_matching
        temp_r, temp_matching = sato_eval.calculate_R_precision(
            text_np, pert_np, top_k=3, sum_all=True
        )
        r_pert += temp_r
        matching_pert += temp_matching

        orig_embeddings.append(orig_np)
        pert_embeddings.append(pert_np)
        embedding_cosine += F.cosine_similarity(
            motion_orig_embedding, motion_pert_embedding, dim=-1
        ).sum().item()
        embedding_l2 += torch.linalg.vector_norm(
            motion_orig_embedding - motion_pert_embedding, dim=-1
        ).sum().item()
        frame_orig += int(motion_lens_orig.sum().item())
        frame_pert += int(motion_lens_pert.sum().item())

        stability = token_stability(
            codes_orig, code_lens_orig, codes_pert, code_lens_pert
        )
        for key, value in stability.items():
            token_counts[key] += value
        samples += len(captions)
        logger.info(
            "seed=%d batch=%d samples=%d/%d first_token=%.4f emb_cos=%.4f",
            seed,
            batch_idx + 1,
            samples,
            len(loader.dataset),
            token_counts["first"] / token_counts["count"],
            embedding_cosine / samples,
        )
        if args.max_samples is not None and samples >= args.max_samples:
            break

    if samples < 2:
        raise ValueError("Evaluation requires at least two resolved KIT caption pairs")
    orig_np = np.concatenate(orig_embeddings, axis=0)
    pert_np = np.concatenate(pert_embeddings, axis=0)
    orig_mu, orig_cov = sato_eval.calculate_activation_statistics(orig_np)
    pert_mu, pert_cov = sato_eval.calculate_activation_statistics(pert_np)
    diversity_times = min(300, samples - 1)

    return {
        "seed": seed,
        "samples": samples,
        "matched_pairs": matched,
        "fid_d": float(sato_eval.calculate_frechet_distance(orig_mu, orig_cov, pert_mu, pert_cov)),
        "embedding_cosine": embedding_cosine / samples,
        "embedding_l2": embedding_l2 / samples,
        "first_token_agreement": token_counts["first"] / token_counts["count"],
        "exact_sequence_agreement": token_counts["exact"] / token_counts["count"],
        "token_position_agreement": token_counts["same_positions"] / token_counts["denom"],
        "prefix_agreement": token_counts["prefix"] / token_counts["denom"],
        "text_clip_cosine": text_cosine / samples,
        "top1_orig": r_orig[0] / samples,
        "top2_orig": r_orig[1] / samples,
        "top3_orig": r_orig[2] / samples,
        "top1_pert": r_pert[0] / samples,
        "top2_pert": r_pert[1] / samples,
        "top3_pert": r_pert[2] / samples,
        "matching_orig": matching_orig / samples,
        "matching_pert": matching_pert / samples,
        "diversity_orig": float(sato_eval.calculate_diversity(orig_np, diversity_times)),
        "diversity_pert": float(sato_eval.calculate_diversity(pert_np, diversity_times)),
        "motion_frames_orig": frame_orig / samples,
        "motion_frames_pert": frame_pert / samples,
        "elapsed_sec": time.time() - started,
    }


def mean_std(rows, key):
    values = [float(row[key]) for row in rows]
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def aggregate(runs, specs):
    summaries = []
    for spec in specs:
        rows = [row for row in runs if row["model"] == spec["name"]]
        summary = {
            "model": spec["name"],
            "kind": spec["kind"],
            "checkpoint": row_checkpoint(spec),
            "n": len(rows),
        }
        for metric in METRICS:
            summary[f"{metric}_mean"], summary[f"{metric}_std"] = mean_std(rows, metric)
        summaries.append(summary)

    deltas = []
    baseline = {row["seed"]: row for row in runs if row["kind"] == "baseline"}
    sato = {row["seed"]: row for row in runs if row["kind"] == "sato_pp"}
    for seed in sorted(set(baseline) & set(sato)):
        row = {"seed": seed}
        for metric in METRICS:
            row[metric] = float(sato[seed][metric]) - float(baseline[seed][metric])
        deltas.append(row)
    delta_summary = {"n": len(deltas)}
    for metric in METRICS:
        if deltas:
            delta_summary[f"{metric}_mean"], delta_summary[f"{metric}_std"] = mean_std(deltas, metric)
    return summaries, deltas, delta_summary


def row_checkpoint(spec):
    return spec.get("checkpoint", spec.get("resume_trans", ""))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_metric(summary, metric):
    return f"{summary[f'{metric}_mean']:.4f} +/- {summary[f'{metric}_std']:.4f}"


def write_report(path, args, pairs, summaries, delta_summary):
    lines = [
        "# HumanML3D -> KIT-ML Cross-Dataset Generalization",
        "",
        "HumanML3D-trained generators are frozen and prompted with caption-locked KIT-ML preserving pairs.",
        "All motion metrics are computed in the HumanML3D evaluator space; native KIT motion is not used,",
        "so the report measures text-domain transfer and perturbation robustness without skeleton retargeting.",
        "",
        "## Protocol",
        "",
        f"- Pair file: `{args.pair_data_path}` ({len(pairs)} preserving pairs)",
        f"- Seeds: `{args.start_seed}` to `{args.start_seed + args.num_repeats - 1}`",
        f"- CFG: `{args.cfg_scale}` with `{args.cfg_uncond}` unconditional feature",
        f"- Sampling: `{args.sample_mode}`, temperature `{args.sample_temperature}`, top-k `{args.top_k}`, top-p `{args.top_p}`",
        "- Original and perturbed captions reuse the same per-batch random seed.",
        "",
        "## Results",
        "",
        "Lower is better for FID-D, embedding L2, and Matching. Higher is better for cosine, token agreement, and Top-k.",
        "Perturbed-generation retrieval uses the original KIT annotation as the semantic target because every pair is preserving.",
        "",
        "| Model | n | FID-D | Emb cosine | Emb L2 | First token | Exact sequence | Top1 orig | Top1 pert | Matching orig | Matching pert |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| `{summary['model']}` | {summary['n']} | {format_metric(summary, 'fid_d')} | "
            f"{format_metric(summary, 'embedding_cosine')} | {format_metric(summary, 'embedding_l2')} | "
            f"{format_metric(summary, 'first_token_agreement')} | {format_metric(summary, 'exact_sequence_agreement')} | "
            f"{format_metric(summary, 'top1_orig')} | {format_metric(summary, 'top1_pert')} | "
            f"{format_metric(summary, 'matching_orig')} | {format_metric(summary, 'matching_pert')} |"
        )
    if delta_summary.get("n"):
        lines.extend([
            "",
            "## Paired Delta",
            "",
            "`SATO++ - baseline`, paired by sampling seed:",
            "",
            f"- FID-D: {format_metric(delta_summary, 'fid_d')}",
            f"- Embedding cosine: {format_metric(delta_summary, 'embedding_cosine')}",
            f"- Embedding L2: {format_metric(delta_summary, 'embedding_l2')}",
            f"- First-token agreement: {format_metric(delta_summary, 'first_token_agreement')}",
            f"- Exact-sequence agreement: {format_metric(delta_summary, 'exact_sequence_agreement')}",
            f"- Perturbed-caption Top1: {format_metric(delta_summary, 'top1_pert')}",
            f"- Perturbed-caption Matching: {format_metric(delta_summary, 'matching_pert')}",
            "",
            "Exact-sequence and contiguous-prefix agreement should be interpreted separately from semantic stability. "
            "A model can preserve the first decision and motion semantics while taking a different later autoregressive token path.",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_args(args, specs):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the current T2M-GPT/CLIP evaluation stack")
    if args.num_repeats < 1 or args.batch_size < 2:
        raise ValueError("--num-repeats must be >= 1 and --batch-size must be >= 2")
    for spec in specs:
        checkpoint = resolve_path(row_checkpoint(spec))
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)


def main():
    args = parse_args()
    args.out_dir = resolve_path(args.out_dir)
    specs = model_specs(args)
    validate_args(args, specs)
    pairs = load_pairs(args.pair_data_path)
    pair_index = build_pair_index(pairs)
    logger = make_logger(args.out_dir)
    logger.info("protocol=%s", json.dumps(vars(args), default=str, sort_keys=True))

    vectorizer = WordVectorizer(str(REPO_ROOT / "glove"), "our_vab")
    loader = dataset_TM_eval.CaptionLockedDATALoader(
        "kit", True, args.batch_size, vectorizer, pairs,
        num_workers=args.num_workers,
    )
    logger.info("caption_locked_stats=%s", json.dumps(loader.dataset.resolution_stats, sort_keys=True))

    wrapper_opt = get_opt(evaluator_opt_path("t2m"), torch.device("cuda"))
    wrapper_opt = configure_evaluator_source(wrapper_opt, "t2m")
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    runs = []
    for spec in specs:
        logger.info("loading model=%s checkpoint=%s", spec["name"], row_checkpoint(spec))
        bundle = load_model_bundle(spec)
        if bundle["args"].dataname != "t2m":
            raise ValueError(f"Expected HumanML3D checkpoint, got {bundle['args'].dataname!r}")
        for run_idx in range(1, args.num_repeats + 1):
            seed = args.start_seed + run_idx - 1
            result = evaluate_seed(bundle, loader, pair_index, eval_wrapper, args, seed, logger)
            result.update({
                "model": spec["name"],
                "kind": spec["kind"],
                "run_idx": run_idx,
                "checkpoint": row_checkpoint(spec),
                "iter": bundle["iter"],
            })
            runs.append(result)
            write_csv(args.out_dir / "runs.csv", runs)
            logger.info("result=%s", json.dumps(result, sort_keys=True))
        del bundle
        torch.cuda.empty_cache()

    summaries, deltas, delta_summary = aggregate(runs, specs)
    write_csv(args.out_dir / "summary.csv", summaries)
    write_csv(args.out_dir / "paired_deltas.csv", deltas)
    payload = {
        "protocol": vars(args),
        "pair_count": len(pairs),
        "loader_stats": loader.dataset.resolution_stats,
        "runs": runs,
        "summary": summaries,
        "paired_delta_summary": delta_summary,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_report(args.out_dir / "README.md", args, pairs, summaries, delta_summary)
    logger.info("completed output=%s", args.out_dir)


if __name__ == "__main__":
    main()
