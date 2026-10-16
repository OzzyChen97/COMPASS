#!/usr/bin/env python
import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from argparse import Namespace
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import wandb

import clip
from dataset import dataset_TM_eval
from dataset.config import (
    configure_evaluator_source,
    dataset_max_motion_length,
    evaluator_opt_path,
)
from models.evaluator_wrapper import EvaluatorModelWrapper
import models.t2m_trans as trans
import models.vqvae as vqvae
from options.get_eval_option import get_opt
import utils.sato_pp_eval as sato_pp_eval
from utils.word_vectorizer import WordVectorizer


METRIC_FIELDS = [
    "fid",
    "fid_p",
    "fid_d",
    "diversity_real",
    "diversity",
    "top1",
    "top2",
    "top3",
    "matching",
    "top1_pert",
    "top2_pert",
    "top3_pert",
    "matching_pert",
    "elapsed_sec",
]

RUN_FIELDS = [
    "model",
    "kind",
    "run_idx",
    "seed",
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
    "top1_pert",
    "top2_pert",
    "top3_pert",
    "matching_pert",
    "perturbed_retrieval_samples",
    "matched_pairs",
    "total_pairs",
    "sample_index_sha256",
    "elapsed_sec",
    "path",
]


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


def default_specs():
    return [
        {
            "name": "baseline_t2mgpt_original",
            "kind": "baseline",
            "resume_pth": "pretrained/VQVAE/net_best_fid.pth",
            "resume_trans": "pretrained/VQTransformer_corruption05/net_best_fid.pth",
            "description": "Original T2M-GPT pretrained baseline",
        },
        {
            "name": "baseline_aug_syn",
            "kind": "baseline",
            "resume_pth": "pretrained/VQVAE/net_best_fid.pth",
            "resume_trans": "output_GPT_Final/baseline_aug_syn/net_best_fid.pth",
            "description": "T2M-GPT fine-tuned with synonym augmentation",
        },
        {
            "name": "sato_pp_allloss_l1_iter130k",
            "kind": "sato_pp",
            "checkpoint": (
                "output_GPT_Final/"
                "sato_pp_allloss_gpu0_teachl1_05_eval5k_resume104k_200k_single/"
                "sato_pp_iter_130000.pth"
            ),
            "description": "SATO++ all-loss L1 checkpoint at 130k iterations",
        },
        {
            "name": "loss_no_preserve_l1_150k",
            "kind": "sato_pp",
            "checkpoint": "ablation/loss_no_preserve_l1_150k/sato_pp_last.pth",
            "description": "Finished ablation: no preserve loss, 150k last",
        },
        {
            "name": "loss_no_attribute_l1_150k",
            "kind": "sato_pp",
            "checkpoint": "ablation/loss_no_attribute_l1_150k/sato_pp_last.pth",
            "description": "Finished ablation: no attribution losses, 150k last",
        },
        {
            "name": "loss_full_allloss_l1_iter120k",
            "kind": "sato_pp",
            "checkpoint": "ablation/loss_full_allloss_l1_150k/sato_pp_iter_120000.pth",
            "description": "In-progress ablation: full all-loss L1 checkpoint at 120k",
        },
        {
            "name": "loss_no_teacher_l1_iter120k",
            "kind": "sato_pp",
            "checkpoint": "ablation/loss_no_teacher_l1_150k/sato_pp_iter_120000.pth",
            "description": "In-progress ablation: no teacher L1 checkpoint at 120k",
        },
        {
            "name": "loss_no_alter_l1_iter140k",
            "kind": "sato_pp",
            "checkpoint": "ablation/loss_no_alter_l1_150k/sato_pp_iter_140000.pth",
            "description": "In-progress ablation: no alteration losses checkpoint at 140k",
        },
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repeat fast-batch perturbation evaluation and aggregate mean/std/variance."
    )
    parser.add_argument(
        "--out-dir",
        default="ablation/repeat20_perturb_fast_batch_20260702",
    )
    parser.add_argument(
        "--report-md",
        default="ablation/repeat20_perturb_fast_batch_20260702.md",
    )
    parser.add_argument(
        "--pair-data-path",
        default="dataset/HumanML3D/test_perb_deterministic.json",
    )
    parser.add_argument("--start-seed", type=int, default=123)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument(
        "--loader-seed",
        type=int,
        default=None,
        help="freeze loader order/caption selection while run seeds control generation",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="diagnostic-only cap on samples per repeat",
    )
    parser.add_argument(
        "--sampling-pair-rng-mode",
        choices=["common", "sequential"],
        default="common",
        help="whether perturbed generation replays original sampling RNG",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--evaluator-checkpoint",
        default=None,
        help=(
            "Optional path to a custom text_mot_match finest.tar to use as the "
            "retrieval ruler (e.g. checkpoints/t2m/text_mot_match_para_aware/"
            "model/finest.tar). Defaults to the standard TM2T matcher."
        ),
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="spaCy model used to build perturbed-caption TM2T GloVe+POS inputs",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader worker count; use 0 on systems with short AF_UNIX path limits.",
    )
    parser.add_argument(
        "--dataset-split",
        choices=["val", "test"],
        default="test",
        help="Dataset split used by the evaluator loader.",
    )
    parser.add_argument("--fast-batch", action="store_true")
    parser.add_argument(
        "--humanml-fast-protocol",
        action="store_true",
        help=(
            "Match eval_sato_pp_batch_checkpoints.py: standard shuffled "
            "evaluation loader, fast batched sampling, and one RNG seed set "
            "at evaluation start without per-batch RNG resets."
        ),
    )
    parser.add_argument(
        "--caption-locked",
        action="store_true",
        help=(
            "Evaluate only pair rows whose (motion_id, text_orig) resolves to "
            "an annotated caption, with deterministic motion crops and no "
            "shuffle/drop-last."
        ),
    )
    parser.add_argument(
        "--caption-locked-length-policy",
        choices=["strict", "repair-all"],
        default="strict",
        help="length handling used by the caption-locked HumanML3D loader",
    )
    parser.add_argument(
        "--expected-sample-index",
        default=None,
        help=(
            "frozen sample-index JSON; its order and exact original/perturbed "
            "captions replace pair-data ordering and are checked sample by sample"
        ),
    )
    parser.add_argument(
        "--allowed-levels",
        nargs="+",
        help="optional perturbation-level allowlist, e.g. P1 P2 P3",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Optional subset of model names. Can be specified multiple times.",
    )
    parser.add_argument(
        "--spec-json",
        default=None,
        help="JSON array of baseline/SATO++ model specs. Replaces built-in specs.",
    )
    parser.add_argument("--external-sato-name", default=None)
    parser.add_argument("--external-sato-vq", default=None)
    parser.add_argument("--external-sato-trans", default=None)
    parser.add_argument("--external-sato-clip", default=None)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--cfg-uncond", choices=["zero", "empty"], default="zero")
    parser.add_argument(
        "--sample-mode",
        choices=["categorical", "greedy"],
        default="categorical",
    )
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--diagnostics-dir",
        default=None,
        help="optional directory for per-seed SATO++ embedding/length NPZ dumps",
    )
    parser.add_argument(
        "--retrieval-embedding-dir",
        default=None,
        help=(
            "optional directory for lightweight per-seed perturbed TM2T text/motion "
            "embedding dumps used for independent R-Precision recomputation"
        ),
    )
    parser.add_argument(
        "--generation-audit-dir",
        default=None,
        help=(
            "optional directory for complete per-seed original/perturbed motion-token "
            "generation audits; formal use requires --fast-batch"
        ),
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return parser.parse_args()


def resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


def build_baseline_args(spec):
    values = {
        "dataname": spec.get("dataname", "t2m"),
        "resume_pth": spec["resume_pth"],
        "resume_trans": spec["resume_trans"],
        "nb_code": 512,
        "code_dim": 512,
        "output_emb_width": 512,
        "down_t": 2,
        "stride_t": 2,
        "width": 512,
        "depth": 3,
        "dilation_growth_rate": 3,
        "quantizer": "ema_reset",
        "mu": 0.99,
        "block_size": 51,
        "embed_dim_gpt": 1024,
        "clip_dim": 512,
        "num_layers": 9,
        "n_head_gpt": 16,
        "drop_out_rate": 0.1,
        "ff_rate": 4,
    }
    values.update(spec.get("args", {}))
    return Namespace(**values)


def load_specs(path):
    with open(resolve_repo_path(path), "r", encoding="utf-8") as handle:
        specs = json.load(handle)
    if not isinstance(specs, list) or not specs:
        raise ValueError("--spec-json must contain a non-empty JSON array")
    required = {"name", "kind"}
    for spec in specs:
        if not required.issubset(spec):
            raise ValueError(f"Invalid model spec: {spec}")
    return specs


def resolve_specs_dataname(specs):
    datanames = set()
    for spec in specs:
        if spec["kind"] == "baseline":
            datanames.add(build_baseline_args(spec).dataname)
        elif spec["kind"] == "sato_pp":
            if spec.get("dataname"):
                datanames.add(spec["dataname"])
            else:
                checkpoint = torch.load(
                    resolve_repo_path(spec["checkpoint"]), map_location="cpu"
                )
                datanames.add(checkpoint["args"]["dataname"])
        else:
            raise ValueError(f"Unknown model kind: {spec['kind']}")
    if len(datanames) != 1:
        raise ValueError(f"All evaluation specs must use one dataset, got {sorted(datanames)}")
    return datanames.pop()


def load_pair_data(path):
    with open(resolve_repo_path(path), "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def canonical_sha256(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_expected_sample_index(path):
    payload = json.loads(Path(resolve_repo_path(path)).read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list) or not samples:
        raise ValueError("--expected-sample-index must contain a non-empty samples list")
    required = {
        "loader_name", "motion_id", "caption", "perturbed_caption",
        "level", "motion_length",
    }
    for idx, row in enumerate(samples):
        missing = required - set(row)
        if missing:
            raise ValueError(f"sample-index row {idx} misses {sorted(missing)}")
        if str(row["loader_name"]) != str(row["motion_id"]):
            raise ValueError(
                f"sample-index row {idx} loader_name/motion_id mismatch: "
                f"{row['loader_name']} != {row['motion_id']}"
            )
    digest = canonical_sha256(samples)
    recorded = payload.get("sample_index_sha256") if isinstance(payload, dict) else None
    if recorded and recorded != digest:
        raise ValueError(
            f"sample-index checksum mismatch: recorded={recorded}, computed={digest}"
        )
    return samples, digest


def pair_data_from_sample_index(samples):
    """Preserve frozen loader order, including deliberate mix replicas."""
    return [
        {
            "pair_id": row.get("pair_id", idx),
            "motion_id": str(row["motion_id"]),
            "text_orig": str(row["caption"]),
            "text_pair": str(row["perturbed_caption"]),
            "level": str(row["level"]),
            "mix_replica": row.get("mix_replica", 0),
        }
        for idx, row in enumerate(samples)
    ]


def validate_pair_mix_matches_index(pair_data, samples, expected_levels=None):
    pair_mix = Counter(
        (
            str(row.get("motion_id", "")),
            str(row.get("text_orig", "")),
            str(row.get("text_pair", "")),
            str(row.get("level", "")),
        )
        for row in pair_data
    )
    index_mix = Counter(
        (
            str(row.get("motion_id", "")),
            str(row.get("caption", "")),
            str(row.get("perturbed_caption", "")),
            str(row.get("level", "")),
        )
        for row in samples
    )
    if pair_mix != index_mix:
        raise ValueError(
            "pair-data and frozen sample index do not have the same caption/level "
            "mixture (including duplicate multiplicity)"
        )
    level_counts = dict(sorted(
        Counter(row[3] for row in index_mix.elements()).items()
    ))
    expected = set(expected_levels or ("P1", "P2", "P3"))
    if not expected or not expected.issubset({"P1", "P2", "P3"}):
        raise ValueError(
            "formal repaired evaluation supports only non-empty P1/P2/P3 "
            f"level sets, got {sorted(expected)}"
        )
    if set(level_counts) != expected:
        raise ValueError(
            "formal repaired evaluation frozen index has unexpected levels: "
            f"expected {sorted(expected)}, got {sorted(level_counts)}"
        )
    return level_counts


def make_logger(out_dir):
    logger = logging.getLogger("repeat20_perturb_fast_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(Path(out_dir) / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_clip_model(load_float):
    clip_model, _ = clip.load("ViT-B/32", device=torch.device("cuda"), jit=False)
    clip.model.convert_weights(clip_model)
    if load_float:
        clip_model.float()
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    return clip_model


def load_vqvae(args, strict):
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
    net.load_state_dict(vq_ckpt["net"], strict=strict)
    net.cuda()
    net.eval()
    return net


def load_transformer(args):
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
    model.cuda()
    model.train()
    return model


def load_model_bundle(spec):
    if spec["kind"] == "baseline":
        args = build_baseline_args(spec)
        clip_model = load_clip_model(load_float=bool(spec.get("clip_pth")))
        if spec.get("clip_pth"):
            clip_ckpt = torch.load(resolve_repo_path(spec["clip_pth"]), map_location="cpu")
            if isinstance(clip_ckpt, dict) and "clip" in clip_ckpt:
                clip_ckpt = clip_ckpt["clip"]
            clip_model.load_state_dict(clip_ckpt, strict=True)
            clip_model.eval()
        net = load_vqvae(args, strict=True)
        trans_encoder = load_transformer(args)
        trans_ckpt = torch.load(resolve_repo_path(args.resume_trans), map_location="cpu")
        trans_encoder.load_state_dict(trans_ckpt["trans"], strict=True)
        ckpt_iter = 0
        checkpoint_name = Path(args.resume_trans).name
        checkpoint_path = args.resume_trans
    elif spec["kind"] == "sato_pp":
        ckpt = torch.load(resolve_repo_path(spec["checkpoint"]), map_location="cpu")
        args = Namespace(**ckpt["args"])
        clip_model = load_clip_model(load_float=True)
        clip_model.load_state_dict(ckpt["clip"], strict=True)
        clip_model.eval()
        net = load_vqvae(args, strict=False)
        trans_encoder = load_transformer(args)
        trans_encoder.load_state_dict(ckpt["trans"], strict=True)
        ckpt_iter = int(ckpt.get("iter", 0))
        checkpoint_name = Path(spec["checkpoint"]).name
        checkpoint_path = spec["checkpoint"]
    else:
        raise ValueError(f"Unknown model kind: {spec['kind']}")

    return {
        "args": args,
        "clip_model": clip_model,
        "net": net,
        "trans_encoder": trans_encoder,
        "iter": ckpt_iter,
        "checkpoint": checkpoint_name,
        "path": checkpoint_path,
    }


def read_rows(csv_path):
    path = Path(csv_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_row(csv_path, row):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def validate_existing_rows(
    rows,
    spec,
    bundle,
    seeds,
    model_dir,
    require_sample_index,
    expected_sample_index_sha256=None,
    generation_audit_dir=None,
    retrieval_embedding_dir=None,
):
    """Reject stale/corrupt append-only state before skipping completed seeds."""
    expected_seeds = set(seeds)
    allowed_seeds = (
        set(range(123, 143))
        if expected_sample_index_sha256 is not None
        else expected_seeds
    )
    expected_path = Path(resolve_repo_path(bundle["path"])).resolve()
    seen = set()
    for row_number, row in enumerate(rows, start=2):
        missing = [field for field in RUN_FIELDS if field not in row]
        if missing:
            raise ValueError(
                f"Malformed existing row {row_number} in {model_dir / 'runs.csv'}: "
                f"missing columns {missing}"
            )
        seed = int(row["seed"])
        if seed not in allowed_seeds:
            raise ValueError(
                f"Unexpected existing seed {seed} in {model_dir / 'runs.csv'}; "
                f"expected {sorted(allowed_seeds)}"
            )
        if seed in seen:
            raise ValueError(
                f"Duplicate existing seed {seed} in {model_dir / 'runs.csv'}"
            )
        seen.add(seed)
        expected_run_idx = (
            seed - 122
            if expected_sample_index_sha256 is not None
            else seeds.index(seed) + 1
        )
        actual_path = Path(resolve_repo_path(row["path"])).resolve()
        expected_values = {
            "model": spec["name"],
            "kind": spec["kind"],
            "run_idx": expected_run_idx,
            "checkpoint": bundle["checkpoint"],
            "iter": bundle["iter"],
            "path": expected_path,
        }
        actual_values = {
            "model": row["model"],
            "kind": row["kind"],
            "run_idx": int(row["run_idx"]),
            "checkpoint": row["checkpoint"],
            "iter": int(row["iter"]),
            "path": actual_path,
        }
        if actual_values != expected_values:
            raise ValueError(
                f"Stale or incompatible existing row for seed {seed} in "
                f"{model_dir / 'runs.csv'}: expected {expected_values}, "
                f"got {actual_values}"
            )
        if require_sample_index:
            index_path = model_dir / "sample_indices" / f"seed_{seed}.json"
            if not index_path.is_file() or not row["sample_index_sha256"]:
                raise ValueError(
                    f"Existing seed {seed} lacks its sample-index evidence: "
                    f"{index_path}"
                )
            if (
                expected_sample_index_sha256 is not None
                and row["sample_index_sha256"] != expected_sample_index_sha256
            ):
                raise ValueError(
                    f"Existing seed {seed} uses a different frozen sample index"
                )
        if generation_audit_dir:
            audit_path = (
                Path(generation_audit_dir) / spec["name"] / f"seed{seed}.json"
            )
            if not audit_path.is_file():
                raise ValueError(
                    f"Existing seed {seed} lacks its generation audit: {audit_path}"
                )
        if retrieval_embedding_dir:
            embedding_path = (
                Path(retrieval_embedding_dir) / spec["name"] / f"seed{seed}.npz"
            )
            if not embedding_path.is_file():
                raise ValueError(
                    f"Existing seed {seed} lacks its retrieval embeddings: "
                    f"{embedding_path}"
                )


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_all_rows(out_dir, specs):
    rows = []
    for spec in specs:
        rows.extend(read_rows(Path(out_dir) / spec["name"] / "runs.csv"))
    return rows


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)

    aggregate = []
    wide = []
    for model_name, model_rows in grouped.items():
        base = {
            "model": model_name,
            "kind": model_rows[0]["kind"],
            "checkpoint": model_rows[0]["checkpoint"],
            "path": model_rows[0]["path"],
            "n": len(model_rows),
        }
        wide_row = dict(base)
        for metric in METRIC_FIELDS:
            values = np.array([float(row[metric]) for row in model_rows], dtype=np.float64)
            mean = float(values.mean())
            population_std = float(values.std(ddof=0))
            population_var = float(values.var(ddof=0))
            sample_std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            sample_var = float(values.var(ddof=1)) if len(values) > 1 else 0.0
            ci95 = float(1.96 * population_std / math.sqrt(len(values)))
            min_value = float(values.min())
            max_value = float(values.max())
            aggregate.append(
                {
                    **base,
                    "metric": metric,
                    "mean": mean,
                    "std": population_std,
                    "var": population_var,
                    "population_std": population_std,
                    "population_var": population_var,
                    "sample_std": sample_std,
                    "sample_var": sample_var,
                    "ci95": ci95,
                    "min": min_value,
                    "max": max_value,
                }
            )
            wide_row[f"{metric}_mean"] = mean
            wide_row[f"{metric}_std"] = population_std
            wide_row[f"{metric}_var"] = population_var
            wide_row[f"{metric}_population_std"] = population_std
            wide_row[f"{metric}_population_var"] = population_var
            wide_row[f"{metric}_sample_std"] = sample_std
            wide_row[f"{metric}_sample_var"] = sample_var
            wide_row[f"{metric}_ci95"] = ci95
        wide.append(wide_row)
    return aggregate, wide


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def write_report(out_dir, report_md, specs, rows, aggregate, wide, args):
    completed = len(rows)
    expected = len(specs) * args.num_repeats
    by_model = {row["model"]: row for row in wide}

    lines = [
        "# Repeat20 Perturbation Fast-Batch Evaluation",
        "",
        "## Status",
        "",
        f"- Output directory: `{args.out_dir}`",
        f"- Pair file: `{args.pair_data_path}`",
        f"- Perturbed-text tokenizer: `{args.spacy_model}` with reference TM2T GloVe+POS preprocessing (hyphen removal, POS tagging, and noun/verb lemmatization)",
        f"- Dataset split: `{getattr(args, 'dataset_split', 'test')}`",
        f"- DataLoader workers: `{getattr(args, 'num_workers', 8)}`",
        f"- Seeds: `{args.start_seed}` to `{args.start_seed + args.num_repeats - 1}`",
        f"- Fast batch: `{args.fast_batch}`",
        f"- Caption locked: `{args.caption_locked}`",
        f"- Caption-locked length policy: `{args.caption_locked_length_policy}`",
        f"- Expected sample index: `{args.expected_sample_index}`",
        f"- HumanML fast protocol: `{args.humanml_fast_protocol}`",
        f"- CFG scale: `{args.cfg_scale}`",
        f"- CFG uncond: `{args.cfg_uncond}`",
        f"- Sample mode: `{args.sample_mode}`",
        f"- Sample temperature: `{args.sample_temperature}`",
        f"- Top-k: `{args.top_k}`",
        f"- Top-p: `{args.top_p}`",
    ]
    per_model_cfg = [
        f"`{spec['name']}`={spec.get('cfg_scale', args.cfg_scale)}/{spec.get('cfg_uncond', args.cfg_uncond)}"
        for spec in specs
    ]
    lines.extend([f"- Per-model cfg: {', '.join(per_model_cfg)}", ""])
    lines.extend(
        [
            f"- Completed runs: `{completed}/{expected}`",
            f"- Intended GPU: `CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}`",
            "",
            "## Models",
            "",
            "| Model | Kind | Checkpoint | Completed |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for spec in specs:
        model_rows = [row for row in rows if row["model"] == spec["name"]]
        checkpoint = spec.get("checkpoint", spec.get("resume_trans", ""))
        lines.append(
            f"| `{spec['name']}` | `{spec['kind']}` | `{checkpoint}` | "
            f"`{len(model_rows)}/{args.num_repeats}` |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Key Metrics",
            "",
            "| Model | n | FID mean±std | FID_P mean±std | FID_D mean±std | Matching mean±std |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for spec in specs:
        row = by_model.get(spec["name"])
        if not row:
            lines.append(f"| `{spec['name']}` | 0 | - | - | - | - |")
            continue
        lines.append(
            f"| `{spec['name']}` | {row['n']} | "
            f"{fmt(row['fid_mean'])}±{fmt(row['fid_std'])} | "
            f"{fmt(row['fid_p_mean'])}±{fmt(row['fid_p_std'])} | "
            f"{fmt(row['fid_d_mean'])}±{fmt(row['fid_d_std'])} | "
            f"{fmt(row['matching_mean'])}±{fmt(row['matching_std'])} |"
        )

    lines.extend(
        [
            "",
            "## Perturbed-Text Retrieval",
            "",
            "Each perturbed-caption TM2T embedding is retrieved against the motion generated from that same perturbed caption, within the unchanged evaluator batch.",
            "",
            "| Model | n | R@1_P mean±std | R@2_P mean±std | R@3_P mean±std | Matching_P mean±std |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for spec in specs:
        row = by_model.get(spec["name"])
        if not row:
            lines.append(f"| `{spec['name']}` | 0 | - | - | - | - |")
            continue
        lines.append(
            f"| `{spec['name']}` | {row['n']} | "
            f"{fmt(row['top1_pert_mean'])}±{fmt(row['top1_pert_std'])} | "
            f"{fmt(row['top2_pert_mean'])}±{fmt(row['top2_pert_std'])} | "
            f"{fmt(row['top3_pert_mean'])}±{fmt(row['top3_pert_std'])} | "
            f"{fmt(row['matching_pert_mean'])}±{fmt(row['matching_pert_std'])} |"
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- All single-run metrics: `{Path(out_dir) / 'all_runs.csv'}`",
            f"- Long aggregate table: `{Path(out_dir) / 'aggregate.csv'}`",
            f"- Wide aggregate table: `{Path(out_dir) / 'summary_wide.csv'}`",
            f"- JSON aggregate: `{Path(out_dir) / 'summary.json'}`",
            f"- Run log: `{Path(out_dir) / 'run.log'}`",
            "",
            "The `std` and `var` columns use population statistics (`ddof=0`); explicit sample-statistic columns are retained for compatibility.",
            "The `ci95` columns use the normal half-width `1.96 * population_std / sqrt(n)`.",
        ]
    )

    Path(report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(report_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(out_dir, "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(out_dir, report_md, specs, args):
    rows = collect_all_rows(out_dir, specs)
    rows = sorted(rows, key=lambda r: (r["model"], int(r["run_idx"])))
    write_csv(Path(out_dir) / "all_runs.csv", RUN_FIELDS, rows)
    aggregate, wide = aggregate_rows(rows)
    aggregate_fields = [
        "model",
        "kind",
        "checkpoint",
        "path",
        "n",
        "metric",
        "mean",
        "std",
        "var",
        "population_std",
        "population_var",
        "sample_std",
        "sample_var",
        "ci95",
        "min",
        "max",
    ]
    write_csv(Path(out_dir) / "aggregate.csv", aggregate_fields, aggregate)
    wide_fields = ["model", "kind", "checkpoint", "path", "n"]
    for metric in METRIC_FIELDS:
        wide_fields.extend([
            f"{metric}_mean",
            f"{metric}_std",
            f"{metric}_var",
            f"{metric}_population_std",
            f"{metric}_population_var",
            f"{metric}_sample_std",
            f"{metric}_sample_var",
            f"{metric}_ci95",
        ])
    write_csv(Path(out_dir) / "summary_wide.csv", wide_fields, wide)
    with open(Path(out_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "statistics": {
                "std_ddof": 0,
                "variance_ddof": 0,
                "sample_std_ddof": 1,
                "sample_variance_ddof": 1,
                "ci95_definition": "1.96 * population_std / sqrt(n)",
            },
            "runs": rows,
            "aggregate": aggregate,
            "summary_wide": wide,
        }, f, indent=2)
    write_report(out_dir, report_md, specs, rows, aggregate, wide, args)
    return rows, aggregate, wide


def evaluate_model_seed(spec, bundle, seed, run_idx, args, pair_data,
                        eval_wrapper, w_vectorizer,
                        perturbed_text_processor, logger):
    set_seed(seed if args.loader_seed is None else args.loader_seed)

    cfg_scale = float(spec.get("cfg_scale", args.cfg_scale))
    cfg_uncond = spec.get("cfg_uncond", args.cfg_uncond)
    sample_mode = spec.get("sample_mode", args.sample_mode)
    sample_categorical = sample_mode == "categorical"
    sample_temperature = float(spec.get("sample_temperature", args.sample_temperature))
    top_k = int(spec.get("top_k", args.top_k))
    top_p = float(spec.get("top_p", args.top_p))

    if cfg_scale <= 0:
        raise ValueError(f"Invalid cfg_scale for {spec['name']}: {cfg_scale}")

    is_test = getattr(args, "dataset_split", "test") == "test"
    num_workers = int(getattr(args, "num_workers", 8))
    if args.caption_locked:
        val_loader = dataset_TM_eval.CaptionLockedDATALoader(
            bundle["args"].dataname,
            is_test,
            args.batch_size,
            w_vectorizer,
            pair_data,
            num_workers=num_workers,
            length_policy=getattr(args, "caption_locked_length_policy", "strict"),
        )
        logger.info(
            "Caption-locked loader stats: %s",
            json.dumps(val_loader.dataset.resolution_stats, sort_keys=True),
        )
    else:
        val_loader = dataset_TM_eval.DATALoader(
            bundle["args"].dataname,
            is_test,
            args.batch_size,
            w_vectorizer,
            num_workers=num_workers,
        )
    expected_samples = (
        len(val_loader.dataset)
        if not val_loader.drop_last
        else len(val_loader) * int(args.batch_size)
    )
    if args.sample_limit is not None:
        expected_samples = min(expected_samples, int(args.sample_limit))
    start = time.time()
    diagnostics_output = None
    if args.diagnostics_dir:
        diagnostics_output = (
            Path(args.diagnostics_dir) / spec["name"] / f"seed{seed}.npz"
        )
    retrieval_embedding_output = None
    if args.retrieval_embedding_dir:
        retrieval_embedding_output = (
            Path(args.retrieval_embedding_dir) / spec["name"] / f"seed{seed}.npz"
        )
    generation_audit_output = None
    if args.generation_audit_dir:
        generation_audit_output = (
            Path(args.generation_audit_dir) / spec["name"] / f"seed{seed}.json"
        )
    metrics, _, _, _, _ = sato_pp_eval.evaluation_sato_pp(
        args.out_dir,
        val_loader,
        bundle["net"],
        bundle["trans_encoder"],
        bundle["clip_model"],
        eval_wrapper,
        logger,
        NullWriter(),
        bundle["iter"],
        pair_data=pair_data,
        sample_categorical=sample_categorical,
        use_fast_batch=args.fast_batch,
        cfg_scale=cfg_scale,
        cfg_uncond=cfg_uncond,
        sample_temperature=sample_temperature,
        top_k=top_k,
        top_p=top_p,
        sampling_seed=(
            None if (args.humanml_fast_protocol or args.caption_locked) else seed
        ),
        caption_locked=args.caption_locked,
        collect_sample_index=(args.humanml_fast_protocol or args.caption_locked),
        diagnostics_output=diagnostics_output,
        retrieval_embedding_output=retrieval_embedding_output,
        retrieval_protocol_fingerprint=getattr(
            args, "_protocol_fingerprint", None
        ),
        retrieval_sample_index_sha256=getattr(
            args, "_expected_sample_index_sha256", None
        ),
        retrieval_seed=seed,
        retrieval_batch_size=args.batch_size,
        generation_audit_output=generation_audit_output,
        generation_audit_seed=seed,
        max_samples=args.sample_limit,
        sampling_pair_rng_mode=args.sampling_pair_rng_mode,
        expected_sample_index=getattr(args, "_expected_sample_index", None),
        perturbed_text_processor=perturbed_text_processor,
    )
    strict_shared_loader = args.humanml_fast_protocol or args.caption_locked
    if strict_shared_loader and (
        int(metrics["matched_pairs"]) != expected_samples
        or int(metrics["total_pairs"]) != expected_samples
        or int(metrics["perturbed_retrieval_samples"]) != expected_samples
    ):
        raise RuntimeError(
            "Formal fast evaluation requires matched_pairs=total_pairs="
            "perturbed_retrieval_samples="
            f"{expected_samples} for {bundle['args'].dataname}, got "
            f"{metrics['matched_pairs']}/{metrics['total_pairs']}/"
            f"{metrics['perturbed_retrieval_samples']}"
        )
    sample_index = metrics.pop("sample_index", None)
    sample_index_sha256 = metrics.pop("sample_index_sha256", "")
    if strict_shared_loader:
        if not isinstance(sample_index, list) or len(sample_index) != expected_samples:
            raise RuntimeError(
                f"Expected {expected_samples} sample-index records, got "
                f"{0 if sample_index is None else len(sample_index)}")
        sample_index_path = (
            Path(args.out_dir) / spec["name"] / "sample_indices"
            / f"seed_{seed}.json"
        )
        sample_index_path.parent.mkdir(parents=True, exist_ok=True)
        sample_index_path.write_text(
            json.dumps({
                "model": spec["name"],
                "seed": seed,
                "sample_count": expected_samples,
                "sample_index_sha256": sample_index_sha256,
                "samples": sample_index,
            }, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    elapsed = time.time() - start
    torch.cuda.empty_cache()

    return {
        "model": spec["name"],
        "kind": spec["kind"],
        "run_idx": run_idx,
        "seed": seed,
        "checkpoint": bundle["checkpoint"],
        "iter": bundle["iter"],
        "fid": metrics["fid"],
        "fid_p": metrics["fid_p"],
        "fid_d": metrics["fid_d"],
        "diversity_real": metrics["diversity_real"],
        "diversity": metrics["diversity"],
        "top1": metrics["top1"],
        "top2": metrics["top2"],
        "top3": metrics["top3"],
        "matching": metrics["matching"],
        "top1_pert": metrics["top1_pert"],
        "top2_pert": metrics["top2_pert"],
        "top3_pert": metrics["top3_pert"],
        "matching_pert": metrics["matching_pert"],
        "perturbed_retrieval_samples": metrics["perturbed_retrieval_samples"],
        "matched_pairs": metrics["matched_pairs"],
        "total_pairs": metrics["total_pairs"],
        "sample_index_sha256": sample_index_sha256,
        "elapsed_sec": elapsed,
        "path": bundle["path"],
    }


def main():
    args = parse_args()
    args._expected_sample_index = None
    args._expected_sample_index_sha256 = None
    args._protocol_fingerprint = None
    if args.expected_sample_index:
        if not args.caption_locked:
            raise ValueError("--expected-sample-index requires --caption-locked")
        args._expected_sample_index, args._expected_sample_index_sha256 = (
            load_expected_sample_index(args.expected_sample_index)
        )
        if args.loader_seed is not None:
            raise ValueError(
                "fixed-index formal evaluation must use each run seed for generation; "
                "do not set --loader-seed"
            )
        if args.sampling_pair_rng_mode != "sequential":
            raise ValueError(
                "fixed-index formal evaluation requires "
                "--sampling-pair-rng-mode sequential"
            )
        if args.caption_locked_length_policy != "repair-all":
            raise ValueError(
                "the repaired fixed index requires "
                "--caption-locked-length-policy repair-all"
            )
        if args.sample_limit is not None:
            raise ValueError("formal fixed-index evaluation does not allow --sample-limit")
        if args.allowed_levels:
            expected_levels = set(args.allowed_levels)
            if not expected_levels or not expected_levels.issubset({"P1", "P2", "P3"}):
                raise ValueError(
                    "fixed-index evaluation --allowed-levels must be a non-empty "
                    "subset of P1/P2/P3"
                )
        if not args.fast_batch:
            raise ValueError("formal fixed-index evaluation requires --fast-batch")
        if (
            args.sample_mode != "categorical"
            or float(args.sample_temperature) != 1.0
            or int(args.top_k) != 0
            or float(args.top_p) != 1.0
        ):
            raise ValueError(
                "formal fixed-index evaluation requires categorical sampling, "
                "temperature=1, top_k=0, top_p=1"
            )
    if args.humanml_fast_protocol:
        if args.caption_locked:
            raise ValueError(
                "--humanml-fast-protocol cannot be combined with --caption-locked"
            )
        args.fast_batch = True
    if args.generation_audit_dir and (
        not args.fast_batch or args._expected_sample_index is None
    ):
        raise ValueError(
            "--generation-audit-dir requires formal fixed-index --fast-batch evaluation"
        )
    if args.retrieval_embedding_dir and (
        not args.fast_batch or args._expected_sample_index is None
    ):
        raise ValueError(
            "--retrieval-embedding-dir requires formal fixed-index --fast-batch evaluation"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(out_dir)
    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled":
        wandb_config = {
            key: value for key, value in vars(args).items()
            if not key.startswith("_")
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or out_dir.name,
            group=args.wandb_group,
            tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()],
            job_type="evaluation",
            config=wandb_config,
            dir=out_dir,
            mode=args.wandb_mode,
            resume="allow",
        )

    specs = load_specs(args.spec_json) if args.spec_json else default_specs()
    if args.external_sato_name:
        if not (args.external_sato_vq and args.external_sato_trans and args.external_sato_clip):
            raise ValueError(
                "--external-sato-name requires --external-sato-vq, "
                "--external-sato-trans, and --external-sato-clip"
            )
        specs.append(
            {
                "name": args.external_sato_name,
                "kind": "baseline",
                "resume_pth": args.external_sato_vq,
                "resume_trans": args.external_sato_trans,
                "clip_pth": args.external_sato_clip,
                "description": (
                    "External SATO checkpoint with separate VQ, transformer, "
                    "and CLIP weights"
                ),
            }
        )
    if args.model:
        wanted = set(args.model)
        specs = [spec for spec in specs if spec["name"] in wanted]
        missing = wanted - {spec["name"] for spec in specs}
        if missing:
            raise ValueError(f"Unknown requested model(s): {sorted(missing)}")
    if args._expected_sample_index is not None:
        expected_cfg_scale = float(args.cfg_scale)
        if expected_cfg_scale <= 0:
            raise ValueError("fixed-index comparison requires a positive CFG scale")
        for spec in specs:
            cfg_scale = float(spec.get("cfg_scale", args.cfg_scale))
            if cfg_scale != expected_cfg_scale:
                raise ValueError(
                    "fixed-index comparison requires every model spec to use the "
                    f"CLI CFG scale {expected_cfg_scale}, got {cfg_scale} for "
                    f"{spec['name']}"
                )

    log_args = {key: value for key, value in vars(args).items() if not key.startswith("_")}
    logger.info("Repeat eval config: %s", json.dumps(log_args, indent=2, sort_keys=True))
    logger.info("Models: %s", json.dumps(specs, indent=2, sort_keys=True))
    write_outputs(out_dir, args.report_md, specs, args)

    pair_data = load_pair_data(args.pair_data_path)
    dataname = resolve_specs_dataname(specs)
    if args._expected_sample_index is not None:
        level_counts = validate_pair_mix_matches_index(
            pair_data,
            args._expected_sample_index,
            expected_levels=args.allowed_levels,
        )
        pair_data = pair_data_from_sample_index(args._expected_sample_index)
        logger.info(
            "Using frozen sample-index order: %d rows, sha256=%s, levels=%s",
            len(pair_data),
            args._expected_sample_index_sha256,
            json.dumps(level_counts, sort_keys=True),
        )
    if args.evaluator_checkpoint:
        evaluator_checkpoint_sha256 = hashlib.sha256(
            Path(args.evaluator_checkpoint).read_bytes()).hexdigest()
    else:
        evaluator_checkpoint_sha256 = "standard-t2m-matcher"
    if args._expected_sample_index is not None:
        protocol = {
            "schema_version": 1,
            "sample_count": len(pair_data),
            "sample_index_path": str(args.expected_sample_index),
            "sample_index_sha256": args._expected_sample_index_sha256,
            "dataset_split": args.dataset_split,
            "level_counts": level_counts,
            "caption_locked": True,
            "length_policy": "repair-all",
            "cfg_scale": float(args.cfg_scale),
            "sampling": {
                "mode": "categorical",
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
            },
            "rng": "sequential",
            "generation_order": "original-first",
            "evaluator_length": "generated_length",
            "evaluator_motion_tensor_frames": dataset_max_motion_length(dataname),
            "perturbed_text_retrieval": {
                "sample_count": len(pair_data),
                "batch_count": math.ceil(len(pair_data) / int(args.batch_size)),
                "batch_size": int(args.batch_size),
                "text_source": "resolved text_pair perturbation",
                "motion_source": (
                    "motion generated from the same resolved text_pair perturbation"
                ),
                "evaluator": (
                    "standard TM2T TextEncoderBiGRUCo/MotionEncoderBiGRUCo"
                    if not args.evaluator_checkpoint
                    else str(Path(args.evaluator_checkpoint).resolve())
                ),
                "evaluator_checkpoint_sha256": evaluator_checkpoint_sha256,
                "word_vectorizer": "glove/our_vab",
                "spacy_model": args.spacy_model,
                "max_text_len": 20,
                "alphabetic_only": not dataname.startswith("motionx322"),
                "caption_preprocessing": (
                    "strip whitespace; remove hyphens; keep alphabetic tokens; "
                    "lemmatize nouns/verbs except literal lowercase left"
                    if not dataname.startswith("motionx322")
                    else (
                        "strip whitespace; remove hyphens; skip spaces/punctuation; "
                        "lemmatize nouns/verbs except literal lowercase left"
                    )
                ),
                "retrieval_scope": "within each evaluator-loader batch",
            },
            "metric_scope": {
                "top1/top2/top3/matching": (
                    "original caption TM2T embedding vs original-caption generation"
                ),
                "top1_pert/top2_pert/top3_pert/matching_pert": (
                    "perturbed caption TM2T embedding vs perturbed-caption generation"
                ),
            },
            "models": [spec["name"] for spec in specs],
        }
        protocol["fingerprint"] = canonical_sha256(protocol)
        args._protocol_fingerprint = protocol["fingerprint"]
        (out_dir / "protocol_manifest.json").write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.allowed_levels and args._expected_sample_index is None:
        allowed_levels = set(args.allowed_levels)
        pair_data = [
            row for row in pair_data
            if str(row.get("level", "")) in allowed_levels
        ]
        if not pair_data:
            raise ValueError(
                f"No pair data remains for levels {sorted(allowed_levels)}"
            )
        logger.info(
            "Filtered pair data to levels %s: %d rows",
            sorted(allowed_levels),
            len(pair_data),
        )
    w_vectorizer = WordVectorizer("./glove", "our_vab")
    dataset_opt_path = evaluator_opt_path(dataname)
    wrapper_opt = get_opt(dataset_opt_path, torch.device("cuda"))
    wrapper_opt = configure_evaluator_source(wrapper_opt, dataname)
    if args.evaluator_checkpoint:
        wrapper_opt.evaluator_matcher_checkpoint = str(
            Path(args.evaluator_checkpoint).resolve())
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    perturbed_text_processor = sato_pp_eval.TM2TTextProcessor(
        w_vectorizer,
        max_text_len=int(getattr(eval_wrapper.opt, "max_text_len", 20)),
        spacy_model=args.spacy_model,
        alphabetic_only=not dataname.startswith("motionx322"),
    )
    seeds = [args.start_seed + idx for idx in range(args.num_repeats)]

    for spec in specs:
        model_dir = out_dir / spec["name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        existing_rows = read_rows(model_dir / "runs.csv")

        logger.info("Loading model %s", spec["name"])
        bundle = load_model_bundle(spec)
        logger.info(
            "Loaded model %s checkpoint %s iter %s",
            spec["name"],
            bundle["path"],
            bundle["iter"],
        )
        validate_existing_rows(
            existing_rows,
            spec,
            bundle,
            seeds,
            model_dir,
            args.humanml_fast_protocol or args.caption_locked,
            expected_sample_index_sha256=args._expected_sample_index_sha256,
            generation_audit_dir=args.generation_audit_dir,
            retrieval_embedding_dir=args.retrieval_embedding_dir,
        )
        completed_seeds = {int(row["seed"]) for row in existing_rows}

        for sequence_run_idx, seed in enumerate(seeds, start=1):
            run_idx = (
                seed - 122
                if args._expected_sample_index_sha256 is not None
                else sequence_run_idx
            )
            if seed in completed_seeds:
                logger.info("Skip completed %s seed %s", spec["name"], seed)
                continue

            logger.info(
                "RepeatEval start model=%s run=%d/%d seed=%d",
                spec["name"],
                run_idx,
                args.num_repeats,
                seed,
            )
            row = evaluate_model_seed(
                spec,
                bundle,
                seed,
                run_idx,
                args,
                pair_data,
                eval_wrapper,
                w_vectorizer,
                perturbed_text_processor,
                logger,
            )
            append_row(model_dir / "runs.csv", row)
            if wandb_run is not None:
                wandb_run.log({
                    "eval/fid": float(row["fid"]),
                    "eval/fid_p": float(row["fid_p"]),
                    "eval/fid_d": float(row["fid_d"]),
                    "eval/diversity": float(row["diversity"]),
                    "eval/top1": float(row["top1"]),
                    "eval/top2": float(row["top2"]),
                    "eval/top3": float(row["top3"]),
                    "eval/matching": float(row["matching"]),
                    "eval/top1_pert": float(row["top1_pert"]),
                    "eval/top2_pert": float(row["top2_pert"]),
                    "eval/top3_pert": float(row["top3_pert"]),
                    "eval/matching_pert": float(row["matching_pert"]),
                    "eval/seed": int(seed),
                    "eval/model": spec["name"],
                    "eval/run_idx": run_idx,
                })
            completed_seeds.add(seed)
            logger.info(
                "RepeatEval done model=%s run=%d seed=%d FID=%.4f FID_P=%.4f "
                "FID_D=%.4f R1_P=%.4f Matching_P=%.4f matched=%s/%s elapsed=%.1fs",
                spec["name"],
                run_idx,
                seed,
                float(row["fid"]),
                float(row["fid_p"]),
                float(row["fid_d"]),
                float(row["top1_pert"]),
                float(row["matching_pert"]),
                row["matched_pairs"],
                row["total_pairs"],
                float(row["elapsed_sec"]),
            )
            write_outputs(out_dir, args.report_md, specs, args)

        del bundle
        torch.cuda.empty_cache()

    rows, aggregate, wide = write_outputs(out_dir, args.report_md, specs, args)
    logger.info("Repeat eval complete: %d rows, %d aggregate rows", len(rows), len(aggregate))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
