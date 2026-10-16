#!/usr/bin/env python
"""Train the Motion-X text-motion evaluator on native motion parameters."""

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from dataset.config import dataset_max_motion_length, get_dataset_spec
from dataset.variable_motion import (
    DistributedLengthBucketBatchSampler,
    downsample_lengths,
    lengths_to_mask,
    pad_motion_arrays,
)

from models.modules import (
    MotionEncoderBiGRUCo,
    MovementConvDecoder,
    MovementConvEncoder,
    TextEncoderBiGRUCo,
)
from utils.distributed_training import barrier, cleanup, setup_distributed
from utils.word_vectorizer import POS_enumerator, WordVectorizer
from utils.plateau_control import PlateauController


DECOMP_NAME = "Decomp_SP001_SM001_H512"
MATCH_NAME = "text_mot_match"
COMP_NAME = "Comp_v6_KLD005"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("decomp", "match", "validate"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset/Motion-X"))
    parser.add_argument("--dataset-key", default="motionx")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-ids", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-project", default="SATO++-MotionX-Ablation")
    parser.add_argument("--wandb-group", default="MotionX-Evaluator-263D-seed123")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-watch-log", choices=("all", "gradients", "parameters", "none"), default="gradients")
    parser.add_argument("--wandb-watch-freq", type=int, default=200)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--feat-bias", type=float, default=5.0)
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--max-motion-length", type=int, default=196)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--unit-length", type=int, default=4)
    parser.add_argument("--max-text-len", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=None)

    parser.add_argument("--decomp-global-batch", type=int, default=128)
    parser.add_argument("--decomp-lr-milestones", type=int, nargs="*", default=(60, 80))
    parser.add_argument("--decomp-lr-gamma", type=float, default=0.5)
    parser.add_argument("--decomp-min-epochs", type=int, default=60)
    parser.add_argument("--decomp-patience", type=int, default=10)
    parser.add_argument("--lambda-sparsity", type=float, default=0.001)
    parser.add_argument("--lambda-smooth", type=float, default=0.001)

    parser.add_argument("--match-global-batch", type=int, default=32)
    parser.add_argument("--match-lr-milestones", type=int, nargs="*", default=(40, 80))
    parser.add_argument("--match-lr-gamma", type=float, default=0.5)
    parser.add_argument("--negative-margin", type=float, default=10.0)
    parser.add_argument("--min-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr-policy", choices=("fixed", "plateau"), default="fixed")
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--plateau-min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--selection-max-distance-ratio", type=float, default=0.75)
    parser.add_argument("--validation-retrieval-every", type=int, default=5)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_logger(output_dir, rank):
    logger = logging.getLogger(f"motionx_evaluator.rank{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)
        file_handler = logging.FileHandler(output_dir / "train.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def load_split_ids(root, split, limit=0):
    ids = [line.strip() for line in (root / f"{split}.txt").read_text().splitlines() if line.strip()]
    return ids[:limit] if limit else ids


def evaluator_stats(root, feat_bias, dataset_key="motionx"):
    spec = get_dataset_spec(dataset_key)
    mean = np.load(root / "Mean.npy", allow_pickle=False).astype(np.float32).copy()
    std = np.load(root / "Std.npy", allow_pickle=False).astype(np.float32).copy()
    expected = (spec.dim_pose,)
    if mean.shape != expected or std.shape != expected:
        raise ValueError(f"Expected {expected} Motion-X statistics, got {mean.shape} and {std.shape}")
    for start, end in spec.evaluator_bias_slices:
        stop = None if end == 0 else end
        std[start:stop] /= feat_bias
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Evaluator Mean/Std contain invalid values")
    return mean, std


class MotionWindowDataset(Dataset):
    def __init__(self, root, split, mean, std, window_size=40, limit=0):
        self.window_size = window_size
        self.motions = []
        lengths = []
        for motion_id in load_split_ids(root, split, limit):
            motion = np.load(root / "new_joint_vecs" / f"{motion_id}.npy", allow_pickle=False)
            if motion.ndim != 2 or motion.shape[1] != 263 or motion.shape[0] < window_size:
                continue
            motion = ((motion.astype(np.float32) - mean) / std).astype(np.float32)
            self.motions.append(motion)
            lengths.append(motion.shape[0] - window_size)
        if not self.motions or sum(lengths) == 0:
            raise RuntimeError(f"No usable {split} motion windows")
        self.cumsum = np.cumsum([0] + lengths, dtype=np.int64)

    def __len__(self):
        return int(self.cumsum[-1])

    def __getitem__(self, item):
        motion_id = int(np.searchsorted(self.cumsum[1:], item, side="right"))
        offset = int(item - self.cumsum[motion_id])
        return self.motions[motion_id][offset:offset + self.window_size]


class MotionSequenceEvaluatorDataset(Dataset):
    def __init__(self, root, split, mean, std, dim_pose, limit=0):
        self.rows = []
        self.lengths = []
        for motion_id in load_split_ids(root, split, limit):
            path = root / "new_joint_vecs" / f"{motion_id}.npy"
            if not path.is_file():
                continue
            shape = np.load(path, mmap_mode="r", allow_pickle=False).shape
            if len(shape) != 2 or shape[1] != dim_pose or shape[0] < 1:
                continue
            self.rows.append(path)
            self.lengths.append(int(shape[0]))
        if not self.rows:
            raise RuntimeError(f"No usable {split} native motion sequences")
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        motion = np.load(self.rows[index], allow_pickle=False).astype(np.float32)
        return ((motion - self.mean) / self.std).astype(np.float32)


def collate_decomp_variable(batch, unit_length=4):
    motion, lengths, frame_mask = pad_motion_arrays(batch, multiple=unit_length)
    latent_lengths = torch.div(
        lengths + unit_length - 1, unit_length, rounding_mode="floor")
    latent_mask = lengths_to_mask(latent_lengths, motion.shape[1] // unit_length)
    return motion, lengths, frame_mask, latent_mask


class TextMotionDataset(Dataset):
    def __init__(self, root, split, mean, std, vectorizer, max_text_len=20,
                 max_motion_length=196, unit_length=4, limit=0,
                 dataset_key="motionx"):
        self.vectorizer = vectorizer
        self.max_text_len = max_text_len
        self.max_motion_length = max_motion_length
        self.unit_length = unit_length
        self.spec = get_dataset_spec(dataset_key)
        self.variable_length = self.spec.variable_length
        rows = []
        for motion_id in load_split_ids(root, split, limit):
            motion_path = root / "new_joint_vecs" / f"{motion_id}.npy"
            text_path = root / "texts" / f"{motion_id}.txt"
            try:
                motion = np.load(motion_path, allow_pickle=False).astype(np.float32)
                if motion.ndim != 2 or motion.shape[1] != self.spec.dim_pose:
                    continue
                if len(motion) < self.spec.min_motion_len:
                    continue
                if not self.variable_length and len(motion) >= 200:
                    continue
                full_captions = []
                segment_rows = []
                for raw_line in text_path.read_text(encoding="utf-8").splitlines():
                    parts = raw_line.strip().split("#")
                    if len(parts) < 4:
                        continue
                    caption, tokens = parts[0], [token for token in parts[1].split() if token]
                    from_tag, to_tag = float(parts[2]), float(parts[3])
                    from_tag = 0.0 if np.isnan(from_tag) else from_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag
                    text = (caption, tokens)
                    if from_tag == 0.0 and to_tag == 0.0:
                        full_captions.append(text)
                    else:
                        clip = motion[
                            int(from_tag * self.spec.fps):int(to_tag * self.spec.fps)]
                        if len(clip) >= self.spec.min_motion_len and (
                                self.variable_length or len(clip) < 200):
                            segment_rows.append((clip, [text]))
                normalized = ((motion - mean) / std).astype(np.float32)
                if full_captions:
                    rows.append((normalized, full_captions))
                for clip, captions in segment_rows:
                    rows.append((((clip - mean) / std).astype(np.float32), captions))
            except (OSError, ValueError, UnicodeError):
                continue
        rows.sort(key=lambda row: len(row[0]))
        self.rows = rows
        self.lengths = [len(row[0]) for row in rows]
        if not rows:
            raise RuntimeError(f"No usable {split} text-motion rows")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        motion, captions = self.rows[index]
        caption, tokens = random.choice(captions)
        tokens = [token if token.count("/") == 1 else "unk/OTHER" for token in tokens]
        if len(tokens) < self.max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens += ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = ["sos/OTHER"] + tokens[:self.max_text_len] + ["eos/OTHER"]
            sent_len = len(tokens)
        word_embeddings, pos_one_hots = zip(*(self.vectorizer[token] for token in tokens))
        word_embeddings = np.stack(word_embeddings).astype(np.float32)
        pos_one_hots = np.stack(pos_one_hots).astype(np.float32)

        motion_length = len(motion)
        if self.variable_length:
            pass
        elif self.unit_length < 10 and np.random.choice(("single", "single", "double")) == "double":
            motion_length = (motion_length // self.unit_length - 1) * self.unit_length
        else:
            motion_length = (motion_length // self.unit_length) * self.unit_length
        motion_length = max(self.unit_length, motion_length)
        if not self.variable_length:
            start = random.randint(0, len(motion) - motion_length)
            motion = motion[start:start + motion_length]
        if not self.variable_length and motion_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_motion_length - motion_length, motion.shape[1]), dtype=np.float32),
            ])
        return word_embeddings, pos_one_hots, caption, sent_len, motion, motion_length, "_".join(tokens)


def collate_text_motion(batch):
    batch.sort(key=lambda item: item[3], reverse=True)
    return torch.utils.data.default_collate(batch)


def collate_text_motion_variable(batch, unit_length=4):
    batch.sort(key=lambda item: item[3], reverse=True)
    word, pos, captions, sent_lengths, motions, _, tokens = zip(*batch)
    padded, motion_lengths, _ = pad_motion_arrays(motions, multiple=unit_length)
    return (
        torch.as_tensor(np.stack(word)), torch.as_tensor(np.stack(pos)),
        list(captions), torch.as_tensor(sent_lengths), padded,
        motion_lengths, list(tokens),
    )


class DecompModel(nn.Module):
    def __init__(self, dim_pose=263, drop_tail=4):
        super().__init__()
        self.drop_tail = int(drop_tail)
        self.movement_encoder = MovementConvEncoder(dim_pose - self.drop_tail, 512, 512)
        self.movement_decoder = MovementConvDecoder(512, 512, dim_pose)

    def forward(self, motion):
        movement_input = motion if self.drop_tail == 0 else motion[..., :-self.drop_tail]
        latent = self.movement_encoder(movement_input)
        return self.movement_decoder(latent), latent


class MatcherModel(nn.Module):
    def __init__(self, device, dim_pose=263, drop_tail=4,
                 variable_length=False, unit_length=4):
        super().__init__()
        self.drop_tail = int(drop_tail)
        self.variable_length = bool(variable_length)
        self.unit_length = int(unit_length)
        self.movement_encoder = MovementConvEncoder(dim_pose - self.drop_tail, 512, 512)
        self.text_encoder = TextEncoderBiGRUCo(300, len(POS_enumerator), 512, 512, device)
        self.motion_encoder = MotionEncoderBiGRUCo(512, 1024, 512, device)

    def forward(self, word_embeddings, pos_one_hots, caption_lengths, motions, motion_lengths):
        align_idx = np.argsort(motion_lengths.tolist())[::-1].copy()
        motions = motions[align_idx]
        motion_lengths = motion_lengths[align_idx]
        with torch.no_grad():
            movement_input = motions if self.drop_tail == 0 else motions[..., :-self.drop_tail]
            movements = self.movement_encoder(movement_input)
        movement_lengths = downsample_lengths(
            motion_lengths, self.unit_length,
            include_partial=self.variable_length).to(motion_lengths.device)
        motion_embeddings = self.motion_encoder(
            movements.detach(), movement_lengths)
        text_embeddings = self.text_encoder(word_embeddings, pos_one_hots, caption_lengths)
        text_embeddings = text_embeddings[align_idx]
        return text_embeddings, motion_embeddings


def decomp_losses(reconstruction, motion, latent, lambda_sparsity, lambda_smooth,
                  frame_mask=None, latent_mask=None):
    if frame_mask is None:
        reconstruction_loss = F.l1_loss(reconstruction, motion)
    else:
        mask = frame_mask.to(reconstruction.dtype).unsqueeze(-1)
        reconstruction_loss = (
            (reconstruction - motion).abs() * mask).sum()
        reconstruction_loss = reconstruction_loss / (
            mask.sum() * motion.shape[-1]).clamp_min(1.0)
    if latent_mask is None:
        sparsity_loss = latent.abs().mean()
        smoothness_loss = F.l1_loss(latent[:, 1:], latent[:, :-1])
    else:
        mask = latent_mask.to(latent.dtype).unsqueeze(-1)
        sparsity_loss = (latent.abs() * mask).sum() / (
            mask.sum() * latent.shape[-1]).clamp_min(1.0)
        pair_mask = (latent_mask[:, 1:] & latent_mask[:, :-1]).to(latent.dtype).unsqueeze(-1)
        smoothness_loss = (
            (latent[:, 1:] - latent[:, :-1]).abs() * pair_mask).sum()
        smoothness_loss = smoothness_loss / (
            pair_mask.sum() * latent.shape[-1]).clamp_min(1.0)
    total = reconstruction_loss + lambda_sparsity * sparsity_loss + lambda_smooth * smoothness_loss
    return total, reconstruction_loss, sparsity_loss, smoothness_loss


def decomp_learning_rate(base_lr, completed_epoch, milestones, gamma):
    decay_count = sum(completed_epoch >= milestone for milestone in milestones)
    return base_lr * gamma ** decay_count


def set_optimizer_lr(optimizers, learning_rate):
    for optimizer in optimizers:
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate


def contrastive_losses(text_embeddings, motion_embeddings, margin, shift):
    positive_distance = F.pairwise_distance(text_embeddings, motion_embeddings)
    negative_motion = motion_embeddings.roll(int(shift), dims=0)
    negative_distance = F.pairwise_distance(text_embeddings, negative_motion)
    positive_loss = positive_distance.square().mean()
    negative_loss = torch.clamp(margin - negative_distance, min=0).square().mean()
    return positive_loss + negative_loss, positive_loss, negative_loss, positive_distance.mean(), negative_distance.mean()


def gather_with_grad(tensor, distributed):
    if not distributed:
        return tensor
    from torch.distributed.nn.functional import all_gather
    return torch.cat(all_gather(tensor), dim=0)


def save_atomic(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_evaluator_contract(args, mean, std):
    spec = get_dataset_spec(args.dataset_key)
    dataset_root = args.checkpoint_root / args.dataset_key
    decomp_meta = dataset_root / DECOMP_NAME / "meta"
    decomp_meta.mkdir(parents=True, exist_ok=True)
    np.save(decomp_meta / "mean.npy", mean)
    np.save(decomp_meta / "std.npy", std)
    comp_root = dataset_root / COMP_NAME
    comp_meta = comp_root / "meta"
    comp_meta.mkdir(parents=True, exist_ok=True)
    np.save(comp_meta / "mean.npy", mean)
    np.save(comp_meta / "std.npy", std)
    option_values = {
        "batch_size": 32,
        "checkpoints_dir": str(args.checkpoint_root),
        "dataset_name": args.dataset_key,
        "decomp_name": DECOMP_NAME,
        "dim_movement_enc_hidden": 512,
        "dim_movement_latent": 512,
        "dim_text_hidden": 512,
        "dim_motion_hidden": 1024,
        "dim_coemb_hidden": 512,
        "feat_bias": args.feat_bias,
        "gpu_id": 0,
        "is_continue": False,
        "is_train": False,
        "max_text_len": args.max_text_len,
        "max_motion_length": dataset_max_motion_length(args.dataset_key),
        "motion_drop_tail": spec.evaluator_drop_tail,
        "motion_input_dim": spec.dim_pose - spec.evaluator_drop_tail,
        "representation": spec.representation,
        "name": COMP_NAME,
        "unit_length": args.unit_length,
    }
    lines = ["------------ Options -------------"]
    lines += [f"{key}: {value}" for key, value in option_values.items()]
    lines.append("-------------- End ----------------")
    (comp_root / "opt.txt").write_text("\n".join(lines) + "\n")


def init_wandb(args, output_dir, job_type, config):
    if args.wandb_mode == "disabled":
        return None
    import wandb
    spec = get_dataset_spec(args.dataset_key)
    default_name = (
        f"MotionX-Decomp-{spec.dim_pose}D-4GPU-seed{args.seed}"
        if job_type == "decomp"
        else f"MotionX-TextMotionMatch-{spec.dim_pose}D-4GPU-seed{args.seed}"
    )
    run_id_path = output_dir / "wandb_run_id.txt"
    run_url_path = output_dir / "wandb_run_url.txt"
    run_id = None
    if run_id_path.is_file():
        run_id = run_id_path.read_text().strip() or None
    elif run_url_path.is_file():
        run_id = run_url_path.read_text().strip().rstrip("/").rsplit("/", 1)[-1] or None
    serializable_config = json.loads(json.dumps(config, default=str))
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name or default_name,
        job_type=job_type,
        tags=["motionx", "evaluator", f"{spec.dim_pose}d", "4gpu", f"seed{args.seed}", job_type],
        config=serializable_config,
        dir=str(output_dir),
        mode=args.wandb_mode,
        id=run_id,
        resume="allow",
    )
    run.define_metric("optimization_step")
    run.define_metric("train/*", step_metric="optimization_step")
    run.define_metric("validation/*", step_metric="optimization_step")
    run.define_metric("system_train/*", step_metric="optimization_step")
    run_id_path.write_text(run.id + "\n")
    run_url_path.write_text((run.url or "") + "\n")
    return run


def loader_kwargs(args, rank, seed_offset=0):
    generator = torch.Generator()
    generator.manual_seed(args.seed + rank * 1009 + seed_offset)
    return {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }


@torch.no_grad()
def validate_decomp(model, loader, device, args):
    model.eval()
    sums = np.zeros(5, dtype=np.float64)
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            motion, _, frame_mask, latent_mask = batch
            frame_mask = frame_mask.to(device, non_blocking=True)
            latent_mask = latent_mask.to(device, non_blocking=True)
        else:
            motion, frame_mask, latent_mask = batch, None, None
        motion = motion.to(device, non_blocking=True).float()
        reconstruction, latent = model(motion)
        losses = decomp_losses(
            reconstruction, motion, latent, args.lambda_sparsity, args.lambda_smooth,
            frame_mask, latent_mask)
        batch = motion.shape[0]
        sums[:4] += np.array([loss.item() for loss in losses]) * batch
        sums[4] += batch
    model.train()
    return sums[:4] / max(1, sums[4])


def retrieval_counts(text_embeddings, motion_embeddings, top_k=3):
    distance = torch.cdist(text_embeddings, motion_embeddings)
    ranking = distance.argsort(dim=1)
    target = torch.arange(len(text_embeddings), device=distance.device)[:, None]
    hits = ranking[:, :top_k].eq(target)
    cumulative = hits.cumsum(dim=1).clamp(max=1).sum(dim=0)
    matching = distance.diagonal().sum()
    return cumulative, matching


@torch.no_grad()
def validate_matcher(model, loader, device, args, include_embeddings=False):
    model.eval()
    sums = np.zeros(10, dtype=np.float64)
    embedding_chunks = []
    shuffled_r1 = 0.0
    for batch in loader:
        word, pos, _, caption_lengths, motions, motion_lengths, _ = batch
        word = word.to(device, non_blocking=True).float()
        pos = pos.to(device, non_blocking=True).float()
        motions = motions.to(device, non_blocking=True).float()
        text_embeddings, motion_embeddings = model(word, pos, caption_lengths, motions, motion_lengths)
        size = len(text_embeddings)
        shift = 1 if size > 1 else 0
        losses = contrastive_losses(text_embeddings, motion_embeddings, args.negative_margin, shift)
        r_counts, matching = retrieval_counts(text_embeddings, motion_embeddings)
        shuffled_counts, _ = retrieval_counts(text_embeddings, motion_embeddings.roll(shift, 0))
        sums[:5] += np.array([loss.item() for loss in losses]) * size
        sums[5:8] += r_counts.cpu().numpy()
        shuffled_r1 += shuffled_counts[0].item()
        sums[8] += matching.item()
        sums[9] += size
        if include_embeddings:
            embedding_chunks.append(motion_embeddings.cpu())
    count = max(1.0, sums[9])
    metrics = {
        "loss": sums[0] / count,
        "positive_loss": sums[1] / count,
        "negative_loss": sums[2] / count,
        "positive_distance": sums[3] / count,
        "negative_distance": sums[4] / count,
        "r1": sums[5] / count,
        "r2": sums[6] / count,
        "r3": sums[7] / count,
        "matching_score": sums[8] / count,
        "shuffled_r1": shuffled_r1 / count,
    }
    model.train()
    embeddings = torch.cat(embedding_chunks).numpy() if embedding_chunks else None
    return metrics, embeddings


def build_decomp_payload(model, optimizer_encoder, optimizer_decoder,
                         completed_epoch, global_step, best_val, stale_epochs,
                         plateau=None):
    return {
        "movement_enc": model.movement_encoder.state_dict(),
        "movement_dec": model.movement_decoder.state_dict(),
        "opt_movement_enc": optimizer_encoder.state_dict(),
        "opt_movement_dec": optimizer_decoder.state_dict(),
        "ep": completed_epoch,
        "total_it": global_step,
        "best_val_loss": best_val,
        "stale_epochs": stale_epochs,
        "plateau": plateau.state_dict() if plateau else None,
    }


def build_match_payload(model, optimizer_text, optimizer_motion,
                        completed_epoch, global_step, best_val, stale_epochs,
                        best_retrieval=float("-inf"), plateau=None):
    return {
        "text_encoder": model.text_encoder.state_dict(),
        "motion_encoder": model.motion_encoder.state_dict(),
        "movement_encoder": model.movement_encoder.state_dict(),
        "opt_text_encoder": optimizer_text.state_dict(),
        "opt_motion_encoder": optimizer_motion.state_dict(),
        "epoch": completed_epoch,
        "iter": global_step,
        "best_val_loss": best_val,
        "stale_epochs": stale_epochs,
        "best_retrieval_score": best_retrieval,
        "plateau": plateau.state_dict() if plateau else None,
    }


def train_decomp(args, distributed, rank, local_rank, world_size, device):
    if args.decomp_global_batch % world_size:
        raise ValueError("Decomp global batch must be divisible by world size")
    local_batch = args.decomp_global_batch // world_size
    max_epochs = args.max_epochs or (100 if get_dataset_spec(args.dataset_key).variable_length else 270)
    output_dir = args.checkpoint_root / args.dataset_key / DECOMP_NAME
    model_dir = output_dir / "model"
    logger = make_logger(output_dir, rank)
    spec = get_dataset_spec(args.dataset_key)
    mean, std = evaluator_stats(args.data_root, args.feat_bias, args.dataset_key)
    if rank == 0:
        write_evaluator_contract(args, mean, std)
    barrier(distributed)

    if spec.variable_length:
        train_dataset = MotionSequenceEvaluatorDataset(
            args.data_root, "train", mean, std, spec.dim_pose, args.limit_ids)
        sampler = DistributedLengthBucketBatchSampler(
            train_dataset.lengths, local_batch, world_size, rank,
            seed=args.seed, shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_dataset, batch_sampler=sampler,
            collate_fn=partial(collate_decomp_variable, unit_length=args.unit_length),
            **loader_kwargs(args, rank))
    else:
        train_dataset = MotionWindowDataset(
            args.data_root, "train", mean, std, args.window_size, args.limit_ids)
        sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
            seed=args.seed, drop_last=True)
        train_loader = DataLoader(
            train_dataset, batch_size=local_batch, sampler=sampler, drop_last=True,
            **loader_kwargs(args, rank))
    if rank == 0:
        if spec.variable_length:
            val_dataset = MotionSequenceEvaluatorDataset(
                args.data_root, "val", mean, std, spec.dim_pose, args.limit_ids)
            val_loader = DataLoader(
                val_dataset, batch_size=args.eval_batch_size, shuffle=False,
                drop_last=False,
                collate_fn=partial(
                    collate_decomp_variable, unit_length=args.unit_length),
                **loader_kwargs(args, rank, 1))
        else:
            val_dataset = MotionWindowDataset(
                args.data_root, "val", mean, std, args.window_size, args.limit_ids)
            val_loader = DataLoader(
                val_dataset, batch_size=args.decomp_global_batch, shuffle=False,
                drop_last=False, **loader_kwargs(args, rank, 1))
    else:
        val_loader = None

    raw_model = DecompModel(spec.dim_pose, spec.evaluator_drop_tail).to(device)
    model = DDP(raw_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else raw_model
    optimizer_encoder = torch.optim.Adam(raw_model.movement_encoder.parameters(), lr=args.lr)
    optimizer_decoder = torch.optim.Adam(raw_model.movement_decoder.parameters(), lr=args.lr)
    start_epoch = 0
    global_step = 0
    best_val = math.inf
    stale_epochs = 0
    plateau = None
    if args.lr_policy == "plateau":
        plateau = PlateauController(
            mode="min", factor=args.plateau_factor,
            lr_patience=args.plateau_patience,
            stop_patience=args.early_stop_patience,
            min_index=args.decomp_min_epochs, min_lr=args.plateau_min_lr,
        )
    latest_path = model_dir / "latest.tar"
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location="cpu")
        raw_model.movement_encoder.load_state_dict(checkpoint["movement_enc"])
        raw_model.movement_decoder.load_state_dict(checkpoint["movement_dec"])
        optimizer_encoder.load_state_dict(checkpoint["opt_movement_enc"])
        optimizer_decoder.load_state_dict(checkpoint["opt_movement_dec"])
        start_epoch = int(checkpoint.get("ep", 0))
        global_step = int(checkpoint.get("total_it", 0))
        best_val = float(checkpoint.get("best_val_loss", math.inf))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        if plateau is not None and checkpoint.get("plateau"):
            plateau.load_state_dict(checkpoint["plateau"])
        logger.info("Resumed Decomp from epoch %d step %d", start_epoch, global_step)

    config = vars(args).copy()
    config.update({
        "world_size": world_size,
        "local_batch": local_batch,
        "global_batch": args.decomp_global_batch,
        "train_windows": len(train_dataset),
        "steps_per_epoch": len(train_loader),
        "effective_max_epochs": max_epochs,
    })
    wandb_run = init_wandb(args, output_dir, "decomp", config) if rank == 0 else None
    if rank == 0 and wandb_run and args.wandb_watch_log != "none":
        wandb_run.watch(raw_model, log=args.wandb_watch_log, log_freq=args.wandb_watch_freq)
    logger.info("Decomp: %d windows, %d steps/epoch, local/global batch %d/%d",
                len(train_dataset), len(train_loader), local_batch, args.decomp_global_batch)

    interval_sums = np.zeros(4, dtype=np.float64)
    interval_started = time.monotonic()
    training_started = interval_started
    for epoch in range(start_epoch, max_epochs):
        if args.lr_policy == "fixed":
            current_lr = decomp_learning_rate(
                args.lr, epoch + 1, args.decomp_lr_milestones,
                args.decomp_lr_gamma)
            set_optimizer_lr((optimizer_encoder, optimizer_decoder), current_lr)
        else:
            current_lr = optimizer_encoder.param_groups[0]["lr"]
        sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            if spec.variable_length:
                motion, _, frame_mask, latent_mask = batch
                frame_mask = frame_mask.to(device, non_blocking=True)
                latent_mask = latent_mask.to(device, non_blocking=True)
            else:
                motion, frame_mask, latent_mask = batch, None, None
            motion = motion.to(device, non_blocking=True).float()
            reconstruction, latent = model(motion)
            losses = decomp_losses(
                reconstruction, motion, latent, args.lambda_sparsity, args.lambda_smooth,
                frame_mask, latent_mask)
            optimizer_encoder.zero_grad(set_to_none=True)
            optimizer_decoder.zero_grad(set_to_none=True)
            losses[0].backward()
            optimizer_encoder.step()
            optimizer_decoder.step()
            global_step += 1
            interval_sums += np.array([loss.detach().item() for loss in losses])
            if global_step % args.log_every == 0:
                totals = torch.tensor(interval_sums, dtype=torch.float64, device=device)
                if distributed:
                    dist.all_reduce(totals)
                elapsed = time.monotonic() - interval_started
                values = (totals.cpu().numpy() / (args.log_every * world_size)).tolist()
                steps_per_second = args.log_every / max(elapsed, 1e-9)
                remaining_steps = (max_epochs - epoch - 1) * len(train_loader) + (len(train_loader) - (global_step % len(train_loader)))
                eta_hours = remaining_steps / max(steps_per_second, 1e-9) / 3600
                if rank == 0:
                    logger.info(
                        "epoch %d/%d step %d loss %.6f rec %.6f sparse %.6f smooth %.6f lr %.2g %.1f step/s ETA %.2fh",
                        epoch + 1, max_epochs, global_step, *values,
                        current_lr, steps_per_second, eta_hours)
                    if wandb_run:
                        wandb_run.log({
                            "train/loss": values[0],
                            "train/reconstruction_loss": values[1],
                            "train/sparsity_loss": values[2],
                            "train/smoothness_loss": values[3],
                            "system_train/steps_per_second": steps_per_second,
                            "system_train/eta_hours": eta_hours,
                            "system_train/elapsed_hours": (time.monotonic() - training_started) / 3600,
                            "optimizer/learning_rate": current_lr,
                            "optimization_step": global_step,
                            "epoch": epoch + 1,
                        })
                interval_sums.fill(0)
                interval_started = time.monotonic()

        barrier(distributed)
        stop_training = False
        if rank == 0:
            validation = validate_decomp(raw_model, val_loader, device, args)
            improved = validation[0] < best_val
            if improved:
                best_val = float(validation[0])
            completed_epoch = epoch + 1
            if plateau is not None:
                decision = plateau.step(
                    validation[0], completed_epoch,
                    [optimizer_encoder, optimizer_decoder],
                )
                stale_epochs = decision.bad_evals
                stop_training = decision.stop
            else:
                if completed_epoch <= args.decomp_min_epochs or improved:
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                stop_training = (
                    completed_epoch >= args.decomp_min_epochs
                    and stale_epochs >= args.decomp_patience
                )
            payload = build_decomp_payload(
                raw_model, optimizer_encoder, optimizer_decoder,
                completed_epoch, global_step, best_val, stale_epochs, plateau)
            save_atomic(payload, latest_path)
            if improved:
                save_atomic(payload, model_dir / "finest.tar")
            if completed_epoch % args.save_every == 0:
                save_atomic(payload, model_dir / f"E{completed_epoch:04d}.tar")
            logger.info(
                "validation epoch %d total %.6f rec %.6f sparse %.6f smooth %.6f best %.6f stale %d stop=%s",
                completed_epoch, *validation, best_val, stale_epochs, stop_training)
            if wandb_run:
                wandb_run.log({
                    "validation/loss": validation[0],
                    "validation/reconstruction_loss": validation[1],
                    "validation/sparsity_loss": validation[2],
                    "validation/smoothness_loss": validation[3],
                    "validation/best_loss": best_val,
                    "validation/stale_epochs": stale_epochs,
                    "optimization_step": global_step,
                    "epoch": completed_epoch,
                })
        stop_tensor = torch.tensor(int(stop_training), device=device)
        if distributed:
            dist.broadcast(stop_tensor, src=0)
            lr_tensor = torch.tensor(
                [optimizer_encoder.param_groups[0]["lr"],
                 optimizer_decoder.param_groups[0]["lr"]],
                dtype=torch.float64, device=device)
            dist.broadcast(lr_tensor, src=0)
            set_optimizer_lr(
                (optimizer_encoder,), float(lr_tensor[0].item()))
            set_optimizer_lr(
                (optimizer_decoder,), float(lr_tensor[1].item()))
        barrier(distributed)
        interval_started = time.monotonic()
        if stop_tensor.item():
            break
    if rank == 0 and wandb_run:
        wandb_run.finish()


def load_decomp_encoder(args, model):
    checkpoint_path = args.checkpoint_root / args.dataset_key / DECOMP_NAME / "model" / "finest.tar"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Decomp checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.movement_encoder.load_state_dict(checkpoint["movement_enc"])
    model.movement_encoder.requires_grad_(False)


def train_match(args, distributed, rank, local_rank, world_size, device):
    if args.match_global_batch % world_size:
        raise ValueError("Matcher global batch must be divisible by world size")
    local_batch = args.match_global_batch // world_size
    max_epochs = args.max_epochs or 300
    output_dir = args.checkpoint_root / args.dataset_key / MATCH_NAME
    model_dir = output_dir / "model"
    logger = make_logger(output_dir, rank)
    spec = get_dataset_spec(args.dataset_key)
    mean, std = evaluator_stats(args.data_root, args.feat_bias, args.dataset_key)
    vectorizer = WordVectorizer("./glove", "our_vab")
    train_dataset = TextMotionDataset(
        args.data_root, "train", mean, std, vectorizer, args.max_text_len,
        args.max_motion_length, args.unit_length, args.limit_ids, args.dataset_key)
    if spec.variable_length:
        sampler = DistributedLengthBucketBatchSampler(
            train_dataset.lengths, local_batch, world_size, rank,
            seed=args.seed, shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_dataset, batch_sampler=sampler,
            collate_fn=partial(
                collate_text_motion_variable, unit_length=args.unit_length),
            **loader_kwargs(args, rank))
    else:
        sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
            seed=args.seed, drop_last=True)
        train_loader = DataLoader(
            train_dataset, batch_size=local_batch, sampler=sampler, drop_last=True,
            collate_fn=collate_text_motion, **loader_kwargs(args, rank))
    if rank == 0:
        val_dataset = TextMotionDataset(
            args.data_root, "val", mean, std, vectorizer, args.max_text_len,
            args.max_motion_length, args.unit_length, args.limit_ids, args.dataset_key)
        val_loader = DataLoader(
            val_dataset, batch_size=(
                args.eval_batch_size if spec.variable_length
                else args.match_global_batch), shuffle=False,
            drop_last=False, collate_fn=(
                partial(collate_text_motion_variable, unit_length=args.unit_length)
                if spec.variable_length else collate_text_motion),
            **loader_kwargs(args, rank, 1))
    else:
        val_loader = None

    raw_model = MatcherModel(
        device, spec.dim_pose, spec.evaluator_drop_tail,
        spec.variable_length, args.unit_length).to(device)
    load_decomp_encoder(args, raw_model)
    text_parameters = list(raw_model.text_encoder.parameters())
    motion_parameters = list(raw_model.motion_encoder.parameters())
    trainable = text_parameters + motion_parameters
    model = DDP(raw_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else raw_model
    optimizer_text = torch.optim.Adam(text_parameters, lr=args.lr)
    optimizer_motion = torch.optim.Adam(motion_parameters, lr=args.lr)
    start_epoch = 0
    global_step = 0
    best_val = math.inf
    best_retrieval = float("-inf")
    stale_epochs = 0
    plateau = None
    if args.lr_policy == "plateau":
        plateau = PlateauController(
            mode="min", factor=args.plateau_factor,
            lr_patience=args.plateau_patience,
            stop_patience=args.early_stop_patience,
            min_index=args.min_epochs, min_lr=args.plateau_min_lr,
        )
    latest_path = model_dir / "latest.tar"
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location="cpu")
        raw_model.text_encoder.load_state_dict(checkpoint["text_encoder"])
        raw_model.motion_encoder.load_state_dict(checkpoint["motion_encoder"])
        raw_model.movement_encoder.load_state_dict(checkpoint["movement_encoder"])
        optimizer_text.load_state_dict(checkpoint["opt_text_encoder"])
        optimizer_motion.load_state_dict(checkpoint["opt_motion_encoder"])
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("iter", 0))
        best_val = float(checkpoint.get("best_val_loss", math.inf))
        best_retrieval = float(checkpoint.get("best_retrieval_score", float("-inf")))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        if plateau is not None and checkpoint.get("plateau"):
            plateau.load_state_dict(checkpoint["plateau"])
        logger.info("Resumed Matcher from epoch %d step %d", start_epoch, global_step)

    config = vars(args).copy()
    config.update({
        "world_size": world_size,
        "local_batch": local_batch,
        "global_batch": args.match_global_batch,
        "train_rows": len(train_dataset),
        "steps_per_epoch": len(train_loader),
        "effective_max_epochs": max_epochs,
    })
    wandb_run = init_wandb(args, output_dir, "text_motion_match", config) if rank == 0 else None
    if rank == 0 and wandb_run and args.wandb_watch_log != "none":
        wandb_run.watch(raw_model, log=args.wandb_watch_log, log_freq=args.wandb_watch_freq)
    logger.info("Matcher: %d rows, %d steps/epoch, local/global batch %d/%d",
                len(train_dataset), len(train_loader), local_batch, args.match_global_batch)

    interval_sums = np.zeros(5, dtype=np.float64)
    interval_started = time.monotonic()
    training_started = interval_started
    stop_training = False
    for epoch in range(start_epoch, max_epochs):
        if args.lr_policy == "fixed":
            current_lr = decomp_learning_rate(
                args.lr, epoch + 1, args.match_lr_milestones,
                args.match_lr_gamma)
            set_optimizer_lr((optimizer_text, optimizer_motion), current_lr)
        else:
            current_lr = optimizer_text.param_groups[0]["lr"]
        sampler.set_epoch(epoch)
        model.train()
        raw_model.movement_encoder.eval()
        for batch in train_loader:
            word, pos, _, caption_lengths, motions, motion_lengths, _ = batch
            word = word.to(device, non_blocking=True).float()
            pos = pos.to(device, non_blocking=True).float()
            motions = motions.to(device, non_blocking=True).float()
            text_embeddings, motion_embeddings = model(
                word, pos, caption_lengths, motions, motion_lengths)
            global_text = gather_with_grad(text_embeddings, distributed)
            global_motion = gather_with_grad(motion_embeddings, distributed)
            shift_tensor = torch.zeros((), dtype=torch.long, device=device)
            if rank == 0:
                shift_tensor.random_(1, len(global_text))
            if distributed:
                dist.broadcast(shift_tensor, src=0)
            losses = contrastive_losses(
                global_text, global_motion, args.negative_margin, int(shift_tensor.item()))
            optimizer_text.zero_grad(set_to_none=True)
            optimizer_motion.zero_grad(set_to_none=True)
            losses[0].backward()
            grad_norm = clip_grad_norm_(trainable, 0.5)
            optimizer_text.step()
            optimizer_motion.step()
            global_step += 1
            interval_sums += np.array([loss.detach().item() for loss in losses])
            if global_step % args.log_every == 0:
                totals = torch.tensor(interval_sums, dtype=torch.float64, device=device)
                if distributed:
                    dist.all_reduce(totals)
                values = (totals.cpu().numpy() / (args.log_every * world_size)).tolist()
                elapsed = time.monotonic() - interval_started
                steps_per_second = args.log_every / max(elapsed, 1e-9)
                remaining_steps = (max_epochs - epoch - 1) * len(train_loader) + (len(train_loader) - (global_step % len(train_loader)))
                eta_hours = remaining_steps / max(steps_per_second, 1e-9) / 3600
                if rank == 0:
                    logger.info(
                        "epoch %d/%d step %d loss %.5f pos %.5f neg %.5f d+ %.4f d- %.4f grad %.3f lr %.2g %.1f step/s ETA %.2fh",
                        epoch + 1, max_epochs, global_step, *values,
                        float(grad_norm), current_lr, steps_per_second, eta_hours)
                    if wandb_run:
                        wandb_run.log({
                            "train/loss": values[0],
                            "train/positive_loss": values[1],
                            "train/negative_loss": values[2],
                            "train/positive_distance": values[3],
                            "train/negative_distance": values[4],
                            "train/grad_norm": float(grad_norm),
                            "system_train/steps_per_second": steps_per_second,
                            "system_train/eta_hours_to_max": eta_hours,
                            "system_train/elapsed_hours": (time.monotonic() - training_started) / 3600,
                            "optimizer/learning_rate": current_lr,
                            "optimization_step": global_step,
                            "epoch": epoch + 1,
                        })
                interval_sums.fill(0)
                interval_started = time.monotonic()

        barrier(distributed)
        metrics = None
        if rank == 0:
            metrics, _ = validate_matcher(raw_model, val_loader, device, args)
            loss_improved = metrics["loss"] < best_val
            if loss_improved:
                best_val = metrics["loss"]
            distance_ratio = metrics["positive_distance"] / max(
                metrics["negative_distance"], 1e-12)
            retrieval_score = metrics["r1"] + metrics["r2"] + metrics["r3"]
            eligible = distance_ratio <= args.selection_max_distance_ratio
            retrieval_improved = eligible and retrieval_score > best_retrieval
            if retrieval_improved:
                best_retrieval = retrieval_score
            completed_epoch = epoch + 1
            control_metric = metrics["loss"]
            if plateau is not None:
                decision = plateau.step(
                    control_metric, completed_epoch,
                    [optimizer_text, optimizer_motion],
                )
                stale_epochs = decision.bad_evals
                stop_training = decision.stop
            else:
                if completed_epoch <= args.min_epochs or loss_improved:
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                stop_training = (
                    completed_epoch >= args.min_epochs
                    and stale_epochs >= args.patience
                )
            payload = build_match_payload(
                raw_model, optimizer_text, optimizer_motion,
                completed_epoch, global_step, best_val, stale_epochs,
                best_retrieval, plateau)
            save_atomic(payload, latest_path)
            if loss_improved:
                save_atomic(payload, model_dir / "loss_best.tar")
                save_atomic(payload, model_dir / "finest.tar")
            if completed_epoch % max(1, args.save_every) == 0:
                save_atomic(payload, model_dir / f"E{completed_epoch:04d}.tar")
            logger.info(
                "validation epoch %d loss %.5f d+ %.4f d- %.4f ratio %.4f R %.4f/%.4f/%.4f retrieval %.4f best_retrieval %.4f best_loss %.5f stale %d stop=%s",
                completed_epoch, metrics["loss"], metrics["positive_distance"],
                metrics["negative_distance"], distance_ratio, metrics["r1"],
                metrics["r2"], metrics["r3"], retrieval_score,
                best_retrieval, best_val, stale_epochs, stop_training)
            if wandb_run:
                wandb_run.log({
                    **{f"validation/{key}": value for key, value in metrics.items()},
                    "validation/best_loss": best_val,
                    "validation/retrieval_score": retrieval_score,
                    "validation/best_retrieval_score": best_retrieval,
                    "validation/distance_ratio": distance_ratio,
                    "validation/retrieval_eligible": int(eligible),
                    "validation/stale_epochs": stale_epochs,
                    "optimization_step": global_step,
                    "epoch": completed_epoch,
                })
        stop_tensor = torch.tensor(int(stop_training), device=device)
        if distributed:
            dist.broadcast(stop_tensor, src=0)
            lr_tensor = torch.tensor(
                [optimizer_text.param_groups[0]["lr"],
                 optimizer_motion.param_groups[0]["lr"]],
                dtype=torch.float64, device=device)
            dist.broadcast(lr_tensor, src=0)
            set_optimizer_lr((optimizer_text,), float(lr_tensor[0].item()))
            set_optimizer_lr((optimizer_motion,), float(lr_tensor[1].item()))
        barrier(distributed)
        if stop_tensor.item():
            break
    if rank == 0 and wandb_run:
        wandb_run.finish()


def validate_final(args, device):
    spec = get_dataset_spec(args.dataset_key)
    mean, std = evaluator_stats(args.data_root, args.feat_bias, args.dataset_key)
    vectorizer = WordVectorizer("./glove", "our_vab")
    dataset = TextMotionDataset(
        args.data_root, "test", mean, std, vectorizer, args.max_text_len,
        args.max_motion_length, args.unit_length, args.limit_ids, args.dataset_key)
    loader = DataLoader(
        dataset, batch_size=(args.eval_batch_size if spec.variable_length else args.match_global_batch),
        shuffle=False, drop_last=False,
        collate_fn=(
            partial(collate_text_motion_variable, unit_length=args.unit_length)
            if spec.variable_length else collate_text_motion),
        **loader_kwargs(args, 0, 2))
    model = MatcherModel(
        device, spec.dim_pose, spec.evaluator_drop_tail,
        spec.variable_length, args.unit_length).to(device)
    checkpoint_path = args.checkpoint_root / args.dataset_key / MATCH_NAME / "model" / "finest.tar"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    required = {"text_encoder", "motion_encoder", "movement_encoder", "epoch", "iter"}
    missing = required - checkpoint.keys()
    if missing:
        raise RuntimeError(f"Evaluator checkpoint is missing keys: {sorted(missing)}")
    model.text_encoder.load_state_dict(checkpoint["text_encoder"])
    model.motion_encoder.load_state_dict(checkpoint["motion_encoder"])
    model.movement_encoder.load_state_dict(checkpoint["movement_encoder"])
    model.eval()
    metrics, embeddings = validate_matcher(model, loader, device, args, include_embeddings=True)
    covariance = np.cov(embeddings, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = max(float(eigenvalues[-1]), 1e-12)
    numerical_rank = int((eigenvalues > largest * 1e-6).sum())
    metrics.update({
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_iter": int(checkpoint["iter"]),
        "embedding_covariance_trace": float(np.trace(covariance)),
        "embedding_numerical_rank": numerical_rank,
        "positive_negative_distance_ratio": metrics["positive_distance"] / max(metrics["negative_distance"], 1e-12),
    })
    gates = {
        "positive_negative_ratio_le_0.75": bool(
            metrics["positive_negative_distance_ratio"] <= 0.75),
        "r1_ge_0.10": bool(metrics["r1"] >= 0.10),
        "r2_ge_0.18": bool(metrics["r2"] >= 0.18),
        "r3_ge_0.25": bool(metrics["r3"] >= 0.25),
        "shuffled_r1_lower": bool(metrics["shuffled_r1"] < metrics["r1"]),
        "embedding_rank_ge_64": bool(numerical_rank >= 64),
        "finite_metrics": bool(all(
            np.isfinite(value) for value in metrics.values()
            if isinstance(value, (int, float)))),
    }
    report = {"metrics": metrics, "gates": gates, "passed": all(gates.values())}
    output_path = args.checkpoint_root / args.dataset_key / "evaluator_validation.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


def main():
    args = parse_args()
    if args.stage == "validate":
        seed_everything(args.seed)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.cuda.set_device(0)
        validate_final(args, torch.device("cuda:0"))
        return

    distributed, rank, local_rank, world_size, device = setup_distributed()
    seed_everything(args.seed + rank)
    try:
        if args.stage == "decomp":
            train_decomp(args, distributed, rank, local_rank, world_size, device)
        else:
            train_match(args, distributed, rank, local_rank, world_size, device)
    finally:
        cleanup(distributed)


if __name__ == "__main__":
    main()
