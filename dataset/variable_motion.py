"""Variable-length motion batching used by the native Motion-X pipeline."""

import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from dataset.config import get_dataset_spec


def round_up(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def lengths_to_mask(lengths, max_length=None, device=None):
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)
    if max_length is None:
        max_length = int(lengths.max().item())
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def downsample_lengths(lengths, unit_length, include_partial=False):
    lengths = torch.as_tensor(lengths, dtype=torch.long)
    if include_partial:
        lengths = lengths + int(unit_length) - 1
    return torch.div(
        lengths, int(unit_length), rounding_mode="floor").clamp_min(1)


def pad_motion_arrays(motions, multiple=1):
    tensors = [torch.as_tensor(motion, dtype=torch.float32) for motion in motions]
    lengths = torch.tensor([len(motion) for motion in tensors], dtype=torch.long)
    max_length = round_up(int(lengths.max().item()), multiple)
    feature_dim = int(tensors[0].shape[1])
    padded = torch.zeros((len(tensors), max_length, feature_dim), dtype=torch.float32)
    for index, motion in enumerate(tensors):
        padded[index, : len(motion)] = motion
    return padded, lengths, lengths_to_mask(lengths, max_length)


def collate_motion_sequences(batch, multiple=4):
    motions, lengths, mask = pad_motion_arrays(batch, multiple=multiple)
    return {"motion": motions, "lengths": lengths, "frame_mask": mask}


class MotionSequenceDataset(Dataset):
    def __init__(self, dataset_name, split="train", limit=0):
        self.spec = get_dataset_spec(dataset_name)
        mean = np.load(self.spec.root / "Mean.npy", allow_pickle=False).astype(np.float32)
        std = np.load(self.spec.root / "Std.npy", allow_pickle=False).astype(np.float32)
        ids = [
            line.strip()
            for line in (self.spec.root / f"{split}.txt").read_text().splitlines()
            if line.strip()
        ]
        if limit:
            ids = ids[:limit]
        self.rows = []
        self.lengths = []
        for motion_id in ids:
            path = self.spec.motion_dir / f"{motion_id}.npy"
            if not path.is_file():
                continue
            shape = np.load(path, mmap_mode="r", allow_pickle=False).shape
            if len(shape) != 2 or shape[1] != self.spec.dim_pose or shape[0] < self.spec.min_motion_len:
                continue
            self.rows.append((motion_id, path))
            self.lengths.append(int(shape[0]))
        if not self.rows:
            raise RuntimeError(f"No valid {dataset_name}/{split} motion sequences")
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        _, path = self.rows[index]
        motion = np.load(path, allow_pickle=False).astype(np.float32)
        return ((motion - self.mean) / self.std).astype(np.float32)


class DistributedLengthBucketBatchSampler(Sampler):
    """Length-bucketed batches with identical step counts on every DDP rank."""

    def __init__(self, lengths, batch_size, num_replicas=1, rank=0, seed=123,
                 shuffle=True, drop_last=True, bucket_multiplier=20):
        self.lengths = list(map(int, lengths))
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.bucket_multiplier = int(bucket_multiplier)
        self.epoch = 0
        self.global_batch = self.batch_size * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.global_batch
        return math.ceil(len(self.lengths) / self.global_batch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        bucket_size = max(self.global_batch, self.global_batch * self.bucket_multiplier)
        ordered = []
        for start in range(0, len(indices), bucket_size):
            bucket = indices[start:start + bucket_size]
            bucket.sort(key=self.lengths.__getitem__)
            batches = [bucket[i:i + self.global_batch] for i in range(0, len(bucket), self.global_batch)]
            if self.drop_last and batches and len(batches[-1]) != self.global_batch:
                batches.pop()
            if self.shuffle:
                rng.shuffle(batches)
            ordered.extend(batches)
        if self.shuffle:
            rng.shuffle(ordered)
        for batch in ordered:
            if len(batch) < self.global_batch:
                if self.drop_last:
                    continue
                batch = batch + batch[: self.global_batch - len(batch)]
            start = self.rank * self.batch_size
            yield batch[start:start + self.batch_size]
