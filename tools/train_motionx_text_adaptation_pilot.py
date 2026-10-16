#!/usr/bin/env python
"""Small Motion-X-322 text/generator adaptation pilot.

This script is intentionally separate from the active training entry points.  It
runs a diagnostic supervised objective on preserving caption pairs only:
clean caption and paraphrase both predict the same cached VQ token sequence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
from argparse import Namespace
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASET_NAME = "motionx322_75"
DATASET_ROOT = REPO_ROOT / "dataset" / "Motion-X-322-MaxToken75"
TOKEN_DIR = DATASET_ROOT / "VQVAE"
BASELINE_TRANS = REPO_ROOT / "output_motionx322" / "GPT_MOTIONX322_MAXTOK75_200K" / "net_best_fid.pth"
BASELINE_VQ = REPO_ROOT / "output_motionx322" / "VQVAE_MOTIONX322" / "net_best_fid.pth"
CLIP_MODEL_NAME = "ViT-B/32"

# Motion-X-322 MaxToken75 architecture, matching existing launchers.
ARCH = {
    "dataname": DATASET_NAME,
    "resume_pth": str(BASELINE_VQ.relative_to(REPO_ROOT)),
    "resume_trans": str(BASELINE_TRANS.relative_to(REPO_ROOT)),
    "vq_name": "VQVAE",
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
    "block_size": 76,
    "embed_dim_gpt": 1024,
    "clip_dim": 512,
    "num_layers": 9,
    "n_head_gpt": 16,
    "drop_out_rate": 0.1,
    "ff_rate": 4,
}


@dataclass
class PairExample:
    pair_index: int
    pair_id: str
    motion_id: str
    text_orig: str
    text_pair: str
    token_rows: np.ndarray
    token_asset: str


@dataclass
class Batch:
    texts: list[str]
    input_indices: torch.Tensor
    targets: torch.Tensor
    lengths: torch.Tensor
    pair_indices: list[int]
    token_row_indices: list[int]
    corruption_sha256: str
    plan_sha256: str


class TrainingDataError(ValueError):
    """Raised when the pilot pair data is unsafe for train-only adaptation."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic Motion-X-322 pilot: train generator, CLIP text, or both "
            "on preserving clean/paraphrase captions mapped to identical VQ tokens."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arm", choices=("generator", "text", "joint"), required=True)
    parser.add_argument("--train-pairs", type=Path, required=True, help="preserving-pair JSONL")
    parser.add_argument("--out-dir", type=Path, required=True, help="new empty output directory")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trans-lr", type=float, default=2e-5)
    parser.add_argument("--clip-lr", type=float, default=1e-6)
    parser.add_argument("--dataset-limit", type=int, default=None, help="optional accepted-pair cap for smoke tests")
    parser.add_argument("--save-every", type=int, default=100, help="fixed checkpoint interval in steps")
    parser.add_argument("--log-every", type=int, default=10, help="stdout/log-file interval in steps")
    parser.add_argument("--pkeep", type=float, default=0.5, help="token keep probability; -1 samples one probability per batch")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="training device; keep default under CUDA_VISIBLE_DEVICES for GPU pilots",
    )
    return parser.parse_args(argv)


def resolve_repo_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def ensure_empty_output_dir(path: Path) -> Path:
    path = resolve_repo_path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty --out-dir: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_split_ids(dataset_root: Path, split: str) -> set[str]:
    path = dataset_root / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing split file: {path}")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing --train-pairs file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingDataError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise TrainingDataError(f"expected object row at {path}:{line_number}")
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def _safe_float(value: str) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return 0.0 if math.isnan(out) else out


def full_motion_captions(text_path: Path) -> set[str]:
    if not text_path.is_file():
        return set()
    captions: set[str] = set()
    with text_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split("#")
            if len(parts) < 4:
                continue
            f_tag = _safe_float(parts[2])
            to_tag = _safe_float(parts[3])
            if f_tag == 0.0 and to_tag == 0.0:
                captions.add(parts[0])
    return captions


def load_token_rows(path: Path, nb_code: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing required token asset: {path}")
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise TrainingDataError(f"failed to load token asset {path}: {exc}") from exc
    if not np.issubdtype(array.dtype, np.integer):
        raise TrainingDataError(f"token asset must contain integer VQ ids: {path} dtype={array.dtype}")
    if array.ndim == 1:
        rows = array.reshape(1, -1)
    elif array.ndim == 2:
        rows = array
    else:
        raise TrainingDataError(f"token asset must be 1D or 2D after numpy load: {path} shape={array.shape}")
    if rows.shape[0] < 1 or rows.shape[1] < 1:
        raise TrainingDataError(f"token asset is empty: {path} shape={rows.shape}")
    rows = rows.astype(np.int64, copy=False)
    min_id = int(rows.min())
    max_id = int(rows.max())
    if min_id < 0 or max_id >= nb_code:
        raise TrainingDataError(
            f"token asset ids out of VQ range [0,{nb_code - 1}]: {path} min={min_id} max={max_id}"
        )
    return np.ascontiguousarray(rows)


def load_training_pairs(
    train_pairs: Path,
    dataset_root: Path = DATASET_ROOT,
    token_dir: Path = TOKEN_DIR,
    nb_code: int = ARCH["nb_code"],
    dataset_limit: int | None = None,
) -> tuple[list[PairExample], dict]:
    dataset_root = resolve_repo_path(dataset_root)
    token_dir = resolve_repo_path(token_dir)
    train_pairs = resolve_repo_path(train_pairs)
    if dataset_limit is not None and dataset_limit <= 0:
        raise TrainingDataError("--dataset-limit must be positive when provided")
    if not token_dir.is_dir():
        raise FileNotFoundError(f"missing required VQ token directory: {token_dir}")

    train_ids = load_split_ids(dataset_root, "train")
    val_ids = load_split_ids(dataset_root, "val") if (dataset_root / "val.txt").exists() else set()
    test_ids = load_split_ids(dataset_root, "test") if (dataset_root / "test.txt").exists() else set()
    rows = read_jsonl(train_pairs)
    required = {"motion_id", "text_orig", "text_pair"}
    missing_required = [
        {"line": row.get("_line_number"), "missing": sorted(required - set(row))}
        for row in rows
        if required - set(row)
    ]
    if missing_required:
        raise TrainingDataError(f"pair rows missing required fields: {missing_required[:5]}")

    nontrain = []
    for row in rows:
        motion_id = str(row["motion_id"])
        if motion_id not in train_ids:
            split = "val" if motion_id in val_ids else "test" if motion_id in test_ids else "unknown"
            nontrain.append({"line": row.get("_line_number"), "motion_id": motion_id, "split": split})
    if nontrain:
        raise TrainingDataError(
            f"rejecting input nontrain motion IDs; examples={nontrain[:10]} total={len(nontrain)}"
        )

    token_cache: dict[str, np.ndarray] = {}
    text_cache: dict[str, set[str]] = {}
    examples: list[PairExample] = []
    report = {
        "train_pairs_path": str(train_pairs),
        "dataset_root": str(dataset_root),
        "token_dir": str(token_dir),
        "raw_rows": len(rows),
        "nontrain_rejected": 0,
        "non_preserve_filtered": 0,
        "annotation_mismatch_filtered": 0,
        "accepted_before_limit": 0,
        "accepted": 0,
        "unique_train_motions": 0,
        "dataset_limit": dataset_limit,
        "token_ndim_counts": {},
        "token_shape_examples": [],
    }

    for row_number, row in enumerate(rows):
        if row.get("relation") != "preserve":
            report["non_preserve_filtered"] += 1
            continue
        motion_id = str(row["motion_id"])
        text_orig = str(row["text_orig"])
        captions = text_cache.get(motion_id)
        if captions is None:
            captions = full_motion_captions(dataset_root / "texts" / f"{motion_id}.txt")
            text_cache[motion_id] = captions
        if text_orig not in captions:
            report["annotation_mismatch_filtered"] += 1
            continue
        report["accepted_before_limit"] += 1
        if dataset_limit is not None and len(examples) >= dataset_limit:
            continue
        token_rows = token_cache.get(motion_id)
        token_asset = token_dir / f"{motion_id}.npy"
        if token_rows is None:
            token_rows = load_token_rows(token_asset, nb_code)
            token_cache[motion_id] = token_rows
            ndim_key = str(1 if token_rows.shape[0] == 1 else 2)
            report["token_ndim_counts"][ndim_key] = report["token_ndim_counts"].get(ndim_key, 0) + 1
            if len(report["token_shape_examples"]) < 8:
                report["token_shape_examples"].append(
                    {"motion_id": motion_id, "shape": list(token_rows.shape), "dtype": str(token_rows.dtype)}
                )
        examples.append(
            PairExample(
                pair_index=row_number,
                pair_id=str(row.get("pair_id", row.get("_line_number", row_number))),
                motion_id=motion_id,
                text_orig=text_orig,
                text_pair=str(row["text_pair"]),
                token_rows=token_rows,
                token_asset=str(token_asset),
            )
        )

    if not examples:
        raise TrainingDataError(
            "no train-safe preserving pairs remain after filtering for relation='preserve' "
            "and exact full-motion text_orig annotation matches"
        )
    report["accepted"] = len(examples)
    report["unique_train_motions"] = len({example.motion_id for example in examples})
    return examples, report


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_model_args(cli_args: argparse.Namespace | None = None) -> Namespace:
    values = dict(ARCH)
    if cli_args is not None:
        values.update(
            arm=cli_args.arm,
            train_pairs=str(resolve_repo_path(cli_args.train_pairs)),
            out_dir=str(resolve_repo_path(cli_args.out_dir)),
            steps=int(cli_args.steps),
            batch_size=int(cli_args.batch_size),
            seed=int(cli_args.seed),
            trans_lr=float(cli_args.trans_lr),
            clip_lr=float(cli_args.clip_lr),
            dataset_limit=cli_args.dataset_limit,
            pkeep=float(cli_args.pkeep),
            diagnostic_pilot=True,
            no_sato_heads_or_losses=True,
        )
    return Namespace(**values)


def is_clip_visual_parameter(name: str) -> bool:
    return name.startswith("visual") or name.startswith("module.visual")


def is_clip_text_parameter(name: str) -> bool:
    stripped = name.removeprefix("module.")
    if stripped == "logit_scale" or stripped.startswith("visual"):
        return False
    return stripped.startswith(("token_embedding", "positional_embedding", "transformer", "ln_final", "text_projection"))


def parameter_count(parameters: Iterable[torch.nn.Parameter]) -> int:
    return int(sum(parameter.numel() for parameter in parameters))


def collect_parameter_groups(trans_encoder: torch.nn.Module, clip_model: torch.nn.Module) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "generator": [],
        "clip_text": [],
        "clip_visual": [],
        "clip_other": [],
    }
    groups["generator"] = [(f"trans.{name}", param) for name, param in trans_encoder.named_parameters()]
    for name, param in clip_model.named_parameters():
        if is_clip_visual_parameter(name):
            groups["clip_visual"].append((f"clip.{name}", param))
        elif is_clip_text_parameter(name):
            groups["clip_text"].append((f"clip.{name}", param))
        else:
            groups["clip_other"].append((f"clip.{name}", param))
    return groups


def configure_trainability(
    trans_encoder: torch.nn.Module,
    clip_model: torch.nn.Module,
    arm: str,
) -> dict:
    if arm not in {"generator", "text", "joint"}:
        raise ValueError(f"unsupported arm: {arm}")
    groups = collect_parameter_groups(trans_encoder, clip_model)
    for _, param in sum(groups.values(), []):
        param.requires_grad = False
    if arm in {"generator", "joint"}:
        for _, param in groups["generator"]:
            param.requires_grad = True
    if arm in {"text", "joint"}:
        for _, param in groups["clip_text"]:
            param.requires_grad = True
    return trainability_summary(groups)


def trainability_summary(groups: dict[str, list[tuple[str, torch.nn.Parameter]]]) -> dict:
    summary = {}
    for group, named_params in groups.items():
        params = [param for _, param in named_params]
        summary[group] = {
            "total": parameter_count(params),
            "trainable": parameter_count(param for param in params if param.requires_grad),
            "tensors": len(params),
            "trainable_tensors": sum(1 for param in params if param.requires_grad),
        }
    summary["all"] = {
        "total": sum(item["total"] for item in summary.values()),
        "trainable": sum(item["trainable"] for item in summary.values()),
    }
    return summary


def expected_trainable_groups(arm: str) -> set[str]:
    if arm == "generator":
        return {"generator"}
    if arm == "text":
        return {"clip_text"}
    if arm == "joint":
        return {"generator", "clip_text"}
    raise ValueError(arm)


def named_parameter_sha256(named_params: Iterable[tuple[str, torch.nn.Parameter]]) -> str:
    digest = hashlib.sha256()
    for name, param in sorted(named_params, key=lambda item: item[0]):
        tensor = param.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def group_hashes(groups: dict[str, list[tuple[str, torch.nn.Parameter]]]) -> dict[str, str]:
    return {name: named_parameter_sha256(params) for name, params in groups.items()}


def verify_frozen_hashes(
    initial: dict[str, str],
    final: dict[str, str],
    arm: str,
) -> dict[str, bool]:
    frozen = set(initial) - expected_trainable_groups(arm)
    result = {group: initial[group] == final[group] for group in sorted(frozen)}
    bad = [group for group, ok in result.items() if not ok]
    if bad:
        raise RuntimeError(f"frozen parameter groups changed unexpectedly: {bad}")
    return result


def gradient_group_norms(groups: dict[str, list[tuple[str, torch.nn.Parameter]]]) -> dict[str, float]:
    norms = {}
    for group, named_params in groups.items():
        total = 0.0
        for _, param in named_params:
            if param.requires_grad and param.grad is not None:
                grad = param.grad.detach()
                if not torch.isfinite(grad).all():
                    raise FloatingPointError(f"non-finite gradient in group {group}")
                total += float(grad.float().pow(2).sum().item())
        norms[group] = math.sqrt(total)
    return norms


def assert_first_backward_reachability(groups: dict[str, list[tuple[str, torch.nn.Parameter]]], arm: str) -> dict[str, float]:
    norms = gradient_group_norms(groups)
    for group in expected_trainable_groups(arm):
        if norms.get(group, 0.0) <= 0.0:
            raise RuntimeError(f"expected non-zero gradient reachability for {group} in {arm} arm")
    for group, named_params in groups.items():
        if group in expected_trainable_groups(arm):
            continue
        leaked = [name for name, param in named_params if param.grad is not None]
        if leaked:
            raise RuntimeError(f"frozen group {group} received gradients: {leaked[:5]}")
    return norms


def build_optimizer(
    trans_encoder: torch.nn.Module,
    clip_model: torch.nn.Module,
    arm: str,
    trans_lr: float,
    clip_lr: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    param_groups = []
    if arm in {"generator", "joint"}:
        params = [param for param in trans_encoder.parameters() if param.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": trans_lr, "name": "generator"})
    if arm in {"text", "joint"}:
        params = [param for name, param in clip_model.named_parameters() if param.requires_grad and is_clip_text_parameter(name)]
        if params:
            param_groups.append({"params": params, "lr": clip_lr, "name": "clip_text"})
    if not param_groups:
        raise RuntimeError(f"no trainable parameters configured for arm={arm}")
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def make_training_batch(
    examples: Sequence[PairExample],
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
    nb_code: int,
    block_size: int,
    pkeep: float,
) -> Batch:
    if pkeep != -1 and not (0.0 <= pkeep <= 1.0):
        raise ValueError("--pkeep must be in [0,1] or -1")
    pair_indices = rng.integers(0, len(examples), size=batch_size, endpoint=False).astype(np.int64)
    token_rows = []
    token_row_indices: list[int] = []
    clean_texts: list[str] = []
    pair_texts: list[str] = []
    for pair_index in pair_indices:
        example = examples[int(pair_index)]
        row_index = int(rng.integers(0, example.token_rows.shape[0], endpoint=False))
        tokens = example.token_rows[row_index]
        if len(tokens) + 1 > block_size:
            raise TrainingDataError(
                f"token sequence exceeds transformer block_size={block_size}: {example.motion_id} length={len(tokens)}"
            )
        token_rows.append(tokens)
        token_row_indices.append(row_index)
        clean_texts.append(example.text_orig)
        pair_texts.append(example.text_pair)

    lengths_np = np.asarray([len(tokens) + 1 for tokens in token_rows], dtype=np.int64)
    max_target_len = int(lengths_np.max())
    end_idx = nb_code
    pad_idx = nb_code + 1
    targets_np = np.full((batch_size, max_target_len), pad_idx, dtype=np.int64)
    for item, tokens in enumerate(token_rows):
        raw_len = len(tokens)
        targets_np[item, :raw_len] = tokens
        targets_np[item, raw_len] = end_idx
    input_np = targets_np[:, :-1].copy()
    keep_probability = float(rng.random()) if pkeep == -1 else float(pkeep)
    keep_mask = rng.random(input_np.shape) < keep_probability
    random_indices = rng.integers(0, nb_code, size=input_np.shape, endpoint=False, dtype=np.int64)
    corrupted_np = np.where(keep_mask, input_np, random_indices).astype(np.int64, copy=False)

    plan = {
        "pair_indices": pair_indices.tolist(),
        "token_row_indices": token_row_indices,
        "lengths": lengths_np.tolist(),
        "keep_probability": keep_probability,
    }
    corruption_sha = hashlib.sha256(
        keep_mask.astype(np.uint8).tobytes() + random_indices.astype(np.int64).tobytes()
    ).hexdigest()
    plan_sha = canonical_sha256({**plan, "corruption_sha256": corruption_sha})

    duplicated_inputs = np.concatenate([corrupted_np, corrupted_np], axis=0)
    duplicated_targets = np.concatenate([targets_np, targets_np], axis=0)
    duplicated_lengths = np.concatenate([lengths_np, lengths_np], axis=0)
    return Batch(
        texts=clean_texts + pair_texts,
        input_indices=torch.as_tensor(duplicated_inputs, dtype=torch.long, device=device),
        targets=torch.as_tensor(duplicated_targets, dtype=torch.long, device=device),
        lengths=torch.as_tensor(duplicated_lengths, dtype=torch.long, device=device),
        pair_indices=pair_indices.tolist(),
        token_row_indices=token_row_indices,
        corruption_sha256=corruption_sha,
        plan_sha256=plan_sha,
    )


def masked_sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError(f"bad logits/target shapes: {tuple(logits.shape)} {tuple(targets.shape)}")
    steps = targets.shape[1]
    if logits.shape[1] < steps:
        raise ValueError(f"logits shorter than targets: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    logits = logits[:, :steps, :].contiguous()
    mask = torch.arange(steps, device=targets.device).unsqueeze(0) < lengths.unsqueeze(1)
    # Padding uses nb_code + 1, which is outside the output vocabulary.
    # Ignore it before cross entropy rather than multiplying its loss by zero.
    safe_targets = targets.masked_fill(~mask, -100)
    losses = F.cross_entropy(logits.view(-1, logits.shape[-1]), safe_targets.reshape(-1), reduction="none", ignore_index=-100)
    losses = losses.view(targets.shape)
    denom = mask.sum().clamp_min(1).to(losses.dtype)
    loss = (losses * mask.to(losses.dtype)).sum() / denom
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        accuracy = ((preds == targets) & mask).sum().to(losses.dtype) / denom
    return loss, accuracy


def compute_paired_supervised_loss(
    clip_model: torch.nn.Module,
    trans_encoder: torch.nn.Module,
    text_tokens: torch.Tensor,
    input_indices: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    clean_batch_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    clip_features = clip_model.encode_text(text_tokens).float()
    logits = trans_encoder(input_indices, clip_features).contiguous()
    clean_loss, clean_acc = masked_sequence_cross_entropy(
        logits[:clean_batch_size], targets[:clean_batch_size], lengths[:clean_batch_size]
    )
    pair_loss, pair_acc = masked_sequence_cross_entropy(
        logits[clean_batch_size:], targets[clean_batch_size:], lengths[clean_batch_size:]
    )
    loss = 0.5 * (clean_loss + pair_loss)
    metrics = {
        "clean_loss": clean_loss.detach(),
        "paraphrase_loss": pair_loss.detach(),
        "clean_accuracy": clean_acc.detach(),
        "paraphrase_accuracy": pair_acc.detach(),
    }
    return loss, metrics


def finite_trainable_parameters(named_params: Iterable[tuple[str, torch.nn.Parameter]]) -> None:
    for name, param in named_params:
        if param.requires_grad and not torch.isfinite(param.detach()).all():
            raise FloatingPointError(f"non-finite parameter: {name}")


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def save_checkpoint(
    path: Path,
    step: int,
    model_args: Namespace,
    cli_args: argparse.Namespace,
    clip_model: torch.nn.Module,
    trans_encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict,
) -> None:
    payload = {
        "iter": int(step),
        "args": vars(model_args),
        "clip": unwrap_module(clip_model).state_dict(),
        "trans": unwrap_module(trans_encoder).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "metadata": metadata,
        "pilot": {
            "schema_version": "motionx-text-adaptation-pilot/v1",
            "arm": cli_args.arm,
            "step": int(step),
            "objective": "plain supervised clean+paraphrase -> same ground-truth VQ tokens, equal weight",
            "sato_heads_or_losses": False,
            "checkpoint_compatibility": "ablation/repeat20_perturb_fast_batch.py kind=sato_pp uses args, clip, trans, iter",
        },
    }
    torch.save(payload, path)


def make_logger(out_dir: Path) -> logging.Logger:
    logger = logging.getLogger("motionx_text_adaptation_pilot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(out_dir / "train.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_clip_model(device: torch.device) -> torch.nn.Module:
    import clip  # Imported lazily so unit tests with fake modules do not need CLIP.

    clip_model, _ = clip.load(CLIP_MODEL_NAME, device=device, jit=False)
    clip_model.float()
    clip_model.eval()
    return clip_model


def load_transformer(model_args: Namespace, device: torch.device) -> torch.nn.Module:
    import models.t2m_trans as trans

    trans_encoder = trans.Text2Motion_Transformer(
        num_vq=model_args.nb_code,
        embed_dim=model_args.embed_dim_gpt,
        clip_dim=model_args.clip_dim,
        block_size=model_args.block_size,
        num_layers=model_args.num_layers,
        n_head=model_args.n_head_gpt,
        drop_out_rate=model_args.drop_out_rate,
        fc_rate=model_args.ff_rate,
    )
    checkpoint = torch.load(resolve_repo_path(model_args.resume_trans), map_location="cpu")
    trans_encoder.load_state_dict(checkpoint["trans"], strict=True)
    trans_encoder.float().to(device)
    trans_encoder.eval()  # strict comparison: dropout disabled even when generator is trainable
    return trans_encoder


def tokenize_texts(texts: list[str], device: torch.device) -> torch.Tensor:
    import clip

    return clip.tokenize(texts, truncate=True).to(device)


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(cli_args: argparse.Namespace) -> dict:
    if cli_args.steps < 1:
        raise ValueError("--steps must be positive")
    if cli_args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if cli_args.save_every < 1 or cli_args.log_every < 1:
        raise ValueError("--save-every and --log-every must be positive")

    out_dir = ensure_empty_output_dir(cli_args.out_dir)
    logger = make_logger(out_dir)
    seed_all(cli_args.seed)
    device = torch.device(cli_args.device)
    model_args = make_model_args(cli_args)

    for required in (BASELINE_TRANS, BASELINE_VQ, TOKEN_DIR):
        if not required.exists():
            raise FileNotFoundError(f"missing required baseline/token asset: {required}")

    examples, filter_report = load_training_pairs(
        cli_args.train_pairs,
        dataset_root=DATASET_ROOT,
        token_dir=TOKEN_DIR,
        nb_code=model_args.nb_code,
        dataset_limit=cli_args.dataset_limit,
    )
    source_metadata = {
        "baseline_transformer": str(BASELINE_TRANS),
        "baseline_transformer_sha256": sha256_file(BASELINE_TRANS),
        "baseline_vqvae": str(BASELINE_VQ),
        "baseline_vqvae_sha256": sha256_file(BASELINE_VQ),
        "train_pairs_sha256": sha256_file(resolve_repo_path(cli_args.train_pairs)),
        "clip_pretrained": CLIP_MODEL_NAME,
        "vq_token_dir": str(TOKEN_DIR),
    }
    logger.info("Filtering report: %s", json.dumps(filter_report, sort_keys=True))

    clip_model = load_clip_model(device)
    trans_encoder = load_transformer(model_args, device)
    clip_model.eval()
    trans_encoder.eval()
    trainability = configure_trainability(trans_encoder, clip_model, cli_args.arm)
    groups = collect_parameter_groups(trans_encoder, clip_model)
    optimizer = build_optimizer(
        trans_encoder,
        clip_model,
        cli_args.arm,
        trans_lr=cli_args.trans_lr,
        clip_lr=cli_args.clip_lr,
        weight_decay=cli_args.weight_decay,
    )
    logger.info("Trainable parameter counts: %s", json.dumps(trainability, sort_keys=True))

    initial_hashes = group_hashes(groups)
    manifest = {
        "schema_version": "motionx-text-adaptation-pilot/v1",
        "arm": cli_args.arm,
        "args": vars(model_args),
        "objective": {
            "type": "plain supervised cross entropy",
            "clean_caption_weight": 0.5,
            "paraphrase_caption_weight": 0.5,
            "target": "same cached full-motion VQ token sequence for text_orig and text_pair",
            "sato_heads_or_losses": False,
            "motion_decoder_training": False,
        },
        "comparison_controls": {
            "local_rng": "numpy.default_rng(seed) samples pair indices, token-row indices, and corruption masks independently of model RNG",
            "clean_samples_every_step": True,
            "generator_dropout": "disabled consistently by keeping trans_encoder.eval() for all arms; gradients still update trainable parameters",
            "precision": "fp32",
            "validation_or_test_training": False,
            "checkpoint_selection": "none; fixed-interval and final checkpoints only",
        },
        "filtering": filter_report,
        "source_metadata": source_metadata,
        "trainability": trainability,
        "frozen_hashes_initial": initial_hashes,
    }
    write_manifest(out_dir / "manifest.json", manifest)

    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "train_losses.csv"
    csv_fields = [
        "step",
        "loss",
        "clean_loss",
        "paraphrase_loss",
        "clean_accuracy",
        "paraphrase_accuracy",
        "grad_norm",
        "batch_plan_sha256",
        "corruption_sha256",
    ]
    local_rng = np.random.default_rng(cli_args.seed)
    schedule_digest = hashlib.sha256()
    first_grad_norms = None
    last_metrics = None

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for step in range(1, cli_args.steps + 1):
            batch = make_training_batch(
                examples,
                cli_args.batch_size,
                local_rng,
                device,
                nb_code=model_args.nb_code,
                block_size=model_args.block_size,
                pkeep=cli_args.pkeep,
            )
            text_tokens = tokenize_texts(batch.texts, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_paired_supervised_loss(
                clip_model,
                trans_encoder,
                text_tokens,
                batch.input_indices,
                batch.targets,
                batch.lengths,
                clean_batch_size=cli_args.batch_size,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {float(loss.detach().cpu())}")
            loss.backward()
            if step == 1:
                first_grad_norms = assert_first_backward_reachability(groups, cli_args.arm)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                [param for group in groups.values() for _, param in group if param.requires_grad],
                max_norm=cli_args.grad_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            finite_trainable_parameters((name, param) for group in groups.values() for name, param in group)
            schedule_digest.update(batch.plan_sha256.encode("ascii"))

            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "clean_loss": float(metrics["clean_loss"].cpu()),
                "paraphrase_loss": float(metrics["paraphrase_loss"].cpu()),
                "clean_accuracy": float(metrics["clean_accuracy"].cpu()),
                "paraphrase_accuracy": float(metrics["paraphrase_accuracy"].cpu()),
                "grad_norm": float(grad_norm_tensor.detach().cpu()),
                "batch_plan_sha256": batch.plan_sha256,
                "corruption_sha256": batch.corruption_sha256,
            }
            writer.writerow(row)
            handle.flush()
            last_metrics = row

            if step % cli_args.log_every == 0 or step == 1 or step == cli_args.steps:
                logger.info(
                    "step=%d loss=%.6f clean=%.6f paraphrase=%.6f grad_norm=%.6f",
                    step,
                    row["loss"],
                    row["clean_loss"],
                    row["paraphrase_loss"],
                    row["grad_norm"],
                )
            if step % cli_args.save_every == 0:
                save_checkpoint(
                    checkpoints_dir / f"motionx_text_adaptation_pilot_iter_{step:06d}.pth",
                    step,
                    model_args,
                    cli_args,
                    clip_model,
                    trans_encoder,
                    optimizer,
                    metadata={**manifest, "first_backward_grad_norms": first_grad_norms, "schedule_sha256_so_far": schedule_digest.hexdigest()},
                )

    final_hashes = group_hashes(groups)
    frozen_verified = verify_frozen_hashes(initial_hashes, final_hashes, cli_args.arm)
    manifest.update(
        {
            "first_backward_grad_norms": first_grad_norms,
            "frozen_hashes_final": final_hashes,
            "frozen_hashes_verified": frozen_verified,
            "schedule_sha256": schedule_digest.hexdigest(),
            "last_metrics": last_metrics,
            "outputs": {
                "loss_csv": str(csv_path),
                "checkpoints_dir": str(checkpoints_dir),
                "final_checkpoint": str(out_dir / "motionx_text_adaptation_pilot_last.pth"),
            },
        }
    )
    save_checkpoint(
        out_dir / "motionx_text_adaptation_pilot_last.pth",
        cli_args.steps,
        model_args,
        cli_args,
        clip_model,
        trans_encoder,
        optimizer,
        metadata=manifest,
    )
    write_manifest(out_dir / "manifest.json", manifest)
    logger.info("Finished pilot; final checkpoint: %s", out_dir / "motionx_text_adaptation_pilot_last.pth")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()