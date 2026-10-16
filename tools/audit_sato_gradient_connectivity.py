#!/usr/bin/env python3
"""Audit SATO++ per-loss gradient connectivity on a fixed real HumanML3D batch."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models.t2m_trans as trans
from dataset.config import get_dataset_spec
from dataset.slot_parser import SlotParser
from models.clip_attribution import (
    compute_grad_input_attribution,
    configure_second_order_attention,
    normalize_attribution,
)
from models.sato_pp_losses import (
    loss_align,
    loss_attribution_diff,
    loss_covariance,
    loss_locality,
    loss_teacher,
    loss_variance,
)
from models.sato_pp_modules import AttributionPooling, PredictorHead


DEFAULT_CHECKPOINT = REPO_ROOT / (
    "output_GPT_Final/"
    "sato_pp_allloss_gpu0_teachl1_05_eval5k_resume104k_200k_single/"
    "sato_pp_iter_130000.pth"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "ablation/attribution_latent_gpu6_20260905/gradient_audit.json"
)

LOSS_NAMES = ("gen", "teacher", "align", "var", "cov", "diff", "loc")
GROUP_NAMES = ("transformer", "clip", "attr_pooling", "predictor")


@dataclass
class SourceRow:
    pair_file_line: int
    motion_id: str
    relation: str
    text_orig: str
    text_pair: str
    edit_type: str
    row: dict[str, Any]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_clip_module():
    try:
        import clip as clip_module
    except ModuleNotFoundError:
        from clip_local import clip as clip_module
    return clip_module


def as_path(repo_root: Path, value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def require_gpu6() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "6":
        raise RuntimeError(
            "This audit must run exclusively with CUDA_VISIBLE_DEVICES=6; "
            f"got {visible!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real SATO++ gradient audit.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expected exactly one visible CUDA device after CUDA_VISIBLE_DEVICES=6; "
            f"got {torch.cuda.device_count()}."
        )


def build_transformer(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = trans.Text2Motion_Transformer(
        num_vq=int(cfg["nb_code"]),
        embed_dim=int(cfg["embed_dim_gpt"]),
        clip_dim=int(cfg["clip_dim"]),
        block_size=int(cfg["block_size"]),
        num_layers=int(cfg["num_layers"]),
        n_head=int(cfg["n_head_gpt"]),
        drop_out_rate=float(cfg["drop_out_rate"]),
        fc_rate=int(cfg["ff_rate"]),
    )
    return model.to(device)


def load_models(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    clip = get_clip_module()

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.float()
    clip_model.load_state_dict(checkpoint["clip"], strict=True)

    transformer = build_transformer(cfg, device)
    transformer.load_state_dict(checkpoint["trans"], strict=True)

    attr_pooling = AttributionPooling(
        clip_dim=int(cfg.get("clip_dim_attr", 512)),
        proj_dim=int(cfg.get("proj_dim", 512)),
        proj_type=str(cfg.get("proj_type", "linear")),
    ).to(device)
    attr_pooling.load_state_dict(checkpoint["attr_pooling"], strict=True)

    predictor = PredictorHead(dim=int(cfg.get("proj_dim", 512))).to(device)
    predictor.load_state_dict(checkpoint["predictor_head"], strict=True)

    teacher_path = as_path(REPO_ROOT, cfg.get("resume_trans"))
    if teacher_path is None or not teacher_path.exists():
        raise FileNotFoundError(
            f"Cannot reconstruct frozen teacher from resume_trans={cfg.get('resume_trans')!r}"
        )
    teacher_state = torch.load(teacher_path, map_location="cpu", weights_only=False)
    teacher = build_transformer(cfg, device)
    teacher.load_state_dict(teacher_state["trans"], strict=True)
    del teacher_state

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()

    unfreeze_clip = bool(cfg.get("unfreeze_clip", True))
    if unfreeze_clip:
        clip_model.train()
        for parameter in clip_model.parameters():
            parameter.requires_grad_(True)
    else:
        clip_model.eval()
        for parameter in clip_model.parameters():
            parameter.requires_grad_(False)

    transformer.train()
    for parameter in transformer.parameters():
        parameter.requires_grad_(True)
    attr_pooling.train()
    for parameter in attr_pooling.parameters():
        parameter.requires_grad_(True)
    predictor.train()
    for parameter in predictor.parameters():
        parameter.requires_grad_(True)

    model_provenance = {
        "checkpoint_path": str(checkpoint_path),
        "teacher_resume_trans_path": str(teacher_path),
        "teacher_resume_trans_sha256": file_sha256(teacher_path),
    }
    return clip_model, transformer, teacher, attr_pooling, predictor, model_provenance


def select_source_rows(
    pair_path: Path,
    data_root: Path,
    vq_name: str,
    n_preserve: int,
    n_alter: int,
) -> list[SourceRow]:
    train_path = data_root / "train.txt"
    train_ids = set(train_path.read_text(encoding="utf-8").split())
    slot_parser = SlotParser()
    preserve: list[SourceRow] = []
    alter: list[SourceRow] = []
    seen_preserve_motion: set[str] = set()

    with pair_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            relation = str(row.get("relation", ""))
            motion_id = str(row.get("motion_id", ""))
            if relation not in {"preserve", "alter"}:
                continue
            if not motion_id or motion_id not in train_ids:
                continue
            token_path = data_root / vq_name / f"{motion_id}.npy"
            if not token_path.exists():
                continue
            text_orig = str(row.get("text_orig", ""))
            text_pair = str(row.get("text_pair", ""))
            if not text_orig or not text_pair:
                continue

            source = SourceRow(
                pair_file_line=line_number,
                motion_id=motion_id,
                relation=relation,
                text_orig=text_orig,
                text_pair=text_pair,
                edit_type=str(row.get("edit_type") or ""),
                row=row,
            )
            if relation == "preserve":
                if motion_id in seen_preserve_motion:
                    continue
                if len(preserve) < n_preserve:
                    preserve.append(source)
                    seen_preserve_motion.add(motion_id)
            elif relation == "alter" and len(alter) < n_alter:
                mask = slot_parser.get_edited_token_mask(text_orig, text_pair, 77)
                if int(mask.sum().item()) > 0:
                    alter.append(source)

            if len(preserve) >= n_preserve and len(alter) >= n_alter:
                break

    if len(preserve) < n_preserve or len(alter) < n_alter:
        raise RuntimeError(
            f"Unable to select fixed audit batch with {n_preserve} preserve and "
            f"{n_alter} alter rows from {pair_path}."
        )
    return preserve + alter


def make_motion_tokens(
    token_path: Path,
    block_size: int,
    nb_code: int,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    raw = np.load(token_path, allow_pickle=False)
    if raw.ndim == 2:
        selected = raw[0]
        selected_sequence_index = 0
    else:
        selected = raw.reshape(-1)
        selected_sequence_index = None
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    if len(selected) > block_size - 1:
        selected = selected[: block_size - 1]
    motion_token_len = int(selected.shape[0])
    if motion_token_len + 1 < block_size:
        padded = np.concatenate(
            [
                selected,
                np.ones((1,), dtype=np.int64) * nb_code,
                np.ones((block_size - 1 - motion_token_len,), dtype=np.int64)
                * (nb_code + 1),
            ],
            axis=0,
        )
    else:
        padded = np.concatenate(
            [selected, np.ones((1,), dtype=np.int64) * nb_code], axis=0
        )
    metadata = {
        "raw_shape": list(raw.shape),
        "selected_sequence_index": selected_sequence_index,
        "motion_token_len_without_eos": motion_token_len,
        "padded_len": int(padded.shape[0]),
        "token_sha256": hashlib.sha256(
            np.asarray(padded, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    return padded.astype(np.int64), motion_token_len, metadata


def build_fixed_batch(
    cfg: dict[str, Any],
    n_preserve: int,
    n_alter: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = get_dataset_spec(str(cfg.get("dataname", "t2m")))
    data_root = Path(spec.root)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    pair_path = as_path(REPO_ROOT, cfg.get("pair_data_path"))
    if pair_path is None or not pair_path.exists():
        raise FileNotFoundError(f"pair_data_path not found: {cfg.get('pair_data_path')!r}")

    vq_name = str(cfg.get("vq_name", "VQVAE"))
    block_size = int(cfg["block_size"])
    nb_code = int(cfg["nb_code"])
    slot_parser = SlotParser()
    sources = select_source_rows(pair_path, data_root, vq_name, n_preserve, n_alter)

    token_rows: list[np.ndarray] = []
    token_lens: list[int] = []
    slot_masks: list[np.ndarray] = []
    source_records: list[dict[str, Any]] = []
    for batch_index, source in enumerate(sources):
        token_path = data_root / vq_name / f"{source.motion_id}.npy"
        padded, motion_token_len, token_meta = make_motion_tokens(
            token_path, block_size, nb_code
        )
        token_rows.append(padded)
        token_lens.append(motion_token_len)
        edited_mask = slot_parser.get_edited_token_mask(
            source.text_orig, source.text_pair, 77
        ).numpy().astype(bool)
        slot_masks.append(edited_mask)
        source_records.append(
            {
                "batch_index": batch_index,
                "pair_file": str(pair_path),
                "pair_file_line": source.pair_file_line,
                "motion_id": source.motion_id,
                "token_path": str(token_path),
                "relation": source.relation,
                "edit_type": source.edit_type,
                "text_orig": source.text_orig,
                "text_pair": source.text_pair,
                "slot_mask_true_count": int(edited_mask.sum()),
                **token_meta,
            }
        )

    batch = {
        "cap_orig": [source.text_orig for source in sources],
        "cap_pair": [source.text_pair for source in sources],
        "relation": [source.relation for source in sources],
        "edit_type": [source.edit_type for source in sources],
        "m_tokens_np": np.stack(token_rows).astype(np.int64),
        "m_tokens_len_np": np.asarray(token_lens, dtype=np.int64),
        "slot_mask_np": np.stack(slot_masks).astype(bool),
    }
    batch_meta = {
        "data_root": str(data_root),
        "pair_data_path": str(pair_path),
        "pair_data_sha256": file_sha256(pair_path),
        "train_split_path": str(data_root / "train.txt"),
        "vq_name": vq_name,
        "selection_policy": (
            "first distinct train-split preserve rows plus first train-split alter "
            "row with nonzero SlotParser mask; deterministic, no random crop"
        ),
        "preserve_count": sum(source.relation == "preserve" for source in sources),
        "alter_count": sum(source.relation == "alter" for source in sources),
        "batch_size": len(sources),
        "sources": source_records,
        "batch_sha256": canonical_sha256(source_records),
    }
    return batch, batch_meta


def module_param_summary(module: torch.nn.Module) -> dict[str, Any]:
    params = list(module.named_parameters())
    trainable = [(name, parameter) for name, parameter in params if parameter.requires_grad]
    return {
        "param_count": len(params),
        "param_numel": int(sum(parameter.numel() for _, parameter in params)),
        "trainable_param_count": len(trainable),
        "trainable_numel": int(sum(parameter.numel() for _, parameter in trainable)),
        "frozen_param_count": len(params) - len(trainable),
        "frozen_numel": int(
            sum(parameter.numel() for _, parameter in params if not parameter.requires_grad)
        ),
        "mode": "train" if module.training else "eval",
    }


def trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def grad_stats_for_group(
    loss: torch.Tensor | float,
    params: list[torch.nn.Parameter],
    summary: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = dict(summary)
    loss_is_tensor = torch.is_tensor(loss)
    loss_requires_grad = bool(loss_is_tensor and loss.requires_grad)
    result["loss_requires_grad"] = loss_requires_grad

    if result["trainable_param_count"] == 0:
        result.update(
            {
                "status": "frozen_no_trainable_parameters",
                "disconnected_param_count": 0,
                "disconnected_numel": 0,
                "connected_param_count": 0,
                "connected_numel": 0,
                "connected_zero_param_count": 0,
                "connected_zero_numel": 0,
                "connected_nonzero_param_count": 0,
                "connected_nonzero_numel": 0,
                "grad_l2_norm": None,
                "grad_l1_norm": None,
                "grad_max_abs": None,
            }
        )
        return result

    if not loss_requires_grad:
        result.update(
            {
                "status": "loss_has_no_grad_fn",
                "disconnected_param_count": result["trainable_param_count"],
                "disconnected_numel": result["trainable_numel"],
                "connected_param_count": 0,
                "connected_numel": 0,
                "connected_zero_param_count": 0,
                "connected_zero_numel": 0,
                "connected_nonzero_param_count": 0,
                "connected_nonzero_numel": 0,
                "grad_l2_norm": 0.0,
                "grad_l1_norm": 0.0,
                "grad_max_abs": 0.0,
            }
        )
        return result

    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )
    disconnected_param_count = 0
    disconnected_numel = 0
    connected_param_count = 0
    connected_numel = 0
    connected_zero_param_count = 0
    connected_zero_numel = 0
    connected_nonzero_param_count = 0
    connected_nonzero_numel = 0
    sum_sq = 0.0
    sum_abs = 0.0
    max_abs = 0.0

    for parameter, grad in zip(params, grads):
        numel = int(parameter.numel())
        if grad is None:
            disconnected_param_count += 1
            disconnected_numel += numel
            continue
        connected_param_count += 1
        connected_numel += numel
        grad_float = grad.detach().float()
        grad_abs = grad_float.abs()
        local_sum_abs = float(grad_abs.sum().item())
        local_max_abs = float(grad_abs.max().item()) if grad_abs.numel() else 0.0
        local_sum_sq = float(grad_float.double().pow(2).sum().item())
        sum_abs += local_sum_abs
        sum_sq += local_sum_sq
        max_abs = max(max_abs, local_max_abs)
        if local_max_abs == 0.0:
            connected_zero_param_count += 1
            connected_zero_numel += numel
        else:
            connected_nonzero_param_count += 1
            connected_nonzero_numel += numel

    result.update(
        {
            "status": "computed",
            "disconnected_param_count": disconnected_param_count,
            "disconnected_numel": disconnected_numel,
            "connected_param_count": connected_param_count,
            "connected_numel": connected_numel,
            "connected_zero_param_count": connected_zero_param_count,
            "connected_zero_numel": connected_zero_numel,
            "connected_nonzero_param_count": connected_nonzero_param_count,
            "connected_nonzero_numel": connected_nonzero_numel,
            "grad_l2_norm": math.sqrt(sum_sq),
            "grad_l1_norm": sum_abs,
            "grad_max_abs": max_abs,
        }
    )
    return result


def tensor_scalar(value: torch.Tensor | float) -> float | None:
    if not torch.is_tensor(value):
        return float(value)
    if value.numel() != 1:
        return None
    return float(value.detach().cpu().item())


def compute_loss_graph(
    cfg: dict[str, Any],
    batch: dict[str, Any],
    clip_model: torch.nn.Module,
    transformer: torch.nn.Module,
    teacher: torch.nn.Module,
    attr_pooling: torch.nn.Module,
    predictor: torch.nn.Module,
    device: torch.device,
    create_graph: bool,
    audit_seed: int,
) -> dict[str, Any]:
    clip = get_clip_module()
    configure_second_order_attention(create_graph)
    random.seed(audit_seed)
    np.random.seed(audit_seed)
    torch.manual_seed(audit_seed)
    torch.cuda.manual_seed_all(audit_seed)

    m_tokens = torch.as_tensor(batch["m_tokens_np"], dtype=torch.long, device=device)
    m_tokens_len = torch.as_tensor(
        batch["m_tokens_len_np"], dtype=torch.long, device=device
    )
    slot_mask = torch.as_tensor(batch["slot_mask_np"], dtype=torch.bool, device=device)
    relation = list(batch["relation"])
    target = m_tokens
    bs = int(target.shape[0])

    text_orig = clip.tokenize(list(batch["cap_orig"]), truncate=True).to(device)
    text_pair = clip.tokenize(list(batch["cap_pair"]), truncate=True).to(device)

    with torch.cuda.amp.autocast(enabled=False):
        H_orig, feat_orig = clip_model.encode_text_with_features(text_orig)
        H_pair, feat_pair = clip_model.encode_text_with_features(text_pair)
        H_orig = H_orig.float()
        H_pair = H_pair.float()
        feat_orig = feat_orig.float()
        feat_pair = feat_pair.float()

    input_index = target[:, :-1]
    pkeep = float(cfg.get("pkeep", 0.5))
    if pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(input_index.shape, device=device))
    else:
        mask = torch.bernoulli(pkeep * torch.ones(input_index.shape, device=device))
    mask = mask.round().to(dtype=torch.int64)
    r_indices = torch.randint_like(input_index, int(cfg["nb_code"]))
    a_indices = mask * input_index + (1 - mask) * r_indices

    cls_pred = transformer(a_indices, feat_orig).contiguous()
    loss_ce = torch.nn.CrossEntropyLoss()
    loss_gen: torch.Tensor | float = 0.0
    for i in range(bs):
        loss_gen = loss_gen + loss_ce(
            cls_pred[i][: m_tokens_len[i] + 1], target[i][: m_tokens_len[i] + 1]
        ) / bs

    with torch.no_grad():
        teacher_pred = teacher(a_indices, feat_orig)
    loss_teach = loss_teacher(
        cls_pred,
        teacher_pred,
        m_tokens_len + 1,
        loss_type=str(cfg.get("teacher_loss_type", "kl")),
    )

    zero = torch.tensor(0.0, device=device)
    loss_align_val: torch.Tensor | float = zero
    loss_diff_val: torch.Tensor | float = zero
    loss_loc_val: torch.Tensor | float = zero
    n_preserve = 0
    n_alter = 0
    u_orig_list: list[torch.Tensor] = []
    u_pair_list: list[torch.Tensor] = []
    alpha_orig_rows: list[torch.Tensor | None] = []
    alpha_pair_rows: list[torch.Tensor | None] = []

    for i, rel in enumerate(relation):
        if rel == "preserve":
            n_preserve += 1
            alpha_orig = compute_grad_input_attribution(
                clip_model,
                transformer,
                text_orig[i : i + 1],
                target[i : i + 1],
                K=int(cfg.get("attribution_K", 16)),
                valid_len=m_tokens_len[i : i + 1],
                create_graph=create_graph,
            )
            alpha_orig = normalize_attribution(alpha_orig, float(cfg.get("tau", 1.0)))
            alpha_pair = compute_grad_input_attribution(
                clip_model,
                transformer,
                text_pair[i : i + 1],
                target[i : i + 1],
                K=int(cfg.get("attribution_K", 16)),
                valid_len=m_tokens_len[i : i + 1],
                create_graph=create_graph,
            )
            alpha_pair = normalize_attribution(alpha_pair, float(cfg.get("tau", 1.0)))
            alpha_orig_rows.append(alpha_orig)
            alpha_pair_rows.append(alpha_pair)
            u_orig = attr_pooling(H_orig[i : i + 1], alpha_orig)
            u_pair = attr_pooling(H_pair[i : i + 1], alpha_pair)
            z_orig = predictor(u_orig)
            z_pair = predictor(u_pair)
            loss_align_val = loss_align_val + loss_align(z_orig, z_pair)
            u_orig_list.append(u_orig)
            u_pair_list.append(u_pair)
        elif rel == "alter":
            n_alter += 1
            alpha_orig = compute_grad_input_attribution(
                clip_model,
                transformer,
                text_orig[i : i + 1],
                target[i : i + 1],
                K=int(cfg.get("attribution_K", 16)),
                valid_len=m_tokens_len[i : i + 1],
                create_graph=create_graph,
            )
            alpha_orig = normalize_attribution(alpha_orig, float(cfg.get("tau", 1.0)))
            alpha_pair = compute_grad_input_attribution(
                clip_model,
                transformer,
                text_pair[i : i + 1],
                target[i : i + 1],
                K=int(cfg.get("attribution_K", 16)),
                valid_len=m_tokens_len[i : i + 1],
                create_graph=create_graph,
            )
            alpha_pair = normalize_attribution(alpha_pair, float(cfg.get("tau", 1.0)))
            alpha_orig_rows.append(alpha_orig)
            alpha_pair_rows.append(alpha_pair)
            mask_i = slot_mask[i].unsqueeze(0)
            loss_diff_val = loss_diff_val + loss_attribution_diff(
                alpha_orig, alpha_pair, mask_i
            )
            loss_loc_val = loss_loc_val + loss_locality(
                alpha_orig, alpha_pair, mask_i
            )

    if n_preserve > 0:
        loss_align_val = loss_align_val / n_preserve
    loss_var_val: torch.Tensor | float = zero
    loss_cov_val: torch.Tensor | float = zero
    if n_preserve >= 2 and len(u_orig_list) >= 2:
        u_orig_cat = torch.cat(u_orig_list, dim=0)
        u_pair_cat = torch.cat(u_pair_list, dim=0)
        u_all = torch.cat([u_orig_cat, u_pair_cat], dim=0)
        loss_var_val = loss_variance(u_all, float(cfg.get("var_threshold", 1.0)))
        loss_cov_val = loss_covariance(u_all)
    if n_alter > 0:
        loss_diff_val = loss_diff_val / n_alter
        loss_loc_val = loss_loc_val / n_alter

    losses = {
        "gen": loss_gen,
        "teacher": loss_teach,
        "align": loss_align_val,
        "var": loss_var_val,
        "cov": loss_cov_val,
        "diff": loss_diff_val,
        "loc": loss_loc_val,
    }

    return {
        "losses": losses,
        "counts": {"n_preserve": n_preserve, "n_alter": n_alter},
        "teacher_source": "resume_trans_baseline_frozen",
        "pkeep_mask": {
            "pkeep": pkeep,
            "kept_count": int(mask.sum().item()),
            "total_count": int(mask.numel()),
            "kept_fraction": float(mask.float().mean().item()),
            "a_indices_sha256": hashlib.sha256(
                a_indices.detach().cpu().numpy().astype(np.int64).tobytes()
            ).hexdigest(),
        },
        "text_token_sha256": {
            "orig": hashlib.sha256(
                text_orig.detach().cpu().numpy().astype(np.int64).tobytes()
            ).hexdigest(),
            "pair": hashlib.sha256(
                text_pair.detach().cpu().numpy().astype(np.int64).tobytes()
            ).hexdigest(),
        },
        "alpha_requires_grad_any": bool(
            any(
                alpha is not None and alpha.requires_grad
                for alpha in [*alpha_orig_rows, *alpha_pair_rows]
            )
        ),
        "activation_requires_grad": {
            "H_orig": bool(H_orig.requires_grad),
            "H_pair": bool(H_pair.requires_grad),
            "feat_orig": bool(feat_orig.requires_grad),
            "feat_pair": bool(feat_pair.requires_grad),
            "cls_pred": bool(cls_pred.requires_grad),
        },
    }


def layerwise_attribution_audit(loss, modules):
    """Differentiate the weighted attribution objective once, then group norms.

    Frozen, disconnected, connected-zero and connected-nonzero are distinct.
    No generation or teacher term contributes to this backward pass.
    """
    entries = []
    for module_name in GROUP_NAMES:
        for name, parameter in modules[module_name].named_parameters():
            if module_name == 'clip' and (name.startswith('visual.') or name == 'logit_scale'):
                continue
            if module_name == 'transformer':
                pieces = name.split('.')
                if len(pieces) >= 4 and pieces[1] == 'blocks':
                    group = '.'.join(pieces[:3])
                elif name.startswith('trans_head.head.'):
                    group = 'generation_head'
                elif name.startswith('trans_head.ln_f.'):
                    group = 'final_layernorm'
                else:
                    group = 'conditioning_and_motion_embeddings'
            else:
                group = {'clip': 'text_encoder', 'attr_pooling': 'projection_head',
                         'predictor': 'predictor_head'}[module_name]
            entries.append((group, parameter, module_name))
    trainable = [p for _, p, _ in entries if p.requires_grad]
    gradients = torch.autograd.grad(loss, trainable, allow_unused=True, retain_graph=True) if loss.requires_grad else (None,) * len(trainable)
    by_id = {id(p): g for p, g in zip(trainable, gradients)}
    grouped = {}
    generator_sq = 0.0
    for name, p, module_name in entries:
        row = grouped.setdefault(name, {'parameter_tensors': 0, 'trainable_tensors': 0,
                                       'frozen_tensors': 0, 'connected_tensors': 0,
                                       'disconnected_tensors': 0, 'gradient_sum_sq': 0.0})
        row['parameter_tensors'] += 1
        if not p.requires_grad:
            row['frozen_tensors'] += 1
            continue
        row['trainable_tensors'] += 1
        grad = by_id[id(p)]
        if grad is None:
            row['disconnected_tensors'] += 1
            continue
        row['connected_tensors'] += 1
        sq = float(grad.detach().double().square().sum())
        row['gradient_sum_sq'] += sq
        if module_name == 'transformer':
            generator_sq += sq
    for row in grouped.values():
        sq = row.pop('gradient_sum_sq')
        row['status'] = ('frozen' if row['trainable_tensors'] == 0 else
                         'disconnected' if row['connected_tensors'] == 0 else
                         'connected_zero' if sq == 0 else 'connected_nonzero')
        row['grad_l2_norm'] = math.sqrt(sq) if row['connected_tensors'] else None
        row['sgd_update_gradient_l2'] = math.sqrt(sq)
    return {'definition': 'lambda_align*align + lambda_var*var + lambda_cov*cov + lambda_diff*diff + lambda_loc*loc; training relation normalization retained',
            'loss_value': tensor_scalar(loss), 'layers': grouped,
            'generator_gradient_l2': math.sqrt(generator_sq),
            'gradient_cosine_precondition': 'undefined_zero_generator_gradient' if generator_sq == 0 else 'requires_explicit_robustness_risk_definition',
            'virtual_generator_only_sgd': 'identity_for_every_step_size_without_other_losses_or_regularization' if generator_sq == 0 else 'not_run_without_robustness_risk_definition'}


def audit_once(
    cfg: dict[str, Any],
    batch: dict[str, Any],
    modules: dict[str, torch.nn.Module],
    device: torch.device,
    create_graph: bool,
    audit_seed: int,
) -> dict[str, Any]:
    graph = compute_loss_graph(
        cfg=cfg,
        batch=batch,
        clip_model=modules["clip"],
        transformer=modules["transformer"],
        teacher=modules["teacher"],
        attr_pooling=modules["attr_pooling"],
        predictor=modules["predictor"],
        device=device,
        create_graph=create_graph,
        audit_seed=audit_seed,
    )
    summaries = {
        name: module_param_summary(modules[name])
        for name in GROUP_NAMES
    }
    params = {name: trainable_parameters(modules[name]) for name in GROUP_NAMES}

    lambda_by_loss = {
        "gen": 1.0,
        "teacher": float(cfg.get("lambda_teach", 0.0)),
        "align": float(cfg.get("lambda_align", 0.0)),
        "var": float(cfg.get("lambda_var", 0.0)),
        "cov": float(cfg.get("lambda_cov", 0.0)),
        "diff": float(cfg.get("lambda_diff", 0.0)),
        "loc": float(cfg.get("lambda_loc", 0.0)),
    }

    loss_attr = sum(lambda_by_loss[name] * graph['losses'][name]
                    for name in ('align', 'var', 'cov', 'diff', 'loc'))
    attribution_total = layerwise_attribution_audit(loss_attr, modules)
    loss_reports: dict[str, Any] = {}
    for loss_name in LOSS_NAMES:
        loss = graph["losses"][loss_name]
        value = tensor_scalar(loss)
        weighted_value = None if value is None else value * lambda_by_loss[loss_name]
        loss_reports[loss_name] = {
            "value": value,
            "lambda_weight": lambda_by_loss[loss_name],
            "weighted_value": weighted_value,
            "requires_grad": bool(torch.is_tensor(loss) and loss.requires_grad),
            "groups": {
                group_name: grad_stats_for_group(
                    loss, params[group_name], summaries[group_name]
                )
                for group_name in GROUP_NAMES
            },
        }

    del graph["losses"]
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "create_graph": create_graph,
        "audit_seed": audit_seed,
        "counts": graph["counts"],
        "pkeep_mask": graph["pkeep_mask"],
        "text_token_sha256": graph["text_token_sha256"],
        "alpha_requires_grad_any": graph["alpha_requires_grad_any"],
        "activation_requires_grad": graph["activation_requires_grad"],
        "teacher_source": graph["teacher_source"],
        "attribution_total": attribution_total,
        "losses": loss_reports,
    }


def compact_args(cfg: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "dataname",
        "pair_data_path",
        "vq_name",
        "resume_pth",
        "resume_trans",
        "resume_sato_pp",
        "exp_name",
        "out_dir",
        "iter",
        "batch_size",
        "total_iter",
        "nb_code",
        "block_size",
        "embed_dim_gpt",
        "clip_dim",
        "clip_dim_attr",
        "num_layers",
        "n_head_gpt",
        "drop_out_rate",
        "ff_rate",
        "lambda_teach",
        "teacher_loss_type",
        "lambda_align",
        "lambda_var",
        "lambda_cov",
        "lambda_diff",
        "lambda_loc",
        "var_threshold",
        "attribution_method",
        "attribution_K",
        "attribution_create_graph",
        "tau",
        "unfreeze_clip",
        "pkeep",
        "proj_type",
        "proj_dim",
        "lr",
        "clip_lr",
        "optimizer",
        "weight_decay",
        "lr_scheduler",
        "gamma",
        "preserve_ratio",
        "dataset_limit",
    ]
    return {key: cfg[key] for key in keys if key in cfg}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit per-loss gradient connectivity for a real SATO++ checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-seed", type=int, default=20260905)
    parser.add_argument("--preserve", type=int, default=2)
    parser.add_argument("--alter", type=int, default=1)
    parser.add_argument(
        "--include-create-graph-true",
        action="store_true",
        help="Run an additional same-batch second-order connectivity probe.",
    )
    parser.add_argument(
        "--allow-non-gpu6",
        action="store_true",
        help="Only for local syntax/unit checks; the real audit should not use this.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_non_gpu6:
        require_gpu6()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The real checkpoint audit requires CUDA.")

    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint["args"])
    checkpoint_iter = int(checkpoint.get("iter", -1))

    actual_create_graph = bool(cfg.get("attribution_create_graph", False))
    cfg_for_false = dict(cfg)
    cfg_for_false["attribution_create_graph"] = actual_create_graph

    batch, batch_meta = build_fixed_batch(cfg, args.preserve, args.alter)
    clip_model, transformer, teacher, attr_pooling, predictor, model_provenance = load_models(
        checkpoint_path, checkpoint, cfg, device
    )
    del checkpoint
    gc.collect()

    modules = {
        "clip": clip_model,
        "transformer": transformer,
        "teacher": teacher,
        "attr_pooling": attr_pooling,
        "predictor": predictor,
    }
    freeze_settings = {
        name: module_param_summary(module)
        for name, module in modules.items()
    }
    freeze_settings["clip"]["actual_unfreeze_clip_arg"] = bool(
        cfg.get("unfreeze_clip", True)
    )
    freeze_settings["teacher"]["source"] = "resume_trans copied/frozen in train_sato_pp.py"

    audits: dict[str, Any] = {
        "create_graph_false_actual": audit_once(
            cfg_for_false,
            batch,
            modules,
            device,
            create_graph=actual_create_graph,
            audit_seed=args.audit_seed,
        )
    }
    audits["create_graph_false_actual"]["actual_cfg"] = True
    audits["create_graph_false_actual"]["checkpoint_arg_present"] = (
        "attribution_create_graph" in cfg
    )
    audits["create_graph_false_actual"]["effective_from_missing_default"] = (
        "attribution_create_graph" not in cfg and actual_create_graph is False
    )

    if args.include_create_graph_true:
        try:
            cfg_for_true = dict(cfg)
            cfg_for_true["attribution_create_graph"] = True
            audits["create_graph_true_diagnostic"] = audit_once(
                cfg_for_true,
                batch,
                modules,
                device,
                create_graph=True,
                audit_seed=args.audit_seed,
            )
            audits["create_graph_true_diagnostic"]["actual_cfg"] = False
            audits["create_graph_true_diagnostic"]["diagnostic_only_not_retraining"] = True
        except RuntimeError as exc:
            audits["create_graph_true_diagnostic"] = {
                "actual_cfg": False,
                "diagnostic_only_not_retraining": True,
                "error": repr(exc),
            }
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()

    output = {
        "schema_version": "sato-gradient-connectivity/v1",
        "repo_root": str(REPO_ROOT),
        "command": sys.argv,
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_current_device": int(torch.cuda.current_device()),
            "cuda_device_name": torch.cuda.get_device_name(0),
        },
        "provenance": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "checkpoint_iter": checkpoint_iter,
            "checkpoint_args_subset": compact_args(cfg),
            "checkpoint_has_attribution_create_graph_arg": "attribution_create_graph" in cfg,
            "effective_attribution_create_graph_actual": actual_create_graph,
            "actual_training_entry_point": {
                "file": "train_sato_pp.py",
                "launcher_out": str(
                    checkpoint_path.parent / "launcher.out"
                ),
                "launcher_declares_entry": "entry=train_sato_pp.py",
                "ignored_output_directory": str(checkpoint_path.parent),
            },
            **model_provenance,
        },
        "freeze_settings": freeze_settings,
        "batch": batch_meta,
        "audits": audits,
        "limitations": [
            "The fixed batch uses real HumanML3D pair rows and precomputed VQVAE tokens, but disables the dataloader's random one-token crop to keep the audit reproducible.",
            "CLIP has zero trainable parameters because the checkpoint config sets unfreeze_clip=false; CLIP parameter gradients are therefore recorded as frozen, not as connected-zero gradients.",
            "The create_graph=true section, when present, is a diagnostic same-batch connectivity probe and is not the checkpoint's recorded training configuration.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "checkpoint_iter": checkpoint_iter}))


if __name__ == "__main__":
    main()