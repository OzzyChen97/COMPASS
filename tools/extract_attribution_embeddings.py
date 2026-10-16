#!/usr/bin/env python3
"""Extract motion-conditioned text representations for clustering analysis.

The output is deliberately model-agnostic JSONL.  Each prompt in a semantic
pair produces one row per representation variant, allowing the downstream
analysis to compare a frozen text representation with Grad x Input and the
learned SATO++ pooling head on exactly the same motions and captions.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models.t2m_trans as trans
import models.vqvae as vqvae
from dataset.config import dataset_mean_std_paths
from models.sato_pp_modules import AttributionPooling


def get_clip_module():
    try:
        import clip as clip_module
    except ModuleNotFoundError:  # Repository-local fallback used in offline environments.
        from clip_local import clip as clip_module
    return clip_module


def iter_records(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            yield from json.load(handle)
        else:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_sample_index(path, level=None, limit=None):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"sample index {path} must contain a non-empty samples list")
    required = {
        "loader_name", "motion_id", "caption", "perturbed_caption",
        "level", "motion_length",
    }
    selected = []
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"sample index row {index} misses {sorted(missing)}")
        if str(row["loader_name"]) != str(row["motion_id"]):
            raise ValueError(f"sample index row {index} loader/motion identity mismatch")
        if level and str(row["level"]).upper() != str(level).upper():
            continue
        selected.append(dict(row))
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        raise ValueError(f"sample index {path} has no rows for level {level!r}")
    full_digest = canonical_sha256(rows)
    recorded = payload.get("sample_index_sha256") if isinstance(payload, dict) else None
    if recorded and recorded != full_digest:
        raise ValueError(
            f"sample index checksum mismatch: recorded={recorded}, computed={full_digest}"
        )
    return selected, full_digest


def rows_from_sample_indices(paths, limit_per_level=None):
    selected = []
    index_hashes = {}
    for path in paths:
        rows, digest = load_sample_index(path, limit=limit_per_level)
        index_hashes[str(rows[0]["level"]).upper()] = digest
        for offset, row in enumerate(rows):
            selected.append({
                "sample_offset": len(selected),
                "motion_id": str(row["motion_id"]),
                "pair_id": row.get("pair_id", offset),
                "text_orig": str(row["caption"]),
                "text_pair": str(row["perturbed_caption"]),
                "relation": "preserve",
                "semantic_class": f"preserve/{str(row['level']).upper()}",
                "level": str(row["level"]).upper(),
                "motion_length": int(row["motion_length"]),
                "length_repair": row.get("length_repair"),
            })
    return selected, index_hashes


def load_vqvae(reference_checkpoint, checkpoint_path, device):
    reference = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(reference["args"])
    args = Namespace(**cfg)
    model = vqvae.HumanVQVAE(
        args,
        cfg["nb_code"], cfg["code_dim"], cfg["output_emb_width"],
        cfg["down_t"], cfg["stride_t"], cfg["width"], cfg["depth"],
        cfg["dilation_growth_rate"],
    ).to(device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("net", state), strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def load_caption_locked_motion(data_root, row, mean, std):
    motion_id = str(row["motion_id"])
    motion = np.load(Path(data_root) / "new_joint_vecs" / f"{motion_id}.npy")
    repair = row.get("length_repair") or {}
    start = int(repair.get("repaired_start", 0))
    end = int(repair.get("repaired_end", len(motion)))
    motion = motion[start:end]
    repair_type = str(repair.get("repair_type", "none"))
    if repair_type == "resample-long-to-maximum" and len(motion) != 196:
        positions = np.linspace(0.0, len(motion) - 1, 196)
        lower = np.floor(positions).astype(np.int64)
        upper = np.minimum(lower + 1, len(motion) - 1)
        weight = (positions - lower).astype(motion.dtype, copy=False)[:, None]
        motion = motion[lower] * (1.0 - weight) + motion[upper] * weight
    m_length = (len(motion) // 4) * 4
    if m_length <= 0:
        raise ValueError(f"motion {motion_id} is too short to encode")
    crop_start = (len(motion) - m_length) // 2
    motion = motion[crop_start : crop_start + m_length]
    return ((motion - mean) / std).astype(np.float32), m_length


def encode_motion_tokens(model, motion, nb_code, block_size, device):
    with torch.no_grad():
        codes = model.encode(torch.as_tensor(motion, device=device).unsqueeze(0))[0]
    codes = codes.detach().cpu().numpy().astype(np.int64)[: block_size - 1]
    valid_len = min(len(codes) + 1, block_size)
    target = np.concatenate([codes, np.asarray([nb_code], dtype=np.int64)])
    if len(target) < block_size:
        target = np.pad(target, (0, block_size - len(target)), constant_values=nb_code + 1)
    return target, valid_len


def semantic_class(row):
    explicit = row.get("semantic_class") or row.get("semantic_label")
    if explicit:
        return str(explicit)
    relation = str(row.get("relation", "unknown"))
    edit_type = row.get("edit_type") or row.get("perturbation_type")
    if relation == "preserve":
        return f"preserve/{edit_type or 'paraphrase'}"
    if relation == "alter":
        return f"alter/{edit_type or 'semantic_edit'}"
    return str(edit_type or relation)


def normalized_entropy(alpha, mask):
    values = alpha[mask.bool()].float().clamp_min(1e-12)
    if values.numel() <= 1:
        return 0.0
    values = values / values.sum()
    entropy = -(values * values.log()).sum()
    return float((entropy / np.log(values.numel())).item())


def build_transformer(cfg, device):
    return trans.Text2Motion_Transformer(
        num_vq=cfg["nb_code"],
        embed_dim=cfg["embed_dim_gpt"],
        clip_dim=cfg["clip_dim"],
        block_size=cfg["block_size"],
        num_layers=cfg["num_layers"],
        n_head=cfg["n_head_gpt"],
        drop_out_rate=cfg["drop_out_rate"],
        fc_rate=cfg["ff_rate"],
    ).to(device).eval()


def load_bundle(reference_checkpoint, transformer_checkpoint, device):
    clip = get_clip_module()
    reference = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
    if "args" not in reference or "clip" not in reference:
        raise ValueError("reference checkpoint must contain args and clip weights")
    cfg = dict(reference["args"])
    transformer_state = reference["trans"]
    if transformer_checkpoint:
        candidate = torch.load(transformer_checkpoint, map_location="cpu", weights_only=False)
        transformer_state = candidate.get("trans", candidate)

    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.float()
    clip_model.load_state_dict(reference["clip"], strict=False)
    clip_model.eval()

    transformer = build_transformer(cfg, device)
    transformer.load_state_dict(transformer_state, strict=True)
    transformer.eval()

    attr_pool = None
    if not transformer_checkpoint and "attr_pooling" in reference:
        attr_pool = AttributionPooling(
            clip_dim=cfg.get("clip_dim_attr", 512),
            proj_dim=cfg.get("proj_dim", 512),
            proj_type=cfg.get("proj_type", "linear"),
        ).to(device)
        attr_pool.load_state_dict(reference["attr_pooling"], strict=True)
        attr_pool.eval()

    for module in [clip_model, transformer, attr_pool]:
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = False
    return clip_model, transformer, attr_pool, cfg


def motion_tokens(data_root, motion_id, nb_code, block_size):
    array = np.load(Path(data_root) / "VQVAE" / f"{motion_id}.npy")
    if array.ndim == 2:
        array = array[0]
    array = array.reshape(-1).astype(np.int64)[: block_size - 1]
    valid_len = min(len(array) + 1, block_size)
    result = np.concatenate([array, np.asarray([nb_code], dtype=np.int64)])
    if len(result) < block_size:
        result = np.pad(result, (0, block_size - len(result)), constant_values=nb_code + 1)
    return result, valid_len


def encode_motion_token_batch(model, motions, lengths, cfg, device):
    max_length = max(lengths)
    batch = np.zeros((len(motions), max_length, motions[0].shape[1]), dtype=np.float32)
    for index, motion in enumerate(motions):
        batch[index, : len(motion)] = motion
    with torch.no_grad():
        all_codes = model.encode(torch.as_tensor(batch, device=device))
    downsample = int(cfg["stride_t"]) ** int(cfg["down_t"])
    output = []
    for codes, length in zip(all_codes, lengths):
        code_length = min(length // downsample, int(cfg["block_size"]) - 1)
        values = codes[:code_length].detach().cpu().numpy().astype(np.int64)
        valid_len = min(len(values) + 1, int(cfg["block_size"]))
        target = np.concatenate([
            values, np.asarray([int(cfg["nb_code"])], dtype=np.int64)
        ])
        if len(target) < int(cfg["block_size"]):
            target = np.pad(
                target, (0, int(cfg["block_size"]) - len(target)),
                constant_values=int(cfg["nb_code"]) + 1,
            )
        output.append((target, valid_len))
    return output


def encode_prompt_batch(
    clip_model, transformer, attr_pool, cfg, texts, targets, valid_lens, device
):
    clip = get_clip_module()
    text_ids = clip.tokenize(list(texts), truncate=True).to(device)
    dtype = clip_model.dtype
    token_input = clip_model.token_embedding(text_ids).type(dtype).detach().requires_grad_(True)
    hidden = token_input + clip_model.positional_embedding.type(dtype)
    hidden = clip_model.transformer(hidden.permute(1, 0, 2)).permute(1, 0, 2)
    token_features = clip_model.ln_final(hidden).type(dtype)
    eos_index = text_ids.argmax(dim=-1)
    batch_indices = torch.arange(token_features.shape[0], device=device)
    pooled = token_features[batch_indices, eos_index]
    if clip_model.text_projection is not None:
        pooled = pooled @ clip_model.text_projection

    target = torch.as_tensor(np.stack(targets), dtype=torch.long, device=device)
    logits = transformer(target[:, :-1], pooled.float())
    max_steps = min(int(cfg.get("attribution_K", 16)), logits.shape[1])
    log_probs = F.log_softmax(logits[:, :max_steps], dim=-1)
    selected = target[:, :max_steps].clamp(0, logits.shape[-1] - 1)
    selected_scores = log_probs.gather(2, selected.unsqueeze(-1)).squeeze(-1)
    step_index = torch.arange(max_steps, device=device).unsqueeze(0)
    lengths = torch.as_tensor(valid_lens, device=device).unsqueeze(1).clamp(max=max_steps)
    step_mask = step_index < lengths
    per_sample_score = (
        selected_scores * step_mask
    ).sum(dim=1) / step_mask.sum(dim=1).clamp_min(1)
    gradient, = torch.autograd.grad(per_sample_score.sum(), token_input)
    raw_attr = (gradient.float() * token_input.detach().float()).abs().sum(dim=-1)

    visible = text_ids.ne(0)
    visible[batch_indices, 0] = False
    visible[batch_indices, eos_index] = False
    masked_attr = raw_attr.masked_fill(~visible, -1e4)
    alpha = F.softmax(masked_attr / float(cfg.get("tau", 1.0)), dim=-1)
    weighted = (token_features.float() * alpha.unsqueeze(-1)).sum(dim=1)

    outputs = {
        "clip_pooled": pooled.detach().float().cpu().numpy(),
        "grad_input_weighted": weighted.detach().float().cpu().numpy(),
    }
    if attr_pool is not None:
        outputs["sato_learned_pool"] = (
            attr_pool(token_features.float(), alpha).detach().float().cpu().numpy()
        )
    entropies = [
        normalized_entropy(alpha[index], visible[index])
        for index in range(len(texts))
    ]
    token_counts = visible.sum(dim=1).detach().cpu().numpy().astype(np.int64)
    return outputs, entropies, token_counts


def encode_prompt(clip_model, transformer, attr_pool, cfg, text, target, valid_len, device):
    clip = get_clip_module()
    text_ids = clip.tokenize([text], truncate=True).to(device)
    dtype = clip_model.dtype
    token_input = clip_model.token_embedding(text_ids).type(dtype).detach().requires_grad_(True)
    hidden = token_input + clip_model.positional_embedding.type(dtype)
    hidden = clip_model.transformer(hidden.permute(1, 0, 2)).permute(1, 0, 2)
    token_features = clip_model.ln_final(hidden).type(dtype)
    eos_index = text_ids.argmax(dim=-1)
    pooled = token_features[torch.arange(token_features.shape[0], device=device), eos_index]
    if clip_model.text_projection is not None:
        pooled = pooled @ clip_model.text_projection

    target = torch.as_tensor(target, dtype=torch.long, device=device).unsqueeze(0)
    logits = transformer(target[:, :-1], pooled.float())
    steps = min(int(cfg.get("attribution_K", 16)), logits.shape[1], valid_len)
    log_probs = F.log_softmax(logits[:, :steps], dim=-1)
    selected = target[:, :steps].clamp(0, logits.shape[-1] - 1)
    score = log_probs.gather(2, selected.unsqueeze(-1)).squeeze(-1).mean()
    gradient, = torch.autograd.grad(score, token_input)
    raw_attr = (gradient.float() * token_input.detach().float()).abs().sum(dim=-1)

    visible = text_ids.ne(0)
    visible[0, 0] = False
    visible[0, eos_index.item()] = False
    masked_attr = raw_attr.masked_fill(~visible, -1e4)
    alpha = F.softmax(masked_attr / float(cfg.get("tau", 1.0)), dim=-1)
    weighted = (token_features.float() * alpha.unsqueeze(-1)).sum(dim=1)

    outputs = {
        "clip_pooled": pooled[0].detach().float().cpu().tolist(),
        "grad_input_weighted": weighted[0].detach().float().cpu().tolist(),
    }
    if attr_pool is not None:
        outputs["sato_learned_pool"] = attr_pool(token_features.float(), alpha)[0].detach().cpu().tolist()
    return outputs, normalized_entropy(alpha[0], visible[0]), int(visible.sum().item())


def select_rows(paths, data_root, per_class, seed):
    grouped = {}
    for path in paths:
        for row in iter_records(path):
            motion_id = str(row.get("motion_id", ""))
            if not motion_id:
                continue
            if not row.get("text_orig") or not row.get("text_pair"):
                continue
            grouped.setdefault(semantic_class(row), []).append(row)
    rng = random.Random(seed)
    selected = []
    for label in sorted(grouped):
        rows = grouped[label]
        rng.shuffle(rows)
        selected.extend(rows[:per_class] if per_class > 0 else rows)
    return selected


def save_npz_records(records, output, metadata):
    records = sorted(
        records,
        key=lambda record: (
            int(record.get("sample_offset", 0)),
            0 if record["prompt_role"] == "orig" else 1,
            str(record["representation"]),
        ),
    )
    variants = sorted({record["representation"] for record in records})
    key_order = []
    grouped = defaultdict(dict)
    for record in records:
        key = (
            str(record["level"]), str(record["motion_id"]), str(record["pair_id"]),
            str(record["prompt_role"]), str(record["text"]),
        )
        if key not in grouped:
            key_order.append(key)
        grouped[key][record["representation"]] = record
    for key in key_order:
        missing = set(variants) - set(grouped[key])
        if missing:
            raise ValueError(f"record {key} misses representations {sorted(missing)}")
    exemplar = [grouped[key][variants[0]] for key in key_order]
    arrays = {
        "schema_version": np.asarray("attribution-motion-alignment/v1"),
        "model": np.asarray(str(metadata["model"])),
        "checkpoint_sha256": np.asarray(str(metadata["checkpoint_sha256"])),
        "sample_index_hashes_json": np.asarray(json.dumps(
            metadata["sample_index_hashes"], sort_keys=True
        )),
        "level": np.asarray([row["level"] for row in exemplar]),
        "motion_id": np.asarray([row["motion_id"] for row in exemplar]),
        "pair_id": np.asarray([str(row["pair_id"]) for row in exemplar]),
        "prompt_role": np.asarray([row["prompt_role"] for row in exemplar]),
        "text": np.asarray([row["text"] for row in exemplar]),
        "attribution_entropy": np.asarray([
            row["attribution_entropy"] for row in exemplar
        ], dtype=np.float32),
        "visible_token_count": np.asarray([
            row["visible_token_count"] for row in exemplar
        ], dtype=np.int64),
        "target_token_sha256": np.asarray([
            row["target_token_sha256"] for row in exemplar
        ]),
    }
    for variant in variants:
        arrays[variant] = np.stack([
            grouped[key][variant]["embedding"] for key in key_order
        ]).astype(np.float32)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def iter_motion_batches(rows, data_root, mean, std, batch_size):
    grouped = defaultdict(list)
    for row in rows:
        motion, length = load_caption_locked_motion(data_root, row, mean, std)
        grouped[int(length)].append((row, motion))
    for length in sorted(grouped):
        items = grouped[length]
        for start in range(0, len(items), batch_size):
            selected = items[start : start + batch_size]
            yield [item[0] for item in selected], [item[1] for item in selected], [length] * len(selected)


def extract_fixed_index_records(
    rows, clip_model, transformer, attr_pool, vq_model, cfg, data_root,
    model_label, device, batch_size,
):
    mean_path, std_path = dataset_mean_std_paths(cfg.get("dataname", "t2m"))
    mean, std = np.load(mean_path), np.load(std_path)
    records = []
    completed = 0
    for batch_rows, motions, lengths in iter_motion_batches(
        rows, data_root, mean, std, batch_size
    ):
        targets = encode_motion_token_batch(
            vq_model, motions, lengths, cfg, device,
        )
        for prompt_role, field in (("orig", "text_orig"), ("pair", "text_pair")):
            outputs, entropies, token_counts = encode_prompt_batch(
                clip_model, transformer, attr_pool, cfg,
                [row[field] for row in batch_rows],
                [item[0] for item in targets],
                [item[1] for item in targets],
                device,
            )
            for index, row in enumerate(batch_rows):
                token_digest = hashlib.sha256(
                    np.asarray(targets[index][0], dtype=np.int64).tobytes()
                ).hexdigest()
                for representation, embeddings in outputs.items():
                    records.append({
                        "sample_offset": int(row["sample_offset"]),
                        "dataset": Path(data_root).name,
                        "model": model_label,
                        "variant": f"{model_label}/{representation}",
                        "representation": representation,
                        "motion_id": str(row["motion_id"]),
                        "pair_id": row["pair_id"],
                        "prompt_role": prompt_role,
                        "text": str(row[field]),
                        "relation": "preserve",
                        "semantic_class": str(row["semantic_class"]),
                        "level": str(row["level"]),
                        "attribution_entropy": entropies[index],
                        "visible_token_count": int(token_counts[index]),
                        "target_token_sha256": token_digest,
                        "embedding": embeddings[index],
                    })
        completed += len(batch_rows)
        print(f"extracted {completed}/{len(rows)} pairs", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-checkpoint", required=True,
                        help="SATO++ checkpoint containing architecture args and CLIP weights")
    parser.add_argument("--transformer-checkpoint",
                        help="optional baseline transformer checkpoint; omit for the SATO++ transformer")
    parser.add_argument("--vq-checkpoint",
                        help="VQ-VAE checkpoint used to encode caption-locked reference motions")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--data-root", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pair-data", nargs="+")
    source.add_argument("--sample-index", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=128)
    parser.add_argument("--limit-per-level", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu-id", type=int, default=4)
    args = parser.parse_args()
    if args.sample_index and not args.vq_checkpoint:
        parser.error("--sample-index requires --vq-checkpoint")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu_id)
    clip_model, transformer, attr_pool, cfg = load_bundle(
        args.reference_checkpoint, args.transformer_checkpoint, device
    )

    if args.sample_index:
        rows, index_hashes = rows_from_sample_indices(
            args.sample_index, args.limit_per_level
        )
        vq_model = load_vqvae(
            args.reference_checkpoint, args.vq_checkpoint, device
        )
        records = extract_fixed_index_records(
            rows, clip_model, transformer, attr_pool, vq_model, cfg,
            args.data_root, args.model_label, device, args.batch_size,
        )
        save_npz_records(records, args.output, {
            "model": args.model_label,
            "checkpoint_sha256": file_sha256(
                args.transformer_checkpoint or args.reference_checkpoint
            ),
            "sample_index_hashes": index_hashes,
        })
    else:
        rows = select_rows(args.pair_data, args.data_root, args.per_class, args.seed)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for pair_index, row in enumerate(rows):
                motion_id = str(row["motion_id"])
                target, valid_len = motion_tokens(
                    args.data_root, motion_id, cfg["nb_code"], cfg["block_size"]
                )
                for prompt_role, field in (("orig", "text_orig"), ("pair", "text_pair")):
                    embeddings, entropy, token_count = encode_prompt(
                        clip_model, transformer, attr_pool, cfg,
                        row[field], target, valid_len, device,
                    )
                    for representation, embedding in embeddings.items():
                        record = {
                            "dataset": Path(args.data_root).name,
                            "model": args.model_label,
                            "variant": f"{args.model_label}/{representation}",
                            "representation": representation,
                            "motion_id": motion_id,
                            "pair_id": row.get("pair_id", f"{motion_id}:{pair_index}"),
                            "prompt_role": prompt_role,
                            "text": row[field],
                            "relation": row.get("relation", "unknown"),
                            "semantic_class": semantic_class(row),
                            "edit_type": row.get("edit_type"),
                            "attribution_entropy": entropy,
                            "visible_token_count": token_count,
                            "embedding": embedding,
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if (pair_index + 1) % 25 == 0:
                    print(f"extracted {pair_index + 1}/{len(rows)} pairs", flush=True)
    print(Path(args.output))


if __name__ == "__main__":
    main()
