#!/usr/bin/env python3
"""Evaluate MotionGPT with the shared T2M-GPT/SATO++ perturbation loader.

The legacy parity path mirrors ``ablation/repeat20_perturb_fast_batch.py``:
the standard shuffled HumanML3D test loader, batch size 32,
``drop_last=True``, motion-id perturbation lookup, and the repository's T2M-GPT
TM2T evaluator.  Caption-locked protocols instead resolve exact
``(motion_id, text_orig)`` rows in deterministic order with ``drop_last=False``
for either HumanML3D or Motion-X-322 MaxToken75.

Formal legacy HumanML3D runs contain 4,640 loader samples; caption-locked runs
derive or explicitly lock their sample count from a frozen loader index. Native
Motion-X-322 Table-3 runs still require all 7,733 unique test motions, while an
explicit caption-locked level protocol may lock a strict subset such as the
7,679-motion P1/P2/P3 intersection. Every sample must resolve a perturbation
without runtime caption fallback. ``--limit`` is an explicitly marked smoke
mode. ``--index-only`` performs the loader/matching audit without loading
MotionGPT or a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import linalg


REPO_ROOT = Path(__file__).resolve().parents[1]
MOTIONGPT_ROOT = Path(__file__).resolve().parent
FORMAL_BATCH_SIZE = 32
FORMAL_SAMPLE_COUNT = 4_640

DATASET_PROTOCOLS = {
    "humanml3d": {
        "dataname": "t2m",
        "expected_samples": 4_640,
        "drop_last": True,
        "pair_match": "exact motion id, then legacy A-W segment prefix removal",
        "evaluator": "t2m",
        "schema_slug": "humanml-loader",
        "require_unique_motions": False,
    },
    "motionx322": {
        "dataname": "motionx322_75",
        "expected_samples": 7_733,
        "drop_last": False,
        "pair_match": "exact motion id only",
        "evaluator": "motionx322",
        "schema_slug": "motionx322-loader",
        "require_unique_motions": True,
    },
}


class LoaderEvaluationError(ValueError):
    pass


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pairs(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise LoaderEvaluationError(f"empty prompt-pair file: {path}")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        if "Extra data" not in str(exc):
            raise
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = next(
            (value[key] for key in ("pairs", "data", "items") if isinstance(value.get(key), list)),
            [value],
        )
    if not isinstance(value, list):
        raise LoaderEvaluationError("prompt-pair file must contain a JSON array or JSONL")
    return value


def normalize_pairs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, row in enumerate(rows):
        motion_id = row.get("motion_id", row.get("id", row.get("name")))
        text_pair = row.get("text_pair", row.get("perturbed", row.get("rewrite")))
        text_orig = row.get("text_orig", row.get("original", row.get("caption", "")))
        if motion_id in (None, "") or not isinstance(text_pair, str) or not text_pair.strip():
            raise LoaderEvaluationError(f"pair {index} lacks motion_id/text_pair")
        normalized.append(
            {
                **row,
                "motion_id": str(motion_id).strip(),
                "text_orig": str(text_orig).strip(),
                "text_pair": text_pair.strip(),
                "level": str(row.get("level", "unknown")),
            }
        )
    return normalized


def normalize_caption_key(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def filter_pairs_by_levels(
    pairs: Sequence[Mapping[str, Any]], allowed_levels: Sequence[str] | None
) -> list[dict[str, Any]]:
    if not allowed_levels:
        return [dict(row) for row in pairs]
    allowed = {str(level).strip() for level in allowed_levels if str(level).strip()}
    if not allowed:
        raise LoaderEvaluationError("--allowed-levels cannot be empty")
    selected = [dict(row) for row in pairs if str(row.get("level", "")) in allowed]
    if not selected:
        raise LoaderEvaluationError(
            f"no perturbation pairs remain after filtering levels {sorted(allowed)}"
        )
    return selected


def pair_identity(path: Path, pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    canonical = [
        {
            "motion_id": row["motion_id"],
            "text_orig": row["text_orig"],
            "text_pair": row["text_pair"],
            "level": row["level"],
        }
        for row in pairs
    ]
    return {
        "path": str(path.resolve()),
        "count": len(pairs),
        "unique_motion_count": len({row["motion_id"] for row in pairs}),
        "sha256": canonical_sha256(canonical),
    }


def build_pair_index(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    # This is deliberately last-write-wins, matching utils.sato_pp_eval.py.
    return {str(row["motion_id"]): row for row in pairs}


def build_caption_pair_index(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in pairs:
        key = (str(row["motion_id"]), normalize_caption_key(str(row["text_orig"])))
        if key in index:
            previous = index[key]
            comparable = ("text_pair", "level")
            if any(str(previous.get(field)) != str(row.get(field)) for field in comparable):
                raise LoaderEvaluationError(
                    "conflicting duplicate caption-aware perturbation key: " + repr(key)
                )
            # An expanded evaluation mix may repeat the exact same pair to
            # reproduce SATO++ loader weights. Resolution remains unambiguous.
            continue
        index[key] = row
    return index


def validate_pair_contract(
    pair_meta: Mapping[str, Any],
    dataset: str,
    expected_samples: int | None = None,
) -> None:
    preset = DATASET_PROTOCOLS[dataset]
    if not preset["require_unique_motions"]:
        return
    expected = int(expected_samples or preset["expected_samples"])
    if pair_meta["count"] != expected or pair_meta["unique_motion_count"] != expected:
        raise LoaderEvaluationError(
            f"formal {dataset} pair file must contain exactly {expected} unique "
            f"motions, got {pair_meta['count']} rows/"
            f"{pair_meta['unique_motion_count']} unique"
        )


def resolve_loader_motion_id(
    name: str, pair_index: Mapping[str, Any], allow_segment_prefix: bool = True
) -> str | None:
    """Resolve legacy random segment prefixes without damaging arbitrary IDs.

    The historical evaluator used the suffix after ``A_`` ... ``W_`` for
    HumanML3D segments (including IDs such as ``M013514``).  Exact lookup is
    attempted first; the prefix is removed only when the suffix exists in the
    pair index.
    """
    value = str(name)
    if value in pair_index:
        return value
    prefix, separator, suffix = value.partition("_")
    if (
        allow_segment_prefix
        and separator
        and len(prefix) == 1
        and "A" <= prefix <= "W"
        and suffix in pair_index
    ):
        return suffix
    return None


def resolve_perturbations(
    names: Sequence[str],
    pair_index: Mapping[str, Mapping[str, Any]],
    allow_segment_prefix: bool = True,
) -> tuple[list[str | None], list[str | None], list[str]]:
    texts: list[str | None] = []
    motion_ids: list[str | None] = []
    levels: list[str] = []
    for name in names:
        motion_id = resolve_loader_motion_id(
            str(name), pair_index, allow_segment_prefix=allow_segment_prefix
        )
        row = pair_index.get(motion_id) if motion_id is not None else None
        texts.append(None if row is None else str(row["text_pair"]))
        motion_ids.append(motion_id)
        levels.append("missing" if row is None else str(row.get("level", "unknown")))
    return texts, motion_ids, levels


def resolve_caption_perturbations(
    names: Sequence[str],
    captions: Sequence[str],
    pair_index: Mapping[tuple[str, str], Mapping[str, Any]],
    motion_ids: Sequence[str] | None = None,
) -> tuple[list[str | None], list[str | None], list[str], list[Mapping[str, Any] | None]]:
    texts: list[str | None] = []
    resolved_motion_ids: list[str | None] = []
    levels: list[str] = []
    rows: list[Mapping[str, Any] | None] = []
    for index, (name, caption) in enumerate(zip(names, captions)):
        motion_id = str(motion_ids[index]) if motion_ids is not None else str(name)
        row = pair_index.get((motion_id, normalize_caption_key(str(caption))))
        texts.append(None if row is None else str(row["text_pair"]))
        resolved_motion_ids.append(None if row is None else motion_id)
        levels.append("missing" if row is None else str(row.get("level", "unknown")))
        rows.append(row)
    return texts, resolved_motion_ids, levels, rows


def protocol_payload(
    pair_meta: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    pair_rng_mode: str = "sequential",
    dataset: str = "humanml3d",
    evaluator_id: str | None = None,
    sampling: Mapping[str, Any] | None = None,
    sampling_kwargs_mode: str = "explicit",
    length_mode: str = "generated",
    prompt_mode: str = "official-default",
    generation_order: str = "original-first",
    loader_mode: str = "legacy-motion-id",
    expected_samples: int | None = None,
    allowed_levels: Sequence[str] | None = None,
    caption_length_policy: str = "strict",
) -> dict[str, Any]:
    preset = DATASET_PROTOCOLS[dataset]
    source_protocol = source_manifest.get("protocol", {})
    return {
        "schema_version": f"motiongpt-{preset['schema_slug']}/v1",
        "dataset": dataset,
        "evaluation_split": "test",
        "loader": {
            "implementation": (
                "dataset.dataset_TM_eval.CaptionLockedDATALoader"
                if loader_mode == "caption-locked"
                else "dataset.dataset_TM_eval.DATALoader"
            ),
            "batch_size": FORMAL_BATCH_SIZE,
            "shuffle": loader_mode != "caption-locked",
            "drop_last": False if loader_mode == "caption-locked" else preset["drop_last"],
            "caption_locked": loader_mode == "caption-locked",
            "expected_samples": int(expected_samples or preset["expected_samples"]),
            "pair_match": (
                "exact normalized (motion_id, text_orig)"
                if loader_mode == "caption-locked"
                else preset["pair_match"]
            ),
            "allowed_levels": list(allowed_levels or []),
            "length_policy": (
                caption_length_policy if loader_mode == "caption-locked" else "standard"
            ),
        },
        "pair_file": dict(pair_meta),
        "sampling": {
            **dict(source_protocol.get("sampling", {}) if sampling is None else sampling),
            "kwargs_mode": sampling_kwargs_mode,
            "implementation": (
                "explicit Hugging Face generate kwargs"
                if sampling_kwargs_mode == "explicit"
                else "implicit Hugging Face generation defaults"
            ),
        },
        "pair_rng_mode": pair_rng_mode,
        "prompt": {
            "mode": prompt_mode,
            "implementation": (
                "Generate motion: <Caption_Placeholder>"
                if prompt_mode == "official-default"
                else "random configured Text-to-Motion instruction template"
            ),
        },
        "generation_order": generation_order,
        "generation_length_mode": length_mode,
        "evaluator_length_mode": (
            "min(generated_length,reference_length)"
            if length_mode == "reference"
            else "generated_length_capped_to_batch_seq"
        ),
        "seeds": list(source_protocol.get("seeds", [])),
        "evaluator": evaluator_id or f"t2m-gpt-tm2t-root:{preset['evaluator']}",
    }


def validate_source_manifest(
    manifest: Mapping[str, Any], seed: int, config: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol = manifest.get("protocol")
    model = manifest.get("model")
    if not isinstance(protocol, Mapping) or not isinstance(model, Mapping):
        raise LoaderEvaluationError("source manifest lacks protocol/model")
    if seed not in protocol.get("seeds", []):
        raise LoaderEvaluationError(f"seed {seed} is not in the source manifest")
    commands = [row for row in manifest.get("commands", []) if int(row.get("seed", -1)) == seed]
    if len(commands) != 1 or Path(commands[0]["config"]).resolve() != config.resolve():
        raise LoaderEvaluationError("--config must match exactly one source-manifest seed entry")
    return protocol, model


def reset_rng(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG touched by loader/model evaluation."""
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def load_motiongpt_config(config_path: Path, assets_path: Path):
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    assets = OmegaConf.load(assets_path)
    config_folder = Path(assets.CONFIG_FOLDER)
    if not config_folder.is_absolute():
        config_folder = MOTIONGPT_ROOT / config_folder
    cfg = OmegaConf.merge(OmegaConf.load(config_folder / "default.yaml"), OmegaConf.load(config_path))
    if not cfg.FULL_CONFIG:
        for module_path in sorted(config_folder.glob("*/*.yaml")):
            node = ".".join(module_path.relative_to(config_folder).with_suffix("").parts)
            OmegaConf.update(cfg, node, OmegaConf.load(module_path), merge=True)
    return OmegaConf.merge(cfg, assets)


def find_assets_config(manifest: Mapping[str, Any], seed: int) -> Path:
    command = next(row["command"] for row in manifest["commands"] if int(row["seed"]) == seed)
    try:
        return Path(command[command.index("--cfg_assets") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise LoaderEvaluationError("source manifest does not record --cfg_assets") from exc


def load_task(
    cfg, batch_size: int, prompt_mode: str = "official-default"
) -> list[dict[str, Any]]:
    if cfg.model.params.condition == "caption":
        task = {"input": ["<Caption_Placeholder>"], "output": [""]}
    elif prompt_mode == "official-default":
        # This is the exact fallback used by MLM.generate_conditional() when
        # DATASET.TASK_PATH is empty in the released HumanML3D test config.
        task = {
            "input": ["Generate motion: <Caption_Placeholder>"],
            "output": [""],
        }
    else:
        configured_path = str(cfg.DATASET.get("TASK_PATH", "")).strip()
        if not configured_path:
            configured_path = "prepare/instructions/template_instructions.json"
        task_path = Path(configured_path)
        if not task_path.is_absolute():
            task_path = MOTIONGPT_ROOT / task_path
        instructions = load_json(task_path)
        tasks = instructions["Text-to-Motion"]
        task = tasks.get("t2m", tasks.get("caption"))
        if task is None:
            raise LoaderEvaluationError("Text-to-Motion template needs t2m or caption")
    return [task] * batch_size


def generate_features(
    model,
    datamodule,
    texts: list[str],
    sampling: str,
    max_frames: int,
    reference_lengths: Sequence[int] | None = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    pass_sampling_kwargs: bool = True,
    audit_rows: list[dict[str, Any]] | None = None,
    prompt_mode: str = "official-default",
):
    import torch

    tasks = load_task(model.hparams.cfg, len(texts), prompt_mode=prompt_mode)
    zero_lengths = [0] * len(texts)
    inputs, _ = model.lm.template_fulfill(tasks, zero_lengths, [""] * len(texts), texts, stage="test")
    generation_kwargs = {}
    if pass_sampling_kwargs:
        generation_kwargs = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
    outputs, output_texts = model.lm.generate_direct(
        inputs,
        max_length=128,
        num_beams=1,
        do_sample=sampling == "categorical",
        **generation_kwargs,
    )
    decoded_rows = []
    lengths = []
    for index, tokens in enumerate(outputs):
        raw_tokens = [int(value) for value in tokens.detach().cpu().tolist()]
        tokens = torch.clamp(tokens, 0, model.hparams.codebook_size - 1)
        if len(tokens) <= 1:
            decoded = torch.zeros((1, 1, datamodule.nfeats), device=model.device)
        else:
            decoded = model.vae.decode(tokens)
        raw_frame_count = int(decoded.shape[1])
        decoded = decoded[:, :max_frames]
        decoded_rows.append(decoded[0])
        lengths.append(int(decoded.shape[1]))
        if audit_rows is not None:
            audit_rows.append({
                "input_text": str(texts[index]),
                "fulfilled_prompt": str(inputs[index]),
                "decoded_output_text": str(output_texts[index]),
                "motion_tokens": raw_tokens,
                "motion_token_count": len(raw_tokens),
                "raw_generated_frame_count": raw_frame_count,
                "generated_frame_count": int(decoded.shape[1]),
                "truncated_to_batch_seq": raw_frame_count > max_frames,
            })
    # Match both official MotionGPT and SATO++: generated motions are placed in
    # the full batch sequence tensor before the evaluator convolution stack.
    features = torch.zeros(
        (len(texts), max_frames, datamodule.nfeats), device=model.device
    )
    for index, decoded in enumerate(decoded_rows):
        features[index, : lengths[index]] = decoded[: lengths[index]]
    # HumanML3D training and evaluator statistics are byte-identical in this
    # workspace; keep the explicit conversion to match official MotionGPT.
    evaluator_lengths = torch.as_tensor(lengths, dtype=torch.long)
    if reference_lengths is not None:
        evaluator_lengths = clamp_evaluator_lengths(
            evaluator_lengths, reference_lengths
        )
    if audit_rows is not None:
        for row, evaluator_length in zip(audit_rows, evaluator_lengths):
            row["evaluator_frame_count"] = int(evaluator_length)
    return datamodule.renorm4t2m(features), evaluator_lengths


def validate_generation_replay_payload(
    payload: Mapping[str, Any], expected_count: int
) -> dict[str, list[Mapping[str, Any]]]:
    """Validate a complete token audit before using it as generated-motion input."""
    replay: dict[str, list[Mapping[str, Any]]] = {}
    for role in ("original", "perturbed"):
        rows = payload.get(role)
        if not isinstance(rows, list) or len(rows) != expected_count:
            actual = len(rows) if isinstance(rows, list) else "missing"
            raise LoaderEvaluationError(
                f"generation replay {role} audit has {actual} rows; "
                f"expected {expected_count}"
            )
        for offset, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise LoaderEvaluationError(
                    f"generation replay {role} row {offset} is not an object"
                )
            if int(row.get("sample_offset", -1)) != offset:
                raise LoaderEvaluationError(
                    f"generation replay {role} order mismatch at offset {offset}"
                )
            if row.get("generation_role") != role:
                raise LoaderEvaluationError(
                    f"generation replay role mismatch at offset {offset}"
                )
            tokens = row.get("motion_tokens")
            if (
                not isinstance(tokens, list)
                or int(row.get("motion_token_count", -1)) != len(tokens)
            ):
                raise LoaderEvaluationError(
                    f"generation replay token audit mismatch at offset {offset}"
                )
        replay[role] = rows
    return replay


def decode_audited_features(
    model,
    datamodule,
    audit_rows: Sequence[Mapping[str, Any]],
    texts: Sequence[str],
    names: Sequence[str],
    motion_ids: Sequence[str | None],
    levels: Sequence[str],
    max_frames: int,
    start_offset: int,
    reference_lengths: Sequence[int] | None = None,
):
    """Decode the exact motion-token sequences retained by a generation audit."""
    import torch

    count = len(texts)
    if not all(len(values) == count for values in (audit_rows, names, motion_ids, levels)):
        raise LoaderEvaluationError("generation replay batch fields have different sizes")
    decoded_rows = []
    lengths = []
    for index, (row, text, name, motion_id, level) in enumerate(
        zip(audit_rows, texts, names, motion_ids, levels)
    ):
        offset = start_offset + index
        expected_identity = {
            "sample_offset": offset,
            "loader_name": str(name),
            "motion_id": str(motion_id),
            "level": str(level),
            "input_text": str(text),
        }
        mismatches = [
            key
            for key, expected in expected_identity.items()
            if str(row.get(key)) != str(expected)
        ]
        if mismatches:
            raise LoaderEvaluationError(
                f"generation replay identity mismatch at offset {offset}: "
                + ", ".join(mismatches)
            )
        raw_tokens = row.get("motion_tokens")
        if not isinstance(raw_tokens, list):
            raise LoaderEvaluationError(
                f"generation replay lacks motion tokens at offset {offset}"
            )
        tokens = torch.as_tensor(raw_tokens, dtype=torch.long, device=model.device)
        tokens = torch.clamp(tokens, 0, model.hparams.codebook_size - 1)
        if len(tokens) <= 1:
            decoded = torch.zeros((1, 1, datamodule.nfeats), device=model.device)
        else:
            decoded = model.vae.decode(tokens)
        raw_frame_count = int(decoded.shape[1])
        decoded = decoded[:, :max_frames]
        generated_frame_count = int(decoded.shape[1])
        if int(row.get("raw_generated_frame_count", -1)) != raw_frame_count:
            raise LoaderEvaluationError(
                f"generation replay raw frame count mismatch at offset {offset}"
            )
        if int(row.get("generated_frame_count", -1)) != generated_frame_count:
            raise LoaderEvaluationError(
                f"generation replay cropped frame count mismatch at offset {offset}"
            )
        decoded_rows.append(decoded[0])
        lengths.append(generated_frame_count)

    features = torch.zeros(
        (count, max_frames, datamodule.nfeats), device=model.device
    )
    for index, decoded in enumerate(decoded_rows):
        features[index, : lengths[index]] = decoded[: lengths[index]]
    evaluator_lengths = torch.as_tensor(lengths, dtype=torch.long)
    if reference_lengths is not None:
        evaluator_lengths = clamp_evaluator_lengths(
            evaluator_lengths, reference_lengths
        )
    for index, (row, evaluator_length) in enumerate(zip(audit_rows, evaluator_lengths)):
        if int(row.get("evaluator_frame_count", -1)) != int(evaluator_length):
            raise LoaderEvaluationError(
                "generation replay evaluator frame count mismatch at offset "
                f"{start_offset + index}"
            )
    return datamodule.renorm4t2m(features), evaluator_lengths


def get_motion_embeddings_in_input_order(evaluator, motions, lengths):
    """Undo EvaluatorModelWrapper's internal descending-length permutation."""
    import torch

    lengths_tensor = torch.as_tensor(lengths, dtype=torch.long)
    align_idx = np.argsort(lengths_tensor.tolist())[::-1].copy()
    sorted_embeddings = evaluator.get_motion_embeddings(motions, lengths_tensor)
    embeddings = torch.empty_like(sorted_embeddings)
    embeddings[torch.as_tensor(align_idx, device=embeddings.device)] = sorted_embeddings
    return embeddings


def clamp_evaluator_lengths(
    generated_lengths: Sequence[int], reference_lengths: Sequence[int]
):
    """Clamp only TM2T lengths to MotionGPT's official reference contract.

    The generated/padded tensor may retain frames beyond the reference motion;
    evaluator lengths are always ``min(generated_length, reference_length)``.
    """
    import torch

    generated = torch.as_tensor(generated_lengths, dtype=torch.long)
    reference = torch.as_tensor(reference_lengths, dtype=torch.long)
    if generated.ndim != 1 or reference.ndim != 1 or generated.shape != reference.shape:
        raise LoaderEvaluationError(
            "generated/reference evaluator lengths must be equal-size 1D arrays"
        )
    if torch.any(generated < 1) or torch.any(reference < 1):
        raise LoaderEvaluationError("evaluator lengths must be positive")
    return torch.minimum(generated, reference)


def slice_batch(batch: Sequence[Any], count: int) -> tuple[Any, ...]:
    sliced = []
    for value in batch:
        sliced.append(value[:count])
    return tuple(sliced)


def build_loader(
    seed: int,
    batch_size: int,
    num_workers: int,
    dataname: str = "t2m",
    is_test: bool = True,
    loader_mode: str = "legacy-motion-id",
    pair_data: Sequence[Mapping[str, Any]] | None = None,
    caption_length_policy: str = "strict",
):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from dataset import dataset_TM_eval
    from utils.word_vectorizer import WordVectorizer

    reset_rng(seed)
    vectorizer = WordVectorizer(str(REPO_ROOT / "glove"), "our_vab")
    if loader_mode == "caption-locked":
        if dataname not in {"t2m", "motionx322_75"}:
            raise LoaderEvaluationError(
                "caption-locked loader supports HumanML3D and Motion-X-322 MaxToken75"
            )
        return dataset_TM_eval.CaptionLockedDATALoader(
            dataname,
            is_test,
            batch_size,
            vectorizer,
            pair_data,
            num_workers=num_workers,
            length_policy=caption_length_policy,
        )
    return dataset_TM_eval.DATALoader(
        dataname, is_test, batch_size, vectorizer, num_workers=num_workers
    )


def build_root_evaluator(device, dataname: str = "t2m"):
    from dataset.config import configure_evaluator_source, evaluator_opt_path
    from models.evaluator_wrapper import EvaluatorModelWrapper
    from options.get_eval_option import get_opt

    options = get_opt(evaluator_opt_path(dataname), device)
    options = configure_evaluator_source(options, dataname)
    ruler_override = os.environ.get("T2MGPT_RULER_CHECKPOINT")
    if ruler_override:
        options.evaluator_matcher_checkpoint = ruler_override
    return EvaluatorModelWrapper(options)


def calculate_activation_statistics(activations: np.ndarray):
    return np.mean(activations, axis=0), np.cov(activations, rowvar=False)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    difference = mu1 - mu2
    covariance_mean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covariance_mean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    return float(
        difference.dot(difference)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * np.trace(covariance_mean)
    )


def calculate_r_precision(text: np.ndarray, motion: np.ndarray, top_k: int = 3):
    squared = (
        -2 * np.dot(text, motion.T)
        + np.sum(np.square(text), axis=1, keepdims=True)
        + np.sum(np.square(motion), axis=1)
    )
    distances = np.sqrt(np.maximum(squared, 0.0))
    ranking = np.argsort(distances, axis=1)
    targets = np.arange(len(text))[:, None]
    correct = np.zeros(len(text), dtype=bool)
    top = []
    for index in range(top_k):
        # Do not mutate in place: appended columns must preserve R@1/R@2
        # rather than all becoming views of the final cumulative R@3 mask.
        correct = correct | (ranking[:, index] == targets[:, 0])
        top.append(correct[:, None].copy())
    return np.concatenate(top, axis=1).sum(axis=0), float(np.trace(distances))


def correct_match_ranks(text: np.ndarray, motion: np.ndarray) -> np.ndarray:
    squared = (
        -2 * np.dot(text, motion.T)
        + np.sum(np.square(text), axis=1, keepdims=True)
        + np.sum(np.square(motion), axis=1)
    )
    ranking = np.argsort(np.sqrt(np.maximum(squared, 0.0)), axis=1)
    return np.argmax(ranking == np.arange(len(text))[:, None], axis=1)


def evaluation_diversity(activations: np.ndarray) -> float:
    times = 300 if len(activations) > 300 else min(100, len(activations) - 1)
    if times < 1:
        raise LoaderEvaluationError("diversity requires at least two samples")
    first = np.random.choice(len(activations), times, replace=False)
    second = np.random.choice(len(activations), times, replace=False)
    return float(np.linalg.norm(activations[first] - activations[second], axis=1).mean())


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_expected_index(
    expected: Mapping[str, Any],
    actual_records: Sequence[Mapping[str, Any]],
    protocol_fingerprint: str,
) -> None:
    if expected.get("protocol_fingerprint") != protocol_fingerprint:
        raise LoaderEvaluationError("expected sample index has a different protocol fingerprint")
    expected_records = expected.get("samples")
    if not isinstance(expected_records, list):
        raise LoaderEvaluationError("expected sample index lacks samples")
    if list(expected_records[: len(actual_records)]) != list(actual_records):
        raise LoaderEvaluationError("loader records differ from the frozen sample index")
    if len(actual_records) == len(expected_records):
        expected_hash = expected.get("sample_index_sha256")
        if expected_hash != canonical_sha256(actual_records):
            raise LoaderEvaluationError("frozen sample-index hash does not match its records")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    preset = DATASET_PROTOCOLS[args.dataset]
    dataname = str(preset["dataname"])
    manifest_path = args.manifest.resolve()
    config_path = args.config.resolve()
    assets_path = args.assets_cfg.resolve() if args.assets_cfg else None
    manifest = load_json(manifest_path)
    source_protocol, model_meta = validate_source_manifest(manifest, args.seed, config_path)
    pair_path = args.pair_file.resolve()
    source_pairs = normalize_pairs(load_pairs(pair_path))
    pairs = filter_pairs_by_levels(source_pairs, args.allowed_levels)
    pair_meta = pair_identity(pair_path, pairs)
    pair_meta["source_count"] = len(source_pairs)
    pair_meta["allowed_levels"] = list(args.allowed_levels or [])
    validate_pair_contract(
        pair_meta,
        args.dataset,
        expected_samples=args.expected_samples,
    )
    pair_index = build_pair_index(pairs)
    caption_pair_index = build_caption_pair_index(pairs)
    loader = build_loader(
        args.seed,
        args.batch_size,
        args.num_workers,
        dataname=dataname,
        loader_mode=args.loader_mode,
        pair_data=pairs,
        caption_length_policy=args.caption_length_policy,
    )
    caption_entry_index = None
    if args.loader_mode == "caption-locked":
        caption_entry_index = {}
        for entry in loader.dataset.entries:
            key = (
                str(entry["motion_id"]),
                normalize_caption_key(str(entry["caption"])),
            )
            previous = caption_entry_index.get(key)
            if previous is not None and previous.get("length_repair") != entry.get("length_repair"):
                raise LoaderEvaluationError(
                    "duplicate caption-locked entries disagree on length repair: " + repr(key)
                )
            caption_entry_index[key] = entry
    expected_loader_samples = (
        len(loader.dataset)
        if not loader.drop_last
        else len(loader) * args.batch_size
    )
    expected_formal_samples = (
        int(args.expected_samples)
        if args.expected_samples is not None
        else (
            expected_loader_samples
            if args.loader_mode == "caption-locked"
            else int(preset["expected_samples"])
        )
    )
    source_expected_samples = source_protocol.get("samples_per_seed")
    if (
        args.limit is None
        and source_expected_samples is not None
        and int(source_expected_samples) != expected_formal_samples
    ):
        raise LoaderEvaluationError(
            "source manifest sample-count contract mismatch: "
            f"{source_expected_samples} != {expected_formal_samples}"
        )
    source_sampling = dict(source_protocol.get("sampling", {}))
    sampling = {
        "mode": args.sample_mode or source_sampling.get("mode", "categorical"),
        "temperature": (
            args.temperature
            if args.temperature is not None
            else float(source_sampling.get("temperature", 1.0))
        ),
        "top_k": (
            args.top_k if args.top_k is not None else int(source_sampling.get("top_k", 0))
        ),
        "top_p": (
            args.top_p if args.top_p is not None else float(source_sampling.get("top_p", 1.0))
        ),
    }
    protocol = protocol_payload(
        pair_meta,
        manifest,
        args.pair_rng_mode,
        dataset=args.dataset,
        evaluator_id=args.protocol_evaluator_id,
        sampling=sampling,
        sampling_kwargs_mode=args.sampling_kwargs_mode,
        length_mode=args.length_mode,
        prompt_mode=args.prompt_mode,
        generation_order=args.generation_order,
        loader_mode=args.loader_mode,
        expected_samples=expected_formal_samples,
        allowed_levels=args.allowed_levels,
        caption_length_policy=args.caption_length_policy,
    )
    protocol_fingerprint = canonical_sha256(protocol)
    expected_index = (
        load_json(args.expected_sample_index.resolve())
        if args.expected_sample_index is not None
        else None
    )
    if expected_index is not None and expected_index.get("protocol_fingerprint") != protocol_fingerprint:
        raise LoaderEvaluationError("expected sample index has a different protocol fingerprint")

    if args.protocol_manifest:
        write_json(
            args.protocol_manifest,
            {
                "kind": "motiongpt_loader_protocol",
                "protocol": protocol,
                "fingerprint": protocol_fingerprint,
                "source_manifest": str(manifest_path),
            },
        )

    if args.generation_audit_input is not None and args.generation_audit_output is not None:
        raise LoaderEvaluationError(
            "--generation-audit-input and --generation-audit-output are mutually exclusive"
        )
    if args.generation_audit_input is not None and args.generation_seed is not None:
        raise LoaderEvaluationError(
            "--generation-seed cannot be used when replaying audited motion tokens"
        )
    if not args.limit and args.batch_size != FORMAL_BATCH_SIZE:
        raise LoaderEvaluationError(f"formal evaluation requires batch size {FORMAL_BATCH_SIZE}")
    if not args.limit and not args.index_only and args.sampling_kwargs_mode != "explicit":
        raise LoaderEvaluationError(
            "formal evaluation requires --sampling-kwargs-mode explicit; legacy mode "
            "silently uses Hugging Face defaults such as top_k=50"
        )
    if not args.limit and not args.index_only:
        formal_contract = {
            "pair_rng_mode": (args.pair_rng_mode, "sequential"),
            "generation_order": (args.generation_order, "original-first"),
            "length_mode": (args.length_mode, "generated"),
            "prompt_mode": (args.prompt_mode, "official-default"),
        }
        mismatches = [
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in formal_contract.items()
            if actual != expected
        ]
        if mismatches:
            raise LoaderEvaluationError(
                "formal SATO++ parity protocol mismatch: " + ", ".join(mismatches)
            )
        if args.generation_audit_input is None:
            if args.generation_audit_output is None or args.generation_audit_batches != 0:
                raise LoaderEvaluationError(
                    "formal evaluation requires either a complete token replay via "
                    "--generation-audit-input or a complete generation audit via "
                    "--generation-audit-output and --generation-audit-batches 0"
                )
    if not args.limit and expected_loader_samples != expected_formal_samples:
        raise LoaderEvaluationError(
            f"formal loader produced {expected_loader_samples}, expected "
            f"{expected_formal_samples}"
        )

    device = None
    model = datamodule = evaluator = None
    perturbed_text_processor = None
    checkpoint_load_audit: dict[str, Any] | None = None
    generation_replay_path = (
        args.generation_audit_input.resolve()
        if args.generation_audit_input is not None
        else None
    )
    generation_replay = (
        validate_generation_replay_payload(
            load_json(generation_replay_path), expected_formal_samples
        )
        if generation_replay_path is not None and not args.index_only
        else None
    )
    if (
        float(sampling.get("temperature", 1.0)) != 1.0
        or int(sampling.get("top_k", 0)) != 0
        or float(sampling.get("top_p", 1.0)) != 1.0
    ):
        raise LoaderEvaluationError("MotionGPT supports only temperature=1, top_k=0, top_p=1")

    if not args.index_only:
        import torch

        device = torch.device(args.device)
        evaluator = build_root_evaluator(device, dataname=dataname)
        from utils.sato_pp_eval import TM2TTextProcessor
        from utils.word_vectorizer import WordVectorizer
        perturbed_text_processor = TM2TTextProcessor(
            WordVectorizer(str(REPO_ROOT / "glove"), "our_vab"),
            max_text_len=int(getattr(evaluator.opt, "max_text_len", 20)),
            spacy_model=args.spacy_model,
            alphabetic_only=not dataname.startswith("motionx322"),
        )
        assets = assets_path if assets_path is not None else find_assets_config(manifest, args.seed)
        previous = Path.cwd()
        os.chdir(MOTIONGPT_ROOT)
        try:
            if str(MOTIONGPT_ROOT) not in sys.path:
                sys.path.insert(0, str(MOTIONGPT_ROOT))
            from mGPT.data.build_data import build_data
            from mGPT.models.build_model import build_model

            cfg = load_motiongpt_config(config_path, assets)
            cfg.SEED_VALUE = args.seed
            cfg.TEST.CHECKPOINTS = model_meta["checkpoint"]
            cfg.TRAIN.STRICT_LOAD = False
            cfg.DATASET.ALLOW_MISSING_WORD_VECTORIZER = True
            # The released HumanML3D checkpoint uses the legacy ``VQVAE``
            # token directory.  Native Motion-X training writes and consumes
            # ``VQVAE_MOTIONX322``; do not silently redirect it to the legacy
            # Motion-X symlink during formal evaluation.
            if args.dataset == "humanml3d":
                cfg.DATASET.CODE_PATH = "VQVAE"
            cfg.DEBUG = False
            datamodule = build_data(cfg, phase="test")
            model = build_model(cfg, datamodule)
            checkpoint_payload = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")
            load_result = model.load_state_dict(checkpoint_payload["state_dict"], strict=False)
            checkpoint_load_audit = {
                "strict": False,
                "missing_key_count": len(load_result.missing_keys),
                "unexpected_key_count": len(load_result.unexpected_keys),
                "missing_keys": list(load_result.missing_keys),
                "unexpected_keys": list(load_result.unexpected_keys),
            }
            model.to(device).eval()
        finally:
            os.chdir(previous)
        # Model construction consumes RNG.  Existing repeat evaluation resets
        # immediately before each seed, so reset here before loader iteration.
        reset_rng(args.seed)

    gt_embeddings = []
    orig_embeddings = []
    pair_embeddings = []
    matched = 0
    processed = 0
    level_counts: Counter[str] = Counter()
    sample_ids: list[str] = []
    sample_index: list[dict[str, Any]] = []
    caption_matched = 0
    r_precision = np.zeros(3, dtype=np.float64)
    perturbed_r_precision = np.zeros(3, dtype=np.float64)
    gt_r_precision = np.zeros(3, dtype=np.float64)
    matching_sum = 0.0
    perturbed_matching_sum = 0.0
    diagnostic_batches: list[dict[str, Any]] = []
    generation_audit: dict[str, list[dict[str, Any]]] | None = (
        {"original": [], "perturbed": []}
        if args.generation_audit_output is not None
        else None
    )
    all_reference_lengths: list[int] = []
    all_original_lengths: list[int] = []
    all_paired_lengths: list[int] = []
    embedding_variants: dict[str, list[Any]] = {
        key: []
        for key in (
            "gt",
            "original_generated",
            "paired_generated",
            "original_reference_clamped",
            "paired_reference_clamped",
            "original_ground_truth",
            "paired_ground_truth",
        )
    }
    text_embeddings: list[Any] = []
    perturbed_text_embeddings: list[Any] = []
    original_motions: list[Any] = []
    retrieval_perturbed_text_embeddings: list[Any] = []
    retrieval_paired_embeddings: list[Any] = []
    retrieval_batch_sizes: list[int] = []

    loader_iterator = iter(loader)
    for batch_index, raw_batch in enumerate(loader_iterator):
        remaining = None if args.limit is None else args.limit - processed
        if remaining is not None and remaining <= 0:
            break
        batch = raw_batch if remaining is None or remaining >= len(raw_batch[2]) else slice_batch(raw_batch, remaining)
        word, pos, captions, sent_lengths, poses, motion_lengths, _, names = batch
        if args.loader_mode == "caption-locked":
            perturbed, motion_ids, levels, resolved_rows = resolve_caption_perturbations(
                names, captions, caption_pair_index
            )
        else:
            perturbed, motion_ids, levels = resolve_perturbations(
                names,
                pair_index,
                allow_segment_prefix=not preset["require_unique_motions"],
            )
            resolved_rows = [
                pair_index.get(motion_id) if motion_id is not None else None
                for motion_id in motion_ids
            ]
        batch_matched = sum(text is not None for text in perturbed)
        matched += batch_matched
        caption_matched += sum(
            row is not None
            and normalize_caption_key(str(caption))
            == normalize_caption_key(str(row.get("text_orig", "")))
            for caption, row in zip(captions, resolved_rows)
        )
        processed += len(captions)
        level_counts.update(levels)
        sample_ids.extend(str(name) for name in names)
        batch_entries = (
            [
                caption_entry_index[
                    (str(name), normalize_caption_key(str(caption)))
                ]
                for name, caption in zip(names, captions)
            ]
            if args.loader_mode == "caption-locked"
            else [None] * len(captions)
        )
        sample_index.extend(
            {
                "loader_name": str(name),
                "motion_id": "<missing>" if motion_id is None else motion_id,
                "caption": str(caption),
                "perturbed_caption": "<missing>" if text is None else str(text),
                "level": level,
                "motion_length": int(length),
                "pair_id": None if row is None else row.get("pair_id"),
                "pair_text_orig": "<missing>" if row is None else str(row.get("text_orig", "")),
                "caption_pair_exact": bool(
                    row is not None
                    and normalize_caption_key(str(caption))
                    == normalize_caption_key(str(row.get("text_orig", "")))
                ),
                "length_repair": (
                    None if entry is None else dict(entry.get("length_repair", {}))
                ),
            }
            for name, motion_id, caption, text, level, length, row, entry in zip(
                names, motion_ids, captions, perturbed, levels, motion_lengths,
                resolved_rows, batch_entries
            )
        )
        if expected_index is not None:
            expected_records = expected_index.get("samples")
            if not isinstance(expected_records, list):
                raise LoaderEvaluationError("expected sample index lacks samples")
            start = len(sample_index) - len(captions)
            if expected_records[start : len(sample_index)] != sample_index[start:]:
                raise LoaderEvaluationError(
                    f"loader records differ from frozen index at batch offset {start}"
                )
        if batch_matched != len(captions):
            missing = [str(name) for name, text in zip(names, perturbed) if text is None]
            raise LoaderEvaluationError(f"unmatched perturbations: {missing[:5]}")
        if args.index_only:
            continue

        import torch

        assert model is not None and datamodule is not None and evaluator is not None and device is not None
        start_offset = processed - len(captions)
        evaluator_reference_lengths = (
            motion_lengths if args.length_mode == "reference" else None
        )
        if generation_replay is not None:
            original_motion, original_lengths = decode_audited_features(
                model,
                datamodule,
                generation_replay["original"][start_offset:processed],
                list(captions),
                names,
                motion_ids,
                levels,
                int(poses.shape[1]),
                start_offset,
                reference_lengths=evaluator_reference_lengths,
            )
            paired_motion, paired_lengths = decode_audited_features(
                model,
                datamodule,
                generation_replay["perturbed"][start_offset:processed],
                [str(text) for text in perturbed],
                names,
                motion_ids,
                levels,
                int(poses.shape[1]),
                start_offset,
                reference_lengths=evaluator_reference_lengths,
            )
        else:
            if args.generation_seed is not None:
                reset_rng(int(args.generation_seed) + batch_index * 1_000_003)
            pre_generation_rng = (
                capture_rng_state() if args.pair_rng_mode == "common" else None
            )
            audit_this_batch = generation_audit is not None and (
                args.generation_audit_batches == 0
                or batch_index < args.generation_audit_batches
            )
            original_audit: list[dict[str, Any]] | None = (
                [] if audit_this_batch else None
            )
            paired_audit: list[dict[str, Any]] | None = (
                [] if audit_this_batch else None
            )
            generation_inputs = {
                "original": (list(captions), original_audit),
                "perturbed": ([str(text) for text in perturbed], paired_audit),
            }
            first_kind, second_kind = (
                ("original", "perturbed")
                if args.generation_order == "original-first"
                else ("perturbed", "original")
            )

            def generate_kind(kind: str):
                texts, audit_rows = generation_inputs[kind]
                return generate_features(
                    model,
                    datamodule,
                    texts,
                    str(sampling["mode"]),
                    int(poses.shape[1]),
                    reference_lengths=evaluator_reference_lengths,
                    temperature=float(sampling.get("temperature", 1.0)),
                    top_k=int(sampling.get("top_k", 0)),
                    top_p=float(sampling.get("top_p", 1.0)),
                    pass_sampling_kwargs=args.sampling_kwargs_mode == "explicit",
                    audit_rows=audit_rows,
                    prompt_mode=args.prompt_mode,
                )

            generated: dict[str, tuple[Any, Any]] = {}
            generated[first_kind] = generate_kind(first_kind)
            post_original_rng = (
                capture_rng_state() if args.pair_rng_mode == "common" else None
            )
            if pre_generation_rng is not None:
                restore_rng_state(pre_generation_rng)
            generated[second_kind] = generate_kind(second_kind)
            original_motion, original_lengths = generated["original"]
            paired_motion, paired_lengths = generated["perturbed"]
            if audit_this_batch:
                assert original_audit is not None and paired_audit is not None
                assert generation_audit is not None
                for row_index, (original_row, paired_row) in enumerate(
                    zip(original_audit, paired_audit)
                ):
                    identity = {
                        "sample_offset": start_offset + row_index,
                        "loader_name": str(names[row_index]),
                        "motion_id": str(motion_ids[row_index]),
                        "level": str(levels[row_index]),
                        "reference_frame_count": int(motion_lengths[row_index]),
                    }
                    original_row.update({**identity, "generation_role": "original"})
                    paired_row.update({**identity, "generation_role": "perturbed"})
                generation_audit["original"].extend(original_audit)
                generation_audit["perturbed"].extend(paired_audit)
            if post_original_rng is not None:
                restore_rng_state(post_original_rng)
        with torch.no_grad():
            text_embedding, gt_embedding = evaluator.get_co_embeddings(
                word, pos, sent_lengths, poses, motion_lengths
            )
            _, original_embedding = evaluator.get_co_embeddings(
                word, pos, sent_lengths, original_motion, original_lengths
            )
            paired_embedding = get_motion_embeddings_in_input_order(
                evaluator, paired_motion, paired_lengths
            )
            if perturbed_text_processor is None:
                raise LoaderEvaluationError(
                    "perturbed-text retrieval processor was not initialized"
                )
            perturbed_text_embedding = perturbed_text_processor.encode(
                evaluator, [str(text) for text in perturbed]
            )
        if perturbed_text_embedding.shape != paired_embedding.shape:
            raise LoaderEvaluationError(
                "perturbed-text and perturbed-motion embedding shapes differ: "
                f"{tuple(perturbed_text_embedding.shape)} vs "
                f"{tuple(paired_embedding.shape)}"
            )
        temp_pert_r, temp_pert_matching = calculate_r_precision(
            perturbed_text_embedding.cpu().numpy(),
            paired_embedding.cpu().numpy(),
            top_k=3,
        )
        perturbed_r_precision += temp_pert_r
        perturbed_matching_sum += float(temp_pert_matching)
        if getattr(args, "retrieval_embedding_output", None) is not None:
            retrieval_perturbed_text_embeddings.append(
                perturbed_text_embedding.cpu()
            )
            retrieval_paired_embeddings.append(paired_embedding.cpu())
            retrieval_batch_sizes.append(len(captions))
        gt_embeddings.append(gt_embedding.cpu())
        orig_embeddings.append(original_embedding.cpu())
        pair_embeddings.append(paired_embedding.cpu())
        all_reference_lengths.extend(int(value) for value in motion_lengths)
        all_original_lengths.extend(int(value) for value in original_lengths)
        all_paired_lengths.extend(int(value) for value in paired_lengths)
        if args.embedding_output is not None:
            text_embeddings.append(text_embedding.cpu())
            perturbed_text_embeddings.append(perturbed_text_embedding.cpu())
            original_motions.append(original_motion.cpu())
            original_clamped = clamp_evaluator_lengths(
                original_lengths, motion_lengths
            )
            paired_clamped = clamp_evaluator_lengths(paired_lengths, motion_lengths)
            with torch.no_grad():
                embedding_variants["gt"].append(gt_embedding.cpu())
                embedding_variants["original_generated"].append(
                    original_embedding.cpu()
                )
                embedding_variants["paired_generated"].append(
                    paired_embedding.cpu()
                )
                embedding_variants["original_reference_clamped"].append(
                    get_motion_embeddings_in_input_order(
                        evaluator, original_motion, original_clamped
                    ).cpu()
                )
                embedding_variants["paired_reference_clamped"].append(
                    get_motion_embeddings_in_input_order(
                        evaluator, paired_motion, paired_clamped
                    ).cpu()
                )
                embedding_variants["original_ground_truth"].append(
                    get_motion_embeddings_in_input_order(
                        evaluator, original_motion, motion_lengths
                    ).cpu()
                )
                embedding_variants["paired_ground_truth"].append(
                    get_motion_embeddings_in_input_order(
                        evaluator, paired_motion, motion_lengths
                    ).cpu()
                )
        temp_r, temp_matching = calculate_r_precision(
            text_embedding.cpu().numpy(), original_embedding.cpu().numpy(), top_k=3
        )
        temp_gt_r, _ = calculate_r_precision(
            text_embedding.cpu().numpy(), gt_embedding.cpu().numpy(), top_k=3
        )
        r_precision += temp_r
        gt_r_precision += temp_gt_r
        matching_sum += float(temp_matching)
        if args.diagnostics_output is not None:
            ranks = correct_match_ranks(
                text_embedding.cpu().numpy(), original_embedding.cpu().numpy()
            )
            gt_ranks = correct_match_ranks(
                text_embedding.cpu().numpy(), gt_embedding.cpu().numpy()
            )
            perturbed_ranks = correct_match_ranks(
                perturbed_text_embedding.cpu().numpy(), paired_embedding.cpu().numpy()
            )
            diagnostic_batches.append(
                {
                    "batch_index": len(diagnostic_batches),
                    "count": len(captions),
                    "temp_R_generated": [int(value) for value in temp_r],
                    "temp_R_ground_truth": [int(value) for value in temp_gt_r],
                    "temp_R_perturbed_text_generated": [
                        int(value) for value in temp_pert_r
                    ],
                    "generated_correct_rank_histogram": {
                        str(rank): int((ranks == rank).sum()) for rank in np.unique(ranks)
                    },
                    "ground_truth_correct_rank_histogram": {
                        str(rank): int((gt_ranks == rank).sum()) for rank in np.unique(gt_ranks)
                    },
                    "perturbed_text_generated_correct_rank_histogram": {
                        str(rank): int((perturbed_ranks == rank).sum())
                        for rank in np.unique(perturbed_ranks)
                    },
                    "reference_motion_lengths": [int(value) for value in motion_lengths],
                    "generated_motion_lengths": [int(value) for value in original_lengths],
                    "paired_motion_lengths": [int(value) for value in paired_lengths],
                }
            )

    smoke = args.limit is not None and args.limit < expected_formal_samples
    if not smoke and processed != expected_formal_samples:
        raise LoaderEvaluationError(
            f"formal run processed {processed}, expected {expected_formal_samples}"
        )
    if not smoke and preset["require_unique_motions"]:
        unique_loader_names = len(set(sample_ids))
        if unique_loader_names != expected_formal_samples:
            raise LoaderEvaluationError(
                f"formal {args.dataset} loader must contain {expected_formal_samples} "
                f"unique motions, got {unique_loader_names}"
            )
    if matched != processed:
        raise LoaderEvaluationError(f"matched {matched}/{processed} loader samples")
    if args.loader_mode == "caption-locked" and caption_matched != processed:
        raise LoaderEvaluationError(
            f"caption-aware matching covered {caption_matched}/{processed} loader samples"
        )
    if not args.index_only and not smoke and args.expected_sample_index is None:
        raise LoaderEvaluationError("formal model evaluation requires --expected-sample-index")
    if args.expected_sample_index is not None:
        validate_expected_index(
            expected_index, sample_index, protocol_fingerprint
        )

    name_order_sha256 = canonical_sha256(sample_ids)
    sample_index_sha256 = canonical_sha256(sample_index)
    common = {
        "schema_version": f"motiongpt-{preset['schema_slug']}-result/v1",
        "dataset": args.dataset,
        "seed": args.seed,
        "smoke": smoke,
        "index_only": bool(args.index_only),
        "matched_count": matched,
        "total_count": processed,
        "formal_sample_count": expected_formal_samples,
        "raw_loader_dataset_count": len(loader.dataset),
        "drop_last_count": len(loader.dataset) - expected_loader_samples,
        # ``fallback`` can be an annotation provenance label in the source
        # pair file.  Runtime fallback is forbidden and reported separately.
        "annotation_level_counts": dict(sorted(level_counts.items())),
        "unmatched_count": processed - matched,
        "caption_matched_count": caption_matched,
        "caption_mismatch_count": processed - caption_matched,
        "runtime_fallback_count": 0,
        "sample_index_sha256": sample_index_sha256,
        "name_order_sha256": name_order_sha256,
        "protocol_fingerprint": protocol_fingerprint,
        "protocol": protocol,
        "source_manifest_fingerprint": source_protocol.get("fingerprint"),
        "model": dict(model_meta),
    }
    if args.sample_index_output:
        write_json(
            args.sample_index_output,
            {
                "schema_version": f"motiongpt-{preset['schema_slug']}-index/v1",
                "seed": args.seed,
                "protocol_fingerprint": protocol_fingerprint,
                "sample_index_sha256": sample_index_sha256,
                "name_order_sha256": name_order_sha256,
                "count": len(sample_index),
                "samples": sample_index,
            },
        )
    if args.index_only:
        return common

    import torch

    gt = torch.cat(gt_embeddings).numpy()
    orig = torch.cat(orig_embeddings).numpy()
    pair = torch.cat(pair_embeddings).numpy()
    gt_mu, gt_cov = calculate_activation_statistics(gt)
    orig_mu, orig_cov = calculate_activation_statistics(orig)
    pair_mu, pair_cov = calculate_activation_statistics(pair)
    metrics = {
        "FID": calculate_frechet_distance(gt_mu, gt_cov, orig_mu, orig_cov),
        "FID_P": calculate_frechet_distance(gt_mu, gt_cov, pair_mu, pair_cov),
        "FID_D": calculate_frechet_distance(orig_mu, orig_cov, pair_mu, pair_cov),
        "Matching": matching_sum / processed,
        "R@1": float(r_precision[0] / processed),
        "R@2": float(r_precision[1] / processed),
        "R@3": float(r_precision[2] / processed),
        "top1_pert": float(perturbed_r_precision[0] / processed),
        "top2_pert": float(perturbed_r_precision[1] / processed),
        "top3_pert": float(perturbed_r_precision[2] / processed),
        "matching_pert": perturbed_matching_sum / processed,
        "Diversity": evaluation_diversity(orig),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise LoaderEvaluationError("non-finite metric produced")
    checkpoint = Path(str(model_meta["checkpoint"])).resolve()
    common["artifacts"] = {
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
        "evaluator_checkpoint": {
            "path": str(
                (REPO_ROOT / f"checkpoints/{preset['evaluator']}/text_mot_match/model/finest.tar").resolve()
            ),
            "sha256": file_sha256(
                REPO_ROOT / f"checkpoints/{preset['evaluator']}/text_mot_match/model/finest.tar"
            ),
        },
    }
    common["checkpoint_load_audit"] = checkpoint_load_audit
    common["perturbed_text_retrieval"] = {
        "sample_count": int(processed),
        "batch_count": int(len(gt_embeddings)),
        "batch_size": int(args.batch_size),
        "text_source": "resolved text_pair perturbation",
        "motion_source": "motion generated from the same resolved text_pair perturbation",
        "evaluator": "standard TM2T TextEncoderBiGRUCo/MotionEncoderBiGRUCo",
        "word_vectorizer": "glove/our_vab",
        "spacy_model": args.spacy_model,
        "max_text_len": int(getattr(evaluator.opt, "max_text_len", 20)),
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
    }
    common["metrics"] = metrics
    common["length_summary"] = {
        "reference_mean": float(np.mean(all_reference_lengths)),
        "original_mean": float(np.mean(all_original_lengths)),
        "perturbed_mean": float(np.mean(all_paired_lengths)),
        "original_reference_mae": float(
            np.mean(np.abs(np.asarray(all_original_lengths) - all_reference_lengths))
        ),
        "perturbed_reference_mae": float(
            np.mean(np.abs(np.asarray(all_paired_lengths) - all_reference_lengths))
        ),
    }
    if args.diagnostics_output is not None:
        diagnostics = {
            "schema_version": f"motiongpt-{preset['schema_slug']}-diagnostics/v1",
            "seed": args.seed,
            "count": processed,
            "generated_R_counts": [int(value) for value in r_precision],
            "ground_truth_R_counts": [int(value) for value in gt_r_precision],
            "perturbed_text_generated_R_counts": [
                int(value) for value in perturbed_r_precision
            ],
            "generated_R": [float(value / processed) for value in r_precision],
            "ground_truth_R": [float(value / processed) for value in gt_r_precision],
            "perturbed_text_generated_R": [
                float(value / processed) for value in perturbed_r_precision
            ],
            "perturbed_text_generated_matching": (
                float(perturbed_matching_sum / processed)
            ),
            "batches": diagnostic_batches,
        }
        write_json(args.diagnostics_output, diagnostics)
    if args.generation_audit_output is not None:
        if generation_audit is None or not generation_audit["original"]:
            raise LoaderEvaluationError("generation audit requested but no batch was generated")
        write_json(args.generation_audit_output, generation_audit)
        common["generation_audit"] = str(args.generation_audit_output.resolve())
    elif generation_replay_path is not None:
        common["generation_audit"] = str(generation_replay_path)
        common["generation_replay"] = {
            "mode": "exact audited motion-token replay",
            "path": str(generation_replay_path),
            "sha256": file_sha256(generation_replay_path),
            "original_count": len(generation_replay["original"]),
            "perturbed_count": len(generation_replay["perturbed"]),
        }
    if args.retrieval_embedding_output is not None:
        if (
            len(retrieval_perturbed_text_embeddings) != len(gt_embeddings)
            or len(retrieval_paired_embeddings) != len(gt_embeddings)
            or sum(retrieval_batch_sizes) != processed
        ):
            raise LoaderEvaluationError(
                "retrieval embedding dump does not cover every evaluator batch"
            )
        args.retrieval_embedding_output.parent.mkdir(
            parents=True, exist_ok=True
        )
        np.savez_compressed(
            args.retrieval_embedding_output,
            perturbed_text_embedding=torch.cat(
                retrieval_perturbed_text_embeddings
            ).numpy(),
            perturbed_motion_embedding=torch.cat(
                retrieval_paired_embeddings
            ).numpy(),
            batch_sizes=np.asarray(retrieval_batch_sizes, dtype=np.int64),
            loader_name=np.asarray(sample_ids),
            schema_version=np.asarray(
                "tm2t-perturbed-retrieval-embeddings/v1"
            ),
            seed=np.asarray(args.seed, dtype=np.int64),
            sample_count=np.asarray(processed, dtype=np.int64),
            batch_size=np.asarray(args.batch_size, dtype=np.int64),
            sample_index_sha256=np.asarray(sample_index_sha256),
            protocol_fingerprint=np.asarray(protocol_fingerprint),
        )
        common["retrieval_embedding_dump"] = {
            "schema_version": "tm2t-perturbed-retrieval-embeddings/v1",
            "path": str(args.retrieval_embedding_output.resolve()),
            "sha256": file_sha256(args.retrieval_embedding_output),
            "sample_count": processed,
            "batch_count": len(retrieval_batch_sizes),
        }
    if args.embedding_output is not None:
        arrays = {
            key: torch.cat(values).numpy()
            for key, values in embedding_variants.items()
        }
        args.embedding_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.embedding_output,
            **arrays,
            text_embedding=torch.cat(text_embeddings).numpy(),
            perturbed_text_embedding=torch.cat(perturbed_text_embeddings).numpy(),
            original_motion=torch.cat(original_motions).numpy(),
            loader_name=np.asarray(sample_ids),
            motion_id=np.asarray([row["motion_id"] for row in sample_index]),
            caption=np.asarray([row["caption"] for row in sample_index]),
            perturbed_caption=np.asarray([
                row["perturbed_caption"] for row in sample_index
            ]),
            level=np.asarray([row["level"] for row in sample_index]),
            reference_length=np.asarray(all_reference_lengths, dtype=np.int64),
            original_length=np.asarray(all_original_lengths, dtype=np.int64),
            perturbed_length=np.asarray(all_paired_lengths, dtype=np.int64),
        )
        common["embedding_dump"] = str(args.embedding_output.resolve())
    return common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="existing MotionGPT seed manifest")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets-cfg", type=Path)
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_PROTOCOLS),
        default="humanml3d",
        help="shared evaluator-loader protocol; defaults preserve HumanML3D behavior",
    )
    parser.add_argument("--pair-file", type=Path, default=REPO_ROOT / "dataset/HumanML3D/test_perb_deterministic.json")
    parser.add_argument(
        "--expected-samples",
        type=int,
        help=(
            "explicit formal sample-count contract for a caption-locked subset; "
            "omitting it preserves the dataset's legacy/default count"
        ),
    )
    parser.add_argument(
        "--loader-mode",
        choices=("legacy-motion-id", "caption-locked"),
        default="legacy-motion-id",
        help=(
            "caption-locked resolves the exact normalized (motion_id, text_orig) "
            "pair and deterministically uses its annotated full motion or segment"
        ),
    )
    parser.add_argument(
        "--allowed-levels",
        nargs="+",
        help="optional perturbation-level allowlist, e.g. P1 P2 P3",
    )
    parser.add_argument(
        "--caption-length-policy",
        choices=("strict", "repair-all"),
        default="strict",
        help=(
            "strict preserves HumanML3D's 40..199-frame filter; repair-all "
            "expands short segments or explicitly resamples otherwise excluded "
            "sequences so every caption-locked P1/P2/P3 definition is evaluated"
        ),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--generation-seed",
        type=int,
        help="independent sampling seed; leaves loader order/caption selection tied to --seed",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_sm",
        help="spaCy model used to reproduce TM2T lemma/POS caption preprocessing",
    )
    parser.add_argument("--sample-mode", choices=("categorical", "greedy"))
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument(
        "--pair-rng-mode",
        choices=("sequential", "common"),
        default="sequential",
        help=(
            "sequential matches the legacy T2M-GPT/SATO++ protocol; common "
            "replays identical random numbers for each original/perturbed pair"
        ),
    )
    parser.add_argument(
        "--protocol-evaluator-id",
        help="override only the evaluator identifier recorded in the protocol fingerprint",
    )
    parser.add_argument(
        "--sampling-kwargs-mode",
        choices=("legacy", "explicit"),
        default="explicit",
        help=(
            "legacy preserves MotionGPT's original generate() defaults; explicit "
            "passes temperature/top-k/top-p from the protocol"
        ),
    )
    parser.add_argument(
        "--length-mode",
        choices=("reference", "generated"),
        default="generated",
        help=(
            "generated matches the SATO++ evaluator contract; reference uses "
            "min(generated, ground-truth) to reproduce MotionGPT validation"
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("official-default", "instruction-template"),
        default="official-default",
        help=(
            "official-default uses MotionGPT's released HumanML3D test prompt; "
            "instruction-template reproduces the previous random-template adapter"
        ),
    )
    parser.add_argument(
        "--generation-order",
        choices=("original-first", "perturbed-first"),
        default="original-first",
        help="perturbed-first is a diagnostic control for sequential RNG order",
    )
    parser.add_argument("--limit", type=int, help="explicit smoke sample count")
    parser.add_argument("--index-only", action="store_true", help="audit loader matching without model generation")
    parser.add_argument("--protocol-manifest", type=Path)
    parser.add_argument("--sample-index-output", type=Path)
    parser.add_argument(
        "--expected-sample-index",
        type=Path,
        help="frozen per-seed index; formal model runs require exact parity",
    )
    parser.add_argument("--diagnostics-output", type=Path)
    parser.add_argument(
        "--generation-audit-input",
        type=Path,
        help=(
            "replay exact original/perturbed motion-token sequences from a complete "
            "prior generation audit instead of sampling again"
        ),
    )
    parser.add_argument("--generation-audit-output", type=Path)
    parser.add_argument(
        "--generation-audit-batches",
        type=int,
        default=1,
        help="number of batches to audit; 0 records every batch",
    )
    parser.add_argument("--embedding-output", type=Path)
    parser.add_argument(
        "--retrieval-embedding-output",
        type=Path,
        help=(
            "lightweight NPZ containing perturbed-caption and corresponding "
            "perturbed-generation TM2T embeddings plus evaluator batch sizes"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.batch_size <= 0
        or args.num_workers < 0
        or (args.expected_samples is not None and args.expected_samples < 3)
        or (args.limit is not None and args.limit <= 0)
        or (args.temperature is not None and args.temperature <= 0)
        or (args.top_k is not None and args.top_k < 0)
        or (args.top_p is not None and not 0 < args.top_p <= 1)
        or args.generation_audit_batches < 0
    ):
        parser.error("invalid batch, worker, limit, or sampling argument")
    try:
        result = evaluate(args)
    except (OSError, json.JSONDecodeError, LoaderEvaluationError) as exc:
        parser.error(str(exc))
    write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
