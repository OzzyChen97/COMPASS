#!/usr/bin/env python3
"""Diagnose MotionGPT and SATO++ robustness outputs on an aligned loader run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import linalg


def activation_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(values, axis=0), np.cov(values, rowvar=False)


def frechet_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_mu, left_cov = activation_statistics(left)
    right_mu, right_cov = activation_statistics(right)
    difference = left_mu - right_mu
    covariance_mean, _ = linalg.sqrtm(left_cov.dot(right_cov), disp=False)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(left_cov.shape[0]) * 1e-6
        covariance_mean = linalg.sqrtm(
            (left_cov + offset).dot(right_cov + offset)
        )
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    return float(
        difference.dot(difference)
        + np.trace(left_cov)
        + np.trace(right_cov)
        - 2 * np.trace(covariance_mean)
    )


def fid_triplet(
    ground_truth: np.ndarray,
    original: np.ndarray,
    perturbed: np.ndarray,
) -> dict[str, float]:
    return {
        "FID": frechet_distance(ground_truth, original),
        "FID_P": frechet_distance(ground_truth, perturbed),
        "FID_D": frechet_distance(original, perturbed),
    }


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def length_summary(
    reference: np.ndarray,
    original: np.ndarray,
    perturbed: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in (
        ("reference", reference),
        ("original", original),
        ("perturbed", perturbed),
    ):
        result[name] = {
            **percentile_summary(values),
            "at_196_fraction": float(np.mean(values >= 196)),
            "multiple_of_4_fraction": float(np.mean(values % 4 == 0)),
        }
    result["original_reference_mae"] = float(
        np.mean(np.abs(original - reference))
    )
    result["perturbed_reference_mae"] = float(
        np.mean(np.abs(perturbed - reference))
    )
    result["pair_length_mae"] = float(np.mean(np.abs(original - perturbed)))
    result["pair_length_equal_fraction"] = float(np.mean(original == perturbed))
    return result


def embedding_summary(values: np.ndarray) -> dict[str, float]:
    _, covariance = activation_statistics(values)
    norms = np.linalg.norm(values, axis=1)
    return {
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "covariance_trace": float(np.trace(covariance)),
        "mean_vector_norm": float(np.linalg.norm(np.mean(values, axis=0))),
    }


def pair_shift(original: np.ndarray, perturbed: np.ndarray) -> dict[str, Any]:
    original_norm = np.linalg.norm(original, axis=1)
    perturbed_norm = np.linalg.norm(perturbed, axis=1)
    denominator = np.maximum(original_norm * perturbed_norm, 1e-12)
    cosine = np.sum(original * perturbed, axis=1) / denominator
    delta = perturbed - original
    l2 = np.linalg.norm(delta, axis=1)
    mean_delta_norm = float(np.linalg.norm(np.mean(delta, axis=0)))
    mean_l2 = float(np.mean(l2))
    return {
        "cosine": percentile_summary(cosine),
        "l2": percentile_summary(l2),
        "exact_equal_fraction": float(np.mean(np.all(original == perturbed, axis=1))),
        "mean_delta_norm": mean_delta_norm,
        "cancellation_ratio": mean_delta_norm / max(mean_l2, 1e-12),
    }


def array_alignment(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"shape_equal": False, "left_shape": left.shape, "right_shape": right.shape}
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if difference.ndim == 1:
        row_l2 = np.abs(difference)
    else:
        row_l2 = np.linalg.norm(difference, axis=1)
    return {
        "shape_equal": True,
        "exact_equal": bool(np.array_equal(left, right)),
        "allclose_1e-6": bool(np.allclose(left, right, atol=1e-6, rtol=1e-6)),
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "row_l2": percentile_summary(row_l2),
    }


def group_metrics(
    levels: np.ndarray,
    ground_truth: np.ndarray,
    original: np.ndarray,
    perturbed: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for level in sorted(set(str(value) for value in levels)):
        mask = levels.astype(str) == level
        result[level] = {
            "count": int(np.sum(mask)),
            "metrics": fid_triplet(
                ground_truth[mask], original[mask], perturbed[mask]
            ),
            "pair_shift": pair_shift(original[mask], perturbed[mask]),
        }
    return result


def verify_metadata(
    motiongpt: Mapping[str, np.ndarray],
    sato: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("loader_name", "motion_id", "level", "reference_length"):
        result[key] = {
            "equal": bool(np.array_equal(motiongpt[key], sato[key])),
            "mismatch_count": int(np.sum(motiongpt[key] != sato[key])),
        }
    return result


def audit_generation(
    path: Path,
    index_path: Path,
    sato: Mapping[str, np.ndarray],
    codebook_size: int,
) -> dict[str, Any]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))["samples"]
    result: dict[str, Any] = {}
    expected_text = {
        "original": [str(row["caption"]) for row in index],
        "perturbed": [str(row["perturbed_caption"]) for row in index],
    }
    sato_text = {
        "original": sato["caption"].astype(str).tolist(),
        "perturbed": sato["perturbed_caption"].astype(str).tolist(),
    }
    for kind in ("original", "perturbed"):
        rows = audit[kind]
        input_texts = [str(row["input_text"]) for row in rows]
        quoted_prompts = [f'Generate motion: "{text}"' for text in input_texts]
        unquoted_prompts = [f"Generate motion: {text}" for text in input_texts]
        token_counts = np.asarray(
            [int(row["motion_token_count"]) for row in rows], dtype=np.int64
        )
        generated_lengths = np.asarray(
            [int(row["generated_frame_count"]) for row in rows], dtype=np.int64
        )
        invalid_tokens = sum(
            int(token < 0 or token >= codebook_size)
            for row in rows
            for token in row["motion_tokens"]
        )
        result[kind] = {
            "count": len(rows),
            "input_matches_frozen_index": input_texts == expected_text[kind],
            "input_matches_sato_npz": input_texts == sato_text[kind],
            "quoted_prompt_count": int(sum(
                str(row["fulfilled_prompt"]) == expected
                for row, expected in zip(rows, quoted_prompts)
            )),
            "unquoted_prompt_count": int(sum(
                str(row["fulfilled_prompt"]) == expected
                for row, expected in zip(rows, unquoted_prompts)
            )),
            "unexpected_prompt_count": int(sum(
                str(row["fulfilled_prompt"]) not in (quoted, unquoted)
                for row, quoted, unquoted in zip(
                    rows, quoted_prompts, unquoted_prompts
                )
            )),
            "token_count_field_mismatch_count": int(sum(
                len(row["motion_tokens"]) != int(row["motion_token_count"])
                for row in rows
            )),
            "decoded_length_mismatch_count": int(sum(
                int(row["raw_generated_frame_count"])
                != (
                    1
                    if int(row["motion_token_count"]) <= 1
                    else 4 * int(row["motion_token_count"])
                )
                for row in rows
            )),
            "single_token_fallback_count": int(sum(
                int(row["motion_token_count"]) <= 1 for row in rows
            )),
            "invalid_motion_token_count": int(invalid_tokens),
            "truncated_count": int(sum(
                bool(row["truncated_to_batch_seq"]) for row in rows
            )),
            "token_count": percentile_summary(token_counts),
            "generated_length": percentile_summary(generated_lengths),
        }
    original_rows = audit["original"]
    perturbed_rows = audit["perturbed"]
    result["pair"] = {
        "exact_token_sequence_fraction": float(np.mean([
            left["motion_tokens"] == right["motion_tokens"]
            for left, right in zip(original_rows, perturbed_rows)
        ])),
        "same_token_count_fraction": float(np.mean([
            left["motion_token_count"] == right["motion_token_count"]
            for left, right in zip(original_rows, perturbed_rows)
        ])),
    }
    return result


def tokenization_audit(
    generation_path: Path,
    sato: Mapping[str, np.ndarray],
    motiongpt_tokenizer_path: Path,
    levels: np.ndarray,
) -> dict[str, Any]:
    from clip.simple_tokenizer import SimpleTokenizer
    from transformers import AutoTokenizer

    audit = json.loads(generation_path.read_text(encoding="utf-8"))
    motiongpt_tokenizer = AutoTokenizer.from_pretrained(
        str(motiongpt_tokenizer_path), legacy=True
    )
    clip_tokenizer = SimpleTokenizer()
    result: dict[str, Any] = {"motiongpt_context_length": 256, "clip_context_length": 77}

    motiongpt_ids: dict[str, list[list[int]]] = {}
    clip_ids: dict[str, list[list[int]]] = {}
    for kind, sato_key in (
        ("original", "caption"),
        ("perturbed", "perturbed_caption"),
    ):
        prompts = [str(row["fulfilled_prompt"]) for row in audit[kind]]
        texts = sato[sato_key].astype(str).tolist()
        raw_motiongpt_ids = [
            motiongpt_tokenizer(prompt, add_special_tokens=True)["input_ids"]
            for prompt in prompts
        ]
        raw_clip_ids = [clip_tokenizer.encode(text) for text in texts]
        motiongpt_ids[kind] = [ids[:256] for ids in raw_motiongpt_ids]
        clip_ids[kind] = [ids[:75] for ids in raw_clip_ids]
        motiongpt_lengths = np.asarray(
            [len(ids) for ids in raw_motiongpt_ids], dtype=np.int64
        )
        clip_lengths = np.asarray(
            [len(ids) + 2 for ids in raw_clip_ids], dtype=np.int64
        )
        result[kind] = {
            "motiongpt_token_length": percentile_summary(motiongpt_lengths),
            "motiongpt_truncated_count": int(np.sum(motiongpt_lengths > 256)),
            "clip_token_length": percentile_summary(clip_lengths),
            "clip_truncated_count": int(np.sum(clip_lengths > 77)),
            "by_level": {},
        }
        for level in sorted(set(str(value) for value in levels)):
            mask = levels.astype(str) == level
            result[kind]["by_level"][level] = {
                "count": int(np.sum(mask)),
                "motiongpt_truncated_count": int(
                    np.sum(motiongpt_lengths[mask] > 256)
                ),
                "clip_truncated_count": int(np.sum(clip_lengths[mask] > 77)),
                "motiongpt_mean_tokens": float(np.mean(motiongpt_lengths[mask])),
                "clip_mean_tokens": float(np.mean(clip_lengths[mask])),
            }
    result["pair"] = {
        "motiongpt_identical_encoded_fraction": float(np.mean([
            left == right
            for left, right in zip(
                motiongpt_ids["original"], motiongpt_ids["perturbed"]
            )
        ])),
        "clip_identical_encoded_fraction": float(np.mean([
            left == right
            for left, right in zip(clip_ids["original"], clip_ids["perturbed"])
        ])),
    }
    return result


def metric_rows(result: Mapping[str, Any]) -> list[str]:
    rows = ["| Model / length | FID | FID_P | FID_D |", "|---|---:|---:|---:|"]
    for name, payload in result["models"].items():
        metrics = payload["metrics"]
        rows.append(
            f"| {name} | {metrics['FID']:.6f} | {metrics['FID_P']:.6f} | "
            f"{metrics['FID_D']:.6f} |"
        )
    return rows


def render_report(result: Mapping[str, Any]) -> str:
    lines = ["# MotionGPT vs SATO++ Seed Diagnostics", "", *metric_rows(result), ""]
    lines.extend(["## Ground Truth Alignment", ""])
    alignment = result["ground_truth_alignment"]
    lines.append(
        f"- exact_equal: `{alignment['exact_equal']}`; "
        f"allclose_1e-6: `{alignment['allclose_1e-6']}`; "
        f"max_abs: `{alignment['max_abs']:.8g}`"
    )
    lines.extend(["", "## Pair Shift", ""])
    for name, payload in result["models"].items():
        shift = payload["pair_shift"]
        lines.append(
            f"- {name}: cosine mean `{shift['cosine']['mean']:.6f}`, "
            f"L2 mean `{shift['l2']['mean']:.6f}`"
        )
    lines.extend(["", "## Corresponding Ground Truth", ""])
    lines.extend([
        "| Model / length | Original cosine | Original L2 | Perturbed cosine | Perturbed L2 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, payload in result["models"].items():
        original = payload["original_to_ground_truth"]
        perturbed = payload["perturbed_to_ground_truth"]
        lines.append(
            f"| {name} | {original['cosine']['mean']:.6f} | "
            f"{original['l2']['mean']:.6f} | "
            f"{perturbed['cosine']['mean']:.6f} | "
            f"{perturbed['l2']['mean']:.6f} |"
        )
    lines.extend(["", "## Lengths", ""])
    for name, payload in result["lengths"].items():
        lines.append(
            f"- {name}: original mean `{payload['original']['mean']:.3f}`, "
            f"perturbed mean `{payload['perturbed']['mean']:.3f}`, "
            f"pair MAE `{payload['pair_length_mae']:.3f}`"
        )
    if "generation_audit" in result:
        lines.extend(["", "## Generation Audit", ""])
        for kind in ("original", "perturbed"):
            payload = result["generation_audit"][kind]
            lines.append(
                f"- {kind}: `{payload['count']}` rows, unexpected prompts "
                f"`{payload['unexpected_prompt_count']}`, invalid tokens "
                f"`{payload['invalid_motion_token_count']}`"
            )
    if "tokenization_audit" in result:
        lines.extend(["", "## Tokenization Audit", ""])
        for kind in ("original", "perturbed"):
            payload = result["tokenization_audit"][kind]
            lines.append(
                f"- {kind}: MotionGPT truncated "
                f"`{payload['motiongpt_truncated_count']}`, CLIP truncated "
                f"`{payload['clip_truncated_count']}`"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motiongpt-npz", type=Path, required=True)
    parser.add_argument("--sato-npz", type=Path, required=True)
    parser.add_argument("--motiongpt-index", type=Path)
    parser.add_argument("--generation-audit", type=Path)
    parser.add_argument("--motiongpt-tokenizer", type=Path)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    with np.load(args.motiongpt_npz) as loaded:
        motiongpt = {key: loaded[key] for key in loaded.files}
    with np.load(args.sato_npz) as loaded:
        sato = {key: loaded[key] for key in loaded.files}

    metadata = verify_metadata(motiongpt, sato)
    if not all(payload["equal"] for payload in metadata.values()):
        raise SystemExit(f"MotionGPT/SATO++ metadata mismatch: {metadata}")

    models = {
        "sato_pp_generated": (
            sato["gt_embedding"],
            sato["original_embedding"],
            sato["perturbed_embedding"],
        ),
        "motiongpt_generated": (
            motiongpt["gt"],
            motiongpt["original_generated"],
            motiongpt["paired_generated"],
        ),
        "motiongpt_reference_clamped": (
            motiongpt["gt"],
            motiongpt["original_reference_clamped"],
            motiongpt["paired_reference_clamped"],
        ),
        "motiongpt_ground_truth_length": (
            motiongpt["gt"],
            motiongpt["original_ground_truth"],
            motiongpt["paired_ground_truth"],
        ),
    }
    levels = motiongpt["level"].astype(str)
    result: dict[str, Any] = {
        "schema_version": "motiongpt-sato-seed-diagnostics/v1",
        "motiongpt_npz": str(args.motiongpt_npz.resolve()),
        "sato_npz": str(args.sato_npz.resolve()),
        "count": int(len(levels)),
        "metadata_alignment": metadata,
        "ground_truth_alignment": array_alignment(
            motiongpt["gt"], sato["gt_embedding"]
        ),
        "models": {},
        "lengths": {
            "motiongpt": length_summary(
                motiongpt["reference_length"],
                motiongpt["original_length"],
                motiongpt["perturbed_length"],
            ),
            "sato_pp": length_summary(
                sato["reference_length"],
                sato["original_length"],
                sato["perturbed_length"],
            ),
        },
    }
    for name, (ground_truth, original, perturbed) in models.items():
        result["models"][name] = {
            "metrics": fid_triplet(ground_truth, original, perturbed),
            "by_level": group_metrics(
                levels, ground_truth, original, perturbed
            ),
            "ground_truth_embedding": embedding_summary(ground_truth),
            "original_embedding": embedding_summary(original),
            "perturbed_embedding": embedding_summary(perturbed),
            "pair_shift": pair_shift(original, perturbed),
            "original_to_ground_truth": pair_shift(ground_truth, original),
            "perturbed_to_ground_truth": pair_shift(ground_truth, perturbed),
            "fid_permutation_invariant_check": {
                "perturbed_roll_1": fid_triplet(
                    ground_truth, original, np.roll(perturbed, 1, axis=0)
                )
            },
        }

    if args.generation_audit or args.motiongpt_index:
        if args.generation_audit is None or args.motiongpt_index is None:
            raise SystemExit(
                "--generation-audit and --motiongpt-index must be provided together"
            )
        result["generation_audit"] = audit_generation(
            args.generation_audit,
            args.motiongpt_index,
            sato,
            args.codebook_size,
        )
        if args.motiongpt_tokenizer is not None:
            result["tokenization_audit"] = tokenization_audit(
                args.generation_audit,
                sato,
                args.motiongpt_tokenizer,
                levels,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(result), encoding="utf-8")


if __name__ == "__main__":
    main()
