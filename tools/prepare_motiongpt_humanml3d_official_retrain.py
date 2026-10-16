#!/usr/bin/env python3
"""Prepare an isolated, baseline-free official MotionGPT HumanML3D retrain."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


VALIDATION_EVALUATOR_SAMPLES = 1504


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def fresh_config(path: Path):
    return OmegaConf.create(
        OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    )


def data_census(dataset_root: Path) -> dict:
    result = {}
    for split in ("train", "val", "test"):
        ids = [
            line.strip()
            for line in (dataset_root / f"{split}.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        token_eligible = 0
        vae_full = 0
        vae_segments = 0
        missing_motion = 0
        for name in ids:
            motion_path = dataset_root / "new_joint_vecs" / f"{name}.npy"
            if not motion_path.is_file():
                missing_motion += 1
                continue
            motion = np.load(motion_path, mmap_mode="r")
            frames = len(motion)
            if 40 <= frames < 200:
                token_eligible += 1
            if 64 <= frames < 200:
                vae_full += 1
            text_path = dataset_root / "texts" / f"{name}.txt"
            if not text_path.is_file():
                continue
            for line in text_path.read_text(encoding="utf-8").splitlines():
                fields = line.strip().split("#")
                if len(fields) < 4:
                    continue
                try:
                    start, end = float(fields[2]), float(fields[3])
                except ValueError:
                    continue
                start = 0.0 if math.isnan(start) else start
                end = 0.0 if math.isnan(end) else end
                if start == 0.0 and end == 0.0:
                    continue
                segment_frames = len(motion[int(start * 20):int(end * 20)])
                if 64 <= segment_frames < 200:
                    vae_segments += 1
        result[split] = {
            "split_ids": len(ids),
            "missing_motion": missing_motion,
            "token_eligible_original_motions": token_eligible,
            "vae_full_motions_ge_64": vae_full,
            "vae_caption_segments_ge_64": vae_segments,
            "vae_examples_ge_64": vae_full + vae_segments,
        }
    result["validation_evaluator"] = {
        "source_split": "val",
        "dataset_name": "t2m",
        "expanded_dataset_examples": 1530,
        "batch_size": 32,
        "drop_last": True,
        "samples_per_validation": VALIDATION_EVALUATOR_SAMPLES,
    }
    return result


def configure_common(cfg, name: str, seed: int, devices: list[int]) -> None:
    cfg.NAME = name
    cfg.SEED_VALUE = seed
    cfg.ACCELERATOR = "gpu"
    cfg.NUM_NODES = 1
    cfg.DEVICE = devices
    cfg.DEBUG = False
    cfg.TRAIN.RESUME = ""
    cfg.TRAIN.PRETRAINED = ""
    cfg.TRAIN.PRETRAINED_VAE = ""
    cfg.TRAIN.SHUFFLE = True
    cfg.EVAL.SPLIT = "val"
    # HumanML3D keeps the full motion arrays in each dataset object. Forking
    # loader workers after CUDA/DDP initialization can leave every rank
    # blocked on the multiprocessing result queue, so use in-process loading.
    cfg.EVAL.NUM_WORKERS = 0
    cfg.TEST.CHECKPOINTS = ""
    cfg.TEST.SPLIT = "test"
    cfg.TEST.NUM_WORKERS = 0
    cfg.LOGGER.TYPE = []
    cfg.LOGGER.WANDB.params.project = None
    cfg.LOGGER.WANDB.params.offline = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    mgpt_root = repo_root / "MotionGPT"
    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else repo_root / "dataset/HumanML3D"
    )
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        parser.error(f"fresh run output already exists and is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "configs"
    config_dir.mkdir()

    tag = args.run_tag.replace("/", "_")
    names = {
        "stage1": f"VQVAE_H3D_{tag}",
        "token": f"TOKENIZE_H3D_{tag}",
        "stage2": f"Pretrain_H3D_{tag}",
        "stage3": f"Instruct_H3D_{tag}",
    }
    token_code_path = names["stage1"]
    base_paths = {
        "stage1": mgpt_root / "configs/config_h3d_stage1.yaml",
        "token": mgpt_root / "configs/config_h3d_stage1.yaml",
        "stage2": mgpt_root / "configs/config_h3d_stage2.yaml",
        "stage3": mgpt_root / "configs/config_h3d_stage3.yaml",
    }
    configs = {}
    for stage, base_path in base_paths.items():
        cfg = fresh_config(base_path)
        configure_common(cfg, names[stage], args.seed, [0] if stage == "token" else [0, 1, 2, 3])
        cfg.DATASET.CODE_PATH = token_code_path
        cfg.METRIC.TYPE = []
        if stage == "stage1":
            cfg.TRAIN.STAGE = "vae"
            cfg.TRAIN.NUM_WORKERS = 0
            cfg.TRAIN.BATCH_SIZE = 64
            cfg.TRAIN.END_EPOCH = 999999
            cfg.TRAIN.MAX_STEPS = 300000
            cfg.EVAL.BATCH_SIZE = 32
            cfg.LOGGER.VAL_EVERY_STEPS = 10
            cfg.LOGGER.VAL_EVERY_N_STEPS = 10000
            cfg.EARLY_STOPPING = {
                "ENABLED": True,
                "MONITOR": "total/val",
                "MODE": "min",
                "MIN_DELTA": 1e-4,
                "PATIENCE": 10,
            }
        elif stage == "token":
            cfg.TRAIN.STAGE = "token"
            cfg.TRAIN.NUM_WORKERS = 0
            cfg.TRAIN.BATCH_SIZE = 1
            cfg.TRAIN.END_EPOCH = 1
            cfg.TRAIN.MAX_STEPS = -1
            cfg.TRAIN.MAX_SAMPLES = -1
            cfg.EVAL.BATCH_SIZE = 1
            cfg.EVAL.NUM_WORKERS = 0
            cfg.TEST.BATCH_SIZE = 1
            cfg.TEST.NUM_WORKERS = 0
            cfg.LOGGER.VAL_EVERY_STEPS = 1
        else:
            is_stage2 = stage == "stage2"
            cfg.TRAIN.STAGE = "lm_pretrain" if is_stage2 else "lm_instruct"
            cfg.TRAIN.NUM_WORKERS = 0
            cfg.TRAIN.BATCH_SIZE = 4
            cfg.TRAIN.END_EPOCH = 999999
            cfg.TRAIN.MAX_STEPS = 200000 if is_stage2 else 100000
            cfg.EVAL.BATCH_SIZE = 16
            cfg.DATASET.TASK_PATH = (
                "prepare/instructions/template_pretrain.json"
                if is_stage2 else
                "prepare/instructions/template_instructions.json"
            )
            cfg.CHECKPOINT_SELECTION = {
                "ENABLED": True,
                "SPLIT": "val",
                "DATASET": "t2m",
                "BATCH_SIZE": 32,
                "NUM_WORKERS": 0,
                "SEED": args.seed,
                "FID_TOLERANCE": 1e-4,
                "PRIMARY": "FID",
                "TIE_BREAK": ["Matching", "R@3"],
                "EXPECTED_SAMPLES": VALIDATION_EVALUATOR_SAMPLES,
                "EARLY_STOPPING": {
                    "ENABLED": True,
                    "PATIENCE": 5 if is_stage2 else 6,
                    "MIN_VALIDATIONS": 8,
                },
            }
            cfg.LOGGER.VAL_EVERY_STEPS = 10 if is_stage2 else 5
            cfg.LOGGER.VAL_EVERY_N_STEPS = 10000 if is_stage2 else 5000
        output = config_dir / f"{stage}.yaml"
        OmegaConf.save(cfg, output)
        configs[stage] = output

    forbidden = (
        "MotionGPT-base", "motiongpt_s3_h3d", "VQTransformer",
        "net_best_fid.pth", "output_motionx322",
    )
    for path in configs.values():
        text = path.read_text(encoding="utf-8")
        found = [item for item in forbidden if item in text]
        if found:
            parser.error(f"forbidden generator dependency in {path}: {found}")

    tracked_sources = [
        mgpt_root / "mGPT/archs/mgpt_vq.py",
        mgpt_root / "mGPT/archs/mgpt_lm.py",
        mgpt_root / "mGPT/models/mgpt.py",
        mgpt_root / "mGPT/models/base.py",
        mgpt_root / "mGPT/data/HumanML3D.py",
        mgpt_root / "mGPT/data/humanml/dataset_m_vq.py",
        mgpt_root / "mGPT/data/humanml/dataset_t2m.py",
        mgpt_root / "mGPT/data/humanml/dataset_t2m_cb.py",
        mgpt_root / "mGPT/data/humanml/dataset_t2m_token.py",
        mgpt_root / "mGPT/callback.py",
        mgpt_root / "train.py",
        mgpt_root / "scripts/get_motion_code.py",
    ]
    census = data_census(dataset_root)
    manifest = {
        "schema_version": "motiongpt-humanml3d-official-retrain/v1",
        "run_tag": tag,
        "seed": args.seed,
        "method": {
            "repository": git_value(mgpt_root, "config", "--get", "remote.origin.url"),
            "git_head": git_value(mgpt_root, "rev-parse", "HEAD"),
            "language_model_initialization": "MotionGPT official deps/flan-t5-base",
            "generator_initialization": "fresh random VQ-VAE; no released generator checkpoint",
            "uses_t2mgpt_generator_checkpoint": False,
            "uses_released_motiongpt_generator_checkpoint": False,
            "allowed_non_generator_assets": [
                "HumanML3D mean/std normalization arrays",
                "official text-motion evaluator weights used only for validation selection",
                "official GloVe word vectors",
            ],
            "forbidden_generator_baselines": list(forbidden),
        },
        "hardware": {"physical_gpu_ids": [0, 1, 2, 3], "num_nodes": 1},
        "batching": {
            "stage1_per_gpu": 64, "stage1_global": 256,
            "stage2_per_gpu": 4, "stage2_global": 16,
            "stage3_per_gpu": 4, "stage3_global": 16,
        },
        "dataloading": {
            "num_workers": 0,
            "reason": (
                "HumanML3D in-memory datasets deadlock in forked workers "
                "after CUDA/DDP initialization; in-process loading preserves "
                "samples and optimization semantics"
            ),
        },
        "early_stopping": {
            "stage1": "min total/val, min_delta=1e-4, patience=10 validations, interval=10000 steps",
            "stage2": "min validation FID, patience=5, minimum 8 validations, interval=10000 steps",
            "stage3": "min validation FID, patience=6, minimum 8 validations, interval=5000 steps",
            "tie_break": "within FID 1e-4: min Matching, then max R@3",
            "test_data_used_for_selection": False,
        },
        "dataset": {
            "root": str(dataset_root),
            "token_code_path": token_code_path,
            "census": census,
        },
        "names": names,
        "configs": {
            stage: {"path": str(path), "sha256": file_sha256(path)}
            for stage, path in configs.items()
        },
        "source_files": {
            str(path.relative_to(repo_root)): file_sha256(path)
            for path in tracked_sources
        },
        "dependency_chain": {
            "stage1": "fresh random MotionGPT VQ-VAE initialization",
            "token": "selected fresh stage1 validation checkpoint only",
            "stage2": "fresh tokens + selected fresh stage1 + official flan-t5-base",
            "stage3": "selected fresh stage2 + selected fresh stage1 + fresh tokens",
        },
    }
    manifest_path = output_root / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
