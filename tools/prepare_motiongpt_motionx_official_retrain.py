#!/usr/bin/env python3
"""Prepare a fresh, official three-stage MotionGPT Motion-X training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    mgpt_root = repo_root / "MotionGPT"
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        parser.error(f"fresh run output already exists and is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "configs"
    config_dir.mkdir()

    tag = args.run_tag.replace("/", "_")
    names = {
        "stage1": f"VQVAE_MOTIONX322_{tag}",
        "token": f"TOKENIZE_MOTIONX322_{tag}",
        "stage2": f"Pretrain_MOTIONX322_{tag}",
        "stage3": f"Instruct_MOTIONX322_{tag}",
    }
    token_code_path = names["stage1"]
    base_paths = {
        "stage1": mgpt_root / "configs/config_motionx322_stage1.yaml",
        "token": mgpt_root / "configs/config_motionx322_token.yaml",
        "stage2": mgpt_root / "configs/config_motionx322_stage2.yaml",
        "stage3": mgpt_root / "configs/config_motionx322_stage3.yaml",
    }
    configs = {}
    for stage, base_path in base_paths.items():
        cfg = fresh_config(base_path)
        cfg.NAME = names[stage]
        cfg.SEED_VALUE = args.seed
        cfg.DEVICE = [0, 1, 2, 3] if stage != "token" else [0]
        cfg.DEBUG = False
        cfg.TRAIN.RESUME = ""
        cfg.TRAIN.PRETRAINED = ""
        cfg.TRAIN.PRETRAINED_VAE = ""
        if stage in {"token", "stage2", "stage3"}:
            cfg.DATASET.CODE_PATH = token_code_path
        output = config_dir / f"{stage}.yaml"
        OmegaConf.save(cfg, output)
        configs[stage] = output

    forbidden = ("T2M-GPT", "output_motionx322", "VQTransformer", "net_best_fid.pth")
    for path in configs.values():
        text = path.read_text(encoding="utf-8")
        found = [item for item in forbidden if item in text]
        if found:
            parser.error(f"forbidden baseline dependency in {path}: {found}")

    dataset_root = repo_root / "dataset/Motion-X-322-MaxToken75"
    split_counts = {}
    for split in ("train", "val", "test"):
        split_path = dataset_root / f"{split}.txt"
        split_counts[split] = sum(
            1 for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    tracked_sources = [
        mgpt_root / "mGPT/archs/mgpt_vq.py",
        mgpt_root / "mGPT/archs/mgpt_lm.py",
        mgpt_root / "mGPT/models/mgpt.py",
        mgpt_root / "mGPT/models/base.py",
        mgpt_root / "mGPT/data/MotionX322.py",
        mgpt_root / "mGPT/callback.py",
        mgpt_root / "train.py",
    ]
    manifest = {
        "schema_version": "motiongpt-motionx322-official-retrain/v1",
        "run_tag": tag,
        "seed": args.seed,
        "method": {
            "repository": git_value(mgpt_root, "config", "--get", "remote.origin.url"),
            "git_head": git_value(mgpt_root, "rev-parse", "HEAD"),
            "stages": [
                "official VQ-VAE motion tokenizer training",
                "official motion token extraction",
                "official MotionGPT language-model pretraining",
                "official MotionGPT instruction tuning",
            ],
            "language_model_initialization": "MotionGPT official deps/flan-t5-base",
            "forbidden_generator_baselines": list(forbidden),
            "uses_t2mgpt_generator_checkpoint": False,
        },
        "early_stopping": {
            "stage1": "min total/val, min_delta=1e-4, patience=10 validations",
            "stage2": "min validation FID, patience=5, minimum 8 validations",
            "stage3": "min validation FID, patience=6, minimum 8 validations",
            "tie_break": "within FID 1e-4: min Matching, then max R@3",
            "test_data_used_for_selection": False,
        },
        "dataset": {
            "root": str(dataset_root),
            "split_counts": split_counts,
            "token_code_path": token_code_path,
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
            "token": "selected stage1 validation checkpoint",
            "stage2": "selected stage1 validation checkpoint + fresh official MotionGPT LM stage",
            "stage3": "selected stage2 validation checkpoint + selected stage1 validation checkpoint",
            "evaluation": "selected stage3 validation checkpoint",
        },
    }
    manifest_path = output_root / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
