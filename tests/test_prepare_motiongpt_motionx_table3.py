import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_table3_preparation_requires_audited_stage3_dependency_chain(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "best-validation.ckpt"
    checkpoint.write_bytes(b"fresh-stage3")
    best = {
        "FID": 1.0,
        "Matching": 2.0,
        "R@3": 0.5,
        "samples": 2552,
        "split": "val",
        "step": 40000,
        "checkpoint": str(checkpoint),
    }
    (checkpoint_dir / "best-validation.json").write_text(json.dumps(best))

    base = tmp_path / "stage3.yaml"
    base.write_text(
        "NAME: stage3\n"
        "DEVICE: [0]\n"
        "DEBUG: false\n"
        "SEED_VALUE: 123\n"
        "TRAIN:\n"
        "  RESUME: ''\n"
        "  PRETRAINED: ''\n"
        "  PRETRAINED_VAE: ''\n"
        "TEST:\n"
        "  CHECKPOINTS: ''\n"
        "  SPLIT: test\n"
        "  BATCH_SIZE: 32\n"
        "DATASET:\n"
        "  CODE_PATH: fresh_tokens\n"
        "CHECKPOINT_SELECTION:\n"
        "  ENABLED: true\n"
    )
    assets = tmp_path / "assets.yaml"
    assets.write_text("FOLDER: unused\n")
    output = tmp_path / "table3"
    command = [
        sys.executable,
        str(ROOT / "tools/prepare_motiongpt_motionx_table3.py"),
        "--checkpoint", str(checkpoint),
        "--base-config", str(base),
        "--assets-config", str(assets),
        "--output-dir", str(output),
    ]
    missing = subprocess.run(command, text=True, capture_output=True)
    assert missing.returncode != 0
    assert "selected-validation.json" in missing.stderr

    import hashlib
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    audited = {
        "stage": "stage3",
        "split": "val",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "stage1_checkpoint_sha256": "stage1-sha",
        "predecessor_checkpoint_sha256": "stage2-sha",
        "uses_t2mgpt_generator_checkpoint": False,
    }
    (checkpoint_dir / "selected-validation.json").write_text(json.dumps(audited))
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["protocol"]["seeds"] == list(range(123, 143))
    assert manifest["model"]["checkpoint_sha256"] == checkpoint_sha
    assert manifest["model"]["stage1_checkpoint_sha256"] == "stage1-sha"
    assert manifest["model"]["stage2_checkpoint_sha256"] == "stage2-sha"
    assert manifest["model"]["uses_t2mgpt_generator_checkpoint"] is False
    assert len(list((output / "configs").glob("*.yaml"))) == 20


def test_table3_preparation_accepts_literal_last_checkpoint(tmp_path):
    import hashlib
    import pickle
    import zipfile

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "last.ckpt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({
            "global_step": 50000,
            "epoch": 1,
        }))
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    audited = {
        "stage": "stage3",
        "split": "val",
        "checkpoint": str(checkpoint_dir / "best-validation.ckpt"),
        "checkpoint_sha256": "validation-best-sha",
        "stage1_checkpoint_sha256": "stage1-sha",
        "predecessor_checkpoint_sha256": "stage2-sha",
        "uses_t2mgpt_generator_checkpoint": False,
    }
    (checkpoint_dir / "selected-validation.json").write_text(json.dumps(audited))

    base = tmp_path / "stage3.yaml"
    base.write_text(
        "NAME: stage3\nDEVICE: [0]\nDEBUG: false\nSEED_VALUE: 123\n"
        "TRAIN:\n  RESUME: ''\n  PRETRAINED: ''\n  PRETRAINED_VAE: ''\n"
        "TEST:\n  CHECKPOINTS: ''\n  SPLIT: test\n  BATCH_SIZE: 32\n"
        "DATASET:\n  CODE_PATH: fresh_tokens\n"
        "CHECKPOINT_SELECTION:\n  ENABLED: true\n"
    )
    assets = tmp_path / "assets.yaml"
    assets.write_text("FOLDER: unused\n")
    output = tmp_path / "table3-last"
    command = [
        sys.executable,
        str(ROOT / "tools/prepare_motiongpt_motionx_table3.py"),
        "--checkpoint", str(checkpoint),
        "--selection-policy", "last",
        "--base-config", str(base),
        "--assets-config", str(assets),
        "--output-dir", str(output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["model"]["checkpoint_sha256"] == checkpoint_sha
    assert manifest["model"]["selection_policy"] == "last"
    assert manifest["model"]["selection_step"] == 50000
    assert manifest["model"]["selection_metrics"] is None
