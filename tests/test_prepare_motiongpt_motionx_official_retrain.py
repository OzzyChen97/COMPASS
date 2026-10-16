import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).parents[1]
SCRIPT = REPO_ROOT / "tools/prepare_motiongpt_motionx_official_retrain.py"


def test_fresh_retrain_configs_are_isolated_and_checkpoint_free(tmp_path):
    output = tmp_path / "fresh"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-tag",
            "UNIT_FRESH_123",
            "--output-root",
            str(output),
            "--repo-root",
            str(REPO_ROOT),
        ],
        check=True,
    )

    manifest = json.loads((output / "source_manifest.json").read_text())
    assert manifest["method"]["uses_t2mgpt_generator_checkpoint"] is False
    assert manifest["dataset"]["token_code_path"] == "VQVAE_MOTIONX322_UNIT_FRESH_123"
    assert manifest["dataset"]["split_counts"] == {
        "train": 41159,
        "val": 2552,
        "test": 7733,
    }
    assert "MotionGPT/mGPT/models/base.py" in manifest["source_files"]

    configs = {
        stage: OmegaConf.load(output / "configs" / f"{stage}.yaml")
        for stage in ("stage1", "token", "stage2", "stage3")
    }
    assert configs["stage1"].NAME == "VQVAE_MOTIONX322_UNIT_FRESH_123"
    assert configs["stage2"].DATASET.CODE_PATH == configs["stage1"].NAME
    assert configs["stage3"].DATASET.CODE_PATH == configs["stage1"].NAME
    for cfg in configs.values():
        assert list(cfg.DEVICE) in ([0], [0, 1, 2, 3])
        assert not str(cfg.TRAIN.get("RESUME", "")).strip()
        assert not str(cfg.TRAIN.get("PRETRAINED", "")).strip()
        assert not str(cfg.TRAIN.get("PRETRAINED_VAE", "")).strip()


def test_fresh_retrain_refuses_nonempty_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("preserve")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-tag",
            "UNIT_FRESH_123",
            "--output-root",
            str(output),
            "--repo-root",
            str(REPO_ROOT),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "non-empty" in result.stderr
    assert (output / "sentinel").read_text() == "preserve"
