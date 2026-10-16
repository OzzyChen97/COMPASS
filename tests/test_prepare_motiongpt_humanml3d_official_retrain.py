import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).parents[1]
SCRIPT = REPO_ROOT / "tools/prepare_motiongpt_humanml3d_official_retrain.py"


def _fake_h3d(root: Path):
    (root / "new_joint_vecs").mkdir(parents=True)
    (root / "texts").mkdir()
    ids = {"train": ["a", "b"], "val": ["c"], "test": ["d"]}
    lengths = {"a": 80, "b": 32, "c": 100, "d": 120}
    for split, names in ids.items():
        (root / f"{split}.txt").write_text("\n".join(names) + "\n")
    for name, frames in lengths.items():
        np.save(root / "new_joint_vecs" / f"{name}.npy", np.zeros((frames, 263)))
        (root / "texts" / f"{name}.txt").write_text(
            "a person moves#person/NOUN moves/VERB#0#0\n"
        )


def test_humanml3d_retrain_configs_are_four_gpu_validation_only_and_fresh(tmp_path):
    dataset_root = tmp_path / "HumanML3D"
    output = tmp_path / "run"
    _fake_h3d(dataset_root)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run-tag", "UNIT_SEED123",
            "--seed", "123",
            "--output-root", str(output),
            "--repo-root", str(REPO_ROOT),
            "--dataset-root", str(dataset_root),
        ],
        check=True,
    )
    manifest = json.loads((output / "source_manifest.json").read_text())
    assert manifest["method"]["uses_t2mgpt_generator_checkpoint"] is False
    assert manifest["method"]["uses_released_motiongpt_generator_checkpoint"] is False
    assert manifest["hardware"]["physical_gpu_ids"] == [0, 1, 2, 3]
    assert manifest["dataloading"]["num_workers"] == 0
    assert manifest["early_stopping"]["test_data_used_for_selection"] is False
    assert manifest["dataset"]["census"]["train"]["token_eligible_original_motions"] == 1

    configs = {
        stage: OmegaConf.load(output / "configs" / f"{stage}.yaml")
        for stage in ("stage1", "token", "stage2", "stage3")
    }
    assert list(configs["stage1"].DEVICE) == [0, 1, 2, 3]
    assert list(configs["stage2"].DEVICE) == [0, 1, 2, 3]
    assert list(configs["stage3"].DEVICE) == [0, 1, 2, 3]
    assert list(configs["token"].DEVICE) == [0]
    assert configs["stage1"].TRAIN.BATCH_SIZE == 64
    assert configs["stage2"].TRAIN.BATCH_SIZE == 4
    assert configs["stage3"].TRAIN.BATCH_SIZE == 4
    assert configs["stage1"].EARLY_STOPPING.MONITOR == "total/val"
    assert configs["stage1"].EARLY_STOPPING.PATIENCE == 10
    for stage, patience in (("stage2", 5), ("stage3", 6)):
        cfg = configs[stage]
        assert cfg.EVAL.SPLIT == "val"
        assert cfg.CHECKPOINT_SELECTION.DATASET == "t2m"
        assert cfg.CHECKPOINT_SELECTION.EXPECTED_SAMPLES == 1504
        assert cfg.CHECKPOINT_SELECTION.EARLY_STOPPING.PATIENCE == patience
        assert cfg.DATASET.CODE_PATH == configs["stage1"].NAME
    for cfg in configs.values():
        assert cfg.TRAIN.NUM_WORKERS == 0
        assert cfg.EVAL.NUM_WORKERS == 0
        assert cfg.TEST.NUM_WORKERS == 0
        assert not str(cfg.TRAIN.get("RESUME", "")).strip()
        assert not str(cfg.TRAIN.get("PRETRAINED", "")).strip()
        assert not str(cfg.TRAIN.get("PRETRAINED_VAE", "")).strip()
        assert not str(cfg.TEST.get("CHECKPOINTS", "")).strip()


def test_humanml3d_retrain_refuses_nonempty_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_text("preserve")
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--run-tag", "UNIT",
            "--output-root", str(output), "--repo-root", str(REPO_ROOT),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "non-empty" in result.stderr
    assert (output / "sentinel").read_text() == "preserve"
