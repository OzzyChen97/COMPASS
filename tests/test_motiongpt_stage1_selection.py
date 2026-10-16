import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "tools/select_motiongpt_stage1_best.py"
SPEC = importlib.util.spec_from_file_location("stage1_selector", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _write_run(tmp_path, *, baseline_free=True, best_name="best.ckpt"):
    experiment = tmp_path / "MotionGPT/experiments/mgpt/FRESH_STAGE1"
    checkpoints = experiment / "checkpoints"
    checkpoints.mkdir(parents=True)
    torch.save({"global_step": 20000, "epoch": 158}, checkpoints / best_name)
    callback_key = (
        "ModelCheckpoint{'monitor': 'total/val', 'mode': 'min', "
        "'every_n_train_steps': 0, 'every_n_epochs': 1}"
    )
    torch.save({
        "global_step": 30000,
        "epoch": 238,
        "callbacks": {
            callback_key: {
                "best_model_score": torch.tensor(0.1949),
                "best_model_path": f"experiments/mgpt/FRESH_STAGE1/checkpoints/{best_name}",
            }
        },
    }, checkpoints / "last.ckpt")
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps({
        "names": {"stage1": "FRESH_STAGE1"},
        "method": {"uses_t2mgpt_generator_checkpoint": not baseline_free},
    }))
    return experiment, manifest


def test_selects_only_validation_loss_callback_and_hashes_artifacts(tmp_path):
    experiment, manifest = _write_run(tmp_path)
    result = module.select_stage1_best(experiment, manifest)
    assert result["split"] == "val"
    assert result["monitor"] == "total/val"
    assert result["mode"] == "min"
    assert result["score"] == pytest.approx(0.1949)
    assert result["global_step"] == 20000
    assert result["epoch"] == 158
    assert result["checkpoint_sha256"] == module.sha256_file(
        experiment / "checkpoints/best.ckpt"
    )
    assert result["uses_t2mgpt_generator_checkpoint"] is False


def test_rejects_manifest_without_baseline_free_proof(tmp_path):
    experiment, manifest = _write_run(tmp_path, baseline_free=False)
    with pytest.raises(RuntimeError, match="baseline-free"):
        module.select_stage1_best(experiment, manifest)


def test_rejects_best_checkpoint_outside_fresh_experiment(tmp_path):
    experiment, manifest = _write_run(tmp_path, best_name="missing.ckpt")
    (experiment / "checkpoints/missing.ckpt").unlink()
    with pytest.raises(FileNotFoundError, match="fresh experiment"):
        module.select_stage1_best(experiment, manifest)
