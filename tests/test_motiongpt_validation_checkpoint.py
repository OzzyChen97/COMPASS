import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf


MOTIONGPT_ROOT = Path(__file__).parents[1] / "MotionGPT"
sys.path.insert(0, str(MOTIONGPT_ROOT))

from mGPT.callback import (  # noqa: E402
    ValidationMetricCheckpoint,
    getCheckpointCallback,
    is_better_validation_candidate,
)


def test_validation_best_rule_uses_fid_tolerance_then_matching_then_r3():
    best = {"FID": 1.0, "Matching": 3.0, "R@3": 0.7}
    assert is_better_validation_candidate(
        {"FID": 0.9998, "Matching": 99.0, "R@3": 0.0}, best
    )
    assert is_better_validation_candidate(
        {"FID": 1.00005, "Matching": 2.9, "R@3": 0.0}, best
    )
    assert is_better_validation_candidate(
        {"FID": 1.00005, "Matching": 3.0, "R@3": 0.8}, best
    )
    assert not is_better_validation_candidate(
        {"FID": 1.0002, "Matching": 0.0, "R@3": 1.0}, best
    )
    assert not is_better_validation_candidate(
        {"FID": 1.00005, "Matching": 3.1, "R@3": 1.0}, best
    )


def _cfg(tmp_path, selection_enabled, metrics):
    return OmegaConf.create({
        "SEED_VALUE": 123,
        "FOLDER_EXP": str(tmp_path),
        "TRAIN": {"STAGE": "lm_pretrain"},
        "EVAL": {"BATCH_SIZE": 32, "NUM_WORKERS": 0},
        "LOGGER": {"VAL_EVERY_STEPS": 10, "VAL_EVERY_N_STEPS": 10000},
        "METRIC": {"TYPE": metrics},
        "CHECKPOINT_SELECTION": {
            "ENABLED": selection_enabled,
            "SPLIT": "val",
            "DATASET": "motionx322_75",
            "FID_TOLERANCE": 1e-4,
            "EARLY_STOPPING": {
                "ENABLED": True,
                "PATIENCE": 3,
                "MIN_VALIDATIONS": 4,
            },
        },
    })


def test_motionx_selector_is_validation_callback_not_train_step_checkpoint(tmp_path):
    callbacks = getCheckpointCallback(_cfg(tmp_path, True, []), logger=None)
    selectors = [item for item in callbacks if isinstance(item, ValidationMetricCheckpoint)]
    assert len(selectors) == 1
    assert selectors[0].split == "val"
    assert selectors[0].fid_tolerance == pytest.approx(1e-4)
    assert selectors[0].early_stopping_enabled is True
    assert selectors[0].early_stopping_patience == 3
    assert selectors[0].early_stopping_min_validations == 4


def test_validation_early_stop_counts_only_consecutive_non_improvements(tmp_path):
    callback = ValidationMetricCheckpoint(_cfg(tmp_path, True, []))
    assert callback.update_early_stopping(True) is False
    assert callback.update_early_stopping(False) is False
    assert callback.update_early_stopping(False) is False
    # At least four validations and three consecutive misses are required.
    assert callback.update_early_stopping(False) is True
    # Any later validation improvement resets patience.
    assert callback.update_early_stopping(True) is False
    assert callback.bad_validation_checks == 0
    state = callback.state_dict()

    restored = ValidationMetricCheckpoint(_cfg(tmp_path, True, []))
    restored.load_state_dict(state)
    assert restored.validation_checks == 5
    assert restored.bad_validation_checks == 0
    assert restored.best is None


def test_validation_early_stop_rejects_invalid_patience(tmp_path):
    cfg = _cfg(tmp_path, True, [])
    cfg.CHECKPOINT_SELECTION.EARLY_STOPPING.PATIENCE = 0
    with pytest.raises(ValueError, match="PATIENCE"):
        ValidationMetricCheckpoint(cfg)


def test_validation_resume_reconciles_history_newer_than_checkpoint(tmp_path):
    callback = ValidationMetricCheckpoint(_cfg(tmp_path, True, []))
    callback.load_state_dict({
        "best": {
            "step": 5000, "epoch": 0, "split": "val", "FID": 2.0,
            "Matching": 8.0, "R@3": 0.4, "samples": 2552,
        },
        "validation_checks": 4,
        "bad_validation_checks": 3,
    })
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    rows = [
        {"selected": True},
        {"selected": False},
        {"selected": False},
        {"selected": False},
        {"selected": False},
    ]
    (checkpoint_dir / "validation_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (checkpoint_dir / "best-validation.json").write_text(json.dumps({
        "step": 5000, "epoch": 0, "split": "val", "FID": 2.0,
        "Matching": 8.0, "R@3": 0.4, "samples": 2552,
    }), encoding="utf-8")

    callback.on_fit_start(None, None)

    assert callback.validation_checks == 5
    assert callback.bad_validation_checks == 4


def test_validation_resume_detects_replayed_step(tmp_path):
    callback = ValidationMetricCheckpoint(_cfg(tmp_path, True, []))
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "validation_metrics.jsonl").write_text(
        json.dumps({"step": 25000, "selected": False}) + "\n",
        encoding="utf-8",
    )

    assert callback.last_recorded_validation_step() == 25000


def test_validation_selector_records_expected_sample_contract(tmp_path):
    cfg = _cfg(tmp_path, True, [])
    cfg.CHECKPOINT_SELECTION.EXPECTED_SAMPLES = 1504
    callback = ValidationMetricCheckpoint(cfg)
    assert callback.dataset == "motionx322_75"
    assert callback.expected_samples == 1504


def test_metric_checkpoints_do_not_inherit_every_n_train_steps(tmp_path):
    callbacks = getCheckpointCallback(
        _cfg(tmp_path, False, ["TM2TMetrics"]), logger=None
    )
    monitored = [item for item in callbacks if getattr(item, "monitor", None)]
    assert {item.monitor for item in monitored} >= {
        "Metrics/FID", "Metrics/R_precision_top_3"
    }
    assert all(item._every_n_train_steps == 0 for item in monitored)


def test_validation_selector_does_not_log_from_on_validation_end():
    source = (MOTIONGPT_ROOT / "mGPT/callback.py").read_text(encoding="utf-8")
    selector_source = source.split("class ValidationMetricCheckpoint", 1)[1].split(
        "class PromptPairRobustnessCheckpoint", 1
    )[0]
    assert "pl_module.log(" not in selector_source
    assert "trainer.callback_metrics.update" in selector_source
