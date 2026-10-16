import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _module():
    path = ROOT / "tools/select_motiongpt_lm_best.py"
    spec = importlib.util.spec_from_file_location("motiongpt_lm_selection", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    stage1_checkpoint = tmp_path / "stage1.ckpt"
    stage1_checkpoint.write_bytes(b"stage1")
    stage1_selection = tmp_path / "stage1-selection.json"
    stage1_selection.write_text(json.dumps({
        "checkpoint": str(stage1_checkpoint),
        "checkpoint_sha256": _sha(stage1_checkpoint),
        "split": "val",
    }))

    experiment = tmp_path / "Pretrain_FRESH"
    checkpoints = experiment / "checkpoints"
    checkpoints.mkdir(parents=True)
    selected_checkpoint = checkpoints / "best-validation.ckpt"
    selected_checkpoint.write_bytes(b"stage2")

    rows = []
    fids = [8.0, 4.0, 4.5, 4.6, 4.7, 4.8, 4.9]
    best = None
    bad = 0
    module = _module()
    for index, fid in enumerate(fids, start=1):
        row = {
            "FID": fid,
            "Matching": 10.0 - index / 10,
            "R@3": index / 100,
            "epoch": index,
            "samples": 2552,
            "split": "val",
            "step": index * 10000,
        }
        improved = module.is_better(row, best, 1e-4)
        bad = 0 if improved else bad + 1
        row.update({
            "selected": improved,
            "validation_checks": index,
            "bad_validation_checks": bad,
            "early_stop": index >= 4 and bad >= 5,
        })
        if improved:
            best = dict(row)
        rows.append(row)
    (checkpoints / "validation_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (checkpoints / "best-validation.json").write_text(json.dumps({
        **{key: best[key] for key in ("FID", "Matching", "R@3", "epoch", "samples", "split", "step")},
        "checkpoint": str(selected_checkpoint),
    }))
    config = experiment / "config_2026_train.yaml"
    config.write_text(
        "TRAIN:\n"
        "  STAGE: lm_pretrain\n"
        f"  PRETRAINED_VAE: {stage1_checkpoint}\n"
        "  PRETRAINED: ''\n"
        "lm:\n"
        "  default:\n"
        "    params:\n"
        "      model_path: ./deps/flan-t5-base\n"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "method": {"uses_t2mgpt_generator_checkpoint": False},
        "names": {"stage2": experiment.name, "stage3": "Instruct_FRESH"},
    }))
    return module, experiment, manifest, stage1_selection


def _select(module, experiment, manifest, stage1_selection):
    return module.select_lm_best(
        experiment,
        manifest,
        "stage2",
        stage1_selection,
        None,
        2552,
        10000,
        200000,
        5,
        4,
        1e-4,
    )


def test_lm_selector_replays_terminal_validation_history(tmp_path):
    module, experiment, manifest, stage1_selection = _fixture(tmp_path)
    payload = _select(module, experiment, manifest, stage1_selection)
    assert payload["early_stopped"] is True
    assert payload["validation_checks"] == 7
    assert payload["step"] == 20000
    assert payload["uses_t2mgpt_generator_checkpoint"] is False
    assert payload["checkpoint_sha256"] == _sha(
        experiment / "checkpoints/best-validation.ckpt"
    )


def test_lm_selector_rejects_tampered_selected_flag(tmp_path):
    module, experiment, manifest, stage1_selection = _fixture(tmp_path)
    history = experiment / "checkpoints/validation_metrics.jsonl"
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    rows[2]["selected"] = True
    history.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="selected flag"):
        _select(module, experiment, manifest, stage1_selection)


def test_lm_selector_rejects_nonterminal_history(tmp_path):
    module, experiment, manifest, stage1_selection = _fixture(tmp_path)
    history = experiment / "checkpoints/validation_metrics.jsonl"
    rows = [json.loads(line) for line in history.read_text().splitlines()][:-1]
    history.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="not terminal"):
        _select(module, experiment, manifest, stage1_selection)
