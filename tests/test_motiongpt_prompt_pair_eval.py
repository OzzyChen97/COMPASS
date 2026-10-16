import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "MotionGPT" / "eval_prompt_pair_robustness.py"
SPEC = importlib.util.spec_from_file_location("motiongpt_prompt_pair_eval", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_resolve_motion_key_prefers_exact_and_matches_segment_caption():
    data = {
        "m1": {"text": [{"caption": "walk"}]},
        "A_m2": {"text": [{"caption": "turn left"}]},
        "B_m2": {"text": [{"caption": "turn right"}]},
    }
    assert module.resolve_motion_key("m1", "anything", data) == "m1"
    assert module.resolve_motion_key("m2", "turn left", data) == "A_m2"
    assert module.resolve_motion_key("missing", "walk", data) is None


def test_select_pair_records_tracks_missing_and_target():
    data = {"m1": {"text": []}, "m2": {"text": []}}
    pairs = [
        {"motion_id": "bad", "text_orig": "a", "text_pair": "b"},
        {"motion_id": "m1", "text_orig": "a", "text_pair": "b"},
        {"motion_id": "m2", "text_orig": "c", "text_pair": "d"},
    ]
    selected, missing = module.select_pair_records(pairs, data, 2)
    assert [row["motion_key"] for row in selected] == ["m1", "m2"]
    assert missing == 1


def test_smoke_metadata_cannot_masquerade_as_formal():
    assert module.smoke_metadata(100, 5, 5) == {
        "smoke": True,
        "matched_count": 5,
        "protocol_matched_count": 100,
    }
    with pytest.raises(module.PairEvaluationError, match="formal evaluation requires"):
        module.smoke_metadata(100, 99, None)


def test_paired_cosine():
    orig = np.array([[1.0, 0.0], [0.0, 1.0]])
    pair = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert module.paired_cosine(orig, pair) == pytest.approx(0.5)


def test_manifest_seed_and_config_are_locked(tmp_path):
    config = tmp_path / "seed123.yaml"
    config.write_text("x: 1", encoding="utf-8")
    manifest = {
        "protocol": {"seeds": [123]},
        "model": {"id": "baseline"},
        "commands": [{"seed": 123, "config": str(config), "command": []}],
    }
    protocol, model = module.validate_manifest_seed(manifest, 123, config)
    assert protocol["seeds"] == [123]
    assert model["id"] == "baseline"
    with pytest.raises(module.PairEvaluationError, match="not part"):
        module.validate_manifest_seed(manifest, 124, config)


def test_load_task_accepts_official_caption_template(tmp_path):
    import json
    from types import SimpleNamespace

    path = tmp_path / "instructions.json"
    task = {"input": ["<Caption_Placeholder>"], "output": ["<Motion_Placeholder>"]}
    path.write_text(json.dumps({"Text-to-Motion": {"caption": task}}), encoding="utf-8")
    cfg = SimpleNamespace(
        model=SimpleNamespace(params=SimpleNamespace(condition="text")),
        DATASET=SimpleNamespace(TASK_PATH=str(path)),
    )
    assert module.load_task(cfg, 2) == [task, task]


def test_normalized_pair_identity_detects_content_changes(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"motion_id": "m1", "text_orig": "walk", "text_pair": "stroll"}) + "\n",
        encoding="utf-8",
    )
    first = module.normalized_pair_identity(path)
    path.write_text(
        json.dumps({"motion_id": "m1", "text_orig": "walk", "text_pair": "run"}) + "\n",
        encoding="utf-8",
    )
    second = module.normalized_pair_identity(path)
    assert first["count"] == second["count"] == 1
    assert first["sha256"] != second["sha256"]
