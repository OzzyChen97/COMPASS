import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "MotionGPT" / "eval_sato_robustness.py"
SPEC = importlib.util.spec_from_file_location("motiongpt_eval_sato_robustness", MODULE_PATH)
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


def _pairs(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"motion_id": "m1", "text_orig": "walk", "text_pair": "stroll"}),
                json.dumps({"id": "m2", "original": "turn left", "perturbed": "rotate left"}),
            ]
        ),
        encoding="utf-8",
    )
    return path


def _protocol_args(tmp_path, *, phase="screening"):
    return Namespace(
        phase=phase,
        seeds=None,
        selection_split="val" if phase == "screening" else "none",
        evaluation_split="val" if phase == "screening" else "test",
        dataset="kit",
        evaluator="motiongpt-tm2t-official",
        pair_file=_pairs(tmp_path),
        matched_count=2,
        sampling_mode="categorical",
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )


def test_loads_jsonl_and_builds_stable_protocol(tmp_path):
    args = _protocol_args(tmp_path)
    protocol = adapter.build_protocol(args)
    assert protocol["pair_file"]["count"] == 2
    assert protocol["pair_file"]["unique_motion_count"] == 2
    assert protocol["seeds"] == [123, 124, 125]
    assert protocol["fingerprint"] == adapter.protocol_fingerprint(protocol)


def test_final_requires_exactly_twenty_seeds(tmp_path):
    args = _protocol_args(tmp_path, phase="final")
    args.seeds = "1,2,3"
    with pytest.raises(adapter.ProtocolError, match="exactly 20"):
        adapter.build_protocol(args)


def test_test_split_checkpoint_selection_is_forbidden(tmp_path):
    args = _protocol_args(tmp_path)
    args.evaluation_split = "test"
    with pytest.raises(adapter.ProtocolError, match="validation-only"):
        adapter.build_protocol(args)


def test_normalize_metrics_accepts_official_and_robustness_names():
    metrics = adapter.normalize_metrics(
        {
            "Metrics/TM2TMetrics/FID/mean": 0.25,
            "Metrics/TM2TMetrics/R_precision_top_1": [0.5],
            "FID_P": 0.3,
            "FID_D": 0.2,
            "RepCos": 0.9,
        }
    )
    assert metrics == {
        "fid": 0.25,
        "r_precision_top1": 0.5,
        "fid_p": 0.3,
        "fid_d": 0.2,
        "representation_cosine": 0.9,
    }


def test_sanitized_ld_path_drops_system_python_torch(monkeypatch, tmp_path):
    fake_python = tmp_path / "env" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("")
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        "/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/nvidia/lib",
    )
    value = adapter.sanitized_ld_library_path(fake_python)
    assert "python3.12/dist-packages/torch" not in value
    assert "/usr/local/nvidia/lib" in value


def test_ingest_checks_seed_matched_count_and_aggregates(tmp_path):
    protocol = adapter.build_protocol(_protocol_args(tmp_path))
    manifest = {
        "model": {"id": "baseline", "variant": "baseline", "checkpoint": "a.ckpt"},
        "protocol": protocol,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    metric_paths = []
    for index, seed in enumerate(protocol["seeds"]):
        path = tmp_path / f"metrics_{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "matched_count": 2,
                    "protocol_fingerprint": protocol["fingerprint"],
                    "FID": 1.0 + index,
                    "FID_P": 0.5 + index,
                }
            ),
            encoding="utf-8",
        )
        metric_paths.append(path)
    result = adapter.ingest(Namespace(manifest=manifest_path, metric_json=metric_paths))
    assert result["aggregate"]["fid"]["mean"] == 2.0
    assert result["aggregate"]["fid_p"]["n"] == 3


def test_ingest_rejects_smoke_result(tmp_path):
    protocol = adapter.build_protocol(_protocol_args(tmp_path))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"model": {"id": "b"}, "protocol": protocol}), encoding="utf-8"
    )
    metric_paths = []
    for seed in protocol["seeds"]:
        path = tmp_path / f"smoke_{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "smoke": True,
                    "seed": seed,
                    "matched_count": 2,
                    "protocol_fingerprint": protocol["fingerprint"],
                    "FID_P": 0.5,
                }
            ),
            encoding="utf-8",
        )
        metric_paths.append(path)
    with pytest.raises(adapter.ProtocolError, match="smoke result"):
        adapter.ingest(Namespace(manifest=manifest_path, metric_json=metric_paths))


def test_compare_rejects_protocol_mismatch(tmp_path):
    protocol_a = adapter.build_protocol(_protocol_args(tmp_path))
    args_b = _protocol_args(tmp_path)
    args_b.sampling_mode = "greedy"
    protocol_b = adapter.build_protocol(args_b)
    baseline = {
        "kind": "evaluation_result",
        "model": {"id": "b"},
        "protocol": protocol_a,
        "runs": [{}, {}, {}],
        "aggregate": {"fid": {"mean": 1.0}},
    }
    sato = {
        "kind": "evaluation_result",
        "model": {"id": "s"},
        "protocol": protocol_b,
        "runs": [{}, {}, {}],
        "aggregate": {"fid": {"mean": 0.8}},
    }
    baseline_path = tmp_path / "baseline.json"
    sato_path = tmp_path / "sato.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    sato_path.write_text(json.dumps(sato), encoding="utf-8")
    with pytest.raises(adapter.ProtocolError, match="protocol differ"):
        adapter.compare(Namespace(baseline=baseline_path, sato_pp=sato_path))


def test_select_only_accepts_validation_screening(tmp_path):
    protocol = adapter.build_protocol(_protocol_args(tmp_path))
    candidates = []
    for name, score in (("a", 0.4), ("b", 0.2)):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "model": {"id": name, "checkpoint": f"{name}.ckpt"},
                    "protocol": protocol,
                    "aggregate": {"fid": {"mean": score}},
                }
            ),
            encoding="utf-8",
        )
        candidates.append(path)
    selected = adapter.select_checkpoint(
        Namespace(results=candidates, metric="fid", direction="min")
    )
    assert selected["selected_checkpoint"] == "b.ckpt"
    assert selected["selection_split"] == "val"
