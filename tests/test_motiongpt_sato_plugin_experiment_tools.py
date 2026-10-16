import csv
import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
METRICS = ("FID", "FID_P", "FID_D", "Matching", "R@1", "R@2", "R@3", "Diversity")


def _module(name):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selector_fixture(tmp_path):
    baseline = tmp_path / "baseline.ckpt"
    stage1 = tmp_path / "stage1.ckpt"
    pair_file = tmp_path / "pairs.jsonl"
    baseline.write_bytes(b"baseline")
    stage1.write_bytes(b"stage1")
    pair_file.write_text('{"relation":"preserve"}\n')

    experiment = tmp_path / "SATO_PP_MOTIONX322_PLUGIN_SEED123"
    checkpoint_dir = experiment / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "best-validation.ckpt"
    checkpoint.write_bytes(b"plugin")
    rows = []
    module = _module("select_motiongpt_sato_plugin_best")
    best = None
    bad = 0
    for index, fid in enumerate(
        [8.0, 4.0, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2], start=1
    ):
        row = {
            "FID": fid,
            "Matching": 8.0 - index / 100,
            "R@3": 0.4 + index / 1000,
            "epoch": index,
            "samples": 2552,
            "split": "val",
            "step": index * 5000,
        }
        improved = module.is_better(row, best, 1e-4)
        bad = 0 if improved else bad + 1
        row.update({
            "selected": improved,
            "validation_checks": index,
            "bad_validation_checks": bad,
            "early_stop": index >= 8 and bad >= 10,
        })
        if improved:
            best = dict(row)
        rows.append(row)
    (checkpoint_dir / "validation_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (checkpoint_dir / "best-validation.json").write_text(json.dumps({
        **{key: best[key] for key in ("FID", "Matching", "R@3", "epoch", "samples", "split", "step")},
        "checkpoint": str(checkpoint),
    }))
    resolved = experiment / "config_fixture_train.yaml"
    resolved.write_text(
        "NAME: SATO_PP_MOTIONX322_PLUGIN_SEED123\n"
        "DEVICE: [0, 1, 2, 3]\n"
        "TRAIN:\n"
        "  STAGE: lm_sato_pp\n"
        f"  PRETRAINED: {baseline}\n"
        f"  PRETRAINED_VAE: {stage1}\n"
        "  STRICT_LOAD: false\n"
        "  MAX_STEPS: 50000\n"
        "LOGGER:\n"
        "  VAL_EVERY_N_STEPS: 5000\n"
        "CHECKPOINT_SELECTION:\n"
        "  SPLIT: val\n"
        "  EARLY_STOPPING:\n"
        "    PATIENCE: 10\n"
        "    MIN_VALIDATIONS: 8\n"
        "SATO:\n"
        "  ATTRIBUTION_CREATE_GRAPH: true\n"
        "  ATTRIBUTION_K: 16\n"
        "  DISTRIBUTED_VAR_COV: true\n"
        "  DISTRIBUTED_RELATION_LOSSES: true\n"
        f"  PAIR_DATA_PATHS: [{pair_file}]\n"
        "model:\n"
        "  target: mGPT.models.mgpt_sato_pp.MotionGPTSATOPP\n"
    )
    frozen = tmp_path / "frozen.yaml"
    frozen.write_text("frozen: true\n")
    source = tmp_path / "source_manifest.json"
    source.write_text(json.dumps({
        "repo_root": str(tmp_path),
        "method": {
            "uses_motiongpt_baseline_checkpoint": True,
            "uses_t2mgpt_generator_checkpoint": False,
        },
        "training": {
            "experiment_dir": str(experiment),
            "experiment_name": experiment.name,
            "attribution_k": 16,
            "distributed_var_cov": True,
            "distributed_relation_losses": True,
            "evaluation_checkpoint_policy": "validation_best",
        },
        "source_files": {},
        "config": {"path": str(frozen), "sha256": _sha(frozen)},
        "baseline_selection": {
            "path": str(tmp_path / "baseline-selection.json"),
            "checkpoint": str(baseline),
            "checkpoint_sha256": _sha(baseline),
        },
        "stage1_selection": {
            "path": str(tmp_path / "stage1-selection.json"),
            "checkpoint": str(stage1),
            "checkpoint_sha256": _sha(stage1),
        },
        "pairs": {
            "path": str(pair_file),
            "sha256": _sha(pair_file),
            "rows": 210140,
            "counts": {"alter": 1201, "preserve": 208939},
        },
    }))
    return module, experiment, source, frozen


def test_sato_selector_replays_terminal_validation_and_provenance(tmp_path):
    module, experiment, source, _ = _selector_fixture(tmp_path)
    payload = module.select_sato_plugin_best(experiment, source)
    assert payload["early_stopped"] is False
    assert payload["validation_checks"] == 10
    assert payload["step"] == 10000
    assert payload["model_target"].endswith("MotionGPTSATOPP")
    assert payload["attribution_create_graph"] is True
    assert payload["uses_motiongpt_baseline_checkpoint"] is True
    assert payload["checkpoint_sha256"] != payload["baseline_checkpoint_sha256"]


def test_sato_selector_rejects_changed_pair_file(tmp_path):
    module, experiment, source, _ = _selector_fixture(tmp_path)
    manifest = json.loads(source.read_text())
    Path(manifest["pairs"]["path"]).write_text("changed\n")
    with pytest.raises(RuntimeError, match="pair file hash mismatch"):
        module.select_sato_plugin_best(experiment, source)


def test_sato_terminal_selector_accepts_100k_last_checkpoint(tmp_path):
    module = _module("select_motiongpt_sato_plugin_last")
    baseline = tmp_path / "baseline.ckpt"
    stage1 = tmp_path / "stage1.ckpt"
    resume = tmp_path / "resume.ckpt"
    pairs = tmp_path / "pairs.jsonl"
    predecessor_manifest = tmp_path / "predecessor.json"
    for path, payload in (
        (baseline, b"baseline"), (stage1, b"stage1"), (resume, b"resume")
    ):
        path.write_bytes(payload)
    pairs.write_text('{"relation":"preserve"}\n')
    predecessor_manifest.write_text("{}\n")

    experiment = tmp_path / "SATO_PP_TERMINAL_100K"
    checkpoint_dir = experiment / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "last.ckpt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({
            "global_step": 100000,
            "epoch": 3,
            "callbacks": {
                "ValidationMetricCheckpoint": {
                    "validation_checks": 19,
                    "bad_validation_checks": 18,
                }
            },
        }))

    rows = []
    best = None
    bad = 0
    for index in range(1, 21):
        row = {
            "FID": 2.0 if index == 1 else 3.0 + index / 100,
            "Matching": 8.0,
            "R@3": 0.4,
            "epoch": index // 10,
            "samples": 2552,
            "split": "val",
            "step": index * 5000,
        }
        improved = module.is_better(row, best, 1e-4)
        bad = 0 if improved else bad + 1
        row.update({
            "selected": improved,
            "validation_checks": index,
            "bad_validation_checks": bad,
            "early_stop": False,
        })
        if improved:
            best = dict(row)
        rows.append(row)
    (checkpoint_dir / "validation_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    frozen = tmp_path / "frozen.yaml"
    frozen.write_text("frozen: true\n")
    resolved = experiment / "config_terminal_train.yaml"
    resolved.write_text(
        f"NAME: {experiment.name}\n"
        "DEVICE: [0, 1, 2, 3]\n"
        "TRAIN:\n"
        "  STAGE: lm_sato_pp\n"
        "  MAX_STEPS: 100000\n"
        f"  RESUME: {tmp_path / 'predecessor_exp'}\n"
        f"  PRETRAINED: {resume}\n"
        f"  PRETRAINED_VAE: {stage1}\n"
        "CHECKPOINT_SELECTION:\n"
        "  EARLY_STOPPING:\n"
        "    ENABLED: false\n"
        "SATO:\n"
        "  ATTRIBUTION_CREATE_GRAPH: true\n"
        "  ATTRIBUTION_K: 16\n"
        "  DISTRIBUTED_VAR_COV: true\n"
        "  DISTRIBUTED_RELATION_LOSSES: true\n"
        f"  PAIR_DATA_PATHS: [{pairs}]\n"
        "model:\n"
        "  target: mGPT.models.mgpt_sato_pp.MotionGPTSATOPP\n"
    )
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "repo_root": str(tmp_path),
        "method": {
            "uses_motiongpt_baseline_checkpoint": True,
            "uses_t2mgpt_generator_checkpoint": False,
        },
        "training": {
            "experiment_dir": str(experiment),
            "experiment_name": experiment.name,
            "max_steps": 100000,
            "early_stopping_enabled": False,
            "evaluation_checkpoint_policy": "last",
            "distributed_var_cov": True,
            "distributed_relation_losses": True,
        },
        "source_files": {},
        "config": {"path": str(frozen), "sha256": _sha(frozen)},
        "baseline_selection": {
            "path": str(tmp_path / "baseline-selection.json"),
            "checkpoint": str(baseline),
            "checkpoint_sha256": _sha(baseline),
        },
        "stage1_selection": {
            "path": str(tmp_path / "stage1-selection.json"),
            "checkpoint": str(stage1),
            "checkpoint_sha256": _sha(stage1),
        },
        "pairs": {
            "path": str(pairs), "sha256": _sha(pairs), "rows": 210140,
            "counts": {"alter": 1201, "preserve": 208939},
        },
        "continuation": {
            "predecessor_experiment_dir": str(tmp_path / "predecessor_exp"),
            "predecessor_source_manifest": str(predecessor_manifest),
            "predecessor_source_manifest_sha256": _sha(predecessor_manifest),
            "resume_checkpoint": str(resume),
            "resume_checkpoint_sha256": _sha(resume),
            "validation_history_rows": 10,
        },
    }))

    payload = module.select_sato_plugin_last(experiment, source)

    assert payload["selection_policy"] == "last"
    assert payload["step"] == 100000
    assert payload["early_stopping_enabled"] is False
    assert payload["checkpoint_metadata"]["global_step"] == 100000


def test_sato_eval_preparation_copies_baseline_protocol(tmp_path):
    module, experiment, source_path, _ = _selector_fixture(tmp_path)
    selection = module.select_sato_plugin_best(experiment, source_path)
    selection_path = experiment / "checkpoints/selected-validation.json"
    selection_path.write_text(json.dumps(selection))
    source = json.loads(source_path.read_text())
    base_config = tmp_path / "plugin.yaml"
    base_config.write_text(
        "NAME: plugin\nDEVICE: [0, 1, 2, 3]\nDEBUG: false\nSEED_VALUE: 123\n"
        "TRAIN:\n  STAGE: lm_sato_pp\n  RESUME: ''\n  PRETRAINED: x\n"
        "  PRETRAINED_VAE: y\n  STRICT_LOAD: false\n"
        "TEST:\n  CHECKPOINTS: ''\n  SPLIT: test\n  BATCH_SIZE: 16\n"
        "DATASET:\n  CODE_PATH: tokens\n"
        "CHECKPOINT_SELECTION:\n  ENABLED: true\n"
        "SATO:\n  DISTRIBUTED_VAR_COV: true\n"
        "  DISTRIBUTED_RELATION_LOSSES: true\n"
        "model:\n  target: mGPT.models.mgpt_sato_pp.MotionGPTSATOPP\n"
    )
    source["config"] = {"path": str(base_config), "sha256": _sha(base_config)}
    source_path.write_text(json.dumps(source))
    selection["source_manifest_sha256"] = _sha(source_path)
    selection_path.write_text(json.dumps(selection))
    protocol = {
        "dataset": "motionx322",
        "samples_per_seed": 7733,
        "seeds": list(range(123, 143)),
        "fingerprint": "locked-protocol",
    }
    baseline_manifest = tmp_path / "baseline_manifest.json"
    shared_implementation = {
        str(ROOT / "MotionGPT/eval_loader_robustness.py"): _sha(
            ROOT / "MotionGPT/eval_loader_robustness.py"
        ),
        str(ROOT / "MotionGPT/scripts/run_motionx322_table3_eval.sh"): _sha(
            ROOT / "MotionGPT/scripts/run_motionx322_table3_eval.sh"
        ),
        str(ROOT / "tools/aggregate_motiongpt_loader_eval.py"): _sha(
            ROOT / "tools/aggregate_motiongpt_loader_eval.py"
        ),
    }
    baseline_manifest.write_text(json.dumps({
        "protocol": protocol,
        "model": {"checkpoint_sha256": selection["baseline_checkpoint_sha256"]},
        "implementation": shared_implementation,
    }))
    assets = tmp_path / "assets.yaml"
    assets.write_text("CONFIG_FOLDER: configs\nFOLDER: experiments\n")
    output = tmp_path / "repeat20"
    command = [
        sys.executable,
        str(ROOT / "tools/prepare_motiongpt_motionx_sato_plugin_eval.py"),
        "--checkpoint", selection["checkpoint"],
        "--plugin-selection", str(selection_path),
        "--source-manifest", str(source_path),
        "--baseline-manifest", str(baseline_manifest),
        "--base-config", str(base_config),
        "--assets-config", str(assets),
        "--output-dir", str(output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["protocol"] == protocol
    assert manifest["model"]["target"].endswith("MotionGPTSATOPP")
    assert manifest["model"]["baseline_checkpoint_sha256"] == selection["baseline_checkpoint_sha256"]
    assert manifest["model"]["distributed_relation_losses"] is True
    assert len(list((output / "configs").glob("*.yaml"))) == 20


def _write_repeat20(run_dir, checkpoint, evaluator, protocol, offset):
    (run_dir / "combined").mkdir(parents=True)
    fields = ["seed", *METRICS, "matched_count", "total_count", "sample_index_sha256"]
    csv_rows = []
    for seed in range(123, 143):
        values = {metric: float(index + seed / 1000 + offset) for index, metric in enumerate(METRICS)}
        sample_hash = f"sample-{seed}"
        row = {
            "dataset": "motionx322",
            "seed": seed,
            "matched_count": 7733,
            "total_count": 7733,
            "unmatched_count": 0,
            "runtime_fallback_count": 0,
            "sample_index_sha256": sample_hash,
            "protocol_fingerprint": protocol,
            "metrics": values,
            "artifacts": {
                "checkpoint": {"sha256": checkpoint},
                "evaluator_checkpoint": {"sha256": evaluator},
            },
            "checkpoint_load_audit": {
                "missing_key_count": 0,
                "unexpected_key_count": 0,
            },
        }
        (run_dir / f"seed{seed}.json").write_text(json.dumps(row))
        csv_rows.append({
            "seed": seed,
            **values,
            "matched_count": 7733,
            "total_count": 7733,
            "sample_index_sha256": sample_hash,
        })
    with (run_dir / "combined/all_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {
        "dataset": "motionx322",
        "n": 20,
        "samples_per_seed": 7733,
        "identities": {"checkpoint": checkpoint, "evaluator": evaluator, "protocol": protocol},
        "summary": {
            metric: {
                "mean": sum(float(row[metric]) for row in csv_rows) / 20,
                "std": 0.1,
            }
            for metric in METRICS
        },
    }
    path = run_dir / "combined/summary.json"
    path.write_text(json.dumps(summary))
    return path


def test_comparison_requires_paired_identity_and_reports_delta(tmp_path):
    baseline_run = tmp_path / "baseline"
    plugin_run = tmp_path / "plugin"
    baseline_checkpoint = "baseline-sha"
    plugin_checkpoint = "plugin-sha"
    evaluator = "evaluator-sha"
    protocol_identity = "protocol-sha"
    baseline_summary = _write_repeat20(
        baseline_run, baseline_checkpoint, evaluator, protocol_identity, 0.0
    )
    plugin_summary = _write_repeat20(
        plugin_run, plugin_checkpoint, evaluator, protocol_identity, 0.25
    )
    protocol = {"seeds": list(range(123, 143)), "samples_per_seed": 7733}
    baseline_manifest = tmp_path / "baseline_manifest.json"
    shared_implementation = {
        str(ROOT / "MotionGPT/eval_loader_robustness.py"): _sha(
            ROOT / "MotionGPT/eval_loader_robustness.py"
        ),
        str(ROOT / "MotionGPT/scripts/run_motionx322_table3_eval.sh"): _sha(
            ROOT / "MotionGPT/scripts/run_motionx322_table3_eval.sh"
        ),
        str(ROOT / "tools/aggregate_motiongpt_loader_eval.py"): _sha(
            ROOT / "tools/aggregate_motiongpt_loader_eval.py"
        ),
    }
    baseline_manifest.write_text(json.dumps({
        "protocol": protocol,
        "model": {"checkpoint_sha256": baseline_checkpoint},
        "implementation": shared_implementation,
    }))
    selection = tmp_path / "selected.json"
    selection.write_text(json.dumps({
        "stage": "sato_plugin",
        "split": "val",
        "checkpoint_sha256": plugin_checkpoint,
    }))
    plugin_manifest = tmp_path / "plugin_manifest.json"
    plugin_manifest.write_text(json.dumps({
        "protocol": protocol,
        "implementation": shared_implementation,
        "model": {
            "target": "mGPT.models.mgpt_sato_pp.MotionGPTSATOPP",
            "checkpoint_sha256": plugin_checkpoint,
            "baseline_checkpoint_sha256": baseline_checkpoint,
            "uses_motiongpt_baseline_checkpoint": True,
            "uses_t2mgpt_generator_checkpoint": False,
            "attribution_create_graph": True,
            "distributed_var_cov": True,
            "distributed_relation_losses": True,
            "selection_metadata": str(selection),
            "selection_metadata_sha256": _sha(selection),
        },
    }))
    module = _module("compare_motiongpt_motionx_sato_plugin")
    result = module.compare(
        baseline_summary,
        plugin_summary,
        baseline_manifest,
        plugin_manifest,
        baseline_run,
        plugin_run,
    )
    assert result["paired"] is True
    assert len(result["paired_seeds"]) == 20
    assert result["metrics"]["FID"]["delta_sato_minus_baseline"] == pytest.approx(0.25)
    assert result["checkpoint_sha256"] == {
        "baseline": baseline_checkpoint,
        "sato_plugin": plugin_checkpoint,
    }
