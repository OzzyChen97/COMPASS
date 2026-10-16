import ast
import copy
import csv
import json
import re
import subprocess
import time
from pathlib import Path

import pytest

from tools import summarize_var_cov_ablation as summary


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_tiny_eval_root(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "NUM_SEEDS", 2)
    monkeypatch.setattr(summary, "EXPECTED_SEEDS", {123, 124})
    monkeypatch.setitem(summary.EXPECTED_PAIRS, "humanml3d", 2)
    monkeypatch.setitem(summary.EXPECTED_ITERS, "humanml3d", 150000)

    for gpu, variant in enumerate(summary.ORDER):
        model = f"humanml3d_{variant}"
        gpu_root = tmp_path / f"gpu{gpu}"
        _write_csv(gpu_root / "summary_wide.csv", [{"model": model, "n": 2}])
        runs = []
        for run_idx, seed in enumerate((123, 124), start=1):
            samples = [{"sample": 0}, {"sample": 1}]
            digest = summary.canonical_sha256(samples)
            runs.append({
                "model": model,
                "seed": seed,
                "run_idx": run_idx,
                "iter": 150000,
                "matched_pairs": 2,
                "total_pairs": 2,
                "sample_index_sha256": digest,
            })
            index_path = gpu_root / model / "sample_indices" / f"seed_{seed}.json"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(json.dumps({
                "model": model,
                "seed": seed,
                "sample_count": 2,
                "sample_index_sha256": digest,
                "samples": samples,
            }))
        _write_csv(gpu_root / model / "runs.csv", runs)
    return tmp_path


def test_strict_summarizer_accepts_complete_aligned_shards(tmp_path, monkeypatch):
    eval_root = _build_tiny_eval_root(tmp_path, monkeypatch)
    rows = summary.read_rows(eval_root, "humanml3d")
    assert set(rows) == set(summary.ORDER)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("matched_pairs", 1, "expected 2/2"),
        ("run_idx", 2, "expected run_idx"),
        ("iter", 149999, "checkpoint iteration mismatch"),
        ("sample_index_sha256", "stale", "sample-index checksum mismatch"),
    ],
)
def test_strict_summarizer_rejects_incomplete_or_stale_rows(
    tmp_path, monkeypatch, column, value, message
):
    eval_root = _build_tiny_eval_root(tmp_path, monkeypatch)
    runs_path = eval_root / "gpu0/humanml3d_full/runs.csv"
    with runs_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][column] = value
    _write_csv(runs_path, rows)
    with pytest.raises(ValueError, match=message):
        summary.read_rows(eval_root, "humanml3d")


def _load_retry_validator():
    root = Path(__file__).resolve().parents[1]
    source = (root / "ablation/repeat20_perturb_fast_batch.py").read_text()
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "validate_existing_rows"
    )
    namespace = {
        "Path": Path,
        "RUN_FIELDS": [
            "model", "kind", "run_idx", "seed", "checkpoint", "iter",
            "fid", "fid_p", "fid_d", "diversity_real", "diversity",
            "top1", "top2", "top3", "matching", "matched_pairs",
            "total_pairs", "sample_index_sha256", "elapsed_sec", "path",
        ],
        "resolve_repo_path": lambda path: str(
            Path(path) if Path(path).is_absolute() else root / path
        ),
    }
    exec(compile(ast.Module([node], []), "<retry-validator>", "exec"), namespace)
    return namespace["validate_existing_rows"]


def test_retry_validator_rejects_duplicate_or_incompatible_rows(tmp_path):
    validate = _load_retry_validator()
    model_dir = tmp_path / "humanml3d_full"
    index_dir = model_dir / "sample_indices"
    index_dir.mkdir(parents=True)
    index_path = index_dir / "seed_123.json"
    index_path.write_text("{}")
    base = {field: "0" for field in [
        "model", "kind", "run_idx", "seed", "checkpoint", "iter",
        "fid", "fid_p", "fid_d", "diversity_real", "diversity",
        "top1", "top2", "top3", "matching", "matched_pairs",
        "total_pairs", "sample_index_sha256", "elapsed_sec", "path",
    ]}
    base.update({
        "model": "humanml3d_full",
        "kind": "sato_pp",
        "run_idx": "1",
        "seed": "123",
        "checkpoint": "sato_pp_final.pth",
        "iter": "150000",
        "sample_index_sha256": "hash",
        "path": "checkpoint.pth",
    })
    spec = {"name": "humanml3d_full", "kind": "sato_pp"}
    bundle = {
        "checkpoint": "sato_pp_final.pth",
        "iter": 150000,
        "path": "checkpoint.pth",
    }
    validate([base], spec, bundle, [123], model_dir, True)

    with pytest.raises(ValueError, match="Duplicate existing seed"):
        validate([base, copy.deepcopy(base)], spec, bundle, [123], model_dir, True)

    stale = copy.deepcopy(base)
    stale["iter"] = "149999"
    with pytest.raises(ValueError, match="Stale or incompatible"):
        validate([stale], spec, bundle, [123], model_dir, True)


def test_evaluation_waits_for_training_lock_release(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = (root / "ablation/run_var_cov_ablation.sh").read_text()
    match = re.search(
        r"wait_training_lock_released\(\) \{\n.*?\n\}", script, re.DOTALL
    )
    assert match is not None
    function_source = match.group(0)
    out = tmp_path / "humanml3d_full_seed123"
    out.mkdir()
    lock = out / "train.lock"

    holder = subprocess.Popen([
        "bash", "-c",
        'exec 9>"$1"; flock 9; sleep 0.5',
        "holder", str(lock),
    ])
    time.sleep(0.1)
    start = time.monotonic()
    subprocess.run(
        [
            "bash", "-c",
            function_source + '\nwait_training_lock_released "$1"',
            "waiter", str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start
    holder.wait(timeout=2)
    assert elapsed >= 0.3


def test_humanml3d_eval_waits_for_all_four_variant_locks():
    root = Path(__file__).resolve().parents[1]
    script = (root / "ablation/run_var_cov_ablation.sh").read_text()
    assert "for variant in full no_var no_cov no_both; do" in script
    assert "wait_training_lock_released" in script
