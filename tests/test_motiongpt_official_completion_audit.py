import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "tools/audit_motiongpt_motionx_official_complete.py"
SPEC = importlib.util.spec_from_file_location("motiongpt_completion_audit", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_completion_audit_selection_requires_validation_hash_and_baseline_free(tmp_path):
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"fresh")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    selection = tmp_path / "selected.json"
    selection.write_text(json.dumps({
        "stage": "stage2",
        "split": "val",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "uses_t2mgpt_generator_checkpoint": False,
    }))
    assert module.checked_selection(selection, "stage2")["checkpoint_sha256"] == digest

    payload = json.loads(selection.read_text())
    payload["checkpoint_sha256"] = "wrong"
    selection.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="SHA-256"):
        module.checked_selection(selection, "stage2")


def test_completion_audit_locks_full_motionx_seed_and_generation_contract():
    assert module.EXPECTED_SEEDS == list(range(123, 143))
    source = MODULE_PATH.read_text(encoding="utf-8")
    for fragment in (
        'expected_samples=7733',
        '"seed_count": 20',
        '"total_original_generations": 20 * 7733',
        '"total_perturbed_generations": 20 * 7733',
        '"pair_rng_mode": "sequential"',
        '"generation_order": "original-first"',
        '"prompt_mode": "official-default"',
        '"generation_length_mode": "generated"',
    ):
        assert fragment in source


def test_final_handoff_requires_completion_audit():
    handoff = (
        ROOT / "tools/handoff_motiongpt_motionx_official_r2_stage3_to_table3.sh"
    ).read_text(encoding="utf-8")
    assert "audit_motiongpt_motionx_official_complete.py" in handoff
    assert "final_completion_audit.json" in handoff
