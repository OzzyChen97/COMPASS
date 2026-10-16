import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "tools/audit_motiongpt_motionx_tokens.py"
SPEC = importlib.util.spec_from_file_location("token_audit", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _fixture(tmp_path):
    data = tmp_path / "data"
    motions = data / "new_joint_vecs"
    tokens = data / "FRESH_TOKENS"
    motions.mkdir(parents=True)
    tokens.mkdir()
    (data / "train.txt").write_text("a\nb\nshort\n")
    np.save(motions / "a.npy", np.zeros((8, 322), dtype=np.float32))
    np.save(motions / "b.npy", np.zeros((13, 322), dtype=np.float32))
    np.save(motions / "short.npy", np.zeros((3, 322), dtype=np.float32))
    np.save(tokens / "a.npy", np.array([[1, 2]], dtype=np.int64))
    np.save(tokens / "b.npy", np.array([[3, 4, 5]], dtype=np.int64))
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"fresh-stage1")
    selection = tmp_path / "best-validation.json"
    selection.write_text(json.dumps({
        "split": "val",
        "monitor": "total/val",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": module.sha256_file(checkpoint),
        "uses_t2mgpt_generator_checkpoint": False,
    }))
    return data, tokens, selection


def test_token_audit_accepts_exact_eligible_coverage(tmp_path):
    data, tokens, selection = _fixture(tmp_path)
    result = module.audit_tokens(data, tokens, selection)
    assert result["status"] == "pass"
    assert result["split_count"] == 3
    assert result["eligible_count"] == 2
    assert result["ineligible_count"] == 1
    assert result["token_file_count"] == 2
    assert result["missing_count"] == 0


def test_token_audit_rejects_missing_or_wrong_length_tokens(tmp_path):
    data, tokens, selection = _fixture(tmp_path)
    (tokens / "a.npy").unlink()
    np.save(tokens / "b.npy", np.array([[1, 2]], dtype=np.int64))
    with pytest.raises(RuntimeError) as exc:
        module.audit_tokens(data, tokens, selection)
    message = str(exc.value)
    assert '"missing_count": 1' in message
    assert '"invalid_token_count": 1' in message


def test_token_audit_rejects_checkpoint_hash_drift(tmp_path):
    data, tokens, selection = _fixture(tmp_path)
    payload = json.loads(selection.read_text())
    Path(payload["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash"):
        module.audit_tokens(data, tokens, selection)
