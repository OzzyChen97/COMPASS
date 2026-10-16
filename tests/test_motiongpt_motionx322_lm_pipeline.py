import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


MOTIONGPT_ROOT = Path(__file__).parents[1] / "MotionGPT"
if str(MOTIONGPT_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONGPT_ROOT))

from mGPT.data.MotionX322 import (
    MotionX322RawEvalDataset,
    MotionX322TextTokenDataset,
    MotionX322TokenDataset,
)


def _selection_module():
    path = MOTIONGPT_ROOT / "eval_sato_robustness.py"
    spec = importlib.util.spec_from_file_location("motiongpt_selection", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_native_root(tmp_path):
    for name in ("new_joint_vecs", "texts", "VQVAE"):
        (tmp_path / name).mkdir()
    np.save(tmp_path / "Mean.npy", np.zeros(322, dtype=np.float32))
    np.save(tmp_path / "Std.npy", np.ones(322, dtype=np.float32))
    for split, name in (("train", "train_id"), ("val", "val_id"), ("test", "test_id")):
        (tmp_path / f"{split}.txt").write_text(name + "\n", encoding="utf-8")
        np.save(
            tmp_path / "new_joint_vecs" / f"{name}.npy",
            np.zeros((80, 322), dtype=np.float32),
        )
        np.save(
            tmp_path / "VQVAE" / f"{name}.npy",
            np.arange(20, dtype=np.int64)[None, :],
        )
        (tmp_path / "texts" / f"{name}.txt").write_text(
            f"A person moves.{name}#a/DET person/NOUN move/VERB#0.0#0.0\n",
            encoding="utf-8",
        )
    instructions = {
        "Text-to-Motion": {
            "t2m": {
                "class": "t2m",
                "input": ["<Caption_Placeholder>"],
                "output": ["<Motion_Placeholder>"],
            }
        }
    }
    instructions_path = tmp_path / "instructions.json"
    instructions_path.write_text(json.dumps(instructions), encoding="utf-8")
    return instructions_path


class MotionGPTMotionX322LMPipelineTests(unittest.TestCase):
    def test_checkpoint_aliases_launch_isolation_and_smoke_contracts(self):
        parser = (MOTIONGPT_ROOT / "mGPT/config.py").read_text()
        self.assertIn('"--pretrained_vae"', parser)
        self.assertIn('"--vae_checkpoint"', parser)
        self.assertIn('"--max_samples"', parser)

        for name in (
            "run_motionx322_tokenize.sh",
            "run_motionx322_stage1.sh",
            "run_motionx322_stage2.sh",
            "run_motionx322_stage3.sh",
        ):
            launcher = (MOTIONGPT_ROOT / "scripts" / name).read_text()
            self.assertIn("PYTHONNOUSERSITE=1", launcher)
            self.assertIn("TMPDIR=", launcher)
            self.assertIn("PYTHONPATH=", launcher)
        tokenizer = (MOTIONGPT_ROOT / "scripts/run_motionx322_tokenize.sh").read_text()
        self.assertIn("--pretrained_vae", tokenizer)
        self.assertIn("--max_samples", tokenizer)

        token_smoke = (MOTIONGPT_ROOT / "configs/config_motionx322_token_smoke.yaml").read_text()
        stage1_smoke = (MOTIONGPT_ROOT / "configs/config_motionx322_stage1_smoke.yaml").read_text()
        stage2_smoke = (MOTIONGPT_ROOT / "configs/config_motionx322_stage2_smoke.yaml").read_text()
        stage3_smoke = (MOTIONGPT_ROOT / "configs/config_motionx322_stage3_smoke.yaml").read_text()
        self.assertIn("MAX_SAMPLES: 2", token_smoke)
        self.assertIn("MAX_SAMPLES: 8", stage1_smoke)
        for smoke in (stage2_smoke, stage3_smoke):
            self.assertIn("MAX_STEPS: 1", smoke)
            self.assertIn("LIMIT_TRAIN_BATCHES: 1", smoke)
            self.assertIn("LIMIT_VAL_BATCHES: 1", smoke)

    def test_stage2_stage3_configs_lock_steps_batches_and_validation(self):
        stage2 = (MOTIONGPT_ROOT / "configs/config_motionx322_stage2.yaml").read_text()
        stage3 = (MOTIONGPT_ROOT / "configs/config_motionx322_stage3.yaml").read_text()
        self.assertIn("MAX_STEPS: 200000", stage2)
        self.assertIn("DEVICE: [0, 1, 2, 3]", stage2)
        self.assertIn("BATCH_SIZE: 4", stage2)
        self.assertIn("lr: 2e-4", stage2)
        self.assertIn("VAL_EVERY_N_STEPS: 10000", stage2)
        self.assertIn("MAX_STEPS: 100000", stage3)
        self.assertIn("DEVICE: [0, 1, 2, 3]", stage3)
        self.assertIn("BATCH_SIZE: 4", stage3)
        self.assertIn("lr: 1e-4", stage3)
        self.assertIn("VAL_EVERY_N_STEPS: 5000", stage3)
        self.assertIn("PATIENCE: 5", stage2)
        self.assertIn("MIN_VALIDATIONS: 8", stage2)
        self.assertIn("PATIENCE: 6", stage3)
        self.assertIn("MIN_VALIDATIONS: 8", stage3)
        self.assertNotIn("SATO:", stage2)
        self.assertNotIn("SATO:", stage3)

    def test_native_lm_loaders_preserve_requested_split_and_variable_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            instructions = _make_native_root(tmp_path)
            zeros = np.zeros(322, dtype=np.float32)
            ones = np.ones(322, dtype=np.float32)
            tokens = MotionX322TokenDataset(tmp_path, "val", zeros, ones, min_motion_length=1)
            self.assertEqual(tokens.name_list, ["val_id"])
            self.assertEqual(tokens[0][0], "val_id")

            text_tokens = MotionX322TextTokenDataset(
                tmp_path, "test", zeros, ones, code_path="VQVAE", task_path=instructions
            )
            self.assertEqual(text_tokens.name_list, ["test_id"])
            self.assertIn(text_tokens[0][1].shape, ((19,), (20,)))

            raw = MotionX322RawEvalDataset(tmp_path, "val", zeros, ones)
            self.assertEqual(raw.name_list, ["val_id"])
            self.assertEqual(raw[0][1].shape, (80, 322))
            self.assertEqual(raw[0][2], 80)

    def test_validation_fid_selection_ties_matching_then_r3(self):
        module = _selection_module()
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            candidates = []
            rows = (("a", 1.0, 0.5, 0.8), ("b", 1.0, 0.4, 0.7), ("c", 1.0, 0.4, 0.9))
            for name, fid, matching, top3 in rows:
                path = tmp_path / f"{name}.json"
                path.write_text(
                    json.dumps({
                        "protocol": {"phase": "screening", "evaluation_split": "val", "fingerprint": "same"},
                        "model": {"id": name, "checkpoint": f"{name}.ckpt"},
                        "aggregate": {
                            "fid": {"mean": fid},
                            "matching": {"mean": matching},
                            "r_precision_top3": {"mean": top3},
                        },
                    }),
                    encoding="utf-8",
                )
                candidates.append(path)
            selected = module.select_checkpoint(Namespace(results=candidates, metric="fid", direction="min"))
            self.assertEqual(selected["selected_checkpoint"], "c.ckpt")
            self.assertEqual(selected["tie_break"], ["matching", "r_precision_top3"])


if __name__ == "__main__":
    unittest.main()
