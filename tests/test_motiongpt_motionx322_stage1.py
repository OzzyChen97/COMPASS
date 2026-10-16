import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


MOTIONGPT_ROOT = Path(__file__).parents[1] / "MotionGPT"
if str(MOTIONGPT_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONGPT_ROOT))

from mGPT.data.MotionX322 import MotionX322DatasetVQ
from mGPT.losses.mgpt import select_velocity_features
from mGPT.callback import getCheckpointCallback
from pytorch_lightning.callbacks import EarlyStopping


class MotionGPTMotionX322Stage1Tests(unittest.TestCase):
    def test_native_velocity_channels_are_pose_and_translation_only(self):
        features = torch.arange(2 * 3 * 322, dtype=torch.float32).reshape(2, 3, 322)
        selected = select_velocity_features(features, num_joints=55)
        expected = torch.cat((features[..., :209], features[..., 309:312]), dim=-1)
        torch.testing.assert_close(selected, expected)
        self.assertEqual(selected.shape[-1], 212)

    def test_legacy_velocity_slices_remain_unchanged(self):
        kit = torch.arange(2 * 3 * 251, dtype=torch.float32).reshape(2, 3, 251)
        human = torch.arange(2 * 3 * 263, dtype=torch.float32).reshape(2, 3, 263)
        torch.testing.assert_close(
            select_velocity_features(kit, num_joints=21), kit[..., 4:64]
        )
        torch.testing.assert_close(
            select_velocity_features(human, num_joints=22), human[..., 4:67]
        )

    def test_motionx_dataset_samples_normalized_fixed_window_lazily(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "new_joint_vecs").mkdir()
            (root / "texts").mkdir()
            (root / "train.txt").write_text("sample\n", encoding="utf-8")
            np.save(root / "Mean.npy", np.ones(322, dtype=np.float32))
            np.save(root / "Std.npy", np.full(322, 2.0, dtype=np.float32))
            motion = np.full((80, 322), 5.0, dtype=np.float32)
            np.save(root / "new_joint_vecs" / "sample.npy", motion)
            np.save(root / "new_joint_vecs" / "sample2.npy", motion)
            (root / "train.txt").write_text("sample\nsample2\n", encoding="utf-8")

            dataset = MotionX322DatasetVQ(
                root,
                "train",
                mean=np.ones(322, dtype=np.float32),
                std=np.full(322, 2.0, dtype=np.float32),
                min_motion_length=64,
                win_size=64,
                max_samples=1,
            )
            self.assertEqual(dataset.name_list, ["sample"])
            _, sampled, length, *_ = dataset[0]
            self.assertEqual(sampled.shape, (64, 322))
            self.assertEqual(length, 64)
            np.testing.assert_allclose(sampled, 2.0)

    def test_motionx_stage1_budget_and_native_target_are_declared(self):
        config = (MOTIONGPT_ROOT / "configs/config_motionx322_stage1.yaml").read_text()
        self.assertIn("MAX_STEPS: 300000", config)
        self.assertIn("VAL_EVERY_N_STEPS: 10000", config)
        self.assertIn("SHUFFLE: true", config)
        self.assertIn("NFEATS: 322", config)
        self.assertIn("mGPT.data.MotionX322.MotionX322DataModule", config)
        self.assertIn("MONITOR: total/val", config)
        self.assertIn("PATIENCE: 10", config)
        default = (MOTIONGPT_ROOT / "configs/default.yaml").read_text()
        self.assertIn("MAX_STEPS: -1", default)

    def test_stage1_early_stopping_is_validation_loss_only(self):
        cfg = OmegaConf.load(MOTIONGPT_ROOT / "configs/config_motionx322_stage1.yaml")
        with tempfile.TemporaryDirectory() as directory:
            cfg.FOLDER_EXP = directory
            callbacks = getCheckpointCallback(cfg)
        early = [item for item in callbacks if isinstance(item, EarlyStopping)]
        self.assertEqual(len(early), 1)
        self.assertEqual(early[0].monitor, "total/val")
        self.assertEqual(early[0].mode, "min")
        self.assertEqual(early[0].patience, 10)

    def test_validation_metrics_are_reduced_across_all_ddp_ranks(self):
        source = (MOTIONGPT_ROOT / "mGPT/models/base.py").read_text()
        validation_hook = source.split("def on_validation_epoch_end", 1)[1].split(
            "def on_test_epoch_end", 1
        )[0]
        self.assertIn("sync_dist=True", validation_hook)
        self.assertIn("rank_zero_only=False", validation_hook)
        log_calls = [
            line.strip() for line in validation_hook.splitlines()
            if "self.log_dict(" in line
        ]
        self.assertEqual(
            log_calls,
            ["self.log_dict(dico, sync_dist=True, rank_zero_only=False)"],
        )
        self.assertIn('"step": float(self.trainer.global_step)', source)


if __name__ == "__main__":
    unittest.main()
