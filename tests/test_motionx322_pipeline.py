import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dataset.config import (
    dataset_max_motion_length,
    dataset_max_token_length,
    get_dataset_spec,
)
from dataset.dataset_tokenize import normalize_and_pad_motion
from dataset.dataset_sato_pp import validate_variable_token_cap
from dataset.variable_motion import (
    collate_motion_sequences,
    downsample_lengths,
)
from models.quantize_cnn import QuantizeEMAReset
from tools.build_sato_pairs import complete_preserve_record, finalize, progress_records
from tools.build_motionx322_token_cap import build_subset
from tools.finalize_test_perb_levels import (
    build_full_coverage_candidates,
    build_full_coverage_mixed,
)
from tools.tokenization_contract import token_digest
from train_t2m_trans_ddp import should_run_periodic_eval
from train_vq_ddp import global_loss_scales
from utils.eval_trans import _sample_motion_indices_batch_greedy


class _DeterministicGreedyTransformer:
    def __init__(self, sequences, num_vq=8, block_size=6):
        self.sequences = sequences
        self.num_vq = num_vq
        self.block_size = block_size

    def __call__(self, prefixes, features):
        step = 0 if isinstance(prefixes, list) else prefixes.shape[1]
        logits = torch.full(
            (features.shape[0], step + 1, self.num_vq + 1),
            -100.0, device=features.device)
        for row, feature in enumerate(features):
            sequence = self.sequences[int(feature.item())]
            token = sequence[step] if step < len(sequence) else self.num_vq
            logits[row, -1, token] = 100.0
        return logits


class MotionX322PipelineTests(unittest.TestCase):
    def test_sato_variable_dataset_rejects_tokens_over_cap(self):
        validate_variable_token_cap(["valid"], [75], 76)
        with self.assertRaisesRegex(ValueError, "invalid has 76"):
            validate_variable_token_cap(["valid", "invalid"], [75, 76], 76)

    def test_periodic_eval_can_start_after_training_warmup(self):
        self.assertFalse(should_run_periodic_eval(0, 100000, 10000))
        self.assertFalse(should_run_periodic_eval(90000, 100000, 10000))
        self.assertTrue(should_run_periodic_eval(100000, 100000, 10000))
        self.assertTrue(should_run_periodic_eval(110000, 100000, 10000))

    def test_batched_greedy_sampling_preserves_per_row_eos(self):
        transformer = _DeterministicGreedyTransformer({
            0: [2, 3, 8],
            1: [4, 8],
            2: [8],
            3: [1, 5, 6, 7, 0],
        })
        features = torch.arange(4, dtype=torch.float32).unsqueeze(1)
        codes, lengths = _sample_motion_indices_batch_greedy(
            transformer, features)
        self.assertEqual(lengths.tolist(), [2, 1, 1, 5])
        self.assertEqual(codes[0, :2].tolist(), [2, 3])
        self.assertEqual(codes[1, :1].tolist(), [4])
        self.assertEqual(codes[2, :1].tolist(), [1])
        self.assertEqual(codes[3, :5].tolist(), [1, 5, 6, 7, 0])

    def test_tokenization_digest_detects_changed_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token_dir = root / "VQVAE"
            token_dir.mkdir()
            (root / "train.txt").write_text("first\nsecond\n", encoding="utf-8")
            np.save(token_dir / "first.npy", np.array([[1, 2, 3]]))
            np.save(token_dir / "second.npy", np.array([[4, 5]]))
            count, original = token_digest(root, "VQVAE")
            self.assertEqual(count, 2)

            np.save(token_dir / "second.npy", np.array([[4, 6]]))
            _, changed = token_digest(root, "VQVAE")
            self.assertNotEqual(original, changed)

    def test_ema_accumulation_matches_one_global_batch_update(self):
        args = SimpleNamespace(mu=0.99)
        full_batch = QuantizeEMAReset(4, 2, args)
        accumulated = QuantizeEMAReset(4, 2, args)
        initial = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ])
        full_batch.init_codebook(initial)
        accumulated.init_codebook(initial)
        encoded = torch.tensor([
            [[0.9, 0.8, 0.0], [0.1, 0.2, 0.9]],
            [[-0.9, 0.0, 0.1], [0.0, -0.8, -0.9]],
        ])
        mask = torch.tensor([[True, True, True], [True, True, False]])

        _, _, full_perplexity = full_batch(encoded, mask=mask)
        accumulated.begin_codebook_accumulation()
        accumulated(encoded[:1], mask=mask[:1])
        accumulated(encoded[1:], mask=mask[1:])
        accumulated_perplexity = accumulated.end_codebook_accumulation()

        torch.testing.assert_close(accumulated.codebook, full_batch.codebook)
        torch.testing.assert_close(accumulated.code_sum, full_batch.code_sum)
        torch.testing.assert_close(accumulated.code_count, full_batch.code_count)
        torch.testing.assert_close(accumulated_perplexity, full_perplexity)

    def test_accumulated_losses_are_weighted_by_valid_elements(self):
        batches = [
            {"lengths": torch.tensor([2])},
            {"lengths": torch.tensor([6])},
        ]
        scales = global_loss_scales(
            batches, unit_length=4, device=torch.device("cpu"),
            distributed=False)
        torch.testing.assert_close(
            scales[:, 0], torch.tensor([0.25, 0.75], dtype=torch.float64))
        torch.testing.assert_close(
            scales[:, 1], torch.tensor([1 / 6, 5 / 6], dtype=torch.float64))
        torch.testing.assert_close(
            scales[:, 2], torch.tensor([1 / 3, 2 / 3], dtype=torch.float64))

    def test_motionx322_dataset_contract(self):
        spec = get_dataset_spec("motionx322")
        self.assertEqual(spec.dim_pose, 322)
        self.assertEqual(spec.fps, 30.0)
        self.assertEqual(spec.representation, "smplx322")
        self.assertTrue(spec.variable_length)
        self.assertEqual(dataset_max_motion_length("motionx322"), 2919)
        self.assertEqual(dataset_max_token_length("motionx322", 4), 731)

    def test_token_cap_subset_filters_splits_without_copying_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "subset"
            source.mkdir()
            for name in ("new_joint_vecs", "texts", "VQVAE"):
                (source / name).mkdir()
            np.save(source / "Mean.npy", np.zeros(322, dtype=np.float32))
            np.save(source / "Std.npy", np.ones(322, dtype=np.float32))
            rows = [
                {"motion_id": "keep", "split": "train", "frames": 300},
                {"motion_id": "drop", "split": "train", "frames": 301},
                {"motion_id": "test", "split": "test", "frames": 1},
            ]
            (source / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8")
            (source / "dataset_meta.json").write_text(
                json.dumps({"dataset": "motionx322", "max_frames": 301}),
                encoding="utf-8")

            summary = build_subset(source, output, 75, 4)

            self.assertEqual(summary["excluded_count"], 1)
            self.assertEqual(summary["max_motion_tokens"], 75)
            self.assertEqual(
                (output / "train.txt").read_text().splitlines(), ["keep"])
            self.assertTrue((output / "new_joint_vecs").is_symlink())
            metadata = json.loads(
                (output / "dataset_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["max_motion_tokens"], 75)

    def test_full_coverage_candidates_keep_partial_levels(self):
        captions = {"motion": ["shared caption"]}
        by_caption = {
            "shared caption": {
                "P1": ["lexical"],
                "P2": ["syntactic"],
            }
        }

        candidates = build_full_coverage_candidates(
            ["motion"], captions, by_caption, 123)

        self.assertEqual(set(candidates["motion"]), {"P1", "P2"})
        self.assertEqual(candidates["motion"]["P1"]["text_pair"], "lexical")

    def test_full_coverage_mix_uses_available_fallback_levels(self):
        candidates = {
            "all": {
                "P1": {"motion_id": "all", "level": "P1"},
                "P2": {"motion_id": "all", "level": "P2"},
                "P3": {"motion_id": "all", "level": "P3"},
            },
            "p1p2": {
                "P1": {"motion_id": "p1p2", "level": "P1"},
                "P2": {"motion_id": "p1p2", "level": "P2"},
            },
            "p1": {
                "P1": {"motion_id": "p1", "level": "P1"},
            },
        }

        rows, fallback_counts = build_full_coverage_mixed(
            ["all", "p1p2", "p1"], candidates, (0.0, 0.0, 1.0), 123)

        self.assertEqual({row["motion_id"] for row in rows}, set(candidates))
        self.assertEqual(len(rows), 3)
        self.assertEqual(fallback_counts, {"P3->P2": 1, "P3->P1": 1})

    def test_normalize_before_padding_keeps_padding_zero(self):
        mean = np.array([10.0, -2.0], dtype=np.float32)
        std = np.array([2.0, 4.0], dtype=np.float32)
        motion = np.array(
            [[12.0, 2.0], [8.0, -6.0], [10.0, -2.0]], dtype=np.float32)
        result = normalize_and_pad_motion(motion, mean, std, unit_length=4)
        np.testing.assert_allclose(
            result[:3],
            np.array([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]]),
        )
        np.testing.assert_array_equal(
            result[3], np.zeros(2, dtype=np.float32))

    def test_variable_collate_preserves_complete_sequences(self):
        first = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
        second = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
        batch = collate_motion_sequences([first, second], multiple=4)
        self.assertEqual(batch["motion"].shape, (2, 8, 3))
        self.assertEqual(batch["lengths"].tolist(), [5, 8])
        np.testing.assert_array_equal(batch["motion"][0, :5].numpy(), first)
        self.assertEqual(int(torch.count_nonzero(batch["motion"][0, 5:])), 0)
        self.assertEqual(
            batch["frame_mask"][0].tolist(), [True] * 5 + [False] * 3)

    def test_partial_tail_is_included_only_for_native_variable_sequences(self):
        lengths = torch.tensor([1, 4, 5, 2919])
        self.assertEqual(
            downsample_lengths(
                lengths, 4, include_partial=True).tolist(),
            [1, 1, 2, 730],
        )
        self.assertEqual(
            downsample_lengths(
                lengths, 4, include_partial=False).tolist(),
            [1, 1, 1, 729],
        )

    def test_pair_progress_uses_latest_complete_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            item = {
                "motion_id": "mx322_test",
                "text_orig": "A person walks forward.",
            }
            progress = tmp_path / "pairs.json.progress.jsonl"
            invalid = {
                "items": [item],
                "preserve": [{
                    "index": 1,
                    "P1": ["Someone walks forward."],
                    "P2": [],
                    "P3": [],
                }],
                "alter": [],
            }
            valid = {
                "items": [item],
                "preserve": [{
                    "index": 1,
                    "P1": ["Someone walks forward."],
                    "P2": ["Forward, a person walks."],
                    "P3": ["The figure makes their way straight ahead."],
                }],
                "alter": [],
            }
            progress.write_text(
                json.dumps(invalid) + "\n" + json.dumps(valid) + "\n",
                encoding="utf-8",
            )
            record = progress_records(progress)[
                ("mx322_test", item["text_orig"])]
            self.assertTrue(complete_preserve_record(record))

            output = tmp_path / "pairs.json"
            finalize(progress, output, "test", 123, {})
            rows = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                [row["level"] for row in rows], ["P1", "P2", "P3"])
            self.assertTrue(
                all(row["motion_id"] == "mx322_test" for row in rows))


if __name__ == "__main__":
    unittest.main()
