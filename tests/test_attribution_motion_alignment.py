import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.analyze_attribution_motion_alignment import (
    grouped_folds,
    holm_adjust,
    linear_cka,
    load_diagnostic,
    paired_sign_permutation,
    pair_displacement,
    ridge_probe,
    rsa_spearman,
    sample_pair_indices,
)


class AttributionMotionAlignmentTest(unittest.TestCase):
    def test_cka_identity_rotation_and_permutation(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(120, 12))
        rotation, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        self.assertAlmostEqual(linear_cka(x, x), 1.0, places=12)
        self.assertAlmostEqual(linear_cka(x, x @ rotation), 1.0, places=12)
        shuffled = x[rng.permutation(len(x))]
        self.assertLess(linear_cka(x, shuffled), 0.35)

    def test_rsa_and_displacement_detect_correspondence(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(80, 10))
        y = 2.0 * x + rng.normal(scale=0.01, size=x.shape)
        pairs = sample_pair_indices(len(x), 4000, 2)
        self.assertGreater(rsa_spearman(x, y, pairs), 0.99)

        delta = rng.normal(scale=0.08, size=x.shape)
        result = pair_displacement(x, x + delta, y, y + 2.0 * delta)
        self.assertGreater(result["correlation"], 0.95)
        self.assertEqual(result["attr_distance"].shape, (80,))

    def test_grouped_ridge_keeps_groups_and_predicts_linear_target(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(100, 6))
        y = x @ rng.normal(size=(6, 5))
        groups = np.asarray([f"m{i // 2}" for i in range(100)])
        masks = grouped_folds(groups, 5, 3)
        for group in np.unique(groups):
            memberships = [bool(mask[groups == group].all()) for mask in masks]
            self.assertEqual(sum(memberships), 1)
        score = ridge_probe(
            x, y, groups, folds=5, alpha=1e-5, seed=3, pca_components=0
        )
        self.assertGreater(score["r2"], 0.999)
        self.assertGreater(score["cosine"], 0.999)

    def test_permutation_is_reproducible_and_holm_monotone(self):
        reference = np.linspace(0.0, 1.0, 20)
        candidate = reference + 0.2
        first = paired_sign_permutation(candidate, reference, 999, 9)
        second = paired_sign_permutation(candidate, reference, 999, 9)
        self.assertEqual(first, second)
        self.assertLess(first["p_value"], 0.01)
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertTrue(all(value >= raw for value, raw in zip(adjusted, [0.01, 0.04, 0.03])))

    def test_diagnostic_rejects_invalid_embedding_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            values = np.zeros((3, 4), dtype=np.float32)
            np.savez_compressed(
                path,
                gt_embedding=values,
                original_embedding=values,
                perturbed_embedding=values,
                motion_id=np.asarray(["a", "b", "c"]),
                caption=np.asarray(["a", "b", "c"]),
                perturbed_caption=np.asarray(["a", "b", "c"]),
                level=np.asarray(["P1", "P1", "P1"]),
            )
            with self.assertRaisesRegex(ValueError, "shape"):
                load_diagnostic(path, "P1")


if __name__ == "__main__":
    unittest.main()