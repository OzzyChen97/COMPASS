import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.analyze_attribution_clustering import (
    analyze,
    compute_metrics,
    generate_synthetic_records,
    paired_permutation_test,
    read_records,
    parse_args,
)


class AttributionClusteringTest(unittest.TestCase):
    def test_metrics_separate_compact_classes(self):
        x = np.asarray([[1, 0], [0.99, 0.02], [-1, 0], [-0.99, -0.02]], dtype=float)
        metrics = compute_metrics(x, ["a", "a", "b", "b"], knn_k=1)
        self.assertLess(metrics["within_class_cosine_distance"], 0.01)
        self.assertGreater(metrics["between_centroid_cosine_distance"], 1.9)
        self.assertGreater(metrics["silhouette"], 0.99)
        self.assertEqual(metrics["loo_knn_accuracy"], 1.0)
        self.assertEqual(metrics["nearest_centroid_accuracy"], 1.0)
        self.assertGreaterEqual(metrics["effective_rank"], 1.0)

    def test_jsonl_alias_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            rows = [
                {"motion_id": "m1", "relation": "speed", "variant": "v", "embedding": [1, 0]},
                {"motion_id": "m2", "relation": "action", "variant": "v", "embedding": [0, 1]},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            records = read_records(path)
        self.assertEqual(records[0]["semantic_class"], "speed")
        self.assertEqual(records[0]["embedding"].shape, (2,))

    def test_cli_accepts_multiple_input_files(self):
        args = parse_args(["--input", "a.jsonl", "b.jsonl", "--output-dir", "out"])
        self.assertEqual([path.name for path in args.input], ["a.jsonl", "b.jsonl"])

    def test_synthetic_analysis_outputs_and_detects_improvement(self):
        records = generate_synthetic_records(seed=11, motions=50, dimension=10)
        comparison = paired_permutation_test(records, "baseline", "sato_pp", repeats=999, seed=3)
        self.assertEqual(comparison["n_pairs"], 50)
        self.assertLess(comparison["mean_difference"], 0.0)
        self.assertLess(comparison["p_value"], 0.05)
        with tempfile.TemporaryDirectory() as tmp:
            first = analyze(
                records, tmp, bootstrap=20, permutations=99, seed=5,
                knn_k=3, reference_variant="baseline", plot_method="pca",
            )
            second_dir = Path(tmp) / "repeat"
            second = analyze(
                records, second_dir, bootstrap=20, permutations=99, seed=5,
                knn_k=3, reference_variant="baseline", plot_method="pca",
            )
            for filename in ("summary.json", "metrics.csv", "comparisons.csv", "attribution_semantic_space.png"):
                self.assertTrue((Path(tmp) / filename).is_file())
            self.assertEqual(first["variants"], second["variants"])
            self.assertEqual(first["paired_permutation_tests"], second["paired_permutation_tests"])
            self.assertLess(
                first["variants"]["sato_pp"]["metrics"]["within_class_cosine_distance"]["value"],
                first["variants"]["baseline"]["metrics"]["within_class_cosine_distance"]["value"],
            )


if __name__ == "__main__":
    unittest.main()
