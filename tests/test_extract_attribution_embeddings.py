import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.extract_attribution_embeddings import (
    load_sample_index,
    normalized_entropy,
    rows_from_sample_indices,
    save_npz_records,
    semantic_class,
)


class AttributionEmbeddingHelpersTest(unittest.TestCase):
    def test_semantic_class_prefers_explicit_label(self):
        self.assertEqual(semantic_class({"semantic_class": "direction/left"}), "direction/left")

    def test_semantic_class_falls_back_to_relation_and_edit_type(self):
        self.assertEqual(
            semantic_class({"relation": "alter", "edit_type": "speed"}),
            "alter/speed",
        )
        self.assertEqual(semantic_class({"relation": "preserve"}), "preserve/paraphrase")

    def test_normalized_entropy_ignores_masked_tokens(self):
        alpha = torch.tensor([0.5, 0.25, 0.25, 0.0])
        mask = torch.tensor([True, True, True, False])
        value = normalized_entropy(alpha, mask)
        self.assertGreater(value, 0.9)
        self.assertLessEqual(value, 1.0)

    def test_fixed_sample_index_schema_and_hash(self):
        rows = [{
            "loader_name": "m1", "motion_id": "m1", "caption": "walk",
            "perturbed_caption": "a person walks", "level": "P1",
            "motion_length": 80, "pair_id": 7,
        }]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            path.write_text(json.dumps({"samples": rows}), encoding="utf-8")
            loaded, digest = load_sample_index(path)
            converted, hashes = rows_from_sample_indices([path])
        self.assertEqual(loaded, rows)
        self.assertEqual(converted[0]["text_orig"], "walk")
        self.assertEqual(converted[0]["semantic_class"], "preserve/P1")
        self.assertEqual(hashes, {"P1": digest})

    def test_npz_output_has_aligned_roles_and_representations(self):
        records = []
        for role in ("orig", "pair"):
            for representation in ("clip_pooled", "grad_input_weighted"):
                records.append({
                    "level": "P1", "motion_id": "m1", "pair_id": 1,
                    "prompt_role": role, "text": role,
                    "representation": representation,
                    "attribution_entropy": 0.5, "visible_token_count": 3,
                    "target_token_sha256": "abc",
                    "embedding": np.ones(4) * (1 if role == "orig" else 2),
                })
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "records.npz"
            save_npz_records(records, output, {
                "model": "test", "checkpoint_sha256": "ckpt",
                "sample_index_hashes": {"P1": "index"},
            })
            with np.load(output, allow_pickle=False) as dump:
                self.assertEqual(str(dump["schema_version"].item()), "attribution-motion-alignment/v1")
                self.assertEqual(dump["clip_pooled"].shape, (2, 4))
                self.assertEqual(dump["prompt_role"].astype(str).tolist(), ["orig", "pair"])


if __name__ == "__main__":
    unittest.main()
