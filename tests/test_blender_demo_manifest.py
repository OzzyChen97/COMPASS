import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_blender_demo", ROOT / "tools/build_blender_demo.py"
)
demo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(demo)


def candidate(case_id, relation, edit_type, score, failure=False):
    common = {
        "case_id": case_id,
        "relation": relation,
        "edit_type": edit_type,
        "prompt": f"prompt for {case_id}",
        "seed": 123,
        "length": 80,
        "selection_score": score,
        "failure_score": score,
        "is_failure": failure,
    }
    common["baseline"] = {"motion_path": f"motions/{case_id}_baseline.npy"}
    common["sato_pp"] = {"motion_path": f"motions/{case_id}_sato.npy"}
    return common


class BlenderDemoManifestTest(unittest.TestCase):
    def test_fixed_selection_and_deterministic_ranking(self):
        raw = [
            candidate("p_low", "preserve", "paraphrase", 1),
            candidate("p_mid", "preserve", "paraphrase", 2),
            candidate("p_high", "preserve", "paraphrase", 3),
            candidate("p_best", "preserve", "paraphrase", 4),
            candidate("direction", "alter", "direction", 5),
            candidate("speed", "alter", "tempo", 6),
            candidate("body", "alter", "bodypart", 7),
            candidate("action_lower", "alter", "action", 1),
            candidate("failure", "alter", "action", 9, failure=True),
        ]
        normalized = [demo.normalize_candidate(row, idx, ROOT) for idx, row in enumerate(raw)]
        selected = demo.select_cases(normalized)
        self.assertEqual(
            [row["case_id"] for row in selected],
            ["p_best", "p_high", "p_mid", "direction", "speed", "body", "failure"],
        )
        self.assertEqual(len({row["case_id"] for row in selected}), 7)

    def test_rejects_mismatched_generation_conditions(self):
        row = candidate("bad", "preserve", "paraphrase", 1)
        row["sato_pp"]["seed"] = 999
        with self.assertRaisesRegex(demo.ManifestError, "seed mismatch"):
            demo.normalize_candidate(row, 0, ROOT)

    def test_missing_blender_is_nonfatal_and_dry_run_writes_nothing(self):
        rows = [
            candidate("p1", "preserve", "paraphrase", 3),
            candidate("p2", "preserve", "paraphrase", 2),
            candidate("p3", "preserve", "paraphrase", 1),
            candidate("d", "alter", "direction", 1),
            candidate("s", "alter", "speed", 1),
            candidate("b", "alter", "body_part", 1),
            candidate("f", "preserve", "paraphrase", 1, failure=True),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            candidates_path = temp / "candidates.json"
            candidates_path.write_text(json.dumps(rows), encoding="utf-8")
            output_dir = temp / "out"
            args = Namespace(
                candidates=str(candidates_path),
                output_dir=str(output_dir),
                render_config=str(ROOT / "MotionGPT/configs/render.yaml"),
                blender=None,
                dry_run=True,
                run_blender=False,
            )
            with mock.patch.dict(demo.os.environ, {}, clear=True), mock.patch.object(
                demo.shutil, "which", return_value=None
            ), mock.patch.object(demo.Path, "is_file", return_value=False):
                manifest = demo.build_manifest(args)
            self.assertEqual(manifest["blender"]["status"], "missing")
            self.assertEqual(len(manifest["selected_cases"]), 7)
            self.assertFalse(output_dir.exists())
            model = manifest["selected_cases"][0]["models"]["baseline"]
            self.assertEqual(model["render_status"], "blocked_missing_blender")
            self.assertIn("contact-sheet", model["postprocess"][0]["command"])

    def test_requires_all_three_alter_buckets(self):
        rows = [
            candidate("p1", "preserve", "paraphrase", 3),
            candidate("p2", "preserve", "paraphrase", 2),
            candidate("p3", "preserve", "paraphrase", 1),
            candidate("d", "alter", "direction", 1),
            candidate("s", "alter", "speed", 1),
            candidate("f", "alter", "action", 1, failure=True),
        ]
        normalized = [demo.normalize_candidate(row, idx, ROOT) for idx, row in enumerate(rows)]
        with self.assertRaisesRegex(demo.ManifestError, "body_part/action"):
            demo.select_cases(normalized)

    def test_reads_jsonl_candidates(self):
        rows = [candidate("one", "preserve", "paraphrase", 1), candidate("two", "alter", "speed", 1)]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            self.assertEqual([row["case_id"] for row in demo._read_records(path)], ["one", "two"])


if __name__ == "__main__":
    unittest.main()
