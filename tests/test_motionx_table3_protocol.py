import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools" / "build_motionx_table3_manifest.py"
SPEC = importlib.util.spec_from_file_location("motionx_table3_manifest", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC is not None
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


@pytest.fixture
def samples_rows():
    return [
        {"motion_id": f"m{i}", "text_pair": "a", "text_orig": "a", "level": "P1", "relation": "preserve", "edit_type": ""}
        for i in range(7733)
    ]


def test_manifest_build_and_counts(tmp_path, samples_rows):
    pair_file = tmp_path / "pairs.json"
    pair_file.write_text("\n".join(module.json.dumps(row) for row in samples_rows) + "\n", encoding="utf-8")

    spec = [
        {
            "name": "motionx322_t2mgpt_baseline_200k_best",
            "kind": "baseline",
            "cfg_scale": 1.0,
            "cfg_uncond": "zero",
        },
        {
            "name": "motionx322_gemini31_sato100k",
            "kind": "sato_pp",
            "cfg_scale": 1.3,
            "cfg_uncond": "zero",
        },
    ]
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(module.json.dumps(spec), encoding="utf-8")

    rows = []
    for model in ("motionx322_t2mgpt_baseline_200k_best", "motionx322_gemini31_sato100k"):
        kind = "baseline" if "baseline" in model else "sato_pp"
        for seed in range(123, 143):
            rows.append(
                {
                    "seed": seed,
                    "model": model,
                    "kind": kind,
                    "checkpoint": "ckpt.pth",
                    "iter": 10000,
                    "path": "x",
                    "matched_pairs": 7733,
                    "total_pairs": 7733,
                    "elapsed_sec": 1.0,
                    "fid": 0.1,
                    "fid_p": 0.2,
                    "fid_d": 0.3,
                    "diversity_real": 0.4,
                    "diversity": 0.5,
                    "top1": 0.6,
                    "top2": 0.7,
                    "top3": 0.8,
                    "matching": 0.9,
                }
            )
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(module.json.dumps({"runs": rows}), encoding="utf-8")

    manifest = module.build_manifest(
        pair_file=pair_file,
        spec_json=spec_file,
        summary_json=summary_file,
        expected_seeds=list(range(123, 143)),
        expected_samples=7733,
    )

    assert manifest["expected_samples"] == 7733
    assert manifest["loader_protocol"]["pair_level_counts"] == {"P1": 7733}
    assert manifest["loader_protocol"]["pair_matching_contract"] == "7733/7733"
    assert manifest["models"]["motionx322_t2mgpt_baseline_200k_best"]["cfg_scale"] == 1.0
    assert manifest["models"]["motionx322_gemini31_sato100k"]["cfg_scale"] == 1.3

    for entry in spec:
        entry.pop("cfg_scale", None)
    spec_file.write_text(module.json.dumps(spec), encoding="utf-8")
    global_cfg_manifest = module.build_manifest(
        pair_file=pair_file,
        spec_json=spec_file,
        summary_json=summary_file,
        expected_seeds=list(range(123, 143)),
        expected_samples=7733,
        default_cfg_scale=1.3,
        default_cfg_uncond="zero",
    )
    assert all(
        item["cfg_scale"] == 1.3
        for item in global_cfg_manifest["models"].values()
    )


def test_manifest_rejects_seed_gap(tmp_path, samples_rows):
    pair_file = tmp_path / "pairs.json"
    pair_file.write_text("\n".join(module.json.dumps(row) for row in samples_rows) + "\n", encoding="utf-8")
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(
        module.json.dumps(
            [
                {
                    "name": "motionx322_t2mgpt_baseline_200k_best",
                    "kind": "baseline",
                }
            ]
        ),
        encoding="utf-8",
    )
    summary_file = tmp_path / "summary.json"
    summary_file.write_text(
        module.json.dumps(
            {
                "runs": [
                    {
                        "seed": 123,
                        "model": "motionx322_t2mgpt_baseline_200k_best",
                        "kind": "baseline",
                        "checkpoint": "ckpt.pth",
                        "iter": 1,
                        "path": "x",
                        "matched_pairs": 7733,
                        "total_pairs": 7733,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Seed mismatch"):
        module.build_manifest(
            pair_file=pair_file,
            spec_json=spec_file,
            summary_json=summary_file,
            expected_seeds=list(range(123, 143)),
            expected_samples=7733,
        )


def test_real_table3_pair_file_has_expected_counts():
    pair_file = Path("dataset/Motion-X-322/test_perb_gemini31_max75_mixed_full.json")
    rows = module.load_pair_rows(pair_file)
    levels = {row["level"] for row in rows}
    assert levels == {"P1", "P2", "P3"}
    assert len(rows) == 7733
    counts = {key: 0 for key in ("P1", "P2", "P3")}
    for row in rows:
        counts[row["level"]] += 1
    assert counts == {"P1": 3558, "P2": 2099, "P3": 2076}
