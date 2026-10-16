import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

MODULE_PATH = Path(__file__).parents[1] / "MotionGPT" / "eval_loader_robustness.py"
SPEC = importlib.util.spec_from_file_location("motiongpt_loader_eval", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

AGGREGATE_PATH = Path(__file__).parents[1] / "tools" / "aggregate_motiongpt_loader_eval.py"
AGGREGATE_SPEC = importlib.util.spec_from_file_location(
    "aggregate_motiongpt_loader_eval", AGGREGATE_PATH
)
aggregate_module = importlib.util.module_from_spec(AGGREGATE_SPEC)
AGGREGATE_SPEC.loader.exec_module(aggregate_module)


def test_segment_prefix_resolution_supports_numeric_and_m_ids():
    index = {"000123": {}, "M013514": {}, "arbitrary_id": {}}
    assert module.resolve_loader_motion_id("A_000123", index) == "000123"
    assert module.resolve_loader_motion_id("E_M013514", index) == "M013514"
    assert module.resolve_loader_motion_id("arbitrary_id", index) == "arbitrary_id"
    assert module.resolve_loader_motion_id("X_not_present", index) is None


def test_resolve_perturbations_tracks_levels_and_missing():
    index = {
        "m1": {"text_pair": "walk slowly", "level": "P1"},
        "M2": {"text_pair": "turn right", "level": "P3"},
    }
    texts, motion_ids, levels = module.resolve_perturbations(["m1", "B_M2", "bad"], index)
    assert texts == ["walk slowly", "turn right", None]
    assert motion_ids == ["m1", "M2", None]
    assert levels == ["P1", "P3", "missing"]


def test_caption_pair_matching_rejects_same_motion_wrong_caption():
    pairs = [{
        "motion_id": "m1",
        "text_orig": "A person walks forward.",
        "text_pair": "A person walks backward.",
        "level": "P2",
    }]
    index = module.build_caption_pair_index(pairs)
    texts, motion_ids, levels, rows = module.resolve_caption_perturbations(
        ["m1", "m1"],
        ["  A PERSON walks forward. ", "A person jumps."],
        index,
    )
    assert texts == ["A person walks backward.", None]
    assert motion_ids == ["m1", None]
    assert levels == ["P2", "missing"]
    assert rows == [pairs[0], None]


def test_p123_filter_removes_fallback_rows():
    pairs = [
        {"motion_id": "a", "level": "P1"},
        {"motion_id": "b", "level": "P2"},
        {"motion_id": "c", "level": "P3"},
        {"motion_id": "d", "level": "fallback"},
    ]
    selected = module.filter_pairs_by_levels(pairs, ["P1", "P2", "P3"])
    assert [row["motion_id"] for row in selected] == ["a", "b", "c"]


def test_protocol_fingerprint_is_stable_and_records_4640_contract():
    pair_meta = {"path": "/pairs.json", "count": 4353, "sha256": "abc"}
    source = {
        "protocol": {
            "sampling": {"mode": "categorical", "temperature": 1.0, "top_k": 0, "top_p": 1.0},
            "seeds": list(range(123, 143)),
        }
    }
    first = module.protocol_payload(pair_meta, source)
    second = module.protocol_payload(pair_meta, source)
    assert first["loader"]["expected_samples"] == 4640
    assert first["loader"]["drop_last"] is True
    assert first["generation_length_mode"] == "generated"
    assert first["evaluator_length_mode"] == "generated_length_capped_to_batch_seq"
    assert first["sampling"]["kwargs_mode"] == "explicit"
    assert first["sampling"]["implementation"] == "explicit Hugging Face generate kwargs"
    assert first["prompt"]["mode"] == "official-default"
    assert first["pair_rng_mode"] == "sequential"
    assert first["generation_order"] == "original-first"
    assert module.canonical_sha256(first) == module.canonical_sha256(second)


def test_caption_locked_protocol_records_exact_pair_contract():
    protocol = module.protocol_payload(
        {"path": "/pairs.json", "count": 4344, "sha256": "abc"},
        {"protocol": {"sampling": {}, "seeds": list(range(123, 143))}},
        loader_mode="caption-locked",
        expected_samples=4184,
        allowed_levels=["P1", "P2", "P3"],
    )
    assert protocol["loader"] == {
        "implementation": "dataset.dataset_TM_eval.CaptionLockedDATALoader",
        "batch_size": 32,
        "shuffle": False,
        "drop_last": False,
        "caption_locked": True,
        "expected_samples": 4184,
        "pair_match": "exact normalized (motion_id, text_orig)",
        "allowed_levels": ["P1", "P2", "P3"],
        "length_policy": "strict",
    }


def test_caption_locked_repair_all_policy_is_fingerprinted():
    protocol = module.protocol_payload(
        {"path": "/pairs.json", "count": 4344, "sha256": "abc"},
        {"protocol": {"sampling": {}, "seeds": list(range(123, 143))}},
        loader_mode="caption-locked",
        expected_samples=4344,
        allowed_levels=["P1", "P2", "P3"],
        caption_length_policy="repair-all",
    )
    assert protocol["loader"]["expected_samples"] == 4344
    assert protocol["loader"]["length_policy"] == "repair-all"


def test_motionx322_protocol_records_native_7733_contract():
    pair_meta = {"path": "/motionx_pairs.json", "count": 7733,
                 "unique_motion_count": 7733, "sha256": "abc"}
    source = {
        "protocol": {
            "sampling": {
                "mode": "categorical", "temperature": 1.0,
                "top_k": 0, "top_p": 1.0,
            },
            "seeds": list(range(123, 143)),
        }
    }
    protocol = module.protocol_payload(
        pair_meta, source, pair_rng_mode="sequential", dataset="motionx322"
    )
    assert protocol["dataset"] == "motionx322"
    assert protocol["loader"]["expected_samples"] == 7733
    assert protocol["loader"]["drop_last"] is False
    assert protocol["loader"]["pair_match"] == "exact motion id only"
    assert protocol["evaluator"].endswith(":motionx322")
    assert protocol["pair_rng_mode"] == "sequential"
    assert protocol["generation_order"] == "original-first"
    assert protocol["generation_length_mode"] == "generated"
    assert protocol["prompt"]["mode"] == "official-default"


def test_motionx322_evaluator_preserves_native_token_code_path():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'if args.dataset == "humanml3d":\n                cfg.DATASET.CODE_PATH = "VQVAE"' in source


def test_motionx322_exact_matching_does_not_strip_segment_like_prefixes():
    index = {"motion": {"text_pair": "perturbed", "level": "P2"}}
    texts, motion_ids, levels = module.resolve_perturbations(
        ["A_motion"], index, allow_segment_prefix=False
    )
    assert texts == [None]
    assert motion_ids == [None]
    assert levels == ["missing"]


def test_motionx322_pair_contract_requires_7733_unique_motions():
    module.validate_pair_contract(
        {"count": 7733, "unique_motion_count": 7733}, "motionx322"
    )
    with pytest.raises(module.LoaderEvaluationError, match="7733 unique"):
        module.validate_pair_contract(
            {"count": 7733, "unique_motion_count": 7732}, "motionx322"
        )
    # Existing HumanML3D pair-definition expansion remains accepted.
    module.validate_pair_contract(
        {"count": 4353, "unique_motion_count": 4353}, "humanml3d"
    )


def test_motionx322_explicit_caption_locked_subset_preserves_native_default():
    module.validate_pair_contract(
        {"count": 7679, "unique_motion_count": 7679},
        "motionx322",
        expected_samples=7679,
    )
    with pytest.raises(module.LoaderEvaluationError, match="7679 unique"):
        module.validate_pair_contract(
            {"count": 7733, "unique_motion_count": 7733},
            "motionx322",
            expected_samples=7679,
        )
    protocol = module.protocol_payload(
        {
            "path": "/motionx_p1.json",
            "count": 7679,
            "unique_motion_count": 7679,
            "sha256": "abc",
        },
        {"protocol": {"sampling": {}, "seeds": list(range(123, 143))}},
        dataset="motionx322",
        loader_mode="caption-locked",
        expected_samples=7679,
        allowed_levels=["P1"],
        caption_length_policy="repair-all",
    )
    assert protocol["loader"]["expected_samples"] == 7679
    assert protocol["loader"]["allowed_levels"] == ["P1"]
    assert protocol["loader"]["drop_last"] is False


def test_rng_capture_restore_replays_and_preserves_post_orig_stream():
    module.reset_rng(123)
    before = module.capture_rng_state()
    orig = (random.random(), np.random.rand(), torch.rand(2))
    after_orig = module.capture_rng_state()
    module.restore_rng_state(before)
    pair = (random.random(), np.random.rand(), torch.rand(2))
    assert orig[0] == pair[0]
    assert orig[1] == pair[1]
    assert torch.equal(orig[2], pair[2])
    module.restore_rng_state(after_orig)
    next_value = (random.random(), np.random.rand(), torch.rand(2))
    module.reset_rng(123)
    random.random(), np.random.rand(), torch.rand(2)
    expected = (random.random(), np.random.rand(), torch.rand(2))
    assert next_value[0] == expected[0]
    assert next_value[1] == expected[1]
    assert torch.equal(next_value[2], expected[2])


def test_frozen_index_validation_accepts_prefix_and_rejects_drift():
    records = [{"loader_name": "a"}, {"loader_name": "b"}]
    frozen = {
        "protocol_fingerprint": "fp",
        "samples": records,
        "sample_index_sha256": module.canonical_sha256(records),
    }
    module.validate_expected_index(frozen, records[:1], "fp")
    module.validate_expected_index(frozen, records, "fp")
    with pytest.raises(module.LoaderEvaluationError, match="differ"):
        module.validate_expected_index(frozen, [{"loader_name": "x"}], "fp")


def test_r_precision_columns_are_distinct_cumulative_snapshots():
    text = np.eye(4, dtype=np.float64)
    # This permutation yields matches both inside and outside top-3.
    motion = text[[0, 2, 3, 1]]
    counts, _ = module.calculate_r_precision(text, motion, top_k=3)
    ranks = module.correct_match_ranks(text, motion)
    assert counts.tolist() == [int((ranks < k).sum()) for k in (1, 2, 3)]
    assert counts.tolist() == [1, 1, 3]


def test_r_precision_uses_text_to_motion_direction_and_diagonal_matching():
    perturbed_text = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    perturbed_generated_motion = np.asarray([[0.0], [3.0], [5.0], [7.0]])

    counts, matching = module.calculate_r_precision(
        perturbed_text, perturbed_generated_motion, top_k=3
    )
    reverse_counts, _ = module.calculate_r_precision(
        perturbed_generated_motion, perturbed_text, top_k=3
    )

    assert counts.tolist() == [1, 2, 3]
    assert reverse_counts.tolist() == [2, 3, 4]
    assert matching == pytest.approx(9.0)


def test_evaluator_lengths_are_min_generated_reference_without_tensor_crop():
    class FakeLM:
        def template_fulfill(self, tasks, lengths, motions, texts, stage):
            return texts, [""] * len(texts)

        def generate_direct(self, inputs, **kwargs):
            return [torch.arange(2), torch.arange(2)], inputs

    class FakeVAE:
        calls = 0

        def decode(self, tokens):
            length = (8, 3)[self.calls]
            self.calls += 1
            return torch.ones(1, length, 2)

    class FakeModel:
        device = torch.device("cpu")
        lm = FakeLM()
        vae = FakeVAE()
        hparams = type("HParams", (), {
            "codebook_size": 512,
            "cfg": type("Cfg", (), {
                "model": type("Model", (), {
                    "params": type("Params", (), {"condition": "caption"})()
                })()
            })(),
        })()

    class FakeDataModule:
        nfeats = 2

        @staticmethod
        def renorm4t2m(features):
            return features

    features, evaluator_lengths = module.generate_features(
        FakeModel(), FakeDataModule(), ["a", "b"], "categorical", 10,
        reference_lengths=torch.tensor([4, 5]),
    )
    # The evaluator receives the same full batch sequence shape as SATO++.
    assert features.shape == (2, 10, 2)
    # Only the lengths consumed by TM2T follow official val_t2m_forward.
    assert evaluator_lengths.tolist() == [4, 3]


def test_generation_audit_replay_decodes_exact_tokens_and_validates_identity():
    class FakeVAE:
        @staticmethod
        def decode(tokens):
            length = 4 * len(tokens)
            value = float(tokens.sum())
            return torch.full((1, length, 2), value)

    class FakeModel:
        device = torch.device("cpu")
        vae = FakeVAE()
        hparams = type("HParams", (), {"codebook_size": 512})()

    class FakeDataModule:
        nfeats = 2

        @staticmethod
        def renorm4t2m(features):
            return features

    rows = [
        {
            "sample_offset": 4,
            "loader_name": "m1",
            "motion_id": "m1",
            "level": "P1",
            "input_text": "walk",
            "motion_tokens": [1, 2],
            "motion_token_count": 2,
            "raw_generated_frame_count": 8,
            "generated_frame_count": 8,
            "evaluator_frame_count": 8,
        },
        {
            "sample_offset": 5,
            "loader_name": "m2",
            "motion_id": "m2",
            "level": "P2",
            "input_text": "turn",
            "motion_tokens": [3, 4, 5],
            "motion_token_count": 3,
            "raw_generated_frame_count": 12,
            "generated_frame_count": 10,
            "evaluator_frame_count": 10,
        },
    ]
    features, lengths = module.decode_audited_features(
        FakeModel(),
        FakeDataModule(),
        rows,
        ["walk", "turn"],
        ["m1", "m2"],
        ["m1", "m2"],
        ["P1", "P2"],
        max_frames=10,
        start_offset=4,
    )
    assert features.shape == (2, 10, 2)
    assert lengths.tolist() == [8, 10]
    assert torch.all(features[0, :8] == 3)
    assert torch.all(features[1] == 12)

    bad_rows = [dict(rows[0]), dict(rows[1])]
    bad_rows[1]["input_text"] = "wrong"
    with pytest.raises(module.LoaderEvaluationError, match="identity mismatch"):
        module.decode_audited_features(
            FakeModel(), FakeDataModule(), bad_rows,
            ["walk", "turn"], ["m1", "m2"], ["m1", "m2"],
            ["P1", "P2"], max_frames=10, start_offset=4,
        )


def test_generation_replay_payload_requires_complete_ordered_roles():
    def rows(role):
        return [
            {
                "sample_offset": offset,
                "generation_role": role,
                "motion_tokens": [offset + 1],
                "motion_token_count": 1,
            }
            for offset in range(2)
        ]

    payload = {"original": rows("original"), "perturbed": rows("perturbed")}
    replay = module.validate_generation_replay_payload(payload, 2)
    assert replay["original"][1]["motion_tokens"] == [2]
    payload["perturbed"][1]["sample_offset"] = 3
    with pytest.raises(module.LoaderEvaluationError, match="order mismatch"):
        module.validate_generation_replay_payload(payload, 2)


def test_length_modes_distinguish_sato_and_motiongpt_contracts():
    parser = module.build_parser()
    rng_action = next(item for item in parser._actions if item.dest == "pair_rng_mode")
    assert rng_action.default == "sequential"
    action = next(item for item in parser._actions if item.dest == "length_mode")
    assert action.default == "generated"
    assert tuple(action.choices) == ("reference", "generated")
    spacy_action = next(item for item in parser._actions if item.dest == "spacy_model")
    assert spacy_action.default == "en_core_web_sm"
    replay_action = next(
        item for item in parser._actions if item.dest == "generation_audit_input"
    )
    assert replay_action.default is None
    retrieval_action = next(
        item for item in parser._actions
        if item.dest == "retrieval_embedding_output"
    )
    assert retrieval_action.default is None
    assert module.clamp_evaluator_lengths([10, 2], [4, 8]).tolist() == [4, 2]
    with pytest.raises(module.LoaderEvaluationError, match="equal-size"):
        module.clamp_evaluator_lengths([10], [4, 8])


def test_official_prompt_does_not_use_random_instruction_templates():
    cfg = type("Cfg", (), {
        "model": type("Model", (), {
            "params": type("Params", (), {"condition": "text"})()
        })(),
        "DATASET": {"TASK_PATH": ""},
    })()
    tasks = module.load_task(cfg, 2, prompt_mode="official-default")
    assert tasks == [
        {"input": ["Generate motion: <Caption_Placeholder>"], "output": [""]},
        {"input": ["Generate motion: <Caption_Placeholder>"], "output": [""]},
    ]


def test_recompute_perturbed_retrieval_from_lightweight_npz(tmp_path):
    path = tmp_path / "retrieval_embeddings.npz"
    names = [f"motion-{index}" for index in range(4)]
    embeddings = np.arange(4, dtype=np.float32)[:, None]
    np.savez_compressed(
        path,
        perturbed_text_embedding=embeddings,
        perturbed_motion_embedding=embeddings.copy(),
        batch_sizes=np.asarray([4], dtype=np.int64),
        loader_name=np.asarray(names),
        schema_version=np.asarray(
            "tm2t-perturbed-retrieval-embeddings/v1"
        ),
        seed=np.asarray(123, dtype=np.int64),
        sample_count=np.asarray(4, dtype=np.int64),
        batch_size=np.asarray(4, dtype=np.int64),
        sample_index_sha256=np.asarray("sample-index"),
        protocol_fingerprint=np.asarray("protocol"),
    )

    metrics, metadata = aggregate_module.recompute_perturbed_retrieval(
        path, expected_count=4, expected_batch_size=4
    )

    assert metrics == {
        "top1_pert": 1.0,
        "top2_pert": 1.0,
        "top3_pert": 1.0,
        "matching_pert": 0.0,
    }
    assert metadata["counts"] == [4, 4, 4]
    assert metadata["name_order_sha256"] == aggregate_module.canonical_sha256(
        names
    )


def test_retrieval_metric_tolerance_keeps_rprecision_strict():
    float32_noise = {
        "top1_pert": 0.0,
        "top2_pert": 0.0,
        "top3_pert": 0.0,
        "matching_pert": 5e-7,
    }
    assert aggregate_module.retrieval_metric_failures(float32_noise) == {}

    rprecision_drift = dict(float32_noise, top1_pert=1e-9)
    assert aggregate_module.retrieval_metric_failures(rprecision_drift) == {
        "top1_pert": 1e-9,
    }

    matching_drift = dict(float32_noise, matching_pert=2e-6)
    assert aggregate_module.retrieval_metric_failures(matching_drift) == {
        "matching_pert": 2e-6,
    }


def test_motionx322_aggregate_uses_population_stats_and_strict_contract(tmp_path, monkeypatch):
    monkeypatch.setitem(aggregate_module.DATASETS, "motionx322", 2)
    index_dir = tmp_path / "sample_indices"
    index_dir.mkdir()
    metrics = {name: float(i + 1) for i, name in enumerate(aggregate_module.METRICS)}
    for seed in range(123, 143):
        index_hash = f"index-{seed}"
        samples = [{
            "loader_name": f"motion-{i}",
            "motion_id": f"motion-{i}",
            "caption": f"original text {i}",
            "perturbed_caption": f"perturbed text {i}",
            "level": "P1",
        } for i in range(2)]
        def audit_rows(role, text_key):
            return [{
                "sample_offset": i,
                "loader_name": sample["loader_name"],
                "motion_id": sample["motion_id"],
                "level": sample["level"],
                "generation_role": role,
                "input_text": sample[text_key],
                "fulfilled_prompt": f"Generate motion: {sample[text_key]}",
                "motion_tokens": [1, 2],
                "motion_token_count": 2,
                "generated_frame_count": 8,
                "evaluator_frame_count": 8,
            } for i, sample in enumerate(samples)]
        audit_path = tmp_path / f"seed{seed}_generation_audit.json"
        audit_path.write_text(json.dumps({
            "original": audit_rows("original", "caption"),
            "perturbed": audit_rows("perturbed", "perturbed_caption"),
        }))
        result = {
            "dataset": "motionx322",
            "seed": seed,
            "matched_count": 2,
            "total_count": 2,
            "unmatched_count": 0,
            "runtime_fallback_count": 0,
            "perturbed_text_retrieval": {
                "sample_count": 2,
                "batch_count": 1,
                "batch_size": 32,
                "text_source": "resolved text_pair perturbation",
                "motion_source": (
                    "motion generated from the same resolved text_pair perturbation"
                ),
                "evaluator": (
                    "standard TM2T TextEncoderBiGRUCo/MotionEncoderBiGRUCo"
                ),
                "word_vectorizer": "glove/our_vab",
                "spacy_model": "en_core_web_sm",
                "max_text_len": 20,
                "alphabetic_only": False,
                "caption_preprocessing": (
                    "strip whitespace; remove hyphens; skip spaces/punctuation; "
                    "lemmatize nouns/verbs except literal lowercase left"
                ),
                "retrieval_scope": "within each evaluator-loader batch",
            },
            "sample_index_sha256": index_hash,
            "protocol": {
                "sampling": aggregate_module.EXPECTED_SAMPLING,
                "pair_rng_mode": "sequential",
                "generation_order": "original-first",
                "prompt": {"mode": "official-default"},
                "generation_length_mode": "generated",
                "evaluator_length_mode": "generated_length_capped_to_batch_seq",
            },
            "generation_audit": str(audit_path),
            "generation_replay": {
                "mode": "exact audited motion-token replay",
                "path": str(audit_path.resolve()),
                "sha256": aggregate_module.file_sha256(audit_path),
                "original_count": 2,
                "perturbed_count": 2,
            },
            "protocol_fingerprint": "protocol",
            "artifacts": {
                "checkpoint": {"sha256": "checkpoint"},
                "evaluator_checkpoint": {"sha256": "evaluator"},
            },
            "metrics": {**metrics, "FID": float(seed)},
        }
        (tmp_path / f"seed{seed}.json").write_text(json.dumps(result))
        (index_dir / f"seed{seed}.json").write_text(json.dumps({
            "count": 2, "sample_index_sha256": index_hash, "samples": samples,
        }))
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(json.dumps({
        "protocol": {
            "seeds": list(range(123, 143)),
            "samples_per_seed": 2,
        },
        "model": {
            "checkpoint_sha256": "checkpoint",
            "uses_t2mgpt_generator_checkpoint": False,
        },
    }))
    rows, summary = aggregate_module.aggregate(
        tmp_path, "motionx322", source_manifest=source_manifest
    )
    with pytest.raises(ValueError, match="lacks a retrieval embedding dump"):
        aggregate_module.aggregate(
            tmp_path,
            "motionx322",
            source_manifest=source_manifest,
            require_retrieval_embeddings=True,
        )
    assert len(rows) == 20
    assert summary["std_ddof"] == 0
    assert summary["variance_ddof"] == 0
    assert summary["summary"]["FID"]["std"] == pytest.approx(
        np.std(np.arange(123, 143), ddof=0)
    )
    assert summary["summary"]["FID"]["variance"] == pytest.approx(
        np.var(np.arange(123, 143), ddof=0)
    )
    assert summary["summary"]["FID"]["sample_std"] == pytest.approx(
        np.std(np.arange(123, 143), ddof=1)
    )
    expected_ci95 = 1.96 * np.std(np.arange(123, 143), ddof=0) / np.sqrt(20)
    assert summary["summary"]["FID"]["ci95"] == pytest.approx(expected_ci95)
    assert summary["summary"]["FID"]["ci95_low"] == pytest.approx(
        np.mean(np.arange(123, 143)) - expected_ci95
    )
    assert summary["summary"]["FID"]["ci95_high"] == pytest.approx(
        np.mean(np.arange(123, 143)) + expected_ci95
    )
    assert "top1_pert" in summary["summary"]
    assert "matching_pert" in summary["summary"]
    assert summary["perturbed_text_retrieval"]["sample_count"] == 2
    assert summary["source_manifest"]["checkpoint_sha256"] == "checkpoint"
    assert summary["generation_source"] == {
        "mode": "audited_token_replay",
        "audited_token_replay_count": 20,
        "distinct_audit_checksum_count": 1,
        "checksums_verified": True,
    }
    seed123_path = tmp_path / "seed123.json"
    seed123_result = json.loads(seed123_path.read_text())
    corrupted = json.loads(json.dumps(seed123_result))
    corrupted["generation_replay"]["sha256"] = "wrong"
    seed123_path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="replay checksum mismatch"):
        aggregate_module.aggregate(
            tmp_path, "motionx322", source_manifest=source_manifest
        )
    seed123_path.write_text(json.dumps(seed123_result))
    source_manifest.write_text(json.dumps({
        "protocol": {
            "seeds": list(range(123, 143)),
            "samples_per_seed": 2,
        },
        "model": {
            "checkpoint_sha256": "wrong",
            "uses_t2mgpt_generator_checkpoint": False,
        },
    }))
    with pytest.raises(ValueError, match="checkpoint hash differs"):
        aggregate_module.aggregate(
            tmp_path, "motionx322", source_manifest=source_manifest
        )


def test_aggregate_rejects_reference_length_contract(tmp_path, monkeypatch):
    monkeypatch.setitem(aggregate_module.DATASETS, "motionx322", 2)
    index_dir = tmp_path / "sample_indices"
    index_dir.mkdir()
    metrics = {name: 1.0 for name in aggregate_module.METRICS}
    for seed in range(123, 143):
        index_hash = f"index-{seed}"
        result = {
            "dataset": "motionx322", "seed": seed,
            "matched_count": 2, "total_count": 2,
            "unmatched_count": 0, "runtime_fallback_count": 0,
            "sample_index_sha256": index_hash,
            "protocol": {
                "sampling": aggregate_module.EXPECTED_SAMPLING,
                "pair_rng_mode": "sequential",
                "generation_order": "original-first",
                "prompt": {"mode": "official-default"},
                "generation_length_mode": "reference",
                "evaluator_length_mode": "min(generated_length,reference_length)",
            },
            "protocol_fingerprint": "legacy-protocol",
            "artifacts": {
                "checkpoint": {"sha256": "checkpoint"},
                "evaluator_checkpoint": {"sha256": "evaluator"},
            },
            "metrics": metrics,
        }
        (tmp_path / f"seed{seed}.json").write_text(json.dumps(result))
        (index_dir / f"seed{seed}.json").write_text(json.dumps({
            "count": 2, "sample_index_sha256": index_hash,
            "samples": [{}, {}],
        }))
    with pytest.raises(ValueError, match="generation length mode"):
        aggregate_module.aggregate(tmp_path, "motionx322")


def test_motionx_launcher_locks_sato_pp_parity_contract():
    launcher = (Path(__file__).parents[1] / "MotionGPT" / "scripts" /
                "run_motionx322_table3_eval.sh").read_text(encoding="utf-8")
    for fragment in (
        "--pair-rng-mode sequential",
        "--sampling-kwargs-mode explicit",
        "--prompt-mode official-default",
        "--generation-order original-first",
        "--length-mode generated",
        "--generation-audit-batches 0",
    ):
        assert fragment in launcher


def test_motionx_level_launchers_lock_independent_7679_contract():
    root = Path(__file__).parents[1]
    t2m_launcher = (
        root / "tools" / "run_motionx322_level_t2mgpt_satopp.sh"
    ).read_text(encoding="utf-8")
    motiongpt_launcher = (
        root / "tools" / "run_motionx322_level_motiongpt.sh"
    ).read_text(encoding="utf-8")
    orchestrator = (
        root / "tools" / "run_motionx322_level_rprecision.sh"
    ).read_text(encoding="utf-8")

    for fragment in (
        "test_perb_gemini31_max75_${LOWER}.json",
        "--expected-sample-index",
        "--allowed-levels \"$LEVEL\"",
        "--caption-locked",
        "--caption-locked-length-policy repair-all",
        "--sampling-pair-rng-mode sequential",
        "--cfg-scale 1.3",
        "--spacy-model en_core_web_sm",
        "--retrieval-embedding-dir",
        "--generation-audit-dir",
    ):
        assert fragment in t2m_launcher
    for fragment in (
        "--expected-samples 7679",
        "test_perb_gemini31_max75_${LOWER}.json",
        "--loader-mode caption-locked",
        "--allowed-levels \"$LEVEL\"",
        "--caption-length-policy repair-all",
        "--pair-rng-mode sequential",
        "--spacy-model en_core_web_sm",
        "--generation-audit-batches 0",
        "--retrieval-embedding-output",
    ):
        assert fragment in motiongpt_launcher
    assert "MOTIONX_LEVEL_GPUS:-0 1 2 3 4 5 6 7" in orchestrator
    assert "MOTIONX_LEVEL_GPU_MIN_FREE_MIB:-8192" in t2m_launcher
    assert "MOTIONX_LEVEL_GPU_MIN_FREE_MIB:-12288" in motiongpt_launcher
    assert "--query-gpu=memory.free" in t2m_launcher
    assert "--query-gpu=memory.free" in motiongpt_launcher
    assert "aggregate_motionx322_level_rprecision.py" in orchestrator
    assert "motiongpt_satopp" in orchestrator


def test_humanml_single_level_launchers_lock_independent_pool_contract():
    root = Path(__file__).parents[1]
    t2m_launcher = (
        root / "tools" / "run_humanml3d_table1_level_t2mgpt_satopp.sh"
    ).read_text(encoding="utf-8")
    motiongpt_launcher = (
        root / "tools" / "run_humanml3d_table1_level_motiongpt.sh"
    ).read_text(encoding="utf-8")
    orchestrator = (
        root / "tools" / "run_humanml3d_table1_level_rprecision.sh"
    ).read_text(encoding="utf-8")

    for fragment in (
        "--expected-sample-index",
        "--allowed-levels",
        "--caption-locked",
        "--caption-locked-length-policy repair-all",
        "--sampling-pair-rng-mode sequential",
        "--cfg-scale 1.0",
        "--diagnostics-dir",
    ):
        assert fragment in t2m_launcher
    for fragment in (
        "LOADER_MODE=caption-locked",
        "CAPTION_LENGTH_POLICY=repair-all",
        'ALLOWED_LEVELS="$LEVEL"',
        "PAIR_RNG_MODE=sequential",
        "LENGTH_MODE=generated",
        "SAMPLING_KWARGS_MODE=explicit",
        "PROMPT_MODE=official-default",
        "retrieval_embeddings/seed${seed}.npz",
    ):
        assert fragment in motiongpt_launcher
    assert "aggregate_humanml3d_table1_level_rprecision.py" in orchestrator
    assert "smoke_jobs" in orchestrator
    assert "formal_jobs" in orchestrator
    assert "job_is_complete" in orchestrator
    assert "active=$((active - 1))" in orchestrator
    assert "((active -= 1))" not in orchestrator


def test_humanml_launcher_uses_caption_locked_p123_contract():
    launcher = (Path(__file__).parents[1] / "MotionGPT" / "scripts" /
                "run_humanml3d_loader_eval_shard.sh").read_text(encoding="utf-8")
    for fragment in (
        'LOADER_MODE="${LOADER_MODE:-caption-locked}"',
        '--loader-mode "${LOADER_MODE}"',
        'ALLOWED_LEVELS="${ALLOWED_LEVELS:-P1 P2 P3}"',
        "--allowed-levels ${ALLOWED_LEVELS}",
        '--caption-length-policy "${CAPTION_LENGTH_POLICY}"',
        '--pair-file "${PAIR_FILE}"',
        'GENERATION_AUDIT_INPUT_ROOT="${GENERATION_AUDIT_INPUT_ROOT:-}"',
        '--generation-audit-input "${generation_audit_input}"',
        '"${generation_audit_args[@]}"',
        'RETRIEVAL_EMBEDDING_ROOT="${RETRIEVAL_EMBEDDING_ROOT:-${OUTPUT_ROOT}/retrieval_embeddings}"',
        '--retrieval-embedding-output "${retrieval_embedding}"',
        "humanml3d_caption_locked_p123_4184_20seed_20260810",
        '--num-workers "${NUM_WORKERS}"',
    ):
        assert fragment in launcher
    table_launcher = (
        Path(__file__).parents[1] / "MotionGPT" / "scripts" /
        "run_humanml3d_table1_eval.sh"
    ).read_text(encoding="utf-8")
    assert "--require-retrieval-embeddings" in table_launcher
