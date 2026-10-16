import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


class _FakeToken:
    def __init__(self, text, pos="OTHER", lemma=None,
                 is_space=False, is_punct=False):
        self.text = text
        self.pos_ = pos
        self.lemma_ = text if lemma is None else lemma
        self.is_space = is_space
        self.is_punct = is_punct


class _FakeNLP:
    def __init__(self, documents):
        self.documents = documents

    def __call__(self, caption):
        return self.documents[caption]


class _FakeVectorizer:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, token):
        word, _ = token.split("/")
        value = self.values.get(word, self.values["unk"])
        return (
            np.asarray([value], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
        )


class _FakeTextEncoder:
    def __call__(self, word_embeddings, pos_one_hots, sent_lens):
        lengths = sent_lens.cpu().tolist()
        assert lengths == sorted(lengths, reverse=True)
        return word_embeddings[:, 1, :]


class _FakeEvaluator:
    device = torch.device("cpu")
    text_encoder = _FakeTextEncoder()


def test_repaired_sato_mix_asset_is_frozen_and_matches_pair_multiplicity():
    root = Path(__file__).resolve().parents[1]
    asset_root = root / (
        "MotionGPT/experiments/robustness/"
        "humanml3d_caption_locked_sato_mix_repair_all_p123_4631_20260810"
    )
    payload = json.loads((asset_root / "sample_index_seed123.json").read_text())
    pair_paths = list(asset_root.glob("pairs*.json"))
    assert len(pair_paths) == 1
    pairs = json.loads(pair_paths[0].read_text())
    samples = payload["samples"]

    encoded = json.dumps(
        samples, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert len(samples) == len(pairs) == 4631
    assert hashlib.sha256(encoded).hexdigest() == payload["sample_index_sha256"]

    def signature(row, index=False):
        return (
            str(row["motion_id"]),
            str(row["caption"] if index else row["text_orig"]),
            str(row["perturbed_caption"] if index else row["text_pair"]),
            str(row["level"]),
        )

    from collections import Counter
    assert Counter(map(signature, pairs)) == Counter(
        signature(row, index=True) for row in samples
    )
    assert Counter(row["level"] for row in samples) == {
        "P1": 2138,
        "P2": 1240,
        "P3": 1253,
    }


def test_independent_level_asset_audit_locks_shared_source_order(tmp_path):
    from tools.audit_humanml3d_table1_level_assets import (
        audit_level_assets,
        canonical_sha256,
    )

    pair_paths = {}
    index_paths = {}
    for level in ("P1", "P2", "P3"):
        pairs = [{
            "motion_id": f"motion-{index}",
            "text_orig": f"original caption {index}",
            "text_pair": f"{level} caption {index}",
            "relation": "preserve",
            "level": level,
            "pair_id": index + 1,
        } for index in range(4)]
        samples = [{
            "loader_name": row["motion_id"],
            "motion_id": row["motion_id"],
            "caption": row["text_orig"],
            "perturbed_caption": row["text_pair"],
            "level": level,
            "motion_length": 40,
            "pair_id": row["pair_id"],
            "pair_text_orig": row["text_orig"],
            "caption_pair_exact": True,
            "length_repair": {"policy": "repair-all", "repair_type": "none"},
        } for row in pairs]
        pair_path = tmp_path / f"pairs_{level.lower()}.json"
        index_path = tmp_path / f"index_{level.lower()}.json"
        pair_path.write_text(json.dumps(pairs))
        index_path.write_text(json.dumps({
            "count": len(samples),
            "protocol_fingerprint": f"protocol-{level}",
            "sample_index_sha256": canonical_sha256(samples),
            "samples": samples,
        }))
        pair_paths[level] = pair_path
        index_paths[level] = index_path

    manifest = audit_level_assets(
        pair_paths, index_paths, expected_count=4, batch_size=4
    )
    repeated = audit_level_assets(
        pair_paths, index_paths, expected_count=4, batch_size=4
    )

    assert manifest == repeated
    assert manifest["batch_sizes"] == [4]
    assert manifest["common_source_order_sha256"]
    assert manifest["assets"]["P1"]["unique_motion_ids"] == 4
    assert manifest["assets"]["P3"]["level_counts"] == {"P3": 4}


def test_level_rprecision_summary_uses_sample_standard_deviation():
    from tools.aggregate_humanml3d_table1_level_rprecision import summarize_runs

    rows = []
    for level in ("P1", "P2", "P3"):
        for model in ("t2mgpt", "motiongpt", "satopp"):
            for seed, value in ((123, 0.25), (124, 0.75)):
                rows.append({
                    "level": level,
                    "model": model,
                    "method": model,
                    "seed": seed,
                    "samples": 4,
                    "top1_pert": value,
                    "top2_pert": value,
                    "top3_pert": value,
                    "sample_index_sha256": f"index-{level}",
                    "online_offline_max_abs_error": 0.0,
                })

    summary = summarize_runs(rows)
    first = summary[0]
    assert len(summary) == 9
    assert first["top1_pert_mean"] == 0.5
    assert first["top1_pert_sample_std"] == np.std([0.25, 0.75], ddof=1)
    assert first["top1_pert_population_std"] == 0.25


def test_repeat_fixed_index_accepts_explicit_single_level_contract():
    from ablation.repeat20_perturb_fast_batch import (
        validate_pair_mix_matches_index,
    )

    pairs = [{
        "motion_id": "motion-1",
        "text_orig": "original",
        "text_pair": "perturbed",
        "level": "P2",
    }]
    samples = [{
        "motion_id": "motion-1",
        "caption": "original",
        "perturbed_caption": "perturbed",
        "level": "P2",
    }]

    assert validate_pair_mix_matches_index(
        pairs, samples, expected_levels=["P2"]
    ) == {"P2": 1}


def test_repeat_fixed_index_resume_accepts_prior_formal_seed(tmp_path):
    from ablation import repeat20_perturb_fast_batch as repeat_eval

    checkpoint_path = tmp_path / "checkpoint.pth"
    model_dir = tmp_path / "model"
    index_path = model_dir / "sample_indices" / "seed_123.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text("{}")
    row = {field: "0" for field in repeat_eval.RUN_FIELDS}
    row.update({
        "model": "model",
        "kind": "baseline",
        "run_idx": "1",
        "seed": "123",
        "checkpoint": checkpoint_path.name,
        "iter": "0",
        "sample_index_sha256": "frozen-index",
        "path": str(checkpoint_path),
    })

    repeat_eval.validate_existing_rows(
        [row],
        {"name": "model", "kind": "baseline"},
        {
            "path": str(checkpoint_path),
            "checkpoint": checkpoint_path.name,
            "iter": 0,
        },
        [124],
        model_dir,
        True,
        expected_sample_index_sha256="frozen-index",
    )


def test_motionx_t2m_command_audit_accepts_launcher_environment_prefix(tmp_path):
    import shlex

    from tools import aggregate_motionx322_level_rprecision as aggregate

    group_root = tmp_path / "group"
    pair_path = tmp_path / "pairs.json"
    index_path = tmp_path / "index.json"
    spec_path = tmp_path / "spec.json"
    command_path = group_root / "command_seed123_123.txt"
    command_path.parent.mkdir(parents=True)
    tokens = [
        "CUDA_VISIBLE_DEVICES=1",
        str(aggregate.T2M_PYTHON),
        str(aggregate.REPO_ROOT / "ablation/repeat20_perturb_fast_batch.py"),
        "--spec-json", str(spec_path),
        "--model", aggregate.T2M_INTERNAL_MODELS["t2mgpt"],
        "--pair-data-path", str(pair_path),
        "--expected-sample-index", str(index_path),
        "--allowed-levels", "P1",
        "--out-dir", str(group_root),
        "--report-md", str(group_root / "report.md"),
        "--start-seed", "123",
        "--num-repeats", "1",
        "--batch-size", "32",
        "--num-workers", "0",
        "--dataset-split", "test",
        "--spacy-model", "en_core_web_sm",
        "--caption-locked",
        "--caption-locked-length-policy", "repair-all",
        "--fast-batch",
        "--sampling-pair-rng-mode", "sequential",
        "--cfg-scale", "1.3",
        "--cfg-uncond", "zero",
        "--sample-mode", "categorical",
        "--sample-temperature", "1",
        "--top-k", "0",
        "--top-p", "1",
        "--retrieval-embedding-dir", str(group_root / "retrieval_embeddings"),
        "--generation-audit-dir", str(group_root / "generation_audits"),
        "--wandb-mode", "disabled",
    ]
    command_path.write_text(shlex.join(tokens) + "\n")
    asset_manifest = {
        "pairs_and_indices": {
            "P1": {
                "pair_path": str(pair_path),
                "index_path": str(index_path),
            }
        },
        "t2m_family_spec": {"path": str(spec_path)},
    }

    audit = aggregate.validate_t2m_command(
        command_path,
        group_root=group_root,
        model_key="t2mgpt",
        level="P1",
        seeds=[123],
        asset_manifest=asset_manifest,
    )

    assert audit["command"][0] == "CUDA_VISIBLE_DEVICES=1"
    assert "--retrieval-embedding-dir" in audit["command"]

    command_path.write_text(
        shlex.join(tokens + ["--diagnostics-dir", str(group_root / "forbidden")])
        + "\n"
    )
    with pytest.raises(ValueError, match="unsupported flags"):
        aggregate.validate_t2m_command(
            command_path,
            group_root=group_root,
            model_key="t2mgpt",
            level="P1",
            seeds=[123],
            asset_manifest=asset_manifest,
        )


def test_motionx_t2m_embedding_audit_uses_lightweight_retrieval_dump(tmp_path):
    from tools import aggregate_motionx322_level_rprecision as aggregate

    samples = [{
        "loader_name": f"motion-{index}",
        "motion_id": f"motion-{index}",
        "caption": f"original {index}",
        "perturbed_caption": f"perturbed {index}",
        "level": "P1",
        "motion_length": 40,
    } for index in range(4)]
    sample_hash = aggregate.canonical_sha256(samples)
    sample_index_path = tmp_path / "sample_index.json"
    sample_index_path.write_text(json.dumps({
        "model": aggregate.T2M_INTERNAL_MODELS["t2mgpt"],
        "seed": 123,
        "sample_count": len(samples),
        "sample_index_sha256": sample_hash,
        "samples": samples,
    }))

    embeddings = np.eye(len(samples), dtype=np.float32)
    embedding_path = tmp_path / "retrieval.npz"
    np.savez_compressed(
        embedding_path,
        perturbed_text_embedding=embeddings,
        perturbed_motion_embedding=embeddings,
        batch_sizes=np.asarray([len(samples)], dtype=np.int64),
        loader_name=np.asarray([row["loader_name"] for row in samples]),
        motion_id=np.asarray([row["motion_id"] for row in samples]),
        caption=np.asarray([row["caption"] for row in samples]),
        perturbed_caption=np.asarray([
            row["perturbed_caption"] for row in samples
        ]),
        level=np.asarray([row["level"] for row in samples]),
        schema_version=np.asarray("tm2t-perturbed-retrieval-embeddings/v1"),
        seed=np.asarray(123, dtype=np.int64),
        sample_count=np.asarray(len(samples), dtype=np.int64),
        batch_size=np.asarray(32, dtype=np.int64),
        sample_index_sha256=np.asarray(sample_hash),
        protocol_fingerprint=np.asarray("protocol-fingerprint"),
    )
    row = {
        "model": aggregate.T2M_INTERNAL_MODELS["t2mgpt"],
        "seed": 123,
        "sample_index_sha256": sample_hash,
        "top1_pert": 1.0,
        "top2_pert": 1.0,
        "top3_pert": 1.0,
        "matching_pert": 0.0,
    }

    audit = aggregate.audit_t2m_seed_embeddings(
        embedding_path,
        sample_index_path,
        row,
        expected_samples=len(samples),
        expected_level="P1",
        expected_sample_index_sha256=sample_hash,
        expected_protocol_fingerprint="protocol-fingerprint",
    )

    assert audit["counts"] == [4, 4, 4]
    assert audit["max_abs_metric_error"] == 0.0


def test_tm2t_text_processor_uses_reference_humanml_lemma_pos_rules():
    from utils.sato_pp_eval import TM2TTextProcessor

    nlp = _FakeNLP({
        "A leftward Walker moved left 90!": [
            _FakeToken("A", "DET", "a"),
            _FakeToken("leftward", "ADV"),
            _FakeToken("Walker", "NOUN", "walker"),
            _FakeToken("moved", "VERB", "move"),
            _FakeToken("left", "NOUN", "leave"),
            _FakeToken("90", "NUM"),
            _FakeToken("!", "PUNCT", is_punct=True),
        ],
    })
    processor = TM2TTextProcessor(
        _FakeVectorizer({
            "A": 1,
            "leftward": 2,
            "walker": 3,
            "move": 4,
            "left": 5,
            "unk": 0,
        }),
        max_text_len=20,
        nlp=nlp,
    )

    assert processor.tokenize("A left-ward Walker moved left 90!") == [
        "A/DET",
        "leftward/ADV",
        "walker/NOUN",
        "move/VERB",
        "left/NOUN",
    ]


def test_tm2t_text_processor_restores_caption_order_after_length_sort():
    from utils.sato_pp_eval import TM2TTextProcessor

    nlp = _FakeNLP({
        "short": [_FakeToken("short", "ADJ")],
        "long phrase": [
            _FakeToken("long", "ADJ"),
            _FakeToken("phrase", "NOUN", "phrase"),
        ],
    })
    processor = TM2TTextProcessor(
        _FakeVectorizer({"short": 11, "long": 22, "phrase": 33, "unk": 0}),
        nlp=nlp,
    )

    embeddings = processor.encode(_FakeEvaluator(), ["short", "long phrase"])
    assert embeddings[:, 0].tolist() == [11.0, 22.0]


def test_repeat_aggregate_reports_population_statistics():
    from ablation import repeat20_perturb_fast_batch as repeat_eval

    rows = []
    for value in (1.0, 3.0):
        row = {
            "model": "model",
            "kind": "sato_pp",
            "checkpoint": "checkpoint",
            "path": "path",
        }
        row.update({metric: value for metric in repeat_eval.METRIC_FIELDS})
        rows.append(row)

    aggregate, wide = repeat_eval.aggregate_rows(rows)
    fid = next(row for row in aggregate if row["metric"] == "fid")

    assert fid["mean"] == 2.0
    assert fid["std"] == 1.0
    assert fid["var"] == 1.0
    assert fid["sample_std"] == np.sqrt(2.0)
    assert fid["sample_var"] == 2.0
    assert wide[0]["fid_population_std"] == 1.0
    assert wide[0]["fid_population_var"] == 1.0


def test_sato_diagnostic_audit_recomputes_batch_local_retrieval(tmp_path):
    from tools.merge_repeat_eval_shards import (
        audit_seed_diagnostics,
        canonical_sha256,
    )

    samples = [{
        "loader_name": f"motion-{index}",
        "motion_id": f"motion-{index}",
        "caption": f"original {index}",
        "perturbed_caption": f"perturbed {index}",
        "level": "P1",
    } for index in range(4)]
    sample_hash = canonical_sha256(samples)
    sample_index_path = tmp_path / "seed_123.json"
    sample_index_path.write_text(json.dumps({
        "sample_index_sha256": sample_hash,
        "samples": samples,
    }))
    diagnostic_path = tmp_path / "seed123.npz"
    embeddings = np.arange(4, dtype=np.float32)[:, None]
    np.savez_compressed(
        diagnostic_path,
        perturbed_text_embedding=embeddings,
        perturbed_embedding=embeddings.copy(),
        loader_name=np.asarray([row["loader_name"] for row in samples]),
        motion_id=np.asarray([row["motion_id"] for row in samples]),
        caption=np.asarray([row["caption"] for row in samples]),
        perturbed_caption=np.asarray([
            row["perturbed_caption"] for row in samples
        ]),
        level=np.asarray([row["level"] for row in samples]),
    )
    row = {
        "seed": 123,
        "sample_index_sha256": sample_hash,
        "top1_pert": 1.0,
        "top2_pert": 1.0,
        "top3_pert": 1.0,
        "matching_pert": 0.0,
    }

    audit = audit_seed_diagnostics(
        diagnostic_path,
        sample_index_path,
        row,
        expected_samples=4,
        batch_size=4,
        require_full_batches=True,
    )

    assert audit["batch_sizes"] == [4]
    assert audit["top1_count"] == 4
    assert audit["max_abs_metric_error"] == 0.0


def test_perturbed_retrieval_is_text_to_generated_motion_and_cumulative():
    from utils.sato_pp_eval import calculate_R_precision

    perturbed_text = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    perturbed_generated_motion = np.asarray([[0.0], [3.0], [5.0], [7.0]])

    counts, matching = calculate_R_precision(
        perturbed_text,
        perturbed_generated_motion,
        top_k=3,
        sum_all=True,
    )
    reverse_counts, _ = calculate_R_precision(
        perturbed_generated_motion,
        perturbed_text,
        top_k=3,
        sum_all=True,
    )

    assert counts.tolist() == [1, 2, 3]
    assert reverse_counts.tolist() == [2, 3, 4]
    assert matching == 9.0
