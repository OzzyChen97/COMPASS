import ast
import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
SATO = ROOT / "Stable-Text-to-Motion-Framework-official"
LAUNCHER = ROOT / "tools" / "run_motionx322_sato50k.sh"
PAIR_BUILDER = ROOT / "tools" / "build_motionx322_sato_token_aligned_pairs.py"
PAIR_MANIFEST = (
    ROOT / "dataset" / "Motion-X-322"
    / "token_aligned_lexical_pairs_sato_clip_train.manifest.json"
)
PAPER_PAIR_BUILDER = (
    ROOT / "tools" / "build_motionx322_sato_paper_faithful_rsr_pairs.py"
)
PAPER_PAIR_MANIFEST = (
    ROOT / "dataset" / "Motion-X-322"
    / "paper_faithful_rsr_pairs_sato_train.manifest.json"
)
PAPER_LAUNCHER = (
    ROOT / "tools" / "run_motionx322_sato_paper_faithful_rsr.sh"
)
EVAL_LAUNCHER = ROOT / "tools" / "run_motionx322_official_sato_table3_eval.sh"


def test_motionx322_sato_launcher_contract():
    text = LAUNCHER.read_text(encoding="utf-8")
    for fragment in (
        "--dataname motionx322_75",
        '--batch-size "${BATCH_SIZE}"',
        '--total-iter "${TOTAL_ITER}"',
        "--attention-loss-weight 0.05",
        "--warm-up-iter 1000",
        "--lr 1e-4",
        'MOTIONX322_SATO_EVAL_ITER:-5000',
        'MOTIONX322_SATO_SAVE_ITER:-5000',
        'MOTIONX322_SATO_EARLY_STOPPING_PATIENCE:-6',
        'MOTIONX322_SATO_EARLY_STOPPING_MIN_VALIDATIONS:-8',
        'MOTIONX322_SATO_PGD_RADIUS:-0.01',
        'MOTIONX322_SATO_PGD_STEPS:-1',
        'MOTIONX322_SATO_PGD_STEP_SIZE:-0.005',
        'MOTIONX322_SATO_PGD_PROBABILITY:-0.25',
        "DEFAULT_TOTAL_ITER=50000",
        "DEFAULT_TOTAL_ITER=100000",
        'DEFAULT_EXP_NAME="motionx322_sato_official_earlystop_seed${SEED}"',
        'DEFAULT_EXP_NAME="motionx322_sato_token_aligned_earlystop_pgd_mild_v2_seed${SEED}"',
        "earlystop-pgd-dry-run",
        "--resume-train-state",
        "--early-stopping",
        "--early-stopping-patience",
        "--early-stopping-min-validations",
        "--pgd",
        "--pgd-radius",
        "--pgd-steps",
        "--pgd-step-size",
        "--pgd-probability",
        "--pgd-random-start",
        "DEFAULT_PRINT_ITER=100",
        "--block-size 76",
        "--K 2",
        "--pair-jsonl",
        "token_aligned_lexical_pairs_sato_clip_train.jsonl",
    ):
        assert fragment in text
    assert "--nproc_per_node=\"${NPROC}\"" in text
    assert 'GPUS="${GPUS:-0,1,2,3,4,5,6,7}"' in text
    assert 'GLOBAL_BATCH_SIZE="${MOTIONX322_SATO_GLOBAL_BATCH_SIZE:-64}"' in text
    assert "sato_pp" not in text.lower()


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_token_aligned_pair_builder_rejects_reordering_and_preserves_true_eot():
    builder = _load_module("motionx_sato_pair_builder", PAIR_BUILDER)
    tokenizer = builder.load_simple_tokenizer(
        SATO / "CLIP" / "clip" / "simple_tokenizer.py")
    train_ids = {"train_motion"}

    accepted, rejection = builder.align_row(
        {
            "pair_id": 1,
            "motion_id": "train_motion",
            "text_orig": "A person walks forward.",
            "text_pair": "A human steps ahead.",
            "relation": "preserve",
            "level": "P1",
        },
        tokenizer, train_ids, {"P1"}, 75,
    )
    assert rejection is None
    assert accepted["attention_aligned"] is True
    assert accepted["eot_position"] == len(accepted["original_bpe"]) + 1
    assert len(accepted["original_bpe"]) == len(accepted["perturbed_bpe"])
    assert all(
        len(replacement["original_bpe"]) == len(replacement["perturbed_bpe"])
        for replacement in accepted["replacements"]
    )

    rejected, reason = builder.align_row(
        {
            "pair_id": 2,
            "motion_id": "train_motion",
            "text_orig": "a woman is doing a Pelvic Floor Squeeze.",
            "text_pair": "Performing a Pelvic Floor Squeeze is a woman.",
            "relation": "preserve",
            "level": "P2",
        },
        tokenizer, train_ids, {"P1"}, 75,
    )
    assert rejected is None
    assert reason["reason"] == "disallowed_source_level"


def test_generated_pair_manifest_has_strict_train_only_invariants():
    manifest = json.loads(PAIR_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["counts"]["accepted_rows"] > 0
    assert manifest["accepted_motion_ids"] > 0
    assert manifest["allowed_levels"] == ["P1"]
    assert manifest["split_overlap"] == {
        "train_test": 0,
        "train_val": 0,
        "val_test": 0,
    }
    assert all(manifest["invariants"].values())
    assert 0 < manifest["accepted_train_motion_coverage"] < 1
    assert 0 < manifest["unchanged_train_motion_ratio"] < 1


def test_motionx_pair_loader_requires_alignment_metadata():
    dataset_text = (SATO / "dataset" / "dataset_TM_train.py").read_text(
        encoding="utf-8")
    for fragment in (
        "def load_attention_aligned_pairs(",
        "if variable_length and not row.get('attention_aligned', False):",
        "Motion-X pair rows must be produced by the strict",
        "if not pairs and not self.variable_length:",
    ):
        assert fragment in dataset_text


def test_attention_uses_true_clip_eot_and_skips_unchanged_fallback():
    train_text = (SATO / "train.py").read_text(encoding="utf-8")
    dataset_text = (SATO / "dataset" / "dataset_TM_train.py").read_text(
        encoding="utf-8")
    for fragment in (
        "eot_positions = text_token.argmax(dim=-1)",
        "final_attention[sample_index, eot_position, 1:eot_position]",
        "aligned_topk_attention_loss(",
        "K=args.K",
        "loss_attention*args.attention_loss_weight",
        "Token-aligned captions produced different CLIP EOT positions",
    ):
        assert fragment in train_text
    assert "def topK_process_paper_original" in train_text
    assert "tokens_lens = [1 + len(txt.split(' '))" in train_text
    assert "if args.paper_faithful_rsr:" in train_text
    assert "attention_aligned = caption_perb != caption" in dataset_text


def test_motionx322_sato_early_stopping_contract():
    train_text = (SATO / "train.py").read_text(encoding="utf-8")
    option_text = (SATO / "options" / "option_transformer.py").read_text(encoding="utf-8")
    for fragment in (
        "validation_metrics.jsonl",
        "training_complete.json",
        "bad_validation_checks >= args.early_stopping_patience",
        "validation_checks >= args.early_stopping_min_validations",
        "dist.broadcast(stop_tensor, src=0)",
        "resume_train_state['optimizer']",
        "Restored validation state:",
    ):
        assert fragment in train_text
    for fragment in (
        "--early-stopping",
        "--early-stopping-patience",
        "--early-stopping-min-validations",
    ):
        assert fragment in option_text


def test_motionx322_sato_pgd_contract():
    train_text = (SATO / "train.py").read_text(encoding="utf-8")
    option_text = (SATO / "options" / "option_transformer.py").read_text(encoding="utf-8")
    attack_text = (SATO / "attack.py").read_text(encoding="utf-8")
    for fragment in (
        "pgd_attacker.perturb(",
        "random.random() < args.pgd_probability",
        "PGD enabled:",
        "PGD. {pgd_ratio:.1%}",
    ):
        assert fragment in train_text
    for fragment in (
        "--pgd-radius",
        "--pgd-steps",
        "--pgd-step-size",
        "--pgd-probability",
        "--pgd-random-start",
    ):
        assert fragment in option_text
    assert "self._clip_(adv_x, x)" in attack_text
    assert "encoder.train(encoder_was_training)" in attack_text


def test_pgd_projects_to_original_feature_ball_and_restores_encoder():
    path = SATO / "attack.py"
    spec = importlib.util.spec_from_file_location("official_sato_attack", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, unused_indices, embeddings):
            return embeddings * self.scale

    encoder = FakeEncoder().train()
    original = torch.tensor([[-0.5, 0.2], [2.0, -1.0]])
    attacker = module.PGDAttacker(
        radius=0.1, steps=3, step_size=0.08, random_start=False,
        norm_type="l-infty", ascending=True,
    )
    adversarial = attacker.perturb(
        device=torch.device("cpu"),
        m_tokens_len=torch.ones(2, dtype=torch.long),
        bs=2,
        criterion=lambda unused_y, prediction, unused_lengths, unused_bs: prediction.sum(),
        x=original,
        y=torch.zeros_like(original),
        a_indices=torch.zeros_like(original, dtype=torch.long),
        encoder=encoder,
    )

    assert torch.max(torch.abs(adversarial - original)).item() <= 0.100001
    assert adversarial[0, 0].item() < 0
    assert adversarial[1, 0].item() > 1
    assert encoder.training is True
    assert all(parameter.requires_grad for parameter in encoder.parameters())


def test_paper_faithful_rsr_launcher_contract():
    text = PAPER_LAUNCHER.read_text(encoding="utf-8")
    for fragment in (
        "--paper-faithful-rsr",
        "--total-iter \"${TOTAL_ITER}\"",
        "TOTAL_ITER=100000",
        "--batch-size \"${LOCAL_BATCH_SIZE}\"",
        "--warm-up-iter 1000",
        "--lr 1e-4",
        "--lr-scheduler 60000",
        "--gamma 0.05",
        "--optimizer adamw",
        "--weight-decay 1e-6",
        "--pkeep 0.5",
        "--K 2",
        "--teacher-loss-weight 0.1",
        "--attention-loss-weight 0.2",
        "--robust-loss-weight 0.05",
        "--block-size 76",
        "paper_faithful_rsr_pairs_sato_train.jsonl",
        "motionx322_sato_paper_faithful_rsr_100k_4gpu3457_seed${SEED}",
        "experiment_manifest.json",
        "Refusing to overwrite existing experiment",
    ):
        assert fragment in text
    assert "--pgd" not in text
    assert "--early-stopping" not in text
    assert "token_aligned_earlystop" not in text


def test_paper_faithful_rsr_training_contract():
    train_text = (SATO / "train.py").read_text(encoding="utf-8")
    option_text = (SATO / "options" / "option_transformer.py").read_text(
        encoding="utf-8")
    dataset_text = (SATO / "dataset" / "dataset_TM_train.py").read_text(
        encoding="utf-8")
    for fragment in (
        "def topK_process_paper_original",
        "True if args.paper_faithful_rsr else random.choice([True, False])",
        "cls_pred_origin if args.paper_faithful_rsr else cls_pred_per",
        "topk_overlap_loss(\n            weight_ori, weight_perb, K=args.K).sum()",
        "sim_loss*args.teacher_loss_weight",
        "adv_loss*args.robust_loss_weight",
        "loss_attention*args.attention_loss_weight",
        "'paper_faithful_rsr': args.paper_faithful_rsr",
    ):
        assert fragment in train_text
    for fragment in (
        "--paper-faithful-rsr",
        "--teacher-loss-weight",
        "--attention-loss-weight",
        "--robust-loss-weight",
        "--paper-faithful-rsr uses RSR, not PGD",
        "--paper-faithful-rsr runs the published fixed 100000 iterations",
    ):
        assert fragment in option_text
    for fragment in (
        "paper_faithful_rsr=False",
        "row.get('paper_faithful_rsr', False)",
        "Paper RSR pairs must preserve whitespace-token length",
    ):
        assert fragment in dataset_text


def test_paper_faithful_rsr_builder_contract():
    builder = _load_module("motionx_sato_paper_rsr_builder", PAPER_PAIR_BUILDER)
    rng = __import__("random").Random(123)
    replacement = builder.manual_replacement("A person walks forward.", rng)
    assert replacement is not None
    index, source, target, pair = replacement
    assert index in (1, 2)
    assert source.lower() in ("person", "walks")
    assert target.lower() != source.lower()
    assert len(pair.split()) == len("A person walks forward.".split())
    assert "forward" in pair.lower()


def test_paper_faithful_rsr_manifest_invariants():
    manifest = json.loads(PAPER_PAIR_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["paper"]["selected_perturbation"] == (
        "Random Synonym Replacement (RSR)")
    assert manifest["paper"]["application_probability"] == 1.0
    assert manifest["official_source"]["commit"] == (
        "aa46405ad1af3c985e4c4e7c18d60d20993ef2a8")
    assert manifest["motionx_adaptation"]["strict_bpe_alignment"] is False
    assert manifest["motionx_adaptation"]["same_whitespace_length_required"] is True
    assert manifest["splits"]["overlap"] == {
        "train_test": 0,
        "train_val": 0,
        "val_test": 0,
    }
    coverage = manifest["coverage"]
    assert coverage["train_caption_rows"] == coverage["rows_written"]
    assert 0.9 < coverage["replacement_caption_coverage"] <= 1.0
    assert coverage["identity_fallback_rows"] >= 0


def test_motionx322_official_sato_table3_eval_contract():
    text = EVAL_LAUNCHER.read_text(encoding="utf-8")
    for fragment in (
        "test_perb_gemini31_max75_mixed_full.json",
        'MOTIONX322_SATO_MODEL_NAME:-motionx322_sato_official_50k_best_fid',
        'MOTIONX322_SATO_MODEL_DESCRIPTION:-Original SATO reproduced on Motion-X with a 50k budget',
        'MOTIONX322_SATO_SPEC:-$OUT_ROOT/sato50k_spec.json',
        'MOTIONX322_SATO_REPORT:-$OUT_ROOT/MOTIONX322_SATO50K_TABLE3_REPEAT20.md',
        "net_best_fid.pth",
        "clip_best.pth",
        "base_count=$((20 / gpu_count))",
        'start_seed="$next_seed"',
        '--num-repeats "$repeat_count"',
        "--num-repeats 20",
        "--expected-matched-pairs 7733",
        "--expected-total-pairs 7733",
        "--cfg-scale 1.3",
        "--sample-mode categorical",
        "--sample-temperature 1",
        "--top-k 0",
        "--top-p 1",
    ):
        assert fragment in text


def test_official_motionx_adapter_parses_without_starting_training():
    files = [
        SATO / "dataset" / "config.py",
        SATO / "dataset" / "variable_motion.py",
        SATO / "dataset" / "dataset_TM_train.py",
        SATO / "dataset" / "dataset_TM_eval.py",
        SATO / "models" / "vqvae.py",
        SATO / "models" / "evaluator_wrapper.py",
        SATO / "options" / "get_eval_option.py",
        SATO / "options" / "option_transformer.py",
        SATO / "attack.py",
        SATO / "eval_trans_per.py",
        SATO / "train.py",
    ]
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_motionx322_metadata_contract():
    path = SATO / "dataset" / "config.py"
    spec = importlib.util.spec_from_file_location("official_sato_dataset_config", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    data = module.get_dataset_spec("motionx322_75")
    assert data.dim_pose == 322
    assert data.variable_length is True
    assert data.root.name == "Motion-X-322-MaxToken75"
    assert module.dataset_max_token_length("motionx322_75", 4) == 76
