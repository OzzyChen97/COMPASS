import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


MODULE_PATH = Path(__file__).parents[1] / "tools" / "train_motionx_text_adaptation_pilot.py"
SPEC = importlib.util.spec_from_file_location("motionx_text_adaptation_pilot", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class FakeClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(4, 4)
        self.token_embedding = nn.Embedding(16, 4)
        self.transformer = nn.Linear(4, 4)
        self.ln_final = nn.LayerNorm(4)
        self.text_projection = nn.Linear(4, 4, bias=False)
        self.logit_scale = nn.Parameter(torch.ones(()))

    def encode_text(self, tokens):
        x = self.token_embedding(tokens).mean(dim=1)
        x = self.transformer(x)
        x = self.ln_final(x)
        return self.text_projection(x)


class FakeGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond = nn.Linear(4, 5)
        self.head = nn.Linear(5, 6)

    def forward(self, input_indices, clip_features):
        steps = input_indices.shape[1] + 1
        hidden = torch.tanh(self.cond(clip_features)).unsqueeze(1).expand(-1, steps, -1)
        return self.head(hidden)


def _requires_grad_by_group(arm):
    clip_model = FakeClip()
    generator = FakeGenerator()
    module.configure_trainability(generator, clip_model, arm)
    groups = module.collect_parameter_groups(generator, clip_model)
    return {
        name: {param.requires_grad for _, param in params}
        for name, params in groups.items()
    }


def test_freeze_matrix_for_generator_text_and_joint_arms():
    generator = _requires_grad_by_group("generator")
    assert generator["generator"] == {True}
    assert generator["clip_text"] == {False}
    assert generator["clip_visual"] == {False}
    assert generator["clip_other"] == {False}

    text = _requires_grad_by_group("text")
    assert text["generator"] == {False}
    assert text["clip_text"] == {True}
    assert text["clip_visual"] == {False}
    assert text["clip_other"] == {False}

    joint = _requires_grad_by_group("joint")
    assert joint["generator"] == {True}
    assert joint["clip_text"] == {True}
    assert joint["clip_visual"] == {False}
    assert joint["clip_other"] == {False}


def test_text_arm_backpropagates_through_frozen_generator_to_clip_text():
    torch.manual_seed(7)
    clip_model = FakeClip()
    generator = FakeGenerator()
    module.configure_trainability(generator, clip_model, "text")
    groups = module.collect_parameter_groups(generator, clip_model)

    text_tokens = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    input_indices = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 5], [1, 2, 5]], dtype=torch.long)
    lengths = torch.tensor([3, 3], dtype=torch.long)
    loss, metrics = module.compute_paired_supervised_loss(
        clip_model,
        generator,
        text_tokens,
        input_indices,
        targets,
        lengths,
        clean_batch_size=1,
    )
    assert torch.isfinite(loss)
    assert metrics["clean_loss"].item() > 0
    loss.backward()

    norms = module.assert_first_backward_reachability(groups, "text")
    assert norms["clip_text"] > 0
    assert norms["generator"] == 0
    assert all(param.grad is None for _, param in groups["generator"])


def test_pair_loader_rejects_nontrain_motion_ids_fail_closed(tmp_path):
    dataset_root = tmp_path / "Motion-X-322-MaxToken75"
    token_dir = dataset_root / "VQVAE"
    (dataset_root / "texts").mkdir(parents=True)
    token_dir.mkdir()
    (dataset_root / "train.txt").write_text("mx_train\n", encoding="utf-8")
    (dataset_root / "val.txt").write_text("mx_val\n", encoding="utf-8")
    (dataset_root / "test.txt").write_text("", encoding="utf-8")
    pair_path = tmp_path / "pairs.jsonl"
    pair_path.write_text(
        json.dumps(
            {
                "pair_id": "bad-val",
                "motion_id": "mx_val",
                "text_orig": "A person walks.",
                "text_pair": "A human walks.",
                "relation": "preserve",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(module.TrainingDataError, match="nontrain motion IDs"):
        module.load_training_pairs(
            pair_path,
            dataset_root=dataset_root,
            token_dir=token_dir,
            nb_code=512,
            dataset_limit=None,
        )


def test_cross_entropy_ignores_out_of_vocabulary_padding_before_loss():
    logits = torch.randn(2, 4, 6, requires_grad=True)
    targets = torch.tensor([[1, 5, 6, 6], [2, 3, 4, 5]])
    lengths = torch.tensor([2, 4])
    loss, _ = module.masked_sequence_cross_entropy(logits, targets, lengths)
    mask = torch.arange(4)[None, :] < lengths[:, None]
    expected = torch.nn.functional.cross_entropy(logits[mask], targets[mask])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))


def test_text_coverage_case_fallback_preserves_known_words():
    from tools.audit_motionx_text_coverage import LookupVectorizer, semantic_flags
    class Vocab:
        word2vec = {'walk': 1, 'NASA': 2}
        def __getitem__(self, token):
            return token
    fallback = LookupVectorizer(Vocab(), 'fallback_lower')
    assert fallback['Walk/VERB'] == 'walk/VERB'
    assert fallback['NASA/PROPN'] == 'NASA/PROPN'
    assert LookupVectorizer(Vocab(), 'current')['Walk/VERB'] == 'Walk/VERB'
    assert 'direction' in semantic_flags('go left', 'go right')
