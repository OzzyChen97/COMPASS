"""Shared-CLIP batching, selection and checkpoint invariants."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.shared_clip_experiment import MODELS, seq_ce, t2m_batch
from tools.launch_shared_clip_pipeline import select_shared, env
from tools.train_motiongpt_full_t5 import epoch_indices


def test_explicit_batch_visits_requested_pairs_without_resampling():
    examples = [SimpleNamespace(token_rows=np.array([[i, i+1]]), text_orig=str(i), text_pair=str(i)) for i in range(8)]
    selected, target, lengths, corrupted = t2m_batch(examples, [4, 1, 7], torch.device('cpu'), pkeep=1.)
    assert [e.text_orig for e in selected] == ['4','1','7']
    assert target.tolist() == [[4,5,512],[1,2,512],[7,8,512]]
    assert lengths.tolist() == [3,3,3]
    torch.testing.assert_close(corrupted, target[:, :-1])


def test_per_sequence_ce_ignores_out_of_vocabulary_padding():
    logits = torch.zeros(2,3,513, requires_grad=True)
    targets = torch.tensor([[2,512,513],[3,4,512]])
    loss = seq_ce(logits, targets, torch.tensor([2,3]))
    assert loss.item() == pytest.approx(np.log(513))
    loss.backward()
    assert logits.grad[0,2].abs().sum() == 0


def test_four_rank_coverage_no_dropped_pairs():
    all_indices = [v for rank in range(4) for v in epoch_indices(201335,4,rank,0)]
    assert len(all_indices) == 201336
    assert len(set(all_indices)) == 201335


def rows(score):
    return [dict(model=m, query=q, R3=score, samples=661) for m in MODELS for q in ['clean','P1','P2','P3']]


def test_selection_is_one_shared_checkpoint_and_requires_all_models():
    selected, scores = select_shared({'warmup':rows(.4), 'joint':rows(.5)})
    assert selected == 'joint'
    assert scores['joint'] == pytest.approx(.5)
    with pytest.raises(AssertionError): select_shared({'bad':rows(.9)[:-1]})
    wrong = rows(.9); wrong[0]['samples'] = 32
    with pytest.raises(AssertionError): select_shared({'bad':wrong})


def test_subprocess_isolates_python_and_requested_gpus(monkeypatch):
    monkeypatch.setenv('PYTHONPATH','/usr/local/lib/python3.12/site-packages')
    e = env('0,1,2,3')
    assert 'PYTHONPATH' not in e
    assert e['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'