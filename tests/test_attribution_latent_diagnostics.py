"""CPU unit tests for controlled attribution/latent diagnostics."""
import numpy as np
import torch
from tools.measure_attribution_latent import (
    aligned_attr_distance, cosine_distance, generate_edit, jsd,
    source_group, words_and_positions, Extractor,
)
from clip_local.simple_tokenizer import SimpleTokenizer


def test_jsd_identity_and_disjoint():
    assert jsd([1, 2], [1, 2]) == 0
    assert np.isclose(jsd([1, 0], [0, 1]), np.log(2))


def test_alignment_handles_reordering_and_substitutions():
    a = [{'word': 'left'}, {'word': 'arm'}]
    b = [{'word': 'arm'}, {'word': 'left'}]
    distance, coverage = aligned_attr_distance(a, [.2, .8], b, [.8, .2])
    assert distance == 0 and coverage == 1
    c = [{'word': 'right'}, {'word': 'arm'}]
    distance, coverage = aligned_attr_distance(a, [.2, .8], c, [.6, .4])
    assert distance > 0 and coverage == .5


def test_duplicate_word_occurrences_have_separate_support():
    a = [{'word': 'arm'}, {'word': 'arm'}]
    assert aligned_attr_distance(a, [.1, .9], a, [.9, .1])[0] > 0


def test_token_positions_are_exact_and_no_specials():
    tok = SimpleTokenizer()
    text = "A person isn't walking counterclockwise."
    records = words_and_positions(text, tok)
    positions = sum([r['positions'] for r in records], [])
    assert positions == list(range(1, len(tok.encode(text)) + 1))


def test_synthetic_edit_is_single_word_and_mirror_group():
    edited, meta = generate_edit('A person walks forward and walks back.', 'action')
    assert edited == 'A person runs forward and walks back.'
    assert meta['old'] == 'walks' and meta['new'] == 'runs'
    assert source_group('M000123') == source_group('000123')
    assert generate_edit('A person sits.', 'direction') is None


def test_cosine_distance_bounds():
    assert np.isclose(cosine_distance([1, 2], [2, 4]), 0)
    assert np.isclose(cosine_distance([1, 0], [-1, 0]), 2)


def test_interventions_match_absolute_norm_across_groups():
    # Exercise real intervention batching without loading a checkpoint or CUDA.
    obj = object.__new__(Extractor)
    obj.device = torch.device('cpu')
    words = [{'word': word, 'positions': [i+1]} for i, word in enumerate(['walks', 'left', 'raises'])]
    h = torch.ones(1, 77, 8)
    encoded = {'words': words, 'word_raw': [1., 2., 3.], 'h': h,
               'ids': torch.zeros(1, 77, dtype=torch.long),
               'target': torch.zeros(1, 3, dtype=torch.long), 'steps': 2,
               'states': np.ones((2, 2, 8), dtype=np.float32),
               'logp': np.log(np.ones((2, 4), dtype=np.float32)/4), 'score': -1.}
    norms = []
    def forward(batch, ids, target, steps):
        norms.extend((batch - h).norm(dim=(-1, -2)).tolist())
        n = len(batch)
        return None, None, torch.full((n, 2, 4), -np.log(4)), torch.full((n,), -1.), torch.ones(n, 2, 2, 8)
    obj.forward = forward
    rows = obj.intervene(encoded, 123, [.1], 2)
    assert len(rows) == 3 and len(norms) == 12
    assert np.allclose(norms, norms[0], atol=1e-6)
    assert all(r['hidden_relative_l2'] == [0., 0.] for r in rows)


def test_rank_correlation_handles_layers_and_controlled_confounder():
    from tools.analyze_attribution_latent import rank_correlation
    rng = np.random.default_rng(5)
    x = np.arange(100.)
    y = np.column_stack([x, -x])
    assert np.allclose(rank_correlation(x, y), [1, -1])
    c = rng.normal(size=(1000, 1))
    x = c[:, 0] + rng.normal(size=1000) * .2
    y = (c[:, 0] + rng.normal(size=1000) * .2)[:, None]
    assert rank_correlation(x, y)[0] > .8
    assert abs(rank_correlation(x, y, c)[0]) < .15


def test_cluster_bootstrap_keeps_all_rewrites_and_holm():
    from tools.analyze_attribution_latent import cluster_indices, bootstrap_indices, holm
    groups = ['a', 'a', 'a', 'b', 'b', 'b']
    idx = bootstrap_indices(cluster_indices(groups), np.random.default_rng(1))
    counts = np.bincount(idx, minlength=6)
    assert counts[0] == counts[1] == counts[2]
    assert counts[3] == counts[4] == counts[5]
    assert np.allclose(holm([.01, .04, .03]), [.03, .06, .06])


def test_exact_step_prefix_and_batch_invariance_with_tiny_models():
    from types import SimpleNamespace
    from models.t2m_trans import Text2Motion_Transformer
    torch.manual_seed(42)
    width = 16
    class TinyClip(torch.nn.Module):
        dtype = torch.float32
        def __init__(self):
            super().__init__()
            self.positional_embedding = torch.nn.Parameter(torch.zeros(77, width))
            self.transformer = torch.nn.Identity()
            self.ln_final = torch.nn.LayerNorm(width)
            self.text_projection = None
    clip = TinyClip().eval()
    trans = Text2Motion_Transformer(num_vq=8, embed_dim=width, clip_dim=width,
                                   block_size=5, num_layers=1, n_head=8,
                                   drop_out_rate=0, fc_rate=2).eval()
    extractor = Extractor(clip, trans, None, {}, None, torch.device('cpu'))
    h = torch.randn(1, 77, width)
    ids = torch.zeros(1, 77, dtype=torch.long)
    ids[0, 4] = 10
    target = torch.tensor([[2, 3, 1, 4, 8]])
    *_, single, states = extractor.forward(h, ids, target, 4)
    *_, batch, batch_states = extractor.forward(h.repeat(2,1,1), ids.repeat(2,1), target.repeat(2,1), 4)
    assert states.shape == (1, 3, 4, width)
    assert torch.allclose(single.expand(2), batch, atol=1e-6)
    assert torch.allclose(states.expand(2,-1,-1,-1), batch_states, atol=1e-6)
    extractor.close()


def test_paper_view_preserves_original_results_and_all_pairs(tmp_path):
    import json
    from tools.analyze_attribution_latent import paper_view
    analysis = tmp_path / 'analysis'
    analysis.mkdir()
    original = analysis / 'layer_association.pdf'
    original.write_bytes(b'original-analysis-sentinel')
    summary = {'relations': {}, 'interventions': {}}
    for model, scale in [('baseline', 1.), ('satopp', .7)]:
        rows = []
        for group in ['a', 'b']:
            for i, level in enumerate(['P1', 'P2', 'P3']):
                rows.append({'pair_key': group + level, 'group': group, 'target_sha256': group,
                             'text_orig': 'walks', 'text_pair': 'is walking', 'split': 'test',
                             'relation': 'preserve', 'level': level, 'hidden_cosine': [.2 * (i+1) * scale] * 19})
        (tmp_path / f'{model}_pairs.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        for relation in ['preserve', 'alter']:
            summary['relations'][f'test/{model}/{relation}'] = {'partial_rho': [.1 * scale] * 19}
        for eps in [.1, .25]:
            summary['interventions'][f'test/{model}/eps{eps}'] = {'metrics': {'final_hidden_relative_l2': {'means': {'high': .06*scale, 'low': .03*scale, 'random': .04*scale}}}}
    payload = json.dumps(summary)
    (analysis / 'summary.json').write_text(payload)
    (tmp_path / 'manifest.json').write_text(json.dumps({'pair_digest': 'test-digest'}))
    paper_view(tmp_path)
    assert original.read_bytes() == b'original-analysis-sentinel'
    assert (analysis / 'summary.json').read_text() == payload
    data = json.loads((analysis / 'paper_view/figure_data_and_captions.json').read_text())
    result = data['figures']['paraphrase_latent_stability']
    assert np.isclose(result['overall_reduction_percent'], 30)
    assert np.allclose(result['reductions_percent'], [30, 30, 30])
    assert len(list((analysis / 'paper_view').glob('*.pdf'))) == 3