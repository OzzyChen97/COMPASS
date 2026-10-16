#!/usr/bin/env python3
"""Cluster-aware analysis of attribution/Transformer diagnostics; no GPU needed."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.measure_attribution_latent import words_and_positions, aligned_attr_distance
from clip_local.simple_tokenizer import SimpleTokenizer

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-attribution-latent')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    x, y = x - x.mean(0), y - y.mean(0)
    denominator = np.sqrt((x*x).sum(0) * (y*y).sum(0))
    return np.divide((x*y).sum(0), denominator, out=np.full_like(denominator, np.nan), where=denominator > 1e-25)


def rank_correlation(x, y, controls=None):
    x = rankdata(np.asarray(x), axis=0)
    y = rankdata(np.asarray(y), axis=0)
    if x.ndim == 1:
        x = x[:, None]
    if controls is not None:
        c = np.column_stack([np.ones(len(x)), rankdata(controls, axis=0)])
        x = x - c @ np.linalg.lstsq(c, x, rcond=None)[0]
        y = y - c @ np.linalg.lstsq(c, y, rcond=None)[0]
    return corr(x, y)


def cluster_indices(groups):
    lookup = defaultdict(list)
    for i, group in enumerate(groups):
        lookup[group].append(i)
    return [np.asarray(lookup[g]) for g in sorted(lookup)]


def bootstrap_indices(clusters, rng):
    chosen = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[i] for i in chosen])


def arrays(rows):
    return (np.asarray([r['attribution_jsd'] for r in rows]),
            np.asarray([r['hidden_cosine'] for r in rows]),
            np.asarray([[r['clip_distance'], r['length_difference'], r['edit_fraction'], r['token_count_orig'], r['steps'], r['uniform_alpha_jsd']] for r in rows]))


def relation_statistics(rows, repetitions, seed):
    x, y, controls = arrays(rows)
    clusters = cluster_indices([r['group'] for r in rows])
    rng = np.random.default_rng(seed)
    raw, partial = rank_correlation(x, y), rank_correlation(x, y, controls)
    boots_raw, boots_partial = [], []
    for _ in range(repetitions):
        take = bootstrap_indices(clusters, rng)
        boots_raw.append(rank_correlation(x[take], y[take]))
        boots_partial.append(rank_correlation(x[take], y[take], controls[take]))
    ci_raw = np.nanpercentile(boots_raw, [2.5, 97.5], axis=0)
    ci_partial = np.nanpercentile(boots_partial, [2.5, 97.5], axis=0)
    # Group-level permutation within relation subtype; keep P1/P2/P3 together.
    type_clusters = defaultdict(list)
    for cluster in clusters:
        signature = tuple(rows[i]['level'] if rows[i]['relation'] == 'preserve' else rows[i]['edit_type'] for i in cluster)
        type_clusters[signature].append(cluster)
    null = []
    for _ in range(repetitions):
        permuted = np.arange(len(rows))
        for block in type_clusters.values():
            order = rng.permutation(len(block))
            for dst, src in zip(block, order):
                permuted[dst] = block[src]
        null.append(rank_correlation(x, y[permuted]))
    null = np.asarray(null)
    p = (1 + (np.abs(null) >= np.abs(raw)).sum(0)) / (len(null) + 1)
    p[~np.isfinite(raw)] = np.nan
    return {'n_pairs': len(rows), 'n_source_groups': len(clusters),
            'rho': raw.tolist(), 'rho_ci_low': ci_raw[0].tolist(), 'rho_ci_high': ci_raw[1].tolist(),
            'partial_rho': partial.tolist(), 'partial_ci_low': ci_partial[0].tolist(), 'partial_ci_high': ci_partial[1].tolist(),
            'permutation_p_raw': p.tolist(), 'permutation_null_mean': null.mean(0).tolist(),
            'mean_hidden_cosine': y.mean(0).tolist(), 'mean_attribution_jsd': float(x.mean()),
            'mean_clip_distance': float(controls[:, 0].mean()),
            'uniform_alpha_rho': rank_correlation(np.asarray([r['uniform_alpha_jsd'] for r in rows]), y).tolist(),
            'weighted_rho': rank_correlation(np.asarray([r['weighted_distance'] for r in rows]), y).tolist(),
            'clip_rho': rank_correlation(np.asarray([r['clip_distance'] for r in rows]), y).tolist(),
            'learned_pool_rho': rank_correlation(np.asarray([r['learned_pool_distance'] for r in rows]), y).tolist() if rows[0]['learned_pool_distance'] is not None else None,
            'uniform_rho': rank_correlation(np.asarray([r['uniform_distance'] for r in rows]), y).tolist(),
            'output_jsd_rho': float(rank_correlation(x, np.asarray([r['output_jsd'] for r in rows]))[0]),
            'mean_entropy': float(np.mean([r[k] for r in rows for k in ('entropy_orig', 'entropy_pair')])),
            'mean_lexical_coverage': float(np.mean([r['lexical_coverage'] for r in rows])),
            'mean_training_visible_mass': float(np.mean([r[k] for r in rows for k in ('training_visible_mass_orig', 'training_visible_mass_pair')]))}


def model_difference(base, sato, repetitions, seed):
    assert [r['pair_key'] for r in base] == [r['pair_key'] for r in sato]
    xb, yb, cb = arrays(base)
    xs, ys, cs = arrays(sato)
    clusters = cluster_indices([r['group'] for r in base])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        idx = bootstrap_indices(clusters, rng)
        values.append([float(rank_correlation(xs[idx], ys[idx, -1])[0] - rank_correlation(xb[idx], yb[idx, -1])[0]),
                       float(rank_correlation(xs[idx], ys[idx, -1:], cs[idx])[0] - rank_correlation(xb[idx], yb[idx, -1:], cb[idx])[0]),
                       float((ys[idx, -1] - yb[idx, -1]).mean())])
    point = [float(rank_correlation(xs, ys[:, -1])[0] - rank_correlation(xb, yb[:, -1])[0]),
             float(rank_correlation(xs, ys[:, -1:], cs)[0] - rank_correlation(xb, yb[:, -1:], cb)[0]),
             float((ys[:, -1] - yb[:, -1]).mean())]
    ci = np.nanpercentile(values, [2.5, 97.5], axis=0)
    return {name: {'satopp_minus_baseline': point[i], 'ci95': ci[:, i].tolist()} for i, name in enumerate(['final_rho', 'final_partial_rho', 'final_hidden_distance'])}


def intervention_statistics(rows, repetitions, seed):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row['group']][row['kind']] = row
    groups = sorted(grouped)
    if any(set(grouped[g]) != {'high', 'low', 'random'} for g in groups):
        raise ValueError('Incomplete intervention comparisons')
    metrics = {'final_hidden_relative_l2': lambda r: r['hidden_relative_l2'][-1],
               'final_hidden_cosine': lambda r: r['hidden_cosine'][-1],
               'output_jsd': lambda r: r['output_jsd'], 'score_abs_change': lambda r: r['score_abs_change']}
    rng = np.random.default_rng(seed)
    result = {'n_groups': len(groups), 'metrics': {}}
    for name, getter in metrics.items():
        a = np.asarray([[getter(grouped[g][kind]) for kind in ('high', 'low', 'random')] for g in groups])
        boots = np.asarray([a[rng.integers(len(a), size=len(a))].mean(0) for _ in range(repetitions)])
        values = {'means': dict(zip(('high', 'low', 'random'), a.mean(0).tolist())),
                  'mean_ci95': dict(zip(('high', 'low', 'random'), np.percentile(boots, [2.5, 97.5], axis=0).T.tolist()))}
        for j, label in ((1, 'high_minus_low'), (2, 'high_minus_random')):
            difference = a[:, 0] - a[:, j]
            null = np.asarray([(difference * rng.choice([-1, 1], size=len(a))).mean() for _ in range(repetitions)])
            values[label] = {'mean': float(difference.mean()), 'ci95': np.percentile(boots[:, 0] - boots[:, j], [2.5, 97.5]).tolist(), 'signflip_p': float((1 + (np.abs(null) >= abs(difference.mean())).sum())/(repetitions+1)), 'positive_fraction': float((difference > 0).mean())}
        values['high_over_low_ratio_of_means'] = float(a[:, 0].mean() / max(a[:, 1].mean(), 1e-30))
        result['metrics'][name] = values
    return result


def holm(values):
    values = np.asarray(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    order = finite[np.argsort(values[finite])]
    result = np.full(len(values), np.nan)
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(order)-rank)*values[i]))
        result[i] = running
    return result.tolist()


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(args):
    root = Path(args.input)
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    pairs = {m: sorted(read_jsonl(root / f'{m}_pairs.jsonl'), key=lambda r: r['pair_key']) for m in ('baseline', 'satopp')}
    tokenizer = SimpleTokenizer()
    uniform_distances = {}
    for model in pairs:
        for row in pairs[model]:
            key = row['pair_key']
            if key not in uniform_distances:
                wa = words_and_positions(row['text_orig'], tokenizer)
                wb = words_and_positions(row['text_pair'], tokenizer)
                aa = np.array([len(w['positions']) for w in wa], dtype=float)
                ab = np.array([len(w['positions']) for w in wb], dtype=float)
                uniform_distances[key] = aligned_attr_distance(wa, aa/aa.sum(), wb, ab/ab.sum())[0]
            row['uniform_alpha_jsd'] = uniform_distances[key]
    for m in pairs:
        completion = json.loads((root / f'{m}_complete.json').read_text())
        if len(pairs[m]) != completion['pairs']:
            raise ValueError(f'{m} extraction incomplete')
    if len(pairs['baseline']) != len(pairs['satopp']):
        raise ValueError('Baseline/SATO sample counts differ')
    for a, b in zip(pairs['baseline'], pairs['satopp']):
        if any(a[k] != b[k] for k in ('pair_key', 'target_sha256', 'text_orig', 'text_pair', 'steps', 'layers')):
            raise ValueError('Baseline/SATO pair mismatch')
        if abs(a['clip_distance'] - b['clip_distance']) > 1e-5:
            raise ValueError('Frozen CLIP control differs across models')
    layers = pairs['baseline'][0]['layers']
    summary = {'schema': 'attribution-latent-analysis/v1', 'manifest_pair_digest': manifest['pair_digest'], 'layers': layers,
               'bootstrap_repetitions': args.bootstrap, 'group_unit': 'source motion ID stripping mirror M prefix',
               'controls': ['frozen CLIP cosine distance', 'absolute token count difference', '1 - lexical alignment coverage', 'original token count', 'valid early motion steps', 'uniform-token attribution aligned JSD'],
               'relations': {}, 'model_differences': {}, 'interventions': {},
               'interpretation_limits': ['The nominal val split overlaps the current train list and is exploratory only; test is disjoint after mirror grouping.', 'Observational correlation is not evidence that attribution losses updated the generator.', 'Altering pairs are rule-generated diagnostics, not validated edited-motion references.', 'Word alignment is lexical and aggregates unmatched words; no oracle semantic alignment.', 'Reported intervals resample motion groups, not independent training seeds.', 'No free-running generated-motion edit compliance is measured.']}
    for split in manifest['args']['splits']:
        for relation in ('preserve', 'alter'):
            subsets = {}
            for model in pairs:
                rows = [r for r in pairs[model] if r['split'] == split and r['relation'] == relation]
                subsets[model] = rows
                key = f'{split}/{model}/{relation}'
                summary['relations'][key] = relation_statistics(rows, args.bootstrap, args.seed)
                summary['relations'][key]['subgroups'] = {}
                for label in sorted({r['level'] if relation == 'preserve' else r['edit_type'] for r in rows}):
                    sub = [r for r in rows if (r['level'] if relation == 'preserve' else r['edit_type']) == label]
                    summary['relations'][key]['subgroups'][label] = relation_statistics(sub, min(300, args.bootstrap), args.seed)
                print(f'ANALYZED {key} n={len(rows)} rho={summary["relations"][key]["rho"][-1]:.4f} partial={summary["relations"][key]["partial_rho"][-1]:.4f}', flush=True)
            summary['model_differences'][f'{split}/{relation}'] = model_difference(subsets['baseline'], subsets['satopp'], args.bootstrap, args.seed)
    for model in pairs:
        rows = read_jsonl(root / f'{model}_interventions.jsonl')
        for split in manifest['args']['splits']:
            for eps in manifest['args']['epsilons']:
                subset = [r for r in rows if r['split'] == split and r['epsilon_relative_to_sentence_median_embedding_norm'] == eps]
                if not subset:
                    continue
                key = f'{split}/{model}/eps{eps}'
                summary['interventions'][key] = intervention_statistics(subset, args.bootstrap, args.seed)
    # Predeclared inference family: test final layer raw association (4 cells),
    # and test final hidden relative-L2 high-v-low/random (2 models x 2 eps x 2).
    inference = []
    for key, value in summary['relations'].items():
        if key.startswith('test/'):
            inference.append({'test': key + '/final_rho', 'p': value['permutation_p_raw'][-1]})
    for key, value in summary['interventions'].items():
        if key.startswith('test/'):
            for contrast in ('high_minus_low', 'high_minus_random'):
                inference.append({'test': key + '/' + contrast, 'p': value['metrics']['final_hidden_relative_l2'][contrast]['signflip_p']})
    for row, adjusted in zip(inference, holm([r['p'] for r in inference])):
        row['holm_p'] = adjusted
    summary['test_inference_family'] = inference
    (out / 'summary.json').write_text(json.dumps(finite_json(summary), indent=2, allow_nan=False))
    with (out / 'layer_metrics.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'model', 'relation', 'layer', 'rho', 'rho_ci_low', 'rho_ci_high', 'partial_rho', 'partial_ci_low', 'partial_ci_high', 'mean_hidden_cosine'])
        for key, result in summary['relations'].items():
            for i, layer in enumerate(layers):
                writer.writerow(key.split('/') + [layer] + [result[k][i] for k in ('rho', 'rho_ci_low', 'rho_ci_high', 'partial_rho', 'partial_ci_low', 'partial_ci_high', 'mean_hidden_cosine')])
    plot(summary, out)
    print('ANALYSIS_COMPLETE ' + str(out / 'summary.json'), flush=True)


def plot(summary, out):
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    split = 'test' if any(k.startswith('test/') for k in summary['relations']) else 'val'
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    bounds = [v for key, result in summary['relations'].items() if key.startswith(split + '/')
              for metric in ('rho_ci_low', 'rho_ci_high', 'partial_ci_low', 'partial_ci_high')
              for v in result[metric] if v is not None and np.isfinite(v)]
    y_limits = (min(0., min(bounds)) - .04, max(0., max(bounds)) + .04)
    for col, relation in enumerate(('preserve', 'alter')):
        for row, metric in enumerate(('rho', 'partial_rho')):
            ax = axes[row, col]
            for model, color in (('baseline', '#555555'), ('satopp', '#2166ac')):
                r = summary['relations'][f'{split}/{model}/{relation}']
                prefix = 'rho' if metric == 'rho' else 'partial'
                x = np.arange(1, len(r[metric])+1)
                ax.plot(x, r[metric], label='T2M-GPT' if model == 'baseline' else 'SATO++', color=color)
                ax.fill_between(x, r[prefix + '_ci_low'], r[prefix + '_ci_high'], color=color, alpha=.14)
            ax.axhline(0, color='gray', linewidth=.6)
            ax.set_title(f'{relation}: ' + ('Spearman correlation' if row == 0 else 'Controlled rank correlation'))
            ax.set_ylabel('Spearman rho (dimensionless)' if row == 0 else 'Partial rank correlation (dimensionless)')
            ax.set_xlabel('Transformer block / final normalization (index)')
            ax.set_xticks([1, 3, 6, 9, 12, 15, 18, 19])
            ax.tick_params(axis='x', labelbottom=True)
            ax.set_ylim(*y_limits)
            ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f'HumanML3D {split.upper()}: attribution JSD vs hidden cosine distance; 95% bootstrap intervals')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'layer_association.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, model in zip(axes, ('baseline', 'satopp')):
        candidates = [(key, value) for key, value in summary['interventions'].items() if key.startswith(f'{split}/{model}/')]
        for j, (key, value) in enumerate(candidates):
            m = value['metrics']['final_hidden_relative_l2']
            means = np.array([m['means'][k] for k in ('high', 'low', 'random')])
            ci = np.array([m['mean_ci95'][k] for k in ('high', 'low', 'random')])
            x = np.arange(3) + (j-.5)*.32
            ax.bar(x, means, width=.30, label=key.split('/')[-1], color=('#777777', '#2166ac')[j % 2])
            ax.errorbar(x, means, yerr=np.maximum(0, np.stack([means-ci[:, 0], ci[:, 1]-means])), fmt='none', color='black', capsize=3)
        ax.set_xticks(range(3), ['High attribution', 'Low attribution', 'Random'])
        ax.set_xlabel('Perturbed single-token content word')
        ax.set_ylabel('Final hidden relative L2 change (dimensionless)')
        ax.set_title('T2M-GPT' if model == 'baseline' else 'SATO++')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle(f'HumanML3D {split.upper()}: equal-norm word intervention; 95% motion bootstrap intervals')
    fig.text(.5, .015, 'Perturbation norm = epsilon x sentence median content-word embedding norm; paired directions and symmetric signs.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .045, 1, 1])
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'word_intervention.{ext}', dpi=180)
    plt.close(fig)


def paper_view(input_root):
    """Render descriptive, mean-only views; keep original analyses untouched.

    Every existing test example is retained within its explicitly named relation.
    Local response is not a deletion AUC, a Lipschitz bound, or CKA.
    """
    import hashlib
    root = Path(input_root)
    out = root / 'analysis' / 'paper_view'
    out.mkdir(exist_ok=True)
    summary_path = root / 'analysis' / 'summary.json'
    summary = json.loads(summary_path.read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    pairs = {model: sorted(read_jsonl(root / f'{model}_pairs.jsonl'), key=lambda r: r['pair_key'])
             for model in ('baseline', 'satopp')}
    if len(pairs['baseline']) != len(pairs['satopp']):
        raise ValueError('Model pair counts differ')
    for a, b in zip(pairs['baseline'], pairs['satopp']):
        if any(a[k] != b[k] for k in ('pair_key', 'target_sha256', 'text_orig', 'text_pair')):
            raise ValueError('Model pair identities differ')
    colors = {'baseline': '#737373', 'satopp': '#2166ac'}
    labels = {'baseline': 'T2M-GPT', 'satopp': 'SATO++'}
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    exports = {'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
               'pair_digest': manifest['pair_digest'], 'subset': 'all available test rows from the frozen 300-motion study, with explicit relation filters; no outcome-based selection',
               'uncertainty_display': 'point estimates only in presentation figures; original confidence intervals retained in ../summary.json',
               'references': [
                   {'title': 'On the Robustness of Interpretability Methods', 'year': 2018,
                    'url': 'https://arxiv.org/abs/1806.08049',
                    'use': 'local perturbation sensitivity and dataset-level aggregation; our metric is hidden response, not their explanation Lipschitz estimate'},
                   {'title': 'RISE: Randomized Input Sampling for Explanation of Black-box Models', 'year': 2018,
                    'url': 'https://arxiv.org/abs/1806.07421',
                    'use': 'perturb-and-measure presentation, Figures 2 and 4; no deletion/insertion AUC is claimed for our two-strength embedding intervention'},
                   {'title': 'Similarity of Neural Network Representations Revisited', 'year': 2019,
                    'url': 'https://proceedings.mlr.press/v97/kornblith19a.html',
                    'use': 'layerwise representation comparison as methodological context; our heatmap displays partial rank correlation, not CKA'},
               ], 'figures': {}}

    def save(fig, name):
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'{name}.{ext}', dpi=220, bbox_inches='tight')
        plt.close(fig)

    levels = ('P1', 'P2', 'P3')
    preserve = {m: [r for r in rows if r['split'] == 'test' and r['relation'] == 'preserve']
                for m, rows in pairs.items()}
    group_count = len({r['group'] for r in preserve['baseline']})
    level_values = {m: [float(np.mean([r['hidden_cosine'][-1] for r in rows if r['level'] == level]))
                        for level in levels] for m, rows in preserve.items()}
    overall = {m: float(np.mean([r['hidden_cosine'][-1] for r in rows])) for m, rows in preserve.items()}
    reductions = [100 * (1 - s / b) for b, s in zip(level_values['baseline'], level_values['satopp'])]
    overall_reduction = 100 * (1 - overall['satopp'] / overall['baseline'])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    x = np.arange(3)
    for j, model in enumerate(('baseline', 'satopp')):
        bars = axes[0].bar(x + (j-.5)*.34, level_values[model], width=.32,
                           color=colors[model], label=labels[model])
        axes[0].bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    peak = max(level_values['baseline'] + level_values['satopp'])
    for i, decrease in enumerate(reductions):
        axes[0].text(i, peak*1.18, f'{decrease:.1f}% lower', ha='center', fontsize=9, color=colors['satopp'])
    axes[0].set_ylim(0, peak*1.34)
    axes[0].set_xticks(x, ['P1: lexical', 'P2: syntactic', 'P3: narrative'])
    axes[0].set_xlabel('Meaning-preserving paraphrase level')
    axes[0].set_ylabel('Mean final hidden cosine distance')
    axes[0].set_title('(a) Paraphrase-induced hidden drift')
    for model in ('baseline', 'satopp'):
        values = np.sort([r['hidden_cosine'][-1] for r in preserve[model]])
        axes[1].step(np.r_[0, values, 2], np.r_[0, np.arange(1, len(values)+1)/len(values), 1],
                     where='post', color=colors[model], label=labels[model], linewidth=2)
    axes[1].set_xlim(0, 2)
    axes[1].set_ylim(0, 1.03)
    axes[1].set_xlabel('Final hidden cosine distance (smaller is more stable)')
    axes[1].set_ylabel('Fraction of preserving pairs')
    axes[1].set_title('(b) All preserving pairs: empirical CDF')
    axes[1].legend(frameon=False, loc='lower right')
    axes[1].text(.98, .48, f'{overall_reduction:.1f}% lower mean drift\n{overall["baseline"]:.3f} to {overall["satopp"]:.3f}',
                 transform=axes[1].transAxes, ha='right', color=colors['satopp'])
    fig.suptitle('HumanML3D test: internal stability under meaning-preserving rewrites', fontsize=12)
    fig.tight_layout(rect=[0, .035, 1, 1])
    fig.text(.5, .005, f'{len(preserve["baseline"])} preserving pairs / {group_count} source-motion groups; same target-motion prefix; point estimates.', ha='center', fontsize=8)
    save(fig, 'paraphrase_latent_stability')
    exports['figures']['paraphrase_latent_stability'] = {
        'levels': list(levels), 'means': level_values, 'reductions_percent': reductions,
        'overall_mean': overall, 'overall_reduction_percent': overall_reduction,
        'caption': 'Hidden-state stability under meaning-preserving paraphrases on HumanML3D. Left: mean cosine displacement for each paraphrase level. Right: empirical cumulative distribution over all preserving pairs. Both models use identical target-motion prefixes. Lower displacement indicates greater internal stability under this diagnostic, not necessarily better generated-motion quality. Error bars are omitted in this descriptive presentation; full uncertainty estimates are retained in the companion analysis.'}

    epsilons = sorted({float(k.rsplit('eps', 1)[1]) for k in summary['interventions'] if k.startswith('test/')})
    kinds = ('high', 'low', 'random')
    fig, axes = plt.subplots(1, len(epsilons), figsize=(10.4, 3.8), sharey=True, squeeze=False)
    response_data = {}
    ymax = max(summary['interventions'][f'test/{m}/eps{e}']['metrics']['final_hidden_relative_l2']['means'][k]
               for m in ('baseline', 'satopp') for e in epsilons for k in kinds) * 100
    for ax, eps in zip(axes[0], epsilons):
        values = {m: [100 * summary['interventions'][f'test/{m}/eps{eps}']['metrics']['final_hidden_relative_l2']['means'][k]
                      for k in kinds] for m in ('baseline', 'satopp')}
        reduction = [100 * (1-s/b) for b, s in zip(values['baseline'], values['satopp'])]
        for j, model in enumerate(('baseline', 'satopp')):
            bars = ax.bar(x + (j-.5)*.34, values[model], width=.32, color=colors[model], label=labels[model])
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
        for i, delta in enumerate(reduction):
            ax.text(i, max(values['baseline'][i], values['satopp'][i]) + ymax*.12,
                    f'{delta:.1f}% lower', ha='center', color=colors['satopp'], fontsize=9)
        ax.set_ylim(0, ymax*1.3)
        ax.set_xticks(x, ['High attribution', 'Low attribution', 'Random'])
        ax.set_xlabel('Perturbed single-token content word')
        ax.set_title(f'Perturbation strength: epsilon = {eps}')
        response_data[str(eps)] = {'mean_relative_l2_percent': values, 'reduction_percent': reduction}
    axes[0, 0].set_ylabel('Mean final hidden relative L2 change (%)')
    axes[0, -1].legend(frameon=False, loc='upper right')
    fig.suptitle('HumanML3D test: lower local hidden-state response at matched input strength', fontsize=12)
    fig.tight_layout(rect=[0, .045, 1, 1])
    fig.text(.5, .005, '100 shared motion groups; 3 paired directions and symmetric signs; epsilon scales the sentence median embedding norm.', ha='center', fontsize=8)
    save(fig, 'attribution_guided_local_response')
    exports['figures']['attribution_guided_local_response'] = {
        'conditions': response_data,
        'caption': 'Attribution-guided local input sensitivity on HumanML3D. Bars report mean final hidden-state relative L2 changes under equal-norm perturbations of high-attribution, low-attribution, and randomly selected content words. Random word identities are shared across models; high/low words are chosen by each model attribution ranking. The same 100 test motion groups, perturbation directions, and symmetric signs are used. Labels show relative reductions in the mean response of SATO++ compared with T2M-GPT. These values measure local hidden-state sensitivity, not deletion AUC, semantic edit accuracy, or explanation robustness.'}

    keys = ['test/baseline/preserve', 'test/satopp/preserve', 'test/baseline/alter', 'test/satopp/alter']
    matrix = np.asarray([summary['relations'][key]['partial_rho'] for key in keys])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6), gridspec_kw={'width_ratios': [2.25, 1]})
    vmin, vmax = min(0., float(matrix.min())), max(.3, float(matrix.max()))
    image = axes[0].imshow(matrix, aspect='auto', cmap='Blues', vmin=vmin, vmax=vmax)
    axes[0].set_yticks(range(4), ['Preserve / T2M-GPT', 'Preserve / SATO++', 'Alter / T2M-GPT', 'Alter / SATO++'])
    positions = [0, 2, 5, 8, 11, 14, 17, 18]
    axes[0].set_xticks(positions, [str(i+1) for i in positions])
    axes[0].set_xlabel('Transformer block / final normalization (index)')
    axes[0].set_title('(a) Layerwise controlled association')
    fig.colorbar(image, ax=axes[0], fraction=.036, pad=.025, label='Partial Spearman rho')
    for j, model in enumerate(('baseline', 'satopp')):
        vals = [summary['relations'][f'test/{model}/{rel}']['partial_rho'][-1] for rel in ('preserve', 'alter')]
        bars = axes[1].bar(np.arange(2)+(j-.5)*.34, vals, width=.32, color=colors[model], label=labels[model])
        axes[1].bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    axes[1].set_xticks([0, 1], ['Preserve', 'Alter'])
    axes[1].set_ylim(min(0, float(matrix[:, -1].min()))-.01, max(.25, float(matrix[:, -1].max())*1.35))
    axes[1].set_xlabel('Prompt-pair relation')
    axes[1].set_ylabel('Partial Spearman rho')
    axes[1].set_title('(b) Final hidden state')
    axes[1].legend(frameon=False, loc='upper left', fontsize=8)
    fig.suptitle('HumanML3D test: relation-specific attribution-to-hidden association', fontsize=12)
    fig.tight_layout(rect=[0, .045, 1, 1])
    fig.text(.5, .005, 'All 900 preserving / 235 rule-edited pairs; six nuisance controls; point estimates, not a CKA similarity matrix.', ha='center', fontsize=8)
    save(fig, 'relation_specific_association')
    exports['figures']['relation_specific_association'] = {
        'rows': keys, 'partial_rho': matrix.tolist(),
        'caption': 'Relation-specific association between lexically aligned attribution-distribution JSD and teacher-forced hidden-state cosine displacement. The heatmap and final-layer bars report partial rank correlations after controlling for frozen CLIP distance, token-count difference, lexical edit fraction, original token count, valid motion-prefix length, and uniform-attribution JSD. Both preserving and rule-edited relations are shown on common color and value scales. The matrix is not CKA; its values describe associations rather than causal training effects.'}
    (out / 'figure_data_and_captions.json').write_text(json.dumps(exports, indent=2, ensure_ascii=False))
    print(json.dumps({'output': str(out), 'mean_preserve_reduction_percent': overall_reduction,
                      'level_reductions_percent': reductions, 'local_response': response_data}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--paper-view', action='store_true', help='Render mean-only presentation figures from completed test results; preserve originals')
    args = p.parse_args()
    if args.paper_view:
        paper_view(args.input)
    else:
        run(args)