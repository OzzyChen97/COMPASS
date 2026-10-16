"""Train/validation-only Motion-X text coverage controls; never reads test captions.

Validation rewrites are reused only for exact matching original captions from the
historical train pair pool. Their motion IDs are independent. All matching clean
or rewrite strings are then excluded from this adaptation's training pairs.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_motionx_random_retrieval import digest, write_json, canonical_order, retrieval, GROUP_SEEDS
from tools.build_motionx322_val_pairs_from_train_pool import basic_filter, deterministic_digest, load_annotations, load_rows, read_split

LEVELS = ('P1', 'P2', 'P3')
DATA = ROOT / 'dataset/Motion-X-322-MaxToken75'
POOL = ROOT / 'dataset/Motion-X-322/preserving_pairs_gemini31_human_recipe_filtered.jsonl'
DEFAULT = ROOT / 'ablation/motionx_text_adaptation_pilot_20260905'


def normalize(text):
    return ' '.join(text.lower().strip().split())


def semantic_flags(orig, pert):
    """Review flags only, not semantic equivalence labels or automatic exclusions."""
    a, b = set(re.findall(r'[a-z]+', orig.lower())), set(re.findall(r'[a-z]+', pert.lower()))
    flags = []
    for family, words in [('direction', {'left', 'right', 'clockwise', 'counterclockwise', 'forward', 'backward'}), ('negation', {'not', 'never', 'without'})]:
        if a & words != b & words:
            flags.append(family)
    if set(re.findall(r'\d+', orig)) != set(re.findall(r'\d+', pert)):
        flags.append('numerals')
    return flags


def prepare(out):
    assets = out / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    splits = {k: set(read_split(DATA, k)) for k in ('train', 'val', 'test')}
    assert not (splits['train'] & splits['val'] or splits['train'] & splits['test'] or splits['val'] & splits['test'])
    source_train = set(read_split(ROOT / 'dataset/Motion-X-322', 'train'))
    rows = load_rows(POOL)
    if any(str(r.get('motion_id')) not in source_train for r in rows):
        raise ValueError('Pool includes nontrain motion IDs')
    by_caption = defaultdict(lambda: defaultdict(list))
    accepted = []
    for row in rows:
        if row.get('relation') == 'preserve' and row.get('level') in LEVELS and basic_filter(row['text_orig'], row['text_pair']):
            by_caption[row['text_orig'].strip()][row['level']].append(row)
            accepted.append(row)
    val = {level: [] for level in LEVELS}
    for mid in sorted(splits['val']):
        anns = load_annotations(DATA / 'texts' / f'{mid}.txt')
        caption = next((a['caption'] for a in anns if a['full_motion'] and all(by_caption[a['caption']].get(l) for l in LEVELS)), None)
        if caption is None:
            continue
        for level in LEVELS:
            choice = min(by_caption[caption][level], key=lambda r: deterministic_digest(123, mid, level, r['text_pair']))
            val[level].append({'motion_id': mid, 'text_orig': caption, 'text_pair': choice['text_pair'], 'relation': 'preserve', 'level': level, 'pair_id': len(val[level]) + 1, 'source_motion_id': choice['motion_id']})
    if len(val['P1']) < 32:
        raise ValueError('Too few independent validation motions')
    held_texts = {normalize(r[k]) for v in val.values() for r in v for k in ('text_orig', 'text_pair')}
    train = [r for r in accepted if r['motion_id'] in splits['train'] and normalize(r['text_orig']) not in held_texts and normalize(r['text_pair']) not in held_texts]
    if not train:
        raise ValueError('Caption-disjoint training set is empty')
    train.sort(key=lambda r: deterministic_digest(123, r['motion_id'], r['level'], r['text_pair']))
    train_path = assets / 'train_preserve.jsonl'
    with train_path.open('w') as f:
        for row in train:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    for level in LEVELS:
        write_json(assets / f'val_{level.lower()}.json', val[level])
    review = []
    flag_counts = {}
    for split, subset in [('train', train), *[(f'val_{k}', v) for k, v in val.items()]]:
        counts = Counter()
        for row in subset:
            flags = semantic_flags(row['text_orig'], row['text_pair'])
            counts.update(flags)
            if flags and sum(r['split'] == split for r in review) < 30:
                review.append({'split': split, 'flags': flags, **row})
        flag_counts[split] = {'rows': len(subset), 'flags': dict(counts)}
    write_json(assets / 'semantic_review_candidates.json', review)
    report = {'source_pool': str(POOL), 'source_pool_sha256': digest(POOL), 'train_pairs': len(train), 'train_motions': len({r['motion_id'] for r in train}), 'train_level_counts': dict(Counter(r['level'] for r in train)), 'val_motions_per_level': {k: len(v) for k, v in val.items()}, 'excluded_train_rows_matching_validation_text': sum(r['motion_id'] in splits['train'] for r in accepted) - len(train), 'train_val_motion_overlap': 0, 'train_val_exact_normalized_text_overlap': 0, 'test_usage': 'IDs only for split integrity; no test captions, perturbations, metrics or model selection', 'validation_limitation': 'Historical train-generated rewrites transferred to distinct val motions by exact caption match. Matching text removed from adaptation train, but baseline pretraining may have seen those texts. Heuristic flags require human semantic review.', 'semantic_review': flag_counts, 'asset_hashes': {p.name: digest(p) for p in assets.glob('*') if p.is_file() and p.name != 'manifest.json'}}
    write_json(assets / 'manifest.json', report)
    print(json.dumps(report, indent=2), flush=True)


class LookupVectorizer:
    def __init__(self, base, mode):
        self.base, self.mode = base, mode
        self.word2vec = base.word2vec

    def lookup_word(self, word):
        if self.mode == 'lower_all' or (self.mode == 'fallback_lower' and word not in self.word2vec):
            return word.lower()
        return word

    def __getitem__(self, token):
        word, pos = token.rsplit('/', 1)
        return self.base[f'{self.lookup_word(word)}/{pos}']


def coverage(out, device):
    from dataset.config import configure_evaluator_source, evaluator_opt_path
    from dataset.dataset_TM_eval import CaptionLockedDATALoader
    from models.evaluator_wrapper import EvaluatorModelWrapper
    from options.get_eval_option import get_opt
    from utils.sato_pp_eval import TM2TTextProcessor
    from utils.word_vectorizer import WordVectorizer
    assets = out / 'assets'
    val = {l: json.loads((assets / f'val_{l.lower()}.json').read_text()) for l in LEVELS}
    train = load_rows(assets / 'train_preserve.jsonl')
    # Fixed train-only sample, balanced across levels. Full validation is used.
    samples = {f'train_{l}': [r for r in train if r['level'] == l][:1024] for l in LEVELS}
    samples.update({f'val_{l}': v for l, v in val.items()})
    base = WordVectorizer(str(ROOT / 'glove'), 'our_vab')
    vectors = {mode: LookupVectorizer(base, mode) for mode in ('current', 'fallback_lower', 'lower_all')}
    proc = {mode: TM2TTextProcessor(v, max_text_len=20, alphabetic_only=False) for mode, v in vectors.items()}
    stats = {}
    for key, sample in samples.items():
        tokens = [proc['current'].tokenize(r['text_pair']) for r in sample]
        counts = Counter(t.rsplit('/', 1)[0] for ts in tokens for t in ts)
        oov = Counter({w: n for w, n in counts.items() if w not in base.word2vec})
        stats[key] = {'pairs': len(sample), 'tokens': sum(counts.values()), 'truncated_fraction': float(np.mean([len(ts) > 20 for ts in tokens])), 'oov_fraction': {mode: sum(n for w, n in counts.items() if v.lookup_word(w) not in base.word2vec) / max(sum(counts.values()), 1) for mode, v in vectors.items()}, 'oov_recoverable_by_lowercase': sum(n for w, n in oov.items() if w.lower() in base.word2vec), 'oov_top30': oov.most_common(30)}
        print('COVERAGE', key, stats[key]['oov_fraction'], flush=True)
    write_json(out / 'coverage_stats.json', stats)
    expected = sorted(r['motion_id'] for r in val['P1'])
    maps = {l: {r['motion_id']: r for r in v} for l, v in val.items()}
    scores, provenance = [], {}
    for ruler in ('standard', 'para'):
        opt = configure_evaluator_source(get_opt(evaluator_opt_path('motionx322_75'), torch.device(device)), 'motionx322_75')
        ckpt = ROOT / ('checkpoints/motionx322/text_mot_match' + ('_para_aware' if ruler == 'para' else '') + '/model/finest.tar')
        opt.evaluator_matcher_checkpoint = str(ckpt)
        wrapper = EvaluatorModelWrapper(opt)
        provenance[ruler] = {'checkpoint': str(ckpt), 'sha256': digest(ckpt)}
        loader = CaptionLockedDATALoader('motionx322_75', False, 32, base, pair_data=val['P1'], num_workers=2, length_policy='strict')
        motion, ids, embeddings = [], [], {f'{mode}/{q}': [] for mode in proc for q in ('clean', *LEVELS)}
        for word, pos, captions, lens, motions, mlens, _, names in loader:
            _, em = wrapper.get_co_embeddings(word, pos, lens, motions, mlens)
            motion.append(em.cpu().numpy()); ids.extend(names)
            for mode, processor in proc.items():
                for q in ('clean', *LEVELS):
                    texts = list(captions) if q == 'clean' else [maps[q][mid]['text_pair'] for mid in names]
                    embeddings[f'{mode}/{q}'].append(processor.encode(wrapper, texts).cpu().numpy())
        ix = canonical_order(ids, expected)
        m = torch.as_tensor(np.concatenate(motion)[ix], device=device)
        for key, chunks in embeddings.items():
            t = torch.as_tensor(np.concatenate(chunks)[ix], device=device)
            values = np.asarray([retrieval(t, m, np.random.default_rng(seed).permutation(len(expected))) for seed in GROUP_SEEDS])
            mode, query = key.split('/')
            scores.append({'ruler': ruler, 'lookup': mode, 'query': query, 'samples': len(expected), **{f'R{k+1}': float(values[:, k].mean()) for k in range(3)}, **{f'R{k+1}_group_std': float(values[:, k].std(ddof=1)) for k in range(3)}})
        print('VALIDATION_DONE', ruler, flush=True)
    with (out / 'coverage_validation.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0])); writer.writeheader(); writer.writerows(scores)
    write_json(out / 'coverage_manifest.json', {'rulers': provenance, 'asset_manifest_sha256': digest(assets / 'manifest.json'), 'group_seeds': GROUP_SEEDS, 'variants': list(proc), 'clean_text_processing': 'raw original caption through same spaCy processor as paraphrases; all variants identical except vector lookup', 'scope': 'inference-only controls; no checkpoint training or replacement', 'extended_vectors': 'No larger pretrained GloVe asset is present locally; no downloaded or randomly initialized vocabulary expansion used.'})
    print('COVERAGE_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=('prepare', 'coverage'))
    p.add_argument('--output', type=Path, default=DEFAULT)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.task == 'prepare':
        prepare(args.output)
    else:
        coverage(args.output, args.device)


if __name__ == '__main__':
    main()