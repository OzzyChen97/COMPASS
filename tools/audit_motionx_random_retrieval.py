"""Motion-X retrieval audit: shared random candidate groups and GT calibration.

Generation is never rerun or modified. Group seeds are independent of generation
seeds; reported generation standard deviations use 20 generation-level means,
not the 400 dependent generation/group combinations as independent observations.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LEVELS = ('P1', 'P2', 'P3')
MODELS = ('t2mgpt', 'satopp', 'motiongpt', 'motiongpt_satopp')
INTERNAL = {'t2mgpt': 'motionx322_t2mgpt_baseline_200k_best',
            'satopp': 'motionx322_gemini31_sato100k'}
GROUP_SEEDS = list(range(2026090500, 2026090520))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def canonical_order(ids, expected):
    ids = [str(x) for x in ids]
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError('Duplicate or mismatched motion IDs')
    index = {mid: i for i, mid in enumerate(ids)}
    return np.asarray([index[mid] for mid in expected], dtype=np.int64)


def retrieval(text, motion, order, sizes=None):
    """Text-to-motion Euclidean R@1/2/3, with explicit tail and stable ties."""
    n = len(text)
    order = np.asarray(order)
    if text.shape != motion.shape or n < 3:
        raise ValueError('Embeddings must have identical shapes and >=3 rows')
    if len(order) != n or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError('Order must be a full permutation')
    if not torch.isfinite(text).all() or not torch.isfinite(motion).all():
        raise ValueError('Non-finite embeddings')
    sizes = list(sizes) if sizes is not None else [32] * (n // 32) + ([n % 32] if n % 32 else [])
    if sum(sizes) != n or min(sizes) < 3:
        raise ValueError('Invalid candidate sizes')
    # Process adjacent equally sized groups together without changing boundaries.
    total = torch.zeros(3, dtype=torch.float64, device=text.device)
    start = 0
    for size in sorted(set(sizes)):
        groups = []
        start = 0
        for b in sizes:
            if b == size:
                groups.append(order[start:start + b])
            start += b
        ix = torch.as_tensor(np.stack(groups), device=text.device)
        a, b = text[ix], motion[ix]
        d = (a * a).sum(-1, keepdim=True) + (b * b).sum(-1).unsqueeze(1) - 2 * torch.bmm(a, b.transpose(1, 2))
        ranks = torch.argsort(d, dim=-1, stable=True)[..., :3]
        hits = ranks == torch.arange(size, device=text.device)[None, :, None]
        total += hits.cumsum(-1).sum((0, 1))
    return (total / n).cpu().numpy()


def metric_row(**kwargs):
    values = kwargs.pop('values')
    return {**kwargs, **{f'R{k+1}': float(values[k]) for k in range(3)}}


def pair_assets(limit=0):
    pairs, sources = {}, {}
    for level in LEVELS:
        path = ROOT / f'dataset/Motion-X-322/test_perb_gemini31_max75_{level.lower()}.json'
        rows = json.loads(path.read_text())
        if len(rows) != 7679 or len({r['motion_id'] for r in rows}) != 7679:
            raise ValueError('Expected 7679 unique motions per level')
        pairs[level] = rows
        sources[level] = {'path': str(path), 'sha256': digest(path)}
    expected = sorted(r['motion_id'] for r in pairs['P1'])
    originals = {r['motion_id']: r['text_orig'] for r in pairs['P1']}
    for level, rows in pairs.items():
        canonical_order([r['motion_id'] for r in rows], expected)
        if any(r['text_orig'] != originals[r['motion_id']] or r['relation'] != 'preserve' for r in rows):
            raise ValueError('Levels must share original captions and preserve semantics')
    if limit:
        keep = set(expected[:limit])
        pairs = {k: [r for r in v if r['motion_id'] in keep] for k, v in pairs.items()}
        expected = expected[:limit]
    return pairs, expected, sources


def cache_path(run_root, model, level, seed):
    if model in INTERNAL:
        return run_root / f't2m_family/{level.lower()}/{model}/retrieval_embeddings/{INTERNAL[model]}/seed{seed}.npz'
    return run_root / f'motiongpt_family/{model}/{level.lower()}/retrieval_embeddings/seed{seed}.npz'


def generated(args, pairs, expected):
    rows, provenance = [], []
    checkpoint = ROOT / 'checkpoints/motionx322/text_mot_match_para_aware/model/finest.tar'
    checkpoint_hash = digest(checkpoint)
    reference_path = args.run_root.parent / 'final/all_runs.csv'
    with reference_path.open() as f:
        reference = {(r['model'], r['level'], int(r['seed'])): r for r in csv.DictReader(f)}
    if len(reference) != 240:
        raise ValueError('Expected 240 original runs for provenance and numerical parity')
    for model in MODELS:
        for level in LEVELS:
            pair_map = {r['motion_id']: r for r in pairs[level]}
            for seed in range(123, 143):
                path = cache_path(args.run_root, model, level, seed)
                with np.load(path, allow_pickle=False) as z:
                    index_path = ROOT / f'ablation/motionx322_level_rprecision/index_audit/{level.lower()}_sample_index_seed123.json'
                    index = json.loads(index_path.read_text())
                    samples = index['samples']
                    actual_hash = hashlib.sha256(json.dumps(samples, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    if actual_hash != index['sample_index_sha256'] or str(z['sample_index_sha256']) != actual_hash:
                        raise ValueError(f'Sample index checksum mismatch: {path}')
                    if z['loader_name'].astype(str).tolist() != [r['loader_name'] for r in samples]:
                        raise ValueError(f'Cached embedding ordering mismatch: {path}')
                    ids = np.asarray([r['motion_id'] for r in samples])
                    order = canonical_order(ids, expected)
                    for i, mid in enumerate(ids):
                        if samples[i]['caption'] != pair_map[mid]['text_orig'] or samples[i]['perturbed_caption'] != pair_map[mid]['text_pair']:
                            raise ValueError(f'Caption mismatch: {path} {mid}')
                    if 'motion_id' in z and z['motion_id'].astype(str).tolist() != ids.tolist():
                        raise ValueError(f'Embedded motion identity mismatch: {path}')
                    t = torch.as_tensor(z['perturbed_text_embedding'], device=args.device)
                    m = torch.as_tensor(z['perturbed_motion_embedding'], device=args.device)
                    old_gpu = retrieval(t, m, np.arange(len(ids)), z['batch_sizes'])
                    # Reproduce the original NumPy kernel exactly for provenance.
                    # GPU reductions/stable sorting differ on near/exact ties.
                    from utils.sato_pp_eval import calculate_R_precision
                    counts = np.zeros(3)
                    start = 0
                    tn, mn = t.cpu().numpy(), m.cpu().numpy()
                    for size in z['batch_sizes']:
                        counts += calculate_R_precision(tn[start:start + size], mn[start:start + size], top_k=3, sum_all=True)[0]
                        start += size
                    old = counts / len(ids)
                    gpu_numpy_error = float(np.max(np.abs(old_gpu - old)))
                    ref = reference[(model, level, seed)]
                    if ref['evaluator_sha256'] != checkpoint_hash or ref['sample_index_sha256'] != actual_hash:
                        raise ValueError(f'Original evaluator/sample provenance mismatch: {path}')
                    parity_error = float(np.max(np.abs(old - [float(ref[f'top{k}_pert']) for k in (1, 2, 3)])))
                    if parity_error > 1e-12:
                        raise ValueError(f'Original metric parity failed: {path}: {parity_error}')
                    rows.append(metric_row(model=model, level=level, generation_seed=seed, group_seed=-1, protocol='source_order', values=old))
                    t, m = t[order], m[order]
                    for gs in GROUP_SEEDS:
                        values = retrieval(t, m, np.random.default_rng(gs).permutation(len(expected)))
                        rows.append(metric_row(model=model, level=level, generation_seed=seed, group_seed=gs, protocol='random32', values=values))
                provenance.append({'path': str(path), 'sha256': digest(path), 'original_metric_max_abs_error': parity_error, 'gpu_numpy_source_max_abs_error': gpu_numpy_error})
            print(f'GENERATED_DONE {model} {level}', flush=True)
    write_csv(args.output / 'generated_runs.csv', rows)
    write_json(args.output / 'generated_provenance.json', {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint), 'sources': provenance})


def calibrate(args, pairs, expected):
    from dataset.config import configure_evaluator_source, evaluator_opt_path
    from dataset.dataset_TM_eval import CaptionLockedDATALoader
    from models.evaluator_wrapper import EvaluatorModelWrapper
    from options.get_eval_option import get_opt
    from utils.sato_pp_eval import TM2TTextProcessor
    from utils.word_vectorizer import WordVectorizer

    checkpoint = ROOT / ('checkpoints/motionx322/text_mot_match' + ('_para_aware' if args.ruler == 'para' else '') + '/model/finest.tar')
    opt = configure_evaluator_source(get_opt(evaluator_opt_path('motionx322_75'), torch.device(args.device)), 'motionx322_75')
    opt.evaluator_matcher_checkpoint = str(checkpoint)
    wrapper = EvaluatorModelWrapper(opt)
    vectorizer = WordVectorizer(str(ROOT / 'glove'), 'our_vab')
    processor = TM2TTextProcessor(vectorizer, max_text_len=20, alphabetic_only=False)
    loader = CaptionLockedDATALoader('motionx322_75', True, 32, vectorizer, pair_data=pairs['P1'], num_workers=2, unit_length=4, length_policy='repair-all')
    maps = {level: {r['motion_id']: r['text_pair'] for r in rows} for level, rows in pairs.items()}
    ids, sizes, gt, clean = [], [], [], []
    pert = {level: [] for level in LEVELS}
    for bi, batch in enumerate(loader):
        word, pos, captions, lens, motions, mlens, _, names = batch
        et, em = wrapper.get_co_embeddings(word, pos, lens, motions, mlens)
        ids.extend(names)
        sizes.append(len(names))
        gt.append(em.cpu().numpy())
        clean.append(et.cpu().numpy())
        for level in LEVELS:
            ep = processor.encode(wrapper, [maps[level][str(mid)] for mid in names])
            pert[level].append(ep.cpu().numpy())
        if bi % 40 == 0:
            print(f'GT_PROGRESS {args.ruler} {bi}/{len(loader)}', flush=True)
    order = canonical_order(ids, expected)
    arrays = {'gt': np.concatenate(gt), 'clean': np.concatenate(clean), **{level: np.concatenate(pert[level]) for level in LEVELS}}
    rows = []
    motion = torch.as_tensor(arrays['gt'], device=args.device)
    for query in ('clean', *LEVELS):
        text = torch.as_tensor(arrays[query], device=args.device)
        rows.append(metric_row(ruler=args.ruler, query=query, group_seed=-1, protocol='source_order', values=retrieval(text, motion, np.arange(len(ids)), sizes)))
        for gs in GROUP_SEEDS:
            permutation = np.random.default_rng(gs).permutation(len(expected))
            rows.append(metric_row(ruler=args.ruler, query=query, group_seed=gs, protocol='random32', values=retrieval(text[order], motion[order], permutation)))
    np.savez_compressed(args.output / f'gt_{args.ruler}.npz', motion_id=np.asarray(expected), **{k: v[order] for k, v in arrays.items()})
    write_csv(args.output / f'gt_{args.ruler}_runs.csv', rows)
    word_stats = {}
    for level in LEVELS:
        tokens = [processor.tokenize(r['text_pair']) for r in pairs[level]]
        flat = [t.split('/')[0] for row in tokens for t in row]
        word_stats[level] = {'captions': len(tokens), 'truncated_fraction': float(np.mean([len(t) > 20 for t in tokens])), 'oov_token_fraction': sum(w not in vectorizer.word2vec for w in flat) / max(len(flat), 1)}
    write_json(args.output / f'gt_{args.ruler}_provenance.json', {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint), 'resolution': loader.dataset.resolution_stats, 'text_stats': word_stats, 'normalization': {k: hashlib.sha256(np.asarray(getattr(opt, k)).tobytes()).hexdigest() for k in ('source_mean', 'source_std', 'evaluator_mean', 'evaluator_std')}})
    print(f'GT_DONE {args.ruler}', flush=True)


def summarize(args):
    rows = list(csv.DictReader(open(args.output / 'generated_runs.csv')))
    summary = []
    for model in MODELS:
        for level in LEVELS:
            for protocol in ('source_order', 'random32'):
                selected = [r for r in rows if r['model'] == model and r['level'] == level and r['protocol'] == protocol]
                means = []
                for seed in range(123, 143):
                    vals = [[float(r[f'R{k}']) for k in (1, 2, 3)] for r in selected if int(r['generation_seed']) == seed]
                    if len(vals) != (20 if protocol == 'random32' else 1):
                        raise ValueError('Incomplete generation/group coverage')
                    means.append(np.mean(vals, axis=0))
                means = np.asarray(means)
                summary.append({'model': model, 'level': level, 'protocol': protocol, 'generation_seeds': 20, 'groups_per_generation': 20 if protocol == 'random32' else 1, **{f'R{k+1}_mean': float(means[:, k].mean()) for k in range(3)}, **{f'R{k+1}_std_across_generation_means': float(means[:, k].std(ddof=1)) for k in range(3)}})
    write_csv(args.output / 'generated_summary.csv', summary)
    gt = []
    for ruler in ('standard', 'para'):
        rows = list(csv.DictReader(open(args.output / f'gt_{ruler}_runs.csv')))
        for query in ('clean', *LEVELS):
            for protocol in ('source_order', 'random32'):
                vals = np.asarray([[float(r[f'R{k}']) for k in (1, 2, 3)] for r in rows if r['query'] == query and r['protocol'] == protocol])
                if len(vals) != (20 if protocol == 'random32' else 1):
                    raise ValueError('Incomplete GT group coverage')
                gt.append({'ruler': ruler, 'query': query, 'protocol': protocol, **{f'R{k+1}_mean': float(vals[:, k].mean()) for k in range(3)}, **{f'R{k+1}_std_across_groups': float(vals[:, k].std(ddof=1)) if len(vals) > 1 else 0.0 for k in range(3)}})
    write_csv(args.output / 'gt_summary.csv', gt)
    write_json(args.output / 'summary.json', {'generated': summary, 'gt': gt, 'note': 'Post-hoc candidate regrouping; cached generated motions unchanged. Generation uncertainty and group uncertainty are distinct. GT is a reference, not a strict upper bound.'})
    print('SUMMARY_DONE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=('generated', 'calibrate', 'summary'))
    p.add_argument('--ruler', choices=('standard', 'para'), default='para')
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--run-root', type=Path, default=ROOT / 'ablation/motionx322_level_rprecision_para_ruler/runs')
    p.add_argument('--output', type=Path, default=ROOT / 'ablation/motionx322_random32_audit_20260905')
    args = p.parse_args()
    os.chdir(ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.task == 'summary':
        summarize(args)
        return
    if args.limit and args.task != 'calibrate':
        raise ValueError('Limit is only supported for isolated calibration smoke tests')
    pairs, expected, sources = pair_assets(args.limit)
    manifest = {'dataset': 'motionx322_75', 'samples': len(expected), 'canonical_motion_ids': expected, 'group_seeds': GROUP_SEEDS, 'batch_size': 32, 'tail': len(expected) % 32, 'tail_policy': 'retain and sample-weight', 'pair_files': sources, 'script_sha256': digest(__file__), 'task': args.task, 'ruler': args.ruler, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'torch_version': torch.__version__}
    write_json(args.output / f'manifest_{args.task}_{args.ruler}.json', manifest)
    if args.task == 'generated':
        generated(args, pairs, expected)
    else:
        calibrate(args, pairs, expected)


if __name__ == '__main__':
    main()