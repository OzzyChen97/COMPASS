"""Validate fixed-budget Motion-X adaptation arms without using test data.

The same generated motions are scored by the unchanged paraphrase-aware ruler
with current and OOV-lowercase-fallback text lookup. No checkpoint selection.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_motionx_random_retrieval import GROUP_SEEDS, canonical_order, digest, retrieval, write_json
from tools.audit_motionx_text_coverage import DEFAULT, LEVELS, LookupVectorizer
from tools.train_motionx_text_adaptation_pilot import make_model_args, load_clip_model, load_transformer, tokenize_texts, BASELINE_TRANS


@torch.no_grad()
def run(args):
    from ablation.repeat20_perturb_fast_batch import load_vqvae
    from dataset.config import configure_evaluator_source, evaluator_opt_path
    from dataset.dataset_TM_eval import CaptionLockedDATALoader
    from models.evaluator_wrapper import EvaluatorModelWrapper
    from options.get_eval_option import get_opt
    from utils.sato_pp_eval import TM2TTextProcessor, generate_motion_batch
    from utils.word_vectorizer import WordVectorizer
    device = torch.device('cuda')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'runs.csv').exists():
        raise FileExistsError('Refusing to overwrite completed validation')
    pairs = {l: json.loads((args.assets / f'val_{l.lower()}.json').read_text()) for l in LEVELS}
    if args.limit:
        keep = set(sorted(r['motion_id'] for r in pairs['P1'])[:args.limit])
        pairs = {l: [r for r in rows if r['motion_id'] in keep] for l, rows in pairs.items()}
    expected = sorted(r['motion_id'] for r in pairs['P1'])
    lookup = {l: {r['motion_id']: r for r in rows} for l, rows in pairs.items()}
    base = WordVectorizer(str(ROOT / 'glove'), 'our_vab')
    processors = {mode: TM2TTextProcessor(LookupVectorizer(base, mode), max_text_len=20, alphabetic_only=False) for mode in ('current', 'fallback_lower')}
    loader = CaptionLockedDATALoader('motionx322_75', False, 32, base, pair_data=pairs['P1'], num_workers=2, length_policy='strict')
    ckpt = ROOT / 'checkpoints/motionx322/text_mot_match_para_aware/model/finest.tar'
    opt = configure_evaluator_source(get_opt(evaluator_opt_path('motionx322_75'), device), 'motionx322_75')
    opt.evaluator_matcher_checkpoint = str(ckpt)
    ruler = EvaluatorModelWrapper(opt)
    model_args = make_model_args()
    clip_model = load_clip_model(device)
    trans = load_transformer(model_args, device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        clip_model.load_state_dict(state['clip'], strict=True)
        trans.load_state_dict(state['trans'], strict=True)
        del state
    clip_model.eval(); trans.eval()
    clip_model.requires_grad_(False); trans.requires_grad_(False)
    net = load_vqvae(model_args, strict=True)
    rows = []
    for seed in args.seeds:
        ids, motions = [], {q: [] for q in args.queries}
        text = {f'{mode}/{q}': [] for mode in processors for q in args.queries}
        lengths = {q: [] for q in args.queries}
        for bi, batch in enumerate(loader):
            word, pos, captions, lens, real, mlens, _, names = batch
            ids.extend(names)
            clean_emb, _ = ruler.get_co_embeddings(word, pos, lens, real, mlens)
            for qi, q in enumerate(args.queries):
                prompts = list(captions) if q == 'clean' else [lookup[q][mid]['text_pair'] for mid in names]
                feat = clip_model.encode_text(tokenize_texts(prompts, device)).float()
                torch.manual_seed(seed + bi * 1000003 + ('clean', *LEVELS).index(q) * 100003)
                pred, plen = generate_motion_batch(net, trans, feat, 300, 322, use_fast_batch=True, cfg_scale=1.3, uncond_feat=torch.zeros_like(feat), strict_fast_path=True)
                _, em = ruler.get_co_embeddings(word, pos, lens, pred, plen)
                motions[q].append(em.cpu().numpy()); lengths[q].extend(plen.cpu().tolist())
                for mode, processor in processors.items():
                    et = clean_emb if q == 'clean' else processor.encode(ruler, prompts)
                    text[f'{mode}/{q}'].append(et.cpu().numpy())
            if bi % 5 == 0:
                print(f'VAL_PROGRESS {args.name} seed={seed} batch={bi}/{len(loader)}', flush=True)
        ix = canonical_order(ids, expected)
        arrays = {'motion_id': np.asarray(expected)}
        for q in args.queries:
            m = torch.as_tensor(np.concatenate(motions[q])[ix], device=device)
            arrays[f'motion_{q}'] = m.cpu().numpy()
            arrays[f'length_{q}'] = np.asarray(lengths[q])[ix]
            for mode in processors:
                t = torch.as_tensor(np.concatenate(text[f'{mode}/{q}'])[ix], device=device)
                arrays[f'text_{mode}_{q}'] = t.cpu().numpy()
                for gs in GROUP_SEEDS:
                    v = retrieval(t, m, np.random.default_rng(gs).permutation(len(expected)))
                    rows.append({'model': args.name, 'generation_seed': seed, 'group_seed': gs, 'query': q, 'lookup': mode, 'samples': len(expected), **{f'R{k+1}': float(v[k]) for k in range(3)}})
        np.savez_compressed(output / f'seed{seed}.npz', **arrays)
        print(f'VAL_SEED_DONE {args.name} {seed}', flush=True)
    with (output / 'runs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = []
    for q in args.queries:
        for mode in processors:
            vals = []
            for seed in args.seeds:
                vals.append(np.mean([[r[f'R{k}'] for k in (1, 2, 3)] for r in rows if r['query'] == q and r['lookup'] == mode and r['generation_seed'] == seed], axis=0))
            v = np.asarray(vals)
            summary.append({'model': args.name, 'query': q, 'lookup': mode, **{f'R{k+1}': float(v[:, k].mean()) for k in range(3)}, **{f'R{k+1}_generation_std': float(v[:, k].std(ddof=1)) if len(v) > 1 else None for k in range(3)}})
    write_json(output / 'summary.json', summary)
    write_json(output / 'manifest.json', {'model': args.name, 'checkpoint': str(args.checkpoint or BASELINE_TRANS), 'checkpoint_sha256': digest(args.checkpoint or BASELINE_TRANS), 'ruler_sha256': digest(ckpt), 'asset_manifest_sha256': digest(args.assets / 'manifest.json'), 'script_sha256': digest(__file__), 'generation_seeds': args.seeds, 'group_seeds': GROUP_SEEDS, 'samples': len(expected), 'queries': args.queries, 'cfg_scale': 1.3, 'cfg_uncond': 'zero', 'text_lookup_controls': list(processors), 'clean_query': 'original annotation tokens unchanged for both lookup variants', 'selection': 'fixed-budget endpoint only; validation split, no test data'})
    print('PILOT_VALIDATION_COMPLETE', args.name, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--assets', type=Path, default=DEFAULT / 'assets')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[123, 124, 125])
    p.add_argument('--queries', nargs='+', choices=('clean', *LEVELS), default=['clean', *LEVELS])
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(args)


if __name__ == '__main__':
    main()