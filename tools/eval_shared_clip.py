"""Validation/test of one model using an identical shared CLIP checkpoint."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.shared_clip_experiment import sources, verify_sources, MODELS, pilot, ASSETS, seq_ce
from models.shared_clip_conditioning import SharedCLIPText, CLIPMotionGPT
from tools.eval_motiongpt_text_adaptation_validation import load_inference, validation_pairs, CACHE
from tools.audit_motionx_random_retrieval import digest, GROUP_SEEDS, canonical_order, retrieval


@torch.no_grad()
def run(a):
    from dataset.config import configure_evaluator_source, evaluator_opt_path
    from dataset.dataset_TM_eval import CaptionLockedDATALoader
    from models.evaluator_wrapper import EvaluatorModelWrapper
    from options.get_eval_option import get_opt
    from utils.word_vectorizer import WordVectorizer
    from utils.sato_pp_eval import TM2TTextProcessor, generate_motion_batch
    from tools.audit_motionx_text_coverage import LookupVectorizer
    from MotionGPT.eval_loader_robustness import generate_features, get_motion_embeddings_in_input_order
    a.output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda'); torch.set_num_threads(2); pilot.seed_all(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    payload = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    spec = payload['metadata']['spec']; verify_sources(spec)
    assert not (a.original and a.split == 'test')
    official = pilot.load_clip_model(device)
    if a.original and a.model == 'satopp':
        original_source = torch.load(spec[a.model]['checkpoint'], map_location='cpu', weights_only=False)
        official.load_state_dict(original_source['clip'], strict=True)
        del original_source
    clip = SharedCLIPText(official).to(device).eval().requires_grad_(False)
    if not a.original: clip.load_state_dict(payload['state']['clip'], strict=True)
    clip_hash = pilot.named_parameter_sha256(clip.named_parameters())
    mgpt = a.model.startswith('motiongpt')
    if mgpt:
        source = spec[a.model]
        model, data = load_inference(source['config'], source['checkpoint'], device)
        if not a.original:
            model.lm = CLIPMotionGPT(model.lm, clip).to(device).eval().requires_grad_(False)
            model.lm.adapter.load_state_dict(payload['state']['adapters'][a.model], strict=True)
    else:
        from ablation.repeat20_perturb_fast_batch import load_vqvae
        ma = pilot.make_model_args(); trans = pilot.load_transformer(ma, device)
        source = torch.load(spec[a.model]['checkpoint'], map_location='cpu', weights_only=False)
        trans.load_state_dict(source['trans'], strict=True); trans.eval().requires_grad_(False)
        net = load_vqvae(ma, strict=True).eval().requires_grad_(False)
        del source
    if a.split == 'val':
        pairs, expected = validation_pairs(); queries = ['clean', 'P1', 'P2', 'P3']
        cache = np.load(CACHE)
        text_cache = {q: {str(mid): cache['text_fallback_lower_'+q][i] for i, mid in enumerate(cache['motion_id'])} for q in queries}
    else:
        assert a.selection is not None, 'Test requires frozen validation selection'
        selection = json.loads(a.selection.read_text())
        assert selection['checkpoint_sha256'] == digest(a.checkpoint)
        pairs = {}
        for q in ['P1', 'P2', 'P3']:
            path = ROOT/f'dataset/Motion-X-322/test_perb_gemini31_max75_{q.lower()}.json'
            rows = json.loads(path.read_text()); rows = rows['pairs'] if isinstance(rows, dict) else rows
            assert len(rows) == 7679 and len({r['motion_id'] for r in rows}) == 7679
            pairs[q] = rows
        expected = sorted(r['motion_id'] for r in pairs['P1']); queries = ['P1', 'P2', 'P3']
        test = pilot.load_split_ids(pilot.DATASET_ROOT, 'test')
        forbidden = pilot.load_split_ids(pilot.DATASET_ROOT, 'train') | pilot.load_split_ids(pilot.DATASET_ROOT, 'val')
        assert set(expected) <= test and not set(expected) & forbidden
        assert all(sorted(r['motion_id'] for r in rows) == expected for rows in pairs.values())
    if a.limit:
        expected = expected[:a.limit]
        pairs = {q: [r for r in rows if r['motion_id'] in set(expected)] for q, rows in pairs.items()}
    maps = {q: {r['motion_id']: r for r in rows} for q, rows in pairs.items()}
    vectorizer = WordVectorizer(str(ROOT/'glove'), 'our_vab')
    processor = TM2TTextProcessor(LookupVectorizer(vectorizer, 'fallback_lower'), max_text_len=20, alphabetic_only=False)
    loader = CaptionLockedDATALoader('motionx322_75', a.split == 'test', 32, vectorizer,
        pair_data=pairs['P1'], num_workers=0, length_policy='strict' if a.split == 'val' else 'repair-all')
    assert len(loader.dataset) == len(expected)
    ruler = ROOT/'checkpoints/motionx322/text_mot_match_para_aware/model/finest.tar'
    opt = configure_evaluator_source(get_opt(evaluator_opt_path('motionx322_75'), device), 'motionx322_75')
    opt.evaluator_matcher_checkpoint = str(ruler); evaluator = EvaluatorModelWrapper(opt)
    if a.split == 'val':
        old = json.loads((CACHE.parent/'manifest.json').read_text())
        assert old['ruler_sha256'] == digest(ruler)
        assert old['asset_manifest_sha256'] == digest(ASSETS/'manifest.json')
    ids = []; motion = {q: [] for q in queries}; text = {q: [] for q in queries}
    diagnostics = []; truncations = {q: 0 for q in queries}
    with (a.output/'prompts.jsonl').open('x') as audit:
        for bi, batch in enumerate(loader):
            word, pos, captions, lens, real, mlens, _, names = batch
            ids.extend(names)
            for qi, q in enumerate(queries):
                prompts = [maps['P1' if q == 'clean' else q][mid]['text_orig' if q == 'clean' else 'text_pair'] for mid in names]
                pilot.seed_all(123+bi*1000003+qi*100003)
                _, flags = clip.tokenize(prompts); truncations[q] += sum(flags)
                generated = []
                if mgpt:
                    pred, plen = generate_features(model, data, prompts, 'categorical', 300, reference_lengths=mlens,
                        temperature=1., top_k=0, top_p=1., audit_rows=generated)
                else:
                    features = clip.encode_texts(prompts).sentence_features
                    pred, plen = generate_motion_batch(net, trans, features, 300, 322, use_fast_batch=True,
                        cfg_scale=1.3, uncond_feat=torch.zeros_like(features), strict_fast_path=True)
                em = get_motion_embeddings_in_input_order(evaluator, pred, plen)
                motion[q].append(em.cpu().numpy())
                te = np.stack([text_cache[q][mid] for mid in names]) if a.split == 'val' else processor.encode(evaluator, prompts).cpu().numpy()
                text[q].append(te)
                for i, mid in enumerate(names):
                    row = dict(motion_id=mid, query=q, generation_text=prompts[i], retrieval_text=prompts[i], truncated=flags[i], length=int(plen[i]))
                    if generated:
                        if not a.original:
                            assert generated[i]['fulfilled_prompt'] == prompts[i], 'CLIP received a different generation text'
                        row['generation'] = generated[i]
                    audit.write(json.dumps(row)+'\n')
                # Fixed first validation minibatch: correct vs cyclically shuffled
                # text, conditioned on identical native VAE-encoded real motions.
                if a.split == 'val' and bi == 0:
                    targets = []
                    for i in range(len(names)):
                        frames = int(mlens[i])//4*4
                        code = model.vae.encode(real[i:i+1, :frames].to(device))[0] if mgpt else net.encode(real[i:i+1, :frames].to(device))
                        targets.append(code[0])
                    for shuffled in [False, True]:
                        captions_diag = prompts[1:]+prompts[:1] if shuffled else prompts
                        losses = []
                        for i, codes in enumerate(targets):
                            if mgpt and a.original:
                                task = {'input':['Generate motion: <Caption_Placeholder>'], 'output':['<Motion_Placeholder>']}
                                value = model.lm.sato_forward_encdec([captions_diag[i]], codes[None], [len(codes)], [task])['outputs'].loss
                            elif mgpt:
                                ts = model.lm.native_lm.motion_token_to_string(codes[None], [len(codes)])
                                value = model.lm.supervised([captions_diag[i]], ts)['outputs'].loss
                            else:
                                target = torch.cat([codes, codes.new_tensor([512])])[None]
                                logits = trans(target[:, :-1], clip.encode_texts([captions_diag[i]]).sentence_features)
                                value = seq_ce(logits, target, target.new_tensor([target.shape[1]]))
                            losses.append(float(value))
                        diagnostics.append(dict(query=q, shuffled_text=shuffled, samples=len(losses), ce=float(np.mean(losses))))
            audit.flush()
            if bi % 5 == 0: print('SHARED_EVAL_PROGRESS', a.model, a.split, bi, len(loader), flush=True)
    ix = canonical_order(ids, expected); summary = []; runs = []; arrays = {'motion_id': np.asarray(expected)}
    for q in queries:
        mv = np.concatenate(motion[q])[ix]; tv = np.concatenate(text[q])[ix]
        arrays['motion_'+q] = mv; arrays['text_'+q] = tv
        values = []
        for gs in GROUP_SEEDS:
            v = retrieval(torch.from_numpy(tv).to(device), torch.from_numpy(mv).to(device), np.random.default_rng(gs).permutation(len(expected)))
            values.append(v)
            runs.append(dict(model=a.model, query=q, group_seed=gs, **{f'R{k+1}': float(v[k]) for k in range(3)}))
        v = np.mean(values, axis=0)
        summary.append(dict(model=a.model, query=q, samples=len(expected), **{f'R{k+1}': float(v[k]) for k in range(3)}))
    np.savez_compressed(a.output/'embeddings.npz', **arrays)
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (a.output/'conditioning.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    with (a.output/'runs.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0])); w.writeheader(); w.writerows(runs)
    manifest = dict(original=a.original, source_checkpoint_sha256=spec[a.model]['sha256'], split=a.split, samples=len(expected), smoke=bool(a.limit), checkpoint_sha256=digest(a.checkpoint),
        clip_sha256=clip_hash, ruler_sha256=digest(ruler), generation_seed=123, group_seeds=GROUP_SEEDS,
        truncated=truncations, generation_retrieval_same_text=True, shared_text_cache_sha256=digest(CACHE) if a.split == 'val' else None)
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('SHARED_EVAL_COMPLETE', a.model, a.split, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', choices=MODELS, required=True); p.add_argument('--split', choices=['val','test'], default='val')
    p.add_argument('--selection', type=Path); p.add_argument('--limit', type=int, default=0)
    p.add_argument('--original', action='store_true')
    run(p.parse_args())


if __name__ == '__main__': main()