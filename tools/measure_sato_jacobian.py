#!/usr/bin/env python3
"""Direct local Jacobian estimates for the frozen HumanML3D checkpoint pair.

J = d G_K / d H_visible: raw final AR hidden states vs pre-contextual CLIP
content-token embeddings, with BOS/EOS/pad and motion prefixes fixed.
Hutchinson estimates ||J||_F^2; JVP/VJP power iteration estimates ||J||_2.
Also report J / ||G_K(x)||_F, where the denominator is a FIXED basepoint
scalar, to distinguish input sensitivity from the previous relative metric.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.extract_attribution_embeddings import load_bundle, file_sha256, canonical_sha256, get_clip_module
from tools.measure_attribution_latent import words_and_positions, STOPWORDS
from clip_local.simple_tokenizer import SimpleTokenizer
from models.clip_attribution import configure_second_order_attention


class LocalJacobian:
    def __init__(self, function, x):
        self.function = function
        self.x = x.detach().requires_grad_(True)
        self.y = function(self.x)

    def jvp(self, direction):
        # Double-backward JVP avoids relying on unsupported forward-AD kernels.
        _, value = torch.autograd.functional.jvp(self.function, self.x,
                                                 direction, create_graph=False, strict=False)
        return value.detach()

    def vjp(self, cotangent):
        return torch.autograd.grad(self.y, self.x, cotangent,
                                   retain_graph=True, create_graph=False)[0].detach()

    def frobenius(self, probes, seed):
        generator = torch.Generator(device=self.x.device).manual_seed(seed)
        estimates = []
        for _ in range(probes):
            z = torch.randint(0, 2, self.y.shape, device=self.x.device,
                              generator=generator).to(self.x.dtype).mul_(2).sub_(1)
            estimates.append(float(self.vjp(z).double().square().sum()))
        values = np.asarray(estimates)
        return {'squared_norm': float(values.mean()), 'probe_values': estimates,
                'probe_standard_error': float(values.std(ddof=1)/np.sqrt(len(values))) if probes > 1 else None}

    def spectral(self, iterations, restarts, seed, tolerance=1e-3):
        generator = torch.Generator(device=self.x.device).manual_seed(seed)
        runs = []
        for _ in range(restarts):
            v = torch.randn(self.x.shape, device=self.x.device, dtype=self.x.dtype, generator=generator)
            v = v / v.norm()
            trace, converged, residual = [], False, float('inf')
            for i in range(iterations):
                jv = self.jvp(v)
                sigma = float(jv.double().norm())
                normal_v = self.vjp(jv)
                nv_norm = float(normal_v.double().norm())
                trace.append(sigma)
                if nv_norm < 1e-25:
                    residual, converged = 0., True
                    break
                residual = float((normal_v.double() - sigma**2 * v.double()).norm() / nv_norm)
                if i >= 3 and abs(trace[-1]-trace[-2])/max(trace[-1], 1e-30) < tolerance and residual < .02:
                    converged = True
                    break
                v = normal_v / normal_v.norm()
            runs.append({'sigma': trace[-1], 'trace': trace, 'normal_operator_residual': residual, 'converged': converged})
        best = max(runs, key=lambda row: row['sigma'])
        return {'norm_estimate': best['sigma'], 'best_converged': best['converged'],
                'best_residual': best['normal_operator_residual'], 'restarts': runs}


def hidden_function(clip_model, transformer, ids, target, steps):
    with torch.no_grad():
        h = clip_model.token_embedding(ids).detach()
    eos = int(ids[0].argmax())
    prefix, suffix = h[:, :1].detach(), h[:, eos:].detach()
    def forward(content):
        token_input = torch.cat([prefix, content, suffix], dim=1)
        x = token_input + clip_model.positional_embedding.type(clip_model.dtype)
        x = clip_model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        features = clip_model.ln_final(x).type(clip_model.dtype)
        pooled = features[:, eos]
        if clip_model.text_projection is not None:
            pooled = pooled @ clip_model.text_projection
        state = transformer.trans_base(target[:, :steps-1], pooled.float())
        state = transformer.trans_head.blocks(state)
        return transformer.trans_head.ln_f(state)[:, :steps]
    return forward, h[:, 1:eos].detach()


def make_selection(source, count, seed):
    manifest = json.loads((source/'manifest.json').read_text())
    all_rows = [json.loads(line) for line in (source/'baseline_pairs.jsonl').read_text().splitlines()]
    intervention_rows = [json.loads(line) for line in (source/'baseline_interventions.jsonl').read_text().splitlines()]
    eligible = sorted({r['group'] for r in intervention_rows if r['split'] == 'test'})
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible)
    selected = eligible[:count]
    if len(selected) != count:
        raise ValueError(f'Only {len(eligible)} shared intervention groups available')
    anchors = {r['group']: r for r in all_rows if r['split'] == 'test' and r['level'] == 'P1'}
    tokenizer = SimpleTokenizer()
    rows = []
    for group in selected:
        row = anchors[group]
        words = words_and_positions(row['text_orig'], tokenizer)
        candidates = [i for i,w in enumerate(words) if len(w['positions']) == 1 and w['word'].isalpha() and w['word'] not in STOPWORDS]
        order = sorted(candidates, key=lambda i: row['word_raw_orig'][i])
        seed_i = seed + int(hashlib.sha256(group.encode()).hexdigest()[:7], 16)
        choices = {'high': order[-1], 'low': order[0], 'random': int(np.random.default_rng(seed_i).choice(candidates))}
        rows.append({'group': group, 'motion_id': row['motion_id'], 'pair_key': row['pair_key'],
                     'text': row['text_orig'], 'steps': row['steps'], 'seed': seed_i,
                     'target_sha256': row['target_sha256'],
                     'positions': {k: words[i]['positions'][0] for k,i in choices.items()},
                     'words': {k: words[i]['word'] for k,i in choices.items()},
                     'candidate_positions': [words[i]['positions'][0] for i in candidates]})
    return rows, manifest


def directions(operator, row, count, fd_epsilons):
    # Use the SAME baseline-defined high/low/random word and direction in both
    # models. Otherwise q could mix input changes with model changes.
    generator = torch.Generator(device=operator.x.device).manual_seed(row['seed'] + 901)
    width = operator.x.shape[-1]
    norms = [float(operator.x[0,p-1].norm()) for p in row['candidate_positions']]
    median_norm = float(np.median(norms))
    output = {kind: [] for kind in row['positions']}
    for draw in range(count):
        direction_word = torch.randn(width, device=operator.x.device, dtype=operator.x.dtype, generator=generator)
        direction_word /= direction_word.norm()
        for kind, position in row['positions'].items():
            delta = torch.zeros_like(operator.x)
            delta[0, position-1] = direction_word
            jv = operator.jvp(delta)
            gain = float(jv.double().norm()/delta.double().norm())
            finite = []
            if draw == 0:
                for epsilon in fd_epsilons:
                    scale = epsilon * median_norm
                    with torch.no_grad():
                        forward = operator.function(operator.x + scale*delta)
                        backward = operator.function(operator.x - scale*delta)
                        derivative = (forward-backward)/(2*scale)
                    error = float((derivative.double()-jv.double()).norm()/jv.double().norm().clamp_min(1e-30))
                    finite.append({'epsilon': epsilon, 'absolute_step_norm': scale,
                                   'central_difference_gain': float(derivative.double().norm()),
                                   'jvp_relative_error': error})
            output[kind].append({'draw': draw, 'gain': gain, 'finite_difference_checks': finite})
    return output


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '6' or torch.cuda.device_count() != 1:
        raise RuntimeError('This experiment requires exclusively CUDA_VISIBLE_DEVICES=6')
    if min(args.samples,args.probes,args.iterations,args.restarts,args.directions) < 1:
        raise ValueError('All estimator counts must be positive')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    configure_second_order_attention(True)
    source, out = Path(args.source), Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    for m in ('baseline','satopp'):
        if (out/f'{m}_jacobian.jsonl').exists():
            raise FileExistsError(f'Refusing to overwrite {m} results')
    selected, manifest = make_selection(source,args.samples,args.seed)
    source_data = np.load(source/'motion_targets.npz',allow_pickle=False)
    for row in selected:
        if hashlib.sha256(source_data[row['motion_id']].tobytes()).hexdigest()!=row['target_sha256']:
            raise ValueError('Target token hash mismatch')
    config = {'schema':'sato-direct-jacobian/v1','args':vars(args),'selected':selected,
              'selection_digest':canonical_sha256(selected),'source_pair_digest':manifest['pair_digest'],
              'definition':'J = d concat(final normalized AR hidden states at K target-prefix positions) / d visible pre-contextual CLIP token embeddings; BOS/EOS/pad held fixed',
              'normalization':'raw J and J divided by frozen basepoint hidden Frobenius norm; not the derivative of a normalized output function',
              'direction_policy':'baseline-defined high/low words and shared random word, same unit Gaussian directions in both models',
              'spectral_scope':'finite power iteration estimate, not a certified maximum or proof of uniform spectral contraction',
              'gpu_physical':6,'checkpoint_sha256':{'baseline':manifest['baseline_sha256'],'satopp':manifest['reference_sha256']},
              'script_sha256':file_sha256(__file__),'torch_version':torch.__version__}
    (out/'manifest.json').write_text(json.dumps(config,indent=2))
    device = torch.device('cuda:0')
    for model in ('baseline','satopp'):
        ckpt = manifest['baseline_checkpoint'] if model=='baseline' else None
        clip_model, transformer, pool, cfg = load_bundle(manifest['reference_checkpoint'],ckpt,device)
        if file_sha256(ckpt or manifest['reference_checkpoint']) != config['checkpoint_sha256'][model]:
            raise ValueError('Checkpoint hash mismatch')
        with (out/f'{model}_jacobian.jsonl').open('w') as handle:
            for i,row in enumerate(selected):
                ids = get_clip_module().tokenize([row['text']],truncate=False).to(device)
                target = torch.as_tensor(source_data[row['motion_id']],device=device).long()[None]
                function, x = hidden_function(clip_model,transformer,ids,target,row['steps'])
                operator = LocalJacobian(function,x)
                base_norm = float(operator.y.detach().double().norm())
                frob = operator.frobenius(args.probes,row['seed'])
                spectral = operator.spectral(args.iterations,args.restarts,row['seed']+501)
                response = directions(operator,row,args.directions,args.fd_epsilons)
                result = dict(row, model=model, input_shape=list(x.shape), output_shape=list(operator.y.shape),
                              base_hidden_norm=base_norm, input_sha256=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest(),
                              frobenius=frob,spectral=spectral,directional=response,
                              frozen_scale_frobenius_squared=frob['squared_norm']/base_norm**2,
                              frozen_scale_spectral=spectral['norm_estimate']/base_norm)
                handle.write(json.dumps(result,allow_nan=False)+'\n');handle.flush()
                print(f'JACOBIAN {model} {i+1}/{len(selected)} F2={frob["squared_norm"]:.4g} spectral={spectral["norm_estimate"]:.4g} converged={spectral["best_converged"]}',flush=True)
                del operator,function,x
            (out/f'{model}_complete.json').write_text(json.dumps({'samples':len(selected),'selection_digest':config['selection_digest']}))
        del clip_model,transformer,pool
        gc.collect();torch.cuda.empty_cache()
    print('JACOBIAN_COMPLETE '+str(out),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default=str(ROOT/'ablation/attribution_latent_gpu6_20260905/main'))
    p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=32)
    p.add_argument('--probes',type=int,default=16)
    p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--restarts',type=int,default=2)
    p.add_argument('--directions',type=int,default=3)
    p.add_argument('--fd-epsilons',type=float,nargs='+',default=[.001,.01])
    p.add_argument('--seed',type=int,default=123)
    run(p.parse_args())