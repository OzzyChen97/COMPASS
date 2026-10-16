#!/usr/bin/env python3
"""GPU-restricted attribution/Transformer-state diagnostics, not training.

Uses per-sample early target likelihood, frozen models, identical motion prefixes,
word-aligned attribution comparisons, and equal-norm symmetric input interventions.
Outputs raw per-pair records so statistical analysis is independently repeatable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.extract_attribution_embeddings import (
    canonical_sha256, file_sha256, get_clip_module, load_bundle,
    load_vqvae, load_caption_locked_motion, encode_motion_tokens,
)
from dataset.config import dataset_mean_std_paths
from clip_local.simple_tokenizer import SimpleTokenizer, basic_clean, whitespace_clean

DEFAULT_REFERENCE = ROOT / 'output_GPT_Final/sato_pp_allloss_gpu0_teachl1_05_eval5k_resume104k_200k_single/sato_pp_iter_130000.pth'
DEFAULT_BASELINE = ROOT / 'pretrained/VQTransformer_corruption05/net_best_fid.pth'
STOPWORDS = set('a an the person individual someone man woman human is are was were be being been its his her their it they he she and or of to in on at with from then while does doing very something some both'.split())
EDIT_MAPS = {
    'direction': {'left': 'right', 'right': 'left', 'forward': 'backward', 'backward': 'forward', 'clockwise': 'counterclockwise', 'counterclockwise': 'clockwise'},
    'action': {'walks': 'runs', 'walking': 'running', 'walk': 'run', 'runs': 'walks', 'running': 'walking', 'run': 'walk', 'sitting': 'standing', 'standing': 'sitting'},
    'bodypart': {'hand': 'foot', 'hands': 'feet', 'arm': 'leg', 'arms': 'legs', 'leg': 'arm', 'legs': 'arms', 'foot': 'hand', 'feet': 'hands'},
}


def source_group(motion_id):
    """Original and mirrored HumanML3D motions must remain one statistical group."""
    return str(motion_id).removeprefix('M')


def words_and_positions(text, tokenizer):
    """Use the actual CLIP regex/BPE, never infer positions by whitespace."""
    import regex
    clean = whitespace_clean(basic_clean(text)).lower()
    words = regex.findall(tokenizer.pat, clean)
    records, ids, pos = [], [], 1
    for word in words:
        encoded = tokenizer.encode(word)
        records.append({'word': word, 'positions': list(range(pos, pos + len(encoded)))})
        ids.extend(encoded)
        pos += len(encoded)
    if ids != tokenizer.encode(text):
        raise ValueError('Word segmentation does not reproduce CLIP tokenization')
    if pos > 76:
        raise ValueError('Text would be truncated by CLIP')
    return records


def jsd(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    def kl(x):
        keep = x > 0
        return np.sum(x[keep] * np.log(x[keep] / m[keep]))
    return float((kl(p) + kl(q)) / 2)


def aligned_attr_distance(words_a, weights_a, words_b, weights_b):
    """Order-independent identical-word occurrence alignment plus unmatched bucket.

    This is lexical, NOT oracle semantic alignment. Unmatched synonyms share one
    bucket; coverage is reported and controlled in partial correlations.
    """
    def indexed(words, weights):
        counts, values = Counter(), {}
        for record, weight in zip(words, weights):
            word = record['word']
            key = (word, counts[word])
            counts[word] += 1
            values[key] = weight
        return values
    a, b = indexed(words_a, weights_a), indexed(words_b, weights_b)
    shared = sorted(set(a) & set(b))
    pa = [a[k] for k in shared] + [sum(v for k, v in a.items() if k not in b)]
    pb = [b[k] for k in shared] + [sum(v for k, v in b.items() if k not in a)]
    coverage = 2 * len(shared) / max(1, len(a) + len(b))
    return jsd(pa, pb), coverage


def cosine_distance(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.clip(1 - np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30), 0, 2))


def generate_edit(text, kind):
    mapping = EDIT_MAPS[kind]
    matches = list(re.finditer(r'\b[A-Za-z]+\b', text))
    for match in matches:
        old = match.group().lower()
        if old in mapping:
            new = mapping[old]
            edited = text[:match.start()] + new + text[match.end():]
            return edited, {'kind': kind, 'old': old, 'new': new, 'char_start': match.start()}
    return None


def prepare_pairs(args, tokenizer):
    data = Path(args.data_root)
    train = {source_group(x) for x in (data / 'train.txt').read_text().split()}
    val = {source_group(x) for x in (data / 'val.txt').read_text().split()}
    test = {source_group(x) for x in (data / 'test.txt').read_text().split()}
    selected, counts, source_hashes = [], {}, {}
    for split in args.splits:
        allowed = val if split == 'val' else test
        candidates = defaultdict(dict)
        if split == 'val':
            paths = [data / 'val_ruler_perb_p123.json']
            records = json.loads(paths[0].read_text())
        else:
            paths = [ROOT / f'ablation/humanml3d_table1_levels/index_audit/p{level}_sample_index_seed123.json' for level in (1, 2, 3)]
            records = []
            for path in paths:
                payload = json.loads(path.read_text())
                if canonical_sha256(payload['samples']) != payload['sample_index_sha256']:
                    raise ValueError('Sample index digest mismatch')
                for row in payload['samples']:
                    # Cached code sequences describe the full motion. Exclude repaired
                    # submotion/resampling captions instead of silently mismatching them.
                    if row.get('length_repair', {}).get('repair_type', 'none') != 'none':
                        continue
                    records.append({'motion_id': row['motion_id'], 'text_orig': row['caption'], 'text_pair': row['perturbed_caption'], 'relation': 'preserve', 'level': row['level'], 'pair_id': row['pair_id']})
        for path in paths:
            source_hashes[str(path)] = file_sha256(path)
        for row in records:
            group = source_group(row['motion_id'])
            if group not in allowed or (split == 'test' and group in train):
                continue
            if split == 'test' and group in val:
                continue
            if not (data / 'new_joint_vecs' / f"{row['motion_id']}.npy").is_file():
                continue
            try:
                for key in ('text_orig', 'text_pair'):
                    words_and_positions(row[key], tokenizer)
            except ValueError:
                continue
            # Pick a deterministic single orientation, avoiding mirrored duplicates.
            key = row['level']
            existing = candidates[group].get(key)
            if existing is None or (row['motion_id'], str(row['pair_id'])) < (existing['motion_id'], str(existing['pair_id'])):
                candidates[group][key] = dict(row)
        eligible = [g for g in sorted(candidates) if set(candidates[g]) >= {'P1', 'P2', 'P3'}]
        random.Random(args.seed + (0 if split == 'val' else 1)).shuffle(eligible)
        chosen = eligible[:args.motions]
        if len(chosen) < args.motions:
            raise ValueError(f'{split}: only {len(chosen)} eligible motion groups, requested {args.motions}')
        n_alter = Counter()
        for group in chosen:
            for level in ('P1', 'P2', 'P3'):
                row = dict(candidates[group][level], split=split, group=group, edit_type='')
                row['pair_key'] = f'{split}/{group}/{level}'
                selected.append(row)
            anchor = candidates[group]['P1']
            # Balanced preference order; at most one synthetic edit per source group.
            kinds = sorted(EDIT_MAPS, key=lambda k: (n_alter[k], k))
            for kind in kinds:
                edit = generate_edit(anchor['text_orig'], kind)
                if edit is None:
                    continue
                text, annotation = edit
                try:
                    words_and_positions(text, tokenizer)
                except ValueError:
                    continue
                row = dict(anchor, text_pair=text, relation='alter', level='controlled_edit', split=split, group=group, edit_type=kind, edit=annotation)
                row['pair_key'] = f'{split}/{group}/alter'
                row['synthetic_edit'] = True
                selected.append(row)
                n_alter[kind] += 1
                break
        counts[split] = {'motion_groups': len(chosen), 'eligible_groups': len(eligible), 'groups_overlapping_current_train_list': len(set(chosen) & train), 'preserve_pairs': 3 * len(chosen), 'synthetic_alter_pairs': sum(n_alter.values()), 'alter_types': dict(n_alter)}
    return selected, {'counts': counts, 'source_hashes': source_hashes, 'test_train_group_exclusion': True, 'val_is_exploratory_training_overlap': True, 'mirrors_grouped': True, 'test_excludes_val_groups': True, 'selection_seed': args.seed, 'target_policy': 'frozen VQVAE encode of deterministic full-motion multiple-of-4 center crop; same uncorrupted target prefix for both prompts; EOS excluded from K'}


class Extractor:
    def __init__(self, clip_model, transformer, pool, cfg, tokenizer, device):
        self.clip, self.trans, self.pool, self.cfg = clip_model, transformer, pool, cfg
        self.tokenizer, self.device = tokenizer, device
        self.names = []
        self.handles = []
        self.captured = []
        for part in ('trans_base', 'trans_head'):
            for i, block in enumerate(getattr(transformer, part).blocks):
                name = f'{part}.blocks.{i}'
                self.names.append(name)
                self.handles.append(block.register_forward_hook(self.capture))
        self.names.append('trans_head.ln_f')
        self.handles.append(transformer.trans_head.ln_f.register_forward_hook(self.capture))

    def capture(self, module, inputs, output):
        self.captured.append(output[:, :self.steps].detach())

    def forward(self, token_input, ids, target, steps):
        self.steps, self.captured = steps, []
        x = token_input + self.clip.positional_embedding.type(self.clip.dtype)
        x = self.clip.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        features = self.clip.ln_final(x).type(self.clip.dtype)
        pooled = features[torch.arange(len(ids), device=self.device), ids.argmax(-1)]
        if self.clip.text_projection is not None:
            pooled = pooled @ self.clip.text_projection
        # Only compute the causal prefix needed for the diagnostic.
        logits = self.trans(target[:, :steps - 1], pooled.float())
        logp = logits[:, :steps].log_softmax(-1)
        score = logp.gather(-1, target[:, :steps, None]).squeeze(-1).mean(-1)
        states = torch.stack(self.captured, dim=1)  # batch, layer, time, width
        return features, pooled, logp, score, states

    def encode(self, text, target_np, valid_len):
        words = words_and_positions(text, self.tokenizer)
        ids = get_clip_module().tokenize([text], truncate=False).to(self.device)
        expected = [49406] + self.tokenizer.encode(text) + [49407]
        if ids[0, :len(expected)].tolist() != expected:
            raise ValueError('Tokenizer mismatch')
        target = torch.as_tensor(target_np, device=self.device).long()[None]
        steps = min(int(self.cfg.get('attribution_K', 16)), valid_len - 1)
        if steps < 1:
            raise ValueError('No valid motion codes')
        h = self.clip.token_embedding(ids).detach().requires_grad_(True)
        features, pooled, logp, score, states = self.forward(h, ids, target, steps)
        gradient, = torch.autograd.grad(score.sum(), h)
        raw = (gradient.float() * h.detach().float()).abs().sum(-1)[0]
        visible = torch.zeros(77, dtype=torch.bool, device=self.device)
        visible[1:len(expected) - 1] = True
        tau = float(self.cfg.get('tau', 1.0))
        alpha = torch.softmax(raw.masked_fill(~visible, -torch.inf) / tau, -1)
        training_alpha = torch.softmax(raw / tau, -1)
        word_alpha = np.asarray([float(alpha[w['positions']].sum()) for w in words])
        word_raw = [float(raw[w['positions']].sum()) for w in words]
        alpha_np = alpha[visible].detach().cpu().numpy().astype(np.float64)
        entropy = float(-(alpha_np * np.log(np.maximum(alpha_np, 1e-300))).sum() / np.log(max(2, len(alpha_np))))
        with torch.no_grad():
            weighted = (features.float() * alpha[None, :, None]).sum(1)[0]
            uniform = features[0, visible].float().mean(0)
            u = self.pool(features.float(), training_alpha[None])[0] if self.pool else None
        return {
            'words': words, 'word_alpha': word_alpha, 'word_raw': word_raw,
            'entropy': entropy, 'visible_count': int(visible.sum()),
            'raw_sum': float(raw[visible].sum()),
            'training_visible_mass': float(training_alpha[visible].sum()),
            'weighted': weighted.cpu().numpy(), 'uniform': uniform.cpu().numpy(),
            'pool': u.cpu().numpy() if u is not None else None,
            'pooled': pooled[0].detach().float().cpu().numpy(),
            'states': states[0].float().cpu().numpy(),
            'logp': logp[0].detach().float().cpu().numpy(), 'score': float(score.detach()),
            'ids': ids.detach(), 'h': h.detach(), 'target': target, 'steps': steps,
        }

    def intervene(self, encoded, seed, epsilons, repeats):
        candidates = [i for i, w in enumerate(encoded['words']) if len(w['positions']) == 1 and w['word'].isalpha() and w['word'] not in STOPWORDS]
        if len(candidates) < 3:
            return []
        rng = np.random.default_rng(seed)
        ordered = sorted(candidates, key=lambda i: encoded['word_raw'][i])
        choices = {'high': ordered[-1], 'low': ordered[0], 'random': int(rng.choice(candidates))}
        median_norm = float(np.median([float(encoded['h'][0, encoded['words'][i]['positions'][0]].norm()) for i in candidates]))
        rows = []
        for eps in epsilons:
            batches, labels = [], []
            for draw in range(repeats):
                vec = rng.normal(size=encoded['h'].shape[-1]).astype(np.float32)
                vec /= np.linalg.norm(vec)
                vec = torch.tensor(vec, device=self.device) * (eps * median_norm)
                for kind, index in choices.items():
                    pos = encoded['words'][index]['positions'][0]
                    for sign in (-1, 1):
                        h = encoded['h'].clone()
                        h[0, pos] += sign * vec
                        batches.append(h)
                        labels.append((kind, index, draw, sign))
            with torch.no_grad():
                h = torch.cat(batches)
                ids = encoded['ids'].expand(len(h), -1)
                target = encoded['target'].expand(len(h), -1)
                _, _, logp, score, states = self.forward(h, ids, target, encoded['steps'])
                base_states = torch.as_tensor(encoded['states'], device=self.device)[None]
                dstate = (1 - F.cosine_similarity(states.float(), base_states, dim=-1)).clamp_min(0).mean(-1)
                relative = ((states.float() - base_states).square().sum((-1, -2)) / base_states.square().sum((-1, -2)).clamp_min(1e-30)).sqrt()
                base_logp = torch.as_tensor(encoded['logp'], device=self.device)[None]
                mix = torch.logaddexp(base_logp, logp) - np.log(2)
                dlogp = .5 * ((base_logp.exp() * (base_logp - mix)).sum(-1) + (logp.exp() * (logp - mix)).sum(-1)).mean(-1)
                score_change = score - encoded['score']
            for kind, index in choices.items():
                mask = [j for j, label in enumerate(labels) if label[0] == kind]
                rows.append({'kind': kind, 'epsilon_relative_to_sentence_median_embedding_norm': eps, 'perturbation_l2': eps * median_norm, 'directions': repeats, 'symmetric_signs': True,
                    'word': encoded['words'][index]['word'], 'token_position': encoded['words'][index]['positions'][0], 'candidate_count': len(candidates), 'word_raw_attribution': encoded['word_raw'][index],
                    'hidden_cosine': dstate[mask].mean(0).cpu().tolist(), 'hidden_relative_l2': relative[mask].mean(0).cpu().tolist(),
                    'output_jsd': float(dlogp[mask].mean()), 'score_abs_change': float(score_change[mask].abs().mean()), 'score_signed_change': float(score_change[mask].mean())})
        return rows

    def close(self):
        for handle in self.handles:
            handle.remove()


def pair_metrics(row, a, b, names, target_hash):
    da, coverage = aligned_attr_distance(a['words'], a['word_alpha'], b['words'], b['word_alpha'])
    sa, sb = a['states'].astype(np.float64), b['states'].astype(np.float64)
    cos = 1 - (sa * sb).sum(-1) / np.maximum(np.linalg.norm(sa, axis=-1) * np.linalg.norm(sb, axis=-1), 1e-30)
    relative = np.sqrt(((sa - sb) ** 2).sum((-1, -2)) / np.maximum((sa ** 2).sum((-1, -2)), 1e-30))
    la, lb = a['logp'].astype(np.float64), b['logp'].astype(np.float64)
    mix = np.logaddexp(la, lb) - np.log(2)
    return dict(row, target_sha256=target_hash, steps=a['steps'], layers=names,
        attribution_jsd=da, lexical_coverage=coverage, edit_fraction=1-coverage,
        token_count_orig=a['visible_count'], token_count_pair=b['visible_count'],
        length_difference=abs(a['visible_count'] - b['visible_count']),
        entropy_orig=a['entropy'], entropy_pair=b['entropy'],
        attribution_words_orig=[w['word'] for w in a['words']], attribution_words_pair=[w['word'] for w in b['words']],
        word_alpha_orig=a['word_alpha'].tolist(), word_alpha_pair=b['word_alpha'].tolist(),
        word_raw_orig=a['word_raw'], word_raw_pair=b['word_raw'],
        training_visible_mass_orig=a['training_visible_mass'], training_visible_mass_pair=b['training_visible_mass'],
        clip_distance=cosine_distance(a['pooled'], b['pooled']),
        weighted_distance=cosine_distance(a['weighted'], b['weighted']),
        uniform_distance=cosine_distance(a['uniform'], b['uniform']),
        learned_pool_distance=cosine_distance(a['pool'], b['pool']) if a['pool'] is not None else None,
        hidden_cosine=np.clip(cos.mean(-1), 0, 2).tolist(), hidden_relative_l2=relative.tolist(),
        output_jsd=float(.5 * ((np.exp(la)*(la-mix)).sum(-1)+(np.exp(lb)*(lb-mix)).sum(-1)).mean()),
        target_score_orig=a['score'], target_score_pair=b['score'])


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '6':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES=6; no other GPU is permitted')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible GPU required')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = SimpleTokenizer()
    pairs, selection = prepare_pairs(args, tokenizer)
    manifest = dict(selection, schema='attribution-latent/v1', pairs=pairs, pair_digest=canonical_sha256(pairs), reference_checkpoint=str(args.reference), reference_sha256=file_sha256(args.reference), baseline_checkpoint=str(args.baseline), baseline_sha256=file_sha256(args.baseline), gpu_physical=6, args=vars(args), torch_version=torch.__version__, numpy_version=np.__version__, intervention='single BPE content word, equal absolute L2, paired Gaussian direction, symmetric signs', attribution_alignment='identical lexical word occurrences plus unmatched aggregate bucket (not semantic alignment)', temperature_policy='checkpoint tau; word diagnostics exclude BOS/EOS/pad; learned pool uses original 77-position normalization', observation='association and input faithfulness only; synthetic altering pairs have no target edited motion')
    manifest['source_code_sha256'] = {str(Path(__file__).resolve()): file_sha256(__file__)}
    manifest['data_split_sha256'] = {split: file_sha256(Path(args.data_root) / f'{split}.txt') for split in ('train', 'val', 'test')}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['pair_digest'] != manifest['pair_digest']:
            raise ValueError('Existing output has different pairs; use a new directory')
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"SELECTED {selection['counts']}", flush=True)
    reference = torch.load(args.reference, map_location='cpu', weights_only=False)
    cfg = dict(reference['args'])
    del reference
    vq = load_vqvae(args.reference, args.vq_checkpoint, device)
    vq_state = torch.load(args.vq_checkpoint, map_location='cpu', weights_only=False)
    vq.load_state_dict(vq_state.get('net', vq_state), strict=True)
    del vq_state
    mean_path, std_path = dataset_mean_std_paths(cfg.get('dataname', 't2m'))
    mean, std = np.load(mean_path), np.load(std_path)
    targets = {}
    for row in pairs:
        motion_id = row['motion_id']
        if motion_id not in targets:
            motion, length = load_caption_locked_motion(args.data_root, row, mean, std)
            targets[motion_id] = encode_motion_tokens(vq, motion, cfg['nb_code'], cfg['block_size'], device)
    np.savez_compressed(out / 'motion_targets.npz', **{mid: values[0] for mid, values in targets.items()},
                        motion_ids=np.array(list(targets)), valid_lengths=np.array([v[1] for v in targets.values()]))
    manifest['vq_checkpoint_sha256'] = file_sha256(args.vq_checkpoint)
    manifest['normalization_sha256'] = {'mean': file_sha256(mean_path), 'std': file_sha256(std_path)}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    del vq
    torch.cuda.empty_cache()
    print(f'TARGETS_ENCODED {len(targets)}', flush=True)
    for model in args.models:
        output_path = out / f'{model}_pairs.jsonl'
        interventions_path = out / f'{model}_interventions.jsonl'
        if output_path.exists() or interventions_path.exists():
            raise FileExistsError(f'Refusing to overwrite {model} records in {out}')
        clip_model, transformer, pool, cfg = load_bundle(args.reference, args.baseline if model == 'baseline' else None, device)
        extractor = Extractor(clip_model, transformer, pool, cfg, tokenizer, device)
        counters = Counter()
        with output_path.open('w') as records, interventions_path.open('w') as interventions:
            for i, row in enumerate(pairs):
                target, valid = targets[row['motion_id']]
                a = extractor.encode(row['text_orig'], target, valid)
                b = extractor.encode(row['text_pair'], target, valid)
                result = pair_metrics(row, a, b, extractor.names, hashlib.sha256(target.tobytes()).hexdigest())
                records.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                if row['level'] == 'P1' and counters[row['split']] < args.intervention_motions:
                    seed = args.seed + int(hashlib.sha256(row['pair_key'].encode()).hexdigest()[:8], 16)
                    tests = extractor.intervene(a, seed, args.epsilons, args.directions)
                    if tests:
                        counters[row['split']] += 1
                        for test in tests:
                            interventions.write(json.dumps(dict(test, split=row['split'], group=row['group'], motion_id=row['motion_id'], pair_key=row['pair_key'], layers=extractor.names), allow_nan=False) + '\n')
                del a, b
                if (i+1) % 50 == 0:
                    records.flush()
                    interventions.flush()
                    print(f'PROGRESS {model} {i+1}/{len(pairs)} intervention_groups={dict(counters)}', flush=True)
        extractor.close()
        del extractor, clip_model, transformer, pool
        gc.collect()
        torch.cuda.empty_cache()
        (out / f'{model}_complete.json').write_text(json.dumps({'pairs': len(pairs), 'intervention_groups': dict(counters), 'pair_digest': manifest['pair_digest']}))
        print(f'COMPLETE {model} pairs={len(pairs)}', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', default=str(DEFAULT_REFERENCE))
    p.add_argument('--baseline', default=str(DEFAULT_BASELINE))
    p.add_argument('--data-root', default=str(ROOT / 'dataset/HumanML3D'))
    p.add_argument('--vq-checkpoint', default=str(ROOT / 'pretrained/VQVAE/net_best_fid.pth'))
    p.add_argument('--output', required=True)
    p.add_argument('--models', nargs='+', choices=['baseline', 'satopp'], default=['baseline', 'satopp'])
    p.add_argument('--splits', nargs='+', choices=['val', 'test'], default=['val', 'test'])
    p.add_argument('--motions', type=int, default=300)
    p.add_argument('--intervention-motions', type=int, default=100)
    p.add_argument('--epsilons', type=float, nargs='+', default=[0.1, 0.25])
    p.add_argument('--directions', type=int, default=3)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--cpu-threads', type=int, default=4)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())