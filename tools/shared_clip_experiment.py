"""Shared-CLIP experiment utilities; no changes to existing training defaults."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.shared_clip_conditioning import SharedCLIPText, CLIPMotionGPT
from models.shared_clip_sato import CLIPMotionGPTSATOObjective
from tools import train_motionx_text_adaptation_pilot as pilot
from tools.train_motionx_four_model_text_adaptation import load_lm
from tools.train_motiongpt_full_t5 import make_batch

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('t2mgpt', 'satopp', 'motiongpt', 'motiongpt_satopp')
ASSETS = ROOT/'ablation/motionx_text_adaptation_pilot_20260905/assets'


def sources():
    spec = json.loads((ROOT/'ablation/motiongpt_full_t5_epochs3_20260906/source_models.json').read_text())
    spec['t2mgpt'] = dict(checkpoint=str(pilot.BASELINE_TRANS), sha256=pilot.sha256_file(pilot.BASELINE_TRANS))
    return spec


def verify_sources(spec):
    for row in spec.values():
        assert pilot.sha256_file(Path(row['checkpoint'])) == row['sha256']


def seq_ce(logits, target, lengths):
    mask = torch.arange(target.shape[1], device=target.device)[None] < lengths[:, None]
    safe = target.masked_fill(~mask, -100)
    loss = F.cross_entropy(logits[:, :target.shape[1]].transpose(1, 2), safe, ignore_index=-100, reduction='none')
    return ((loss * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()


def t2m_batch(examples, indices, device, pkeep=.5):
    selected = [examples[i] for i in indices]
    lengths = torch.tensor([len(e.token_rows[0])+1 for e in selected], device=device)
    assert int(lengths.max()) <= 76
    targets = torch.full((len(selected), int(lengths.max())), 513, device=device, dtype=torch.long)
    for i, e in enumerate(selected):
        n = len(e.token_rows[0])
        targets[i, :n] = torch.as_tensor(e.token_rows[0], device=device)
        targets[i, n] = 512
    inputs = targets[:, :-1]
    corrupted = torch.where(torch.rand(inputs.shape, device=device) < pkeep,
                            inputs, torch.randint(512, inputs.shape, device=device))
    return selected, targets, lengths, corrupted


class SharedSystem(nn.Module):
    def __init__(self, spec, device):
        super().__init__()
        verify_sources(spec)
        official = pilot.load_clip_model(device)
        self.clip = SharedCLIPText(official).float().to(device).eval()
        self.trans = nn.ModuleDict()
        self.branches = nn.ModuleDict()
        self.cfg = {}
        for name in MODELS[:2]:
            model = pilot.load_transformer(pilot.make_model_args(), device)
            source = torch.load(spec[name]['checkpoint'], map_location='cpu', weights_only=False)
            model.load_state_dict(source['trans'], strict=True)
            self.trans[name] = model.requires_grad_(False).eval()
            if name == 'satopp':
                from models.sato_pp_modules import AttributionPooling, PredictorHead
                args = SimpleNamespace(**source['args'])
                self.sato_args = args
                width = source['attr_pooling']['phi.weight'].shape[1] if 'phi.weight' in source['attr_pooling'] else self.clip.width
                self.pool = AttributionPooling(width, args.proj_dim, args.proj_type).to(device)
                self.head = PredictorHead(args.proj_dim).to(device)
                self.pool.load_state_dict(source['attr_pooling'], strict=True)
                self.head.load_state_dict(source['predictor_head'], strict=True)
                self.pool.requires_grad_(False).eval(); self.head.requires_grad_(False).eval()
                # Complete frozen original SATO++ checkpoint as teacher anchor.
                # Unlike native training's evolving input features, this teacher
                # has a fixed text encoder; record this intentional adaptation.
                teacher_official = pilot.load_clip_model(device)
                teacher_official.load_state_dict(source['clip'], strict=True)
                self.teacher_clip = SharedCLIPText(teacher_official).requires_grad_(False).eval()
                self.teacher_trans = copy.deepcopy(model).requires_grad_(False).eval()
            del source
        for name in MODELS[2:]:
            lm, source, cfg = load_lm(spec[name]['config'], spec[name]['checkpoint'])
            self.cfg[name] = cfg
            branch = CLIPMotionGPT(lm, self.clip).float().to(device).train()
            self.branches[name] = branch
            if name.endswith('satopp'):
                teacher = copy.deepcopy(lm).to(device).eval()
                self.mgpt_sato = CLIPMotionGPTSATOObjective(branch, teacher, source, cfg).to(device).train()
            del source
        self.spec = spec

    def adapter_parameters(self):
        return [p for branch in self.branches.values() for p in branch.adapter.parameters()]

    def frozen_hashes(self):
        modules = {**{n: m for n, m in self.trans.items()},
                   **{n: b.native_lm for n, b in self.branches.items()},
                   'teacher_clip': self.teacher_clip, 'teacher_trans': self.teacher_trans,
                   'pool': self.pool, 'head': self.head,
                   'mgpt_teacher': self.mgpt_sato._sato_teacher.lm,
                   'mgpt_pool': self.mgpt_sato.attr_pooling, 'mgpt_head': self.mgpt_sato.predictor_head}
        return {n: pilot.named_parameter_sha256(m.named_parameters()) for n, m in modules.items()}

    def compact_state(self):
        return {'clip': self.clip.state_dict(),
                'adapters': {n: b.adapter.state_dict() for n, b in self.branches.items()}}

    def load_compact(self, state):
        self.clip.load_state_dict(state['clip'], strict=True)
        for n, b in self.branches.items():
            b.adapter.load_state_dict(state['adapters'][n], strict=True)

    def loss(self, name, examples, indices, device, stage):
        if name.startswith('motiongpt'):
            batch = make_batch(examples, indices, device)
            branch = self.branches[name]
            if name.endswith('satopp') and stage == 'joint':
                loss = self.mgpt_sato(batch)
                scale = 1 + float(self.cfg[name].SATO.LAMBDA_PAIR_GEN)
                return loss/scale, {k: float(v) for k, v in self.mgpt_sato.metrics.items()}
            targets = branch.native_lm.motion_token_to_string(batch['motion'], batch['length'])
            out = branch.supervised(batch['text']+batch['text_pair'], targets+targets)
            from tools.train_motionx_four_model_text_adaptation import sequence_ce
            loss = sequence_ce(out['outputs'].logits, out['labels'])
            return loss, {'generation': float(loss.detach())}
        selected, target, lengths, corrupted = t2m_batch(examples, indices, device)
        clean = [e.text_orig for e in selected]; pair = [e.text_pair for e in selected]
        features = self.clip.encode_texts(clean+pair, require_input_grad=name == 'satopp')
        logits = self.trans[name](torch.cat([corrupted, corrupted]), features.sentence_features)
        n = len(indices)
        gen = .5*(seq_ce(logits[:n], target, lengths)+seq_ce(logits[n:], target, lengths))
        metrics = {'generation': float(gen.detach())}
        if name == 't2mgpt':
            return gen, metrics
        from models.sato_pp_losses import loss_teacher, loss_align, loss_variance, loss_covariance
        with torch.no_grad():
            teacher_feature = self.teacher_clip.encode_texts(clean).sentence_features
            teacher_logits = self.teacher_trans(corrupted, teacher_feature)
        teach = loss_teacher(logits[:n], teacher_logits, lengths, self.sato_args.teacher_loss_type)
        # Match native per-sample early-token attribution (uncorrupted motion).
        ar_logits = self.trans[name](torch.cat([target[:, :-1], target[:, :-1]]), features.sentence_features)
        lens = torch.cat([lengths-1, lengths-1]).clamp(min=1, max=self.sato_args.attribution_K)
        width = min(int(lens.max()), ar_logits.shape[1])
        labels = torch.cat([target, target])[:, :width]
        valid = torch.arange(width, device=device)[None] < lens[:, None]
        labels = labels.masked_fill(~valid, 0)
        logp = F.log_softmax(ar_logits[:, :width], -1).gather(-1, labels[..., None]).squeeze(-1)
        score = ((logp * valid).sum(1)/valid.sum(1)).sum()
        grad, = torch.autograd.grad(score, features.input_embeddings, create_graph=True, retain_graph=True)
        attr = (grad * features.input_embeddings.detach()).abs().sum(-1)
        # Native T2M SATO normalizes over full CLIP context; preserve this rule.
        alpha = torch.softmax(attr / self.sato_args.tau, -1)
        u = self.pool(features.token_features, alpha)
        align = loss_align(self.head(u[:n]), self.head(u[n:]))
        if n >= 2:
            var, cov = loss_variance(u, self.sato_args.var_threshold), loss_covariance(u)
        else:
            var = cov = gen.new_zeros(())
        total = gen + self.sato_args.lambda_teach*teach + self.sato_args.lambda_align*align
        total = total + self.sato_args.lambda_var*var + self.sato_args.lambda_cov*cov
        metrics.update(teacher=float(teach.detach()), align=float(align.detach()),
                       var=float(var.detach()), cov=float(cov.detach()))
        return total, metrics


def load_examples(eligible, system, limit=0):
    data = {}; reports = {}; cache = {}
    for name in MODELS:
        path = pilot.TOKEN_DIR if not name.startswith('motiongpt') else pilot.DATASET_ROOT/str(system.cfg[name].DATASET.CODE_PATH)
        nb = 512 if not name.startswith('motiongpt') else system.branches[name].native_lm.m_codebook_size
        key = (str(path), nb)
        if key not in cache:
            cache[key] = pilot.load_training_pairs(eligible, token_dir=path, nb_code=nb, dataset_limit=limit or None)
        data[name], reports[name] = cache[key]
    keys = [[(e.motion_id, e.pair_id, e.text_orig, e.text_pair) for e in data[n]] for n in MODELS]
    assert all(k == keys[0] for k in keys)
    forbidden = pilot.load_split_ids(pilot.DATASET_ROOT, 'val') | pilot.load_split_ids(pilot.DATASET_ROOT, 'test')
    assert not {e.motion_id for e in data['t2mgpt']} & forbidden
    return data, reports