"""Preflight only: real shared CLIP / native MotionGPT checkpoints, train data only."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.shared_clip_conditioning import SharedCLIPText, CLIPMotionGPT
from models.shared_clip_sato import CLIPMotionGPTSATOObjective
from tools import train_motionx_text_adaptation_pilot as pilot
from tools.train_motionx_four_model_text_adaptation import load_lm
from tools.train_motiongpt_full_t5 import make_batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['motiongpt', 'motiongpt_satopp'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    pilot.seed_all(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    device = torch.device('cuda')
    spec = json.loads((ROOT/'ablation/motiongpt_full_t5_epochs3_20260906/source_models.json').read_text())[a.model]
    assert pilot.sha256_file(Path(spec['checkpoint'])) == spec['sha256']
    native, source, cfg = load_lm(spec['config'], spec['checkpoint'])
    teacher = copy.deepcopy(native).float().to(device).eval() if a.model.endswith('satopp') else None
    official = pilot.load_clip_model(device)
    shared = SharedCLIPText(official).float().to(device).eval()
    branch = CLIPMotionGPT(native, shared).float().to(device).train()
    rows_path = ROOT/'ablation/motionx_text_adaptation_pilot_20260905/assets/train_preserve.jsonl'
    train_ids = pilot.load_split_ids(pilot.DATASET_ROOT, 'train')
    forbidden = pilot.load_split_ids(pilot.DATASET_ROOT, 'val') | pilot.load_split_ids(pilot.DATASET_ROOT, 'test')
    examples = []
    with rows_path.open() as f:
        for line in f:
            r = json.loads(line)
            assert r['motion_id'] in train_ids and r['motion_id'] not in forbidden
            assert r['relation'] == 'preserve'
            path = pilot.DATASET_ROOT / str(cfg.DATASET.CODE_PATH) / (r['motion_id'] + '.npy')
            if not path.exists():
                continue
            tokens = np.load(path, allow_pickle=False)
            token_row = tokens[0] if tokens.ndim > 1 else tokens
            examples.append(SimpleNamespace(token_rows=[token_row], **r))
            if len(examples) == 2:
                break
    assert len(examples) == 2
    batch = make_batch(examples, [0, 1], device)
    ids, truncated = shared.tokenize(batch['text'])
    with torch.no_grad():
        error = (shared(ids).sentence_features - official.encode_text(ids)).abs().max().item()
    assert error <= 1e-5, error
    before = pilot.named_parameter_sha256(branch.native_lm.named_parameters())
    teacher_hash = pilot.named_parameter_sha256(teacher.named_parameters()) if teacher is not None else None
    report = dict(model=a.model, split='train', smoke=True, samples=2, clip_parity_max_error=error,
                  truncated=truncated, source=spec)
    with patch.object(branch.language_model.encoder, 'forward', side_effect=AssertionError('Student T5 encoder used')):
        if teacher is not None:
            objective = CLIPMotionGPTSATOObjective(branch, teacher, source, cfg).to(device).train()
            loss = objective(batch)
            report['losses'] = {k: float(v) for k, v in objective.metrics.items()}
        else:
            out = branch.sato_forward_encdec(batch['text'], batch['motion'], batch['length'], batch['tasks'])
            loss = out['outputs'].loss
        assert torch.isfinite(loss)
        loss.backward()
        report['loss'] = float(loss.detach())
        report['clip_gradient_norm'] = float(shared.token_embedding.weight.grad.norm())
        report['adapter_gradient_norm'] = float(branch.adapter[1].weight.grad.norm())
        assert report['clip_gradient_norm'] > 0 and report['adapter_gradient_norm'] > 0
        assert all(p.grad is None for p in branch.native_lm.parameters())
        assert before == pilot.named_parameter_sha256(branch.native_lm.named_parameters())
        if teacher is not None:
            assert teacher_hash == pilot.named_parameter_sha256(teacher.named_parameters())
            assert all(p.grad is None for p in teacher.parameters())
            assert all(p.grad is None for p in objective.attr_pooling.parameters())
            assert all(p.grad is None for p in objective.predictor_head.parameters())
        branch.generate_direct(batch['text'], max_length=8, do_sample=False)
    report['frozen_verified'] = True
    report['student_encoder_bypassed'] = True
    report['cuda_peak_allocated'] = torch.cuda.max_memory_allocated()
    (a.output/'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('SHARED_CLIP_PREFLIGHT_OK', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()