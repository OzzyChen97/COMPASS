"""One complete epoch of four-model shared CLIP training, with fail-closed resume."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.shared_clip_experiment import SharedSystem, MODELS, load_examples, sources, ASSETS, pilot
from tools.train_motiongpt_full_t5 import epoch_indices, synchronize_gradients, atomic_json
from tools.train_motionx_four_model_text_adaptation import filter_native_short_pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--stage', choices=['warmup', 'joint'], required=True)
    p.add_argument('--epoch', type=int, default=1)
    p.add_argument('--resume', type=Path)
    p.add_argument('--microbatch', type=int, default=2)
    p.add_argument('--accumulate', type=int, default=4)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=0)
    a = p.parse_args()
    if min(a.microbatch, a.accumulate) < 1 or min(a.limit, a.max_steps) < 0:
        p.error('Invalid batch or smoke budget')
    if a.epoch not in ([1] if a.stage == 'warmup' else [1,2,3]):
        p.error('Epoch outside fixed training protocol')
    rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0)); torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1: dist.init_process_group('nccl', timeout=timedelta(hours=1))
    torch.set_num_threads(2); pilot.seed_all(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if rank == 0: a.output.mkdir(parents=True, exist_ok=False)
    if world > 1: dist.barrier()
    spec = sources()
    system = SharedSystem(spec, device)
    eligible = a.assets/'eligible_train.jsonl'
    if rank == 0 and not eligible.exists():
        a.assets.mkdir(parents=True, exist_ok=True)
        path = pilot.DATASET_ROOT/str(system.cfg['motiongpt'].DATASET.CODE_PATH)
        report = filter_native_short_pairs(ASSETS/'train_preserve.jsonl', path, pilot.DATASET_ROOT, eligible)
        atomic_json(a.assets/'native_filter.json', report)
    if world > 1: dist.barrier()
    data, reports = load_examples(eligible, system, a.limit)
    n = len(data['t2mgpt'])
    if not a.limit: assert n == 201335
    system.clip.requires_grad_(a.stage == 'joint')
    params = system.adapter_parameters()
    groups = [{'params': params, 'lr': 1e-4, 'peak': 1e-4}]
    if a.stage == 'joint':
        params = list(system.clip.parameters()) + params
        groups.append({'params': list(system.clip.parameters()), 'lr': 1e-6, 'peak': 1e-6})
    assert len(params) == len({id(x) for x in params})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    metadata = dict(spec=spec, eligible_sha256=pilot.sha256_file(eligible), world_size=world,
                    microbatch=a.microbatch, accumulate=a.accumulate, samples=n,
                    frozen=system.frozen_hashes(), stage=a.stage, epoch=a.epoch,
                    limit=a.limit, max_steps=a.max_steps,
                    teacher_policy='frozen original full checkpoints; no evolving CLIP input to teachers',
                    objective='four equal model weights; equal clean/pair CE; native SATO auxiliaries',
                    token_row_policy='first native token row; full pair coverage each epoch',
                    reports=reports)
    global_step = 0
    if a.resume:
        saved = torch.load(a.resume, map_location='cpu', weights_only=False)
        for key in ['spec', 'eligible_sha256', 'world_size', 'microbatch', 'accumulate', 'samples', 'frozen', 'limit', 'max_steps']:
            assert saved['metadata'][key] == metadata[key], key
        assert saved['complete_epoch'], 'Cannot resume truncated smoke epoch'
        if a.stage == saved['metadata']['stage']:
            assert a.epoch == saved['metadata']['epoch']+1
            optimizer.load_state_dict(saved['optimizer']); global_step = saved['global_step']
        else:
            assert saved['metadata']['stage'] == 'warmup' and a.stage == 'joint' and a.epoch == 1
        system.load_compact(saved['state'])
    else:
        assert a.epoch == 1 and (a.stage == 'warmup' or a.max_steps), 'Joint training requires warmup checkpoint'
    # Rank-independent initialization, then rank-specific deterministic sampling.
    if world > 1:
        for value in system.compact_state()['clip'].values(): dist.broadcast(value, 0)
        for branch in system.branches.values():
            for value in branch.adapter.state_dict().values(): dist.broadcast(value, 0)
    initial_clip_hash = pilot.named_parameter_sha256(system.clip.named_parameters())
    indices = epoch_indices(n, world, rank, a.epoch-1, 123)
    chunks = [indices[i:i+a.microbatch] for i in range(0, len(indices), a.microbatch)]
    steps_epoch = math.ceil(len(chunks)/a.accumulate)
    total_steps = steps_epoch*(3 if a.stage == 'joint' else 1)
    pilot.seed_all(123+rank+a.epoch*100003+(10000000 if a.stage == 'joint' else 0))
    atomic_json(a.output/f'coverage_rank{rank}.json', dict(indices_sha256=pilot.canonical_sha256(indices), count=len(indices), unique=len(set(indices))))
    if rank == 0: atomic_json(a.output/'manifest.json', metadata)
    active = MODELS if a.stage == 'joint' else MODELS[2:]
    print(f'SHARED_EPOCH_START rank={rank} stage={a.stage} epoch={a.epoch} samples={n}', flush=True)
    reach = {}
    with (a.output/f'loss_rank{rank}.jsonl').open('x') as log:
        for window in range(0, len(chunks), a.accumulate):
            optimizer.zero_grad(set_to_none=True)
            batches = chunks[window:window+a.accumulate]
            denom = sum(map(len, batches)); metrics = {}
            for ix in batches:
                for name in active:
                    loss, terms = system.loss(name, data[name], ix, device, a.stage)
                    if not torch.isfinite(loss): raise FloatingPointError((name, terms))
                    if window == 0 and name not in reach:
                        probe = system.clip.token_embedding.weight if a.stage == 'joint' else system.branches[name].adapter[1].weight
                        gradient, = torch.autograd.grad(loss, probe, retain_graph=True)
                        reach[name] = float(gradient.norm()); assert reach[name] > 0
                    (loss * len(ix)/denom/len(active)).backward()
                    metrics[name] = terms
            synchronize_gradients(params, world)
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            global_step += 1
            warm = min(100, total_steps)
            scale = global_step/warm if global_step <= warm else .1+.9*.5*(1+math.cos(math.pi*min(1.,(global_step-warm)/max(1,total_steps-warm))))
            for group in optimizer.param_groups: group['lr'] = group['peak']*scale
            optimizer.step()
            record = dict(step=global_step, microstep=window+len(batches), grad_norm=float(norm), metrics=metrics)
            log.write(json.dumps(record)+'\n'); log.flush()
            if rank == 0 and (window == 0 or global_step % 25 == 0): print('SHARED_TRAIN', json.dumps(record), flush=True)
            if a.max_steps and window//a.accumulate+1 >= a.max_steps: break
    assert metadata['frozen'] == system.frozen_hashes(), 'Frozen source or teacher changed'
    if a.stage == 'warmup':
        # Also check CLIP did not acquire gradients in adapter-only training.
        assert all(p.grad is None for p in system.clip.parameters())
        assert initial_clip_hash == pilot.named_parameter_sha256(system.clip.named_parameters())
    digest = pilot.named_parameter_sha256([('clip.'+k,v) for k,v in system.clip.named_parameters()] +
        [(name+'.'+k,v) for name,b in system.branches.items() for k,v in b.adapter.named_parameters()])
    if world > 1:
        hashes = [None]*world; dist.all_gather_object(hashes, digest); assert len(set(hashes)) == 1
    import random
    import numpy as np
    rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
           'cuda': torch.cuda.get_rng_state(device)}
    rng_states = [None]*world
    if world > 1: dist.all_gather_object(rng_states, rng)
    else: rng_states[0] = rng
    atomic_json(a.output/f'gradient_rank{rank}.json', reach)
    if rank == 0:
        state = system.compact_state()
        # Strict round-trip before publishing ready marker.
        payload = dict(state=state, optimizer=optimizer.state_dict(), global_step=global_step,
                       metadata=metadata, complete_epoch=not a.max_steps, student_sha256=digest, rng_states=rng_states)
        temp = a.output/'checkpoint.tmp'; torch.save(payload, temp)
        loaded = torch.load(temp, map_location='cpu', weights_only=False)
        system.load_compact(loaded['state'])
        temp.replace(a.output/'checkpoint.pth')
        atomic_json(a.output/'ready.json', dict(checkpoint=str((a.output/'checkpoint.pth').resolve()),
                     sha256=pilot.sha256_file(a.output/'checkpoint.pth'), complete_epoch=not a.max_steps,
                     frozen_verified=True, shared_rank_hash=digest))
        print('SHARED_EPOCH_COMPLETE', a.stage, a.epoch, flush=True)
    if world > 1: dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__': main()