"""Run shared CLIP train/validation stages on GPUs 0-3, then frozen test."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.shared_clip_experiment import MODELS, pilot


def select_shared(results):
    scores = {}
    for name, rows in results.items():
        expected = {(m, q) for m in MODELS for q in ['clean', 'P1', 'P2', 'P3']}
        assert {(r['model'], r['query']) for r in rows} == expected and len(rows) == 16
        assert all(r['samples'] == 661 and 0 <= r['R3'] <= 1 for r in rows)
        scores[name] = sum(r['R3'] for r in rows if r['query'] != 'clean') / 12
    chosen = max(scores, key=scores.get)
    return chosen, scores


def env(gpus):
    e = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
             PYTHONNOUSERSITE='1', TOKENIZERS_PARALLELISM='false', NCCL_DEBUG='WARN',
             LD_LIBRARY_PATH='/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64')
    e.pop('PYTHONPATH', None)
    return e


def wait_free(gpus):
    while True:
        rows = subprocess.check_output(['nvidia-smi','-i',gpus,'--query-gpu=memory.used','--format=csv,noheader,nounits'], text=True, timeout=20)
        if all(int(x.strip()) < 1024 for x in rows.splitlines()): return
        print('SHARED_GPU_WAIT', gpus, flush=True); time.sleep(60)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    logs = a.output/'logs'; logs.mkdir()
    snapshot = a.output/'source_snapshot'; snapshot.mkdir()
    paths = ['tools/shared_clip_experiment.py','tools/train_shared_clip_epoch.py','tools/eval_shared_clip.py',
             'tools/launch_shared_clip_pipeline.py','models/shared_clip_conditioning.py','models/shared_clip_sato.py']
    hashes = {}
    for file in paths:
        shutil.copy2(ROOT/file, snapshot/Path(file).name); hashes[file] = pilot.sha256_file(ROOT/file)
    protocol = dict(gpus=[0,1,2,3], warmup_epochs=1, joint_epochs=3, seed=123,
        microbatch=2, accumulate=4, effective_batch_pairs=32, source_hashes=hashes,
        shared_clip='official ViT-B/32 text only', clip_lr=1e-6, adapter_lr=1e-4,
        test_after_validation_selection=True,
        teacher_policy='each original complete model checkpoint, frozen text and motion networks',
        t2m_sato_changes='equal clean/pair CE; fixed full teacher; native full-context attribution; local minibatch variance/covariance',
        randomness='deterministic epoch/rank reseeding; checkpoints only at full epoch boundaries')
    (a.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    def run(command, gpus, logname):
        for file, expected in hashes.items(): assert pilot.sha256_file(ROOT/file) == expected, 'Source changed during experiment'
        wait_free(gpus)
        with (logs/logname).open('x') as log:
            print('SHARED_JOB_START', logname, gpus, flush=True)
            subprocess.run([sys.executable, '-u']+command, cwd=ROOT, env=env(gpus), stdout=log, stderr=subprocess.STDOUT, check=True)
        print('SHARED_JOB_DONE', logname, flush=True)
    def evaluate(checkpoint, stage, split='val', selection=None, original=False):
        def one(i, model):
            dest = a.output/'evaluation'/stage/model
            cmd = ['tools/eval_shared_clip.py','--model',model,'--checkpoint',str(checkpoint),
                   '--output',str(dest),'--split',split]
            if selection: cmd += ['--selection',str(selection)]
            if original: cmd += ['--original']
            run(cmd, str(i), f'{stage}_{model}.log')
            return dest
        with ThreadPoolExecutor(max_workers=4) as pool:
            dirs = list(pool.map(lambda im: one(*im), enumerate(MODELS)))
        manifests = [json.loads((d/'manifest.json').read_text()) for d in dirs]
        for key in ['checkpoint_sha256','ruler_sha256','samples','smoke'] + ([] if original else ['clip_sha256']):
            assert len({json.dumps(m[key]) for m in manifests}) == 1, key
        assert not manifests[0]['smoke']
        return [row for d in dirs for row in json.loads((d/'summary.json').read_text())]
    checkpoints = {}; results = {}; previous = None
    for stage, epoch in [('warmup',1),('joint',1),('joint',2),('joint',3)]:
        name = f'{stage}{epoch}'; dest = a.output/'training'/name
        cmd = ['-m','torch.distributed.run','--nproc_per_node=4','--master_addr=127.0.0.1','--master_port=29713',
               'tools/train_shared_clip_epoch.py','--output',str(dest),'--assets',str(a.output/'assets'),
               '--stage',stage,'--epoch',str(epoch),'--microbatch','2','--accumulate','4']
        if previous: cmd += ['--resume',str(previous)]
        run(cmd, '0,1,2,3', name+'.log')
        ready = json.loads((dest/'ready.json').read_text()); assert ready['complete_epoch'] and ready['frozen_verified']
        previous = dest/'checkpoint.pth'; assert pilot.sha256_file(previous) == ready['sha256']
        checkpoints[name] = previous
        results[name] = evaluate(previous, name)
        if stage == 'warmup':
            controls = evaluate(previous, 'original_controls', original=True)
            (a.output/'original_controls.json').write_text(json.dumps(controls, indent=2)+'\n')
        (a.output/'validation_results.json').write_text(json.dumps(results, indent=2)+'\n')
    chosen, scores = select_shared(results)
    validation_rows = [dict(variant=name, **r) for name, rows in {'original':controls, **results}.items() for r in rows]
    with (a.output/'validation_summary.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(validation_rows[0])); w.writeheader(); w.writerows(validation_rows)
    selection = dict(selected=chosen, scores=scores, checkpoint=str(checkpoints[chosen]),
                     checkpoint_sha256=pilot.sha256_file(checkpoints[chosen]), split='val', samples=661)
    path = a.output/'selection.json'; path.write_text(json.dumps(selection, indent=2)+'\n')
    test = evaluate(checkpoints[chosen], 'selected_test', 'test', path)
    (a.output/'test_summary.json').write_text(json.dumps(test, indent=2)+'\n')
    with (a.output/'test_summary.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(test[0])); w.writeheader(); w.writerows(test)
    print('SHARED_PIPELINE_COMPLETE', flush=True)


if __name__ == '__main__': main()