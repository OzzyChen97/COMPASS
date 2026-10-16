"""Consistent replacement of the HumanML3D SATO++ row with the v3c 120k checkpoint.

The paper reports FID variants from the source encoder and every other column
from the same checkpoint after a fixed 3000-step text-only adaptation. Swapping
only the FID columns would mix two generators in one row, so this driver
regenerates the whole dependency set:

  A  3000-step text adaptation of the new source checkpoint
  B  HumanML3D adapted evaluation  -> Matching_P, R@1/2/3_P, Diversity_O
  C  transfer adapted evaluation   -> Matching_P, R@_P, paired L2, first token
  D  transfer source evaluation    -> FID_D under source encoders
  E  baseline control: the synonym-augmentation run's last checkpoint, to show
     how much of the baseline's clean-FID advantage is checkpoint selection

The HumanML3D source-encoder FID for the new checkpoint already exists in
ablation/satopp_source_fid_sweep_20260912/runs/v3c_iter_120k.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.launch_shared_clip_pipeline import env
from tools.audit_motionx_random_retrieval import digest

BASE_ASSETS = ROOT / 'ablation/source_encoder_fid_cfg13_20260910/assets/humanml_satopp.json'
TRAIN_PAIRS = ROOT / 'ablation/humanml_independent_cfg13_20260907/assets/train_preserve.jsonl'
NEW_SOURCE = ROOT / 'output_GPT_Final/sato_v3c_l1_teach_low_sato_2gpu/sato_pp_iter_120000.pth'
BASELINE_CONTROL = ROOT / 'output_GPT_Final/baseline_aug_syn/net_last.pth'
MIN_FREE = 34000


def wait_gpu(gpu, tag):
    while True:
        free = int(subprocess.check_output(
            ['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
            text=True, timeout=30).strip())
        if free >= MIN_FREE:
            return
        print('GPU_WAIT', tag, gpu, free, flush=True)
        time.sleep(60)


def run(cmd, gpu, log, tag):
    wait_gpu(gpu, tag)
    print('STEP_START', tag, 'gpu', gpu, flush=True)
    started = time.time()
    with log.open('a') as f:
        r = subprocess.run([sys.executable, '-u'] + cmd, cwd=ROOT, env=env(str(gpu)),
                           stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        print('STEP_FAIL', tag, 'rc', r.returncode, flush=True)
        raise SystemExit(f'{tag} failed; see {log}')
    print('STEP_DONE', tag, '%.0fs' % (time.time() - started), flush=True)


def write_assets(path, key, checkpoint, extra=None):
    manifest = json.loads(BASE_ASSETS.read_text())
    entry = {'checkpoint': str(checkpoint), 'sha256': digest(checkpoint),
             'encoder': 'CLIP', 'official_clip': True}
    if extra:
        entry.update(extra)
    manifest['ready_models'][key] = entry
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--adapt-gpu', type=int, default=3)
    p.add_argument('--side-gpu', type=int, nargs='+', default=[6, 7])
    a = p.parse_args()
    root = a.root.resolve()
    for sub in ('assets', 'logs', 'runs'):
        (root / sub).mkdir(parents=True, exist_ok=True)

    (root / 'spec.json').write_text(json.dumps(
        {'satopp': {'checkpoint': str(NEW_SOURCE), 'sha256': digest(NEW_SOURCE)}}, indent=2) + '\n')
    source_assets = write_assets(root / 'assets/source.json', 'humanml/satopp', NEW_SOURCE)
    control_assets = write_assets(root / 'assets/baseline_control.json', 'humanml/t2mgpt', BASELINE_CONTROL)

    def side_jobs(gpu_index):
        gpu = a.side_gpu[gpu_index]
        if gpu_index == 0:
            out = root / 'runs/transfer_source'
            if not (out / 'summary.json').exists():
                run(['tools/eval_unified_mixed.py', '--assets', str(source_assets), '--dataset', 'transfer',
                     '--model', 'satopp', '--output', str(out)], gpu, root / 'logs/transfer_source.log',
                    'D_transfer_source')
        else:
            out = root / 'runs/baseline_control'
            if not (out / 'summary.json').exists():
                run(['tools/eval_unified_mixed.py', '--assets', str(control_assets), '--dataset', 'humanml',
                     '--model', 't2mgpt', '--output', str(out)], gpu, root / 'logs/baseline_control.log',
                    'E_baseline_control')

    def main_chain(_):
        adapt_dir = root / 'adaptation/satopp'
        if not (adapt_dir / 'manifest.json').exists():
            run(['tools/train_humanml_independent_text.py', '--model', 'satopp', '--spec',
                 str(root / 'spec.json'), '--train-pairs', str(TRAIN_PAIRS), '--output',
                 str(adapt_dir), '--steps', '3000'], a.adapt_gpu, root / 'logs/adaptation.log', 'A_adaptation')
        meta = json.loads((adapt_dir / 'manifest.json').read_text())
        assert meta['frozen_verified'] and not meta['smoke'], 'adaptation manifest failed its own checks'
        adapted = Path(meta['checkpoint'])
        assert digest(adapted) == meta['checkpoint_sha256']
        adapted_assets = write_assets(root / 'assets/adapted.json', 'humanml/satopp', adapted,
                                      {'official_clip': False})
        for tag, dataset, out in [('B_humanml_adapted', 'humanml', root / 'runs/humanml_adapted'),
                                  ('C_transfer_adapted', 'transfer', root / 'runs/transfer_adapted')]:
            if (out / 'summary.json').exists():
                continue
            run(['tools/eval_unified_mixed.py', '--assets', str(adapted_assets), '--dataset', dataset,
                 '--model', 'satopp', '--output', str(out)], a.adapt_gpu, root / 'logs' / f'{tag}.log', tag)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(main_chain, 0)] + [pool.submit(side_jobs, i) for i in range(len(a.side_gpu))]
        for f in futures:
            f.result()

    collected = {}
    for name in ('humanml_adapted', 'transfer_adapted', 'transfer_source', 'baseline_control'):
        f = root / 'runs' / name / 'summary.json'
        if f.exists():
            collected[name] = json.loads(f.read_text())
    sweep = json.loads((ROOT / 'ablation/satopp_source_fid_sweep_20260912/runs/v3c_iter_120k/summary.json').read_text())
    collected['humanml_source_fid'] = sweep
    (root / 'summary.json').write_text(json.dumps(collected, indent=2) + '\n')
    print('REPLACEMENT_COMPLETE', sorted(collected), flush=True)


if __name__ == '__main__':
    main()