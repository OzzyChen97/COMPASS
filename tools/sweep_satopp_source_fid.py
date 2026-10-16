"""Sweep HumanML3D SATO++ source checkpoints under the frozen source-encoder FID protocol.

Reuses tools/eval_unified_mixed.py unchanged so every produced row is directly
comparable with ablation/source_encoder_fid_cfg13_20260910/summary.json.

Baseline T2M-GPT reference on the same 4,380-pair mixture:
    FID 0.0805   FID_P 0.2328   FID_D 0.1902
A candidate "wins" only when it is strictly below all three.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.launch_shared_clip_pipeline import env
from tools.audit_motionx_random_retrieval import digest

BASE_ASSETS = ROOT / 'ablation/source_encoder_fid_cfg13_20260910/assets/humanml_satopp.json'
BASELINE = {'FID': 0.08048532397945962, 'FID_P': 0.2327624198771332, 'FID_D': 0.19023732267453397}
PAPER_ROW = {'FID': 0.1954, 'FID_P': 0.1900, 'FID_D': 0.1205}

# Ordered by expected clean-FID promise: runs tuned for quality retention first.
CANDIDATES = [
    ('v3c_best_fid_130k', 'output_GPT_Final/sato_v3c_l1_teach_low_sato_2gpu/sato_pp_best_fid.pth'),
    ('v4b_best_fid_40k', 'output_GPT_Final/sato_v4b_kl_teach_sato_strong_2gpu/sato_pp_best_fid.pth'),
    ('v3c_iter_180k', 'output_GPT_Final/sato_v3c_l1_teach_low_sato_2gpu/sato_pp_iter_180000.pth'),
    ('v4a_best_fid', 'output_GPT_Final/sato_v4a_kl_teach_balanced_2gpu/sato_pp_best_fid.pth'),
    ('v3c_iter_80k', 'output_GPT_Final/sato_v3c_l1_teach_low_sato_2gpu/sato_pp_iter_80000.pth'),
    ('v4b_iter_160k', 'output_GPT_Final/sato_v4b_kl_teach_sato_strong_2gpu/sato_pp_iter_160000.pth'),
    ('v3c_iter_120k', 'output_GPT_Final/sato_v3c_l1_teach_low_sato_2gpu/sato_pp_iter_120000.pth'),
    ('v4b_iter_100k', 'output_GPT_Final/sato_v4b_kl_teach_sato_strong_2gpu/sato_pp_iter_100000.pth'),
    ('v3c_iter_40k', 'output_GPT_Final/sato_v3c_l1_teach_low_sato_2gpu/sato_pp_iter_40000.pth'),
    ('v4a_iter_40k', 'output_GPT_Final/sato_v4a_kl_teach_balanced_2gpu/sato_pp_iter_40000.pth'),
    ('main_iter_110k', 'output_GPT_Final/sato_pp_allloss_gpu0_teachl1_05_eval5k_resume104k_200k_single/sato_pp_iter_110000.pth'),
    ('main_iter_120k', 'output_GPT_Final/sato_pp_allloss_gpu0_teachl1_05_eval5k_resume104k_200k_single/sato_pp_iter_120000.pth'),
    ('main_last', 'output_GPT_Final/sato_pp_allloss_gpu0_teachl1_05_eval5k_resume104k_200k_single/sato_pp_last.pth'),
]


def free_mib(gpu):
    out = subprocess.check_output(
        ['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        text=True, timeout=30)
    return int(out.strip())


def build_assets(tag, checkpoint, out_dir):
    manifest = json.loads(BASE_ASSETS.read_text())
    manifest['ready_models']['humanml/satopp'] = {
        'checkpoint': str(checkpoint),
        'sha256': digest(checkpoint),
        'encoder': 'CLIP',
        'official_clip': True,
    }
    path = out_dir / f'{tag}.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs='+', default=[3, 6, 7])
    p.add_argument('--min-free-mib', type=int, default=34000)
    p.add_argument('--limit', type=int, default=0, help='smoke sample cap; 0 means full 4380')
    p.add_argument('--only', nargs='*', default=None, help='restrict to these tags')
    a = p.parse_args()
    root = a.root.resolve()
    (root / 'assets').mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(parents=True, exist_ok=True)

    todo = [(t, ROOT / c) for t, c in CANDIDATES if a.only is None or t in a.only]
    missing = [t for t, c in todo if not c.exists()]
    if missing:
        raise SystemExit(f'missing checkpoints: {missing}')

    jobs = queue.Queue()
    for tag, checkpoint in todo:
        jobs.put((tag, checkpoint))
    (root / 'protocol.json').write_text(json.dumps(dict(
        purpose='find a SATO++ source checkpoint beating T2M-GPT on FID, FID_P and FID_D at once',
        baseline=BASELINE, paper_row=PAPER_ROW,
        engine_sha256=digest(ROOT / 'tools/eval_unified_mixed.py'),
        base_assets=str(BASE_ASSETS), base_assets_sha256=digest(BASE_ASSETS),
        candidates=[t for t, _ in todo], gpus=a.gpus, min_free_mib=a.min_free_mib,
        limit=a.limit, shared_gpu_policy='never preempt; only start when free memory exceeds the guard',
    ), indent=2) + '\n')

    def worker(gpu):
        while True:
            try:
                tag, checkpoint = jobs.get_nowait()
            except queue.Empty:
                return
            out = root / 'runs' / tag
            if (out / 'summary.json').exists():
                print('SWEEP_SKIP', tag, flush=True)
                continue
            while free_mib(gpu) < a.min_free_mib:
                time.sleep(60)
            assets = build_assets(tag, checkpoint, root / 'assets')
            command = [sys.executable, '-u', 'tools/eval_unified_mixed.py', '--assets', str(assets),
                       '--dataset', 'humanml', '--model', 'satopp', '--output', str(out)]
            if (out / 'run.json').exists():
                command.append('--resume')
            if a.limit:
                command += ['--limit', str(a.limit)]
            print('SWEEP_START', tag, 'gpu', gpu, flush=True)
            started = time.time()
            with (root / 'logs' / f'{tag}.log').open('a') as log:
                r = subprocess.run(command, cwd=ROOT, env=env(str(gpu)), stdout=log,
                                   stderr=subprocess.STDOUT)
            if r.returncode:
                print('SWEEP_FAIL', tag, 'rc', r.returncode, flush=True)
                continue
            s = json.loads((out / 'summary.json').read_text())
            win = all(s[k] < BASELINE[k] for k in BASELINE)
            print('SWEEP_DONE', tag, 'FID %.4f FID_P %.4f FID_D %.4f' % (s['FID'], s['FID_P'], s['FID_D']),
                  'WIN' if win else 'no', '%.0fs' % (time.time() - started), flush=True)

    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        list(pool.map(worker, a.gpus))

    rows = []
    for tag, _ in todo:
        f = root / 'runs' / tag / 'summary.json'
        if f.exists():
            s = json.loads(f.read_text())
            s['tag'] = tag
            s['beats_baseline_all_three'] = all(s[k] < BASELINE[k] for k in BASELINE)
            rows.append(s)
    (root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('SWEEP_COMPLETE', len(rows), 'winners',
          [r['tag'] for r in rows if r['beats_baseline_all_three']], flush=True)


if __name__ == '__main__':
    main()