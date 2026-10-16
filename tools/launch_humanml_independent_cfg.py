"""HumanML3D adaptation and CFG tests, concurrent with Motion-X (GPUs 4-7)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.humanml_independent_assets import MODELS,pilot
from tools.launch_shared_clip_pipeline import env


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();a.root=a.root.resolve()
    spec=a.root/'source_models.json';pairs=a.root/'assets/train_preserve.jsonl'
    protocol=dict(dataset='humanml3d',gpus=[4,5,6,7],independent_text_encoders=True,
        training_steps=3000,learning_rate=1e-6,warmup_steps=100,generation_cfg=1.3,
        generation_seed=123,selection='fixed 3000-step endpoint; not selected on test',
        concurrent_with_motionx=True,teacher_losses_in_append_adaptation=False,
        note='encoder-only supervised adaptation; preserves original SATO source model, no shared CLIP')
    (a.root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def run(cmd,gpu,log):
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=20).strip())
            if free>=32768:break
            print('HUMAN_GPU_WAIT',gpu,free,flush=True);time.sleep(60)
        with log.open('x') as f:subprocess.run([sys.executable,'-u']+cmd,cwd=ROOT,env=env(str(gpu)),stdout=f,stderr=subprocess.STDOUT,check=True)
    def worker(im):
        i,m=im;gpu=i+4;out=a.root/'main'/m
        smoke=json.loads((a.root/'smoke2'/m/'manifest.json').read_text());assert smoke['frozen_verified']
        print('HUMAN_FORMAL_START',m,gpu,flush=True)
        run(['tools/train_humanml_independent_text.py','--model',m,'--spec',str(spec),
             '--train-pairs',str(pairs),'--output',str(out),'--steps','3000'],gpu,a.root/'logs'/f'train_{m}.log')
        meta=json.loads((out/'manifest.json').read_text());assert meta['frozen_verified'] and not meta['smoke']
        checkpoint=Path(meta['checkpoint']);assert pilot.sha256_file(checkpoint)==meta['checkpoint_sha256']
        print('HUMAN_FORMAL_TRAIN_DONE',m,flush=True)
        for level in ['P1','P2','P3']:
            dest=a.root/'test'/m/level
            run(['tools/eval_humanml_independent_cfg.py','--model',m,'--spec',str(spec),'--checkpoint',str(checkpoint),
                 '--level',level,'--output',str(dest)],gpu,a.root/'logs'/f'test_{m}_{level}.log')
            result=json.loads((dest/'manifest.json').read_text())
            assert result['cfg_scale']==1.3 and result['samples']==4380 and not result['smoke']
            print('HUMAN_FORMAL_TEST_DONE',m,level,flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,enumerate(MODELS)))
    rows=[]
    for m in MODELS:
        for level in ['P1','P2','P3']:
            rows.extend(json.loads((a.root/'test'/m/level/'summary.json').read_text()))
    with (a.root/'humanml_summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('HUMAN_INDEPENDENT_CFG_COMPLETE',flush=True)


if __name__=='__main__':main()