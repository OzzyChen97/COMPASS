"""First stage: Motion-X four independent encoders, identical CFG scale 1.3."""
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
from tools.launch_shared_clip_pipeline import env
from tools.audit_motionx_random_retrieval import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    logs=a.output/'logs';logs.mkdir()
    old=ROOT/'ablation/motionx_four_model_text_adapted_test_20260905'
    spec=json.loads((old/'source_models.json').read_text())
    choices=json.loads((ROOT/'ablation/motiongpt_text_improvement_validation_20260906/selection.json').read_text())
    models=['t2mgpt','satopp','motiongpt','motiongpt_satopp']
    paths={'t2mgpt':ROOT/'ablation/motionx_text_adaptation_extended_3000_20260905/main/text/motionx_text_adaptation_pilot_last.pth',
           'satopp':old/'main/satopp/adapted.pth',
           **{m:Path(choices[m]['checkpoint']) for m in models[2:]}}
    manifest={m:dict(checkpoint=str(paths[m]),sha256=digest(paths[m]),encoder='T5' if m.startswith('motiongpt') else 'CLIP') for m in models}
    for m in models[2:]:
        assert manifest[m]['sha256']==choices[m]['checkpoint_sha256']
        manifest[m]['config']=spec[m]['config']
    protocol=dict(dataset='motionx322',models=manifest,cfg_scale=1.3,seed=123,
                  selection='reuse previous validation-selected independent text-only checkpoints; no new test selection',
                  motiongpt_unconditional_prompt='Generate motion: ',
                  next_dataset='humanml3d; separate adaptation assets required, not started by this first-stage launcher')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def worker(i,m):
        for level in ['P1','P2','P3']:
            assert digest(paths[m])==manifest[m]['sha256']
            while True:
                free=int(subprocess.check_output(['nvidia-smi','-i',str(i),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=20).strip())
                if free >= (32768 if m.startswith('motiongpt') else 65536):break
                print('INDEPENDENT_GPU_WAIT',i,free,flush=True);time.sleep(60)
            out=a.output/'test'/m/level
            cmd=[sys.executable,'-u','tools/eval_motionx_independent_cfg.py','--model',m,'--level',level,
                 '--checkpoint',str(paths[m]),'--output',str(out),'--seed','123']
            if m.startswith('motiongpt'):cmd+=['--config',spec[m]['config']]
            with (logs/f'{m}_{level}.log').open('x') as log:
                print('INDEPENDENT_TEST_START',m,level,i,flush=True)
                subprocess.run(cmd,cwd=ROOT,env=env(str(i)),stdout=log,stderr=subprocess.STDOUT,check=True)
            meta=json.loads((out/'manifest.json').read_text())
            assert meta['samples']==7679 and not meta['smoke'] and meta['cfg_scale']==1.3
            assert meta['checkpoint_sha256']==manifest[m]['sha256']
            print('INDEPENDENT_TEST_DONE',m,level,flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda im:worker(*im),enumerate(models)))
    rows=[]
    for m in models:
        for level in ['P1','P2','P3']:
            rows.extend(json.loads((a.output/'test'/m/level/'summary.json').read_text()))
    with (a.output/'motionx_summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.output/'motionx_complete.json').write_text(json.dumps(dict(complete=True,models=4,levels=3,cfg_scale=1.3))+'\n')
    print('INDEPENDENT_MOTIONX_COMPLETE',flush=True)


if __name__=='__main__':main()