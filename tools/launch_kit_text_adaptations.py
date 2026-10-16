"""KIT generator-only text adaptation; evaluator validation is a separate gate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.launch_shared_clip_pipeline import env


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,default=4)
    a=p.parse_args();a.root=a.root.resolve();logs=a.root/'logs';logs.mkdir(exist_ok=False)
    for model in ['t2mgpt','satopp','motiongpt']:
        for stage,steps,limit in [('smoke',2,32),('formal',3000,0)]:
            while True:
                free=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=20).strip())
                if free>=12288:break
                print('KIT_GPU_WAIT',a.gpu,free,flush=True);time.sleep(30)
            out=a.root/stage/model
            cmd=[sys.executable,'-u','tools/kit_independent_native.py','--model',model,'--spec',str(a.root/'sources.json'),
                 '--train-pairs',str(a.root/'train_available.jsonl'),'--output',str(out),'--steps',str(steps)]
            if limit:cmd+=['--limit',str(limit)]
            print('KIT_TEXT_JOB_START',model,stage,a.gpu,flush=True)
            with (logs/f'{stage}_{model}.log').open('x') as log:
                subprocess.run(cmd,cwd=ROOT,env=env(str(a.gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
            meta=json.loads((out/'manifest.json').read_text());assert meta['dataset']=='kit' and meta['frozen_verified']
            print('KIT_TEXT_JOB_DONE',model,stage,flush=True)
    (a.root/'generators_complete.json').write_text(json.dumps(dict(complete=True,models=3,steps=3000,
        kit_ruler_and_test_pending=True))+'\n')
    print('KIT_GENERATOR_ADAPTATIONS_COMPLETE',flush=True)


if __name__=='__main__':main()