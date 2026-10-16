"""Queue component adaptations/retests and isolated Motion-X input-side controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.launch_shared_clip_pipeline import env
from tools.audit_motionx_random_retrieval import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();a.root=a.root.resolve()
    remaining=a.root/'remaining';logs=remaining/'logs';logs.mkdir(exist_ok=False)
    spec=remaining/'component_sources.json';source=json.loads(spec.read_text())
    base=json.loads((a.root/'assets/manifest.json').read_text());jobs=queue.Queue()
    for name in ['t2mgpt','satopp']:jobs.put(('diagnostic',name))
    for name in source:jobs.put(('component',name))
    def run(args,gpu,label):
        with (logs/f'{label}.log').open('x') as log:
            subprocess.run([sys.executable,'-u']+args,cwd=ROOT,env=env(str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
    def worker(gpu):
        while not jobs.empty():
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=20).strip())
            if free<32768:time.sleep(30);continue
            try:kind,name=jobs.get_nowait()
            except queue.Empty:return
            print('REMAINING_JOB_START',kind,name,gpu,flush=True)
            if kind=='diagnostic':
                run(['tools/eval_unified_mixed.py','--assets',str(remaining/f'diagnostic_{name}_assets.json'),'--dataset','motionx',
                     '--model',name,'--output',str(remaining/'motionx_original_controls'/name)],gpu,f'diagnostic_{name}')
            else:
                train=['tools/train_component_text.py','--model',name,'--spec',str(spec),
                       '--train-pairs',str(ROOT/'ablation/humanml_independent_cfg13_20260907/assets/train_preserve.jsonl')]
                smoke=remaining/'smoke'/name
                run(train+['--output',str(smoke),'--steps','2','--limit','32'],gpu,f'smoke_{name}')
                assert json.loads((smoke/'manifest.json').read_text())['frozen_verified']
                out=remaining/'adapted'/name
                run(train+['--output',str(out),'--steps','3000'],gpu,f'train_{name}')
                meta=json.loads((out/'manifest.json').read_text());assert meta['frozen_verified'] and not meta['smoke']
                cp=Path(meta['checkpoint']);assert digest(cp)==meta['checkpoint_sha256']
                updated=json.loads(json.dumps(base))
                updated['ready_models']['humanml/satopp']=dict(checkpoint=str(cp),sha256=digest(cp),encoder='CLIP',variant=name,
                     source_checkpoint=source[name],text_adaptation_steps=3000)
                path=remaining/f'{name}_eval_assets.json';path.write_text(json.dumps(updated,indent=2)+'\n')
                run(['tools/eval_unified_mixed.py','--assets',str(path),'--dataset','humanml','--model','satopp',
                     '--output',str(remaining/'component_evaluation'/name)],gpu,f'eval_{name}')
            print('REMAINING_JOB_DONE',kind,name,gpu,flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
    (remaining/'complete.json').write_text(json.dumps(dict(components=5,motionx_diagnostic_controls=2,kit_pending=True))+'\n')
    print('COMPONENT_AND_FID_CONTROL_STAGE_COMPLETE',flush=True)


if __name__=='__main__':main()