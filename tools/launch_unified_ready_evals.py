"""Schedule currently ready unified mixed evaluations across all eight GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.launch_shared_clip_pipeline import env
from tools.audit_motionx_random_retrieval import digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();a.root=a.root.resolve();assets=a.root/'assets/manifest.json'
    for ds,m in [('humanml','t2mgpt'),('motionx','motiongpt'),('transfer','satopp')]:
        d=json.loads((a.root/'smoke'/ds/m/'summary.json').read_text());assert d['smoke'] and d['samples']==32
    jobs=queue.Queue()
    for ds in ['humanml','motionx','transfer']:
        for m in (['t2mgpt','satopp'] if ds=='transfer' else ['t2mgpt','satopp','motiongpt','motiongpt_satopp']):jobs.put((ds,m))
    script_hash=digest(ROOT/'tools/eval_unified_mixed.py');asset_hash=digest(assets)
    (a.root/'ready_stage_protocol.json').write_text(json.dumps(dict(jobs=10,gpus=list(range(8)),asset_sha256=asset_hash,
        script_sha256=script_hash,cfg_scale=1.3,kit_and_components_pending=True),indent=2)+'\n')
    def worker(gpu):
        while not jobs.empty():
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=20).strip())
            if free<32768:time.sleep(30);continue
            try:ds,m=jobs.get_nowait()
            except queue.Empty:return
            assert digest(assets)==asset_hash and digest(ROOT/'tools/eval_unified_mixed.py')==script_hash
            out=a.root/'runs'/ds/m
            command=[sys.executable,'-u','tools/eval_unified_mixed.py','--assets',str(assets),'--dataset',ds,'--model',m,'--output',str(out)]
            with (a.root/'logs'/f'formal_{ds}_{m}.log').open('x') as log:
                print('UNIFIED_FORMAL_START',ds,m,gpu,flush=True)
                subprocess.run(command,cwd=ROOT,env=env(str(gpu)),stdout=log,stderr=subprocess.STDOUT,check=True)
            print('UNIFIED_FORMAL_DONE',ds,m,gpu,flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
    rows=[json.loads(p.read_text()) for p in sorted((a.root/'runs').glob('*/*/summary.json'))]
    assert len(rows)==10 and all(not r['smoke'] for r in rows)
    (a.root/'ready_stage_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('UNIFIED_READY_STAGE_COMPLETE',flush=True)


if __name__=='__main__':main()