"""Run source-encoder FID controls without changing adapted results or the paper."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import queue
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.audit_motionx_random_retrieval import digest,GROUP_SEEDS
from tools.eval_unified_mixed import distribution_metrics
from tools.launch_shared_clip_pipeline import env
BASE=ROOT/'ablation/unified_retest_cfg13_20260909'
MODELS=('t2mgpt','satopp','motiongpt','motiongpt_satopp')


def read(path):return json.loads(Path(path).read_text())


def lock(path,value):
    if path.exists():assert read(path)==value,path
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def build_jobs(out):
    base=read(BASE/'remaining/kit/evaluation_assets.json');jobs=[]
    def add(dataset,name,model,source,provenance,adapted,reuse=None):
        source=dict(source)
        assert digest(source['checkpoint'])==source['sha256']
        if not model.startswith('motiongpt'):source['official_clip']=True
        source.update(text_adaptation_steps=0,source_provenance=str(provenance),source_provenance_sha256=digest(provenance))
        manifest=json.loads(json.dumps(base))
        key=('humanml' if dataset=='transfer' else dataset)+'/'+model
        manifest['ready_models'][key]=source
        path=out/'assets'/f'{dataset}_{name}.json';lock(path,manifest)
        jobs.append(dict(dataset=dataset,name=name,model=model,assets=str(path),source=source,
            adapted_result=str(adapted),output=str(reuse or out/'runs'/dataset/name),reuse=reuse is not None))
    for dataset in ('humanml','motionx','kit'):
        for model in (MODELS if dataset!='kit' else MODELS[:3]):
            if dataset=='humanml':
                provenance=ROOT/f'ablation/humanml_independent_cfg13_20260907/main/{model}/manifest.json'
                meta=read(provenance);assert meta['frozen_verified'];source=meta['source']
                adapted=BASE/f'runs/humanml/{model}'
            elif dataset=='kit':
                provenance=BASE/f'remaining/kit/formal/{model}/manifest.json'
                meta=read(provenance);assert meta['frozen_verified'];source=meta['source']
                adapted=BASE/f'remaining/kit/evaluations/{model}'
            else:
                adapted=BASE/f'runs/motionx/{model}'
                if model in MODELS[:2]:
                    provenance=BASE/f'remaining/diagnostic_{model}_assets.json'
                    source=read(provenance)['ready_models']['motionx/'+model]
                else:
                    provenance=ROOT/f'ablation/motionx_four_model_text_adapted_test_20260905/main/{model}/manifest.json'
                    source=read(provenance)['source']
            reuse=BASE/f'remaining/motionx_original_controls/{model}' if dataset=='motionx' and model in MODELS[:2] else None
            add(dataset,model,model,source,provenance,adapted,reuse)
    provenance=BASE/'remaining/component_sources.json'
    for name,source in read(provenance).items():
        assert read(BASE/f'remaining/adapted/{name}/manifest.json')['frozen_verified']
        add('humanml',name,'satopp',source,provenance,BASE/f'remaining/component_evaluation/{name}')
    for model in MODELS[:2]:
        provenance=ROOT/f'ablation/humanml_independent_cfg13_20260907/main/{model}/manifest.json'
        add('transfer',model,model,read(provenance)['source'],provenance,BASE/f'runs/transfer/{model}')
    assert len(jobs)==18
    lock(out/'jobs.json',jobs)
    lock(out/'protocol.json',dict(text_encoder='each exact source checkpoint before additional 3000-step adaptation; original CLIP only when absent in source',
        generation_seed=123,cfg_scale=1.3,fixed_pairs='unchanged unified mixed sets',evaluator='unchanged frozen dataset evaluator',
        transfer='HumanML source model; only FID_D defined without real reference',new_runs=16,reused_runs=2,
        preserve_adapted_results=True,paper_modified=False,engine_sha256=digest(ROOT/'tools/eval_unified_mixed.py')))
    return jobs


def verify(job):
    path=Path(job['output']);s=read(path/'summary.json');r=read(path/'run.json')
    assets=read(job['assets']);ds=assets['datasets'][job['dataset']]
    assert s['samples']==ds['samples'] and not s['smoke'] and s['cfg_scale']==1.3
    assert r['seed']==123 and r['group_seeds']==GROUP_SEEDS
    assert s['checkpoint_sha256']==job['source']['sha256'] and s['pair_sha256']==ds['pair_sha256']
    z=np.load(path/'embeddings.npz',allow_pickle=False);az=np.load(Path(job['adapted_result'])/'embeddings.npz',allow_pickle=False)
    assert np.array_equal(z['motion_id'],az['motion_id']) and np.array_equal(z['reference_length'],az['reference_length'])
    assert len(set(z['motion_id']))==s['samples']
    if job['dataset']!='transfer':assert np.array_equal(z['real'],az['real'])
    assert s['ruler_sha256']==read(Path(job['adapted_result'])/'summary.json')['ruler_sha256']
    metrics=distribution_metrics(z.get('real'),z['original'],z['perturbed'])
    errors={k:abs(s[k]-v) for k,v in metrics.items()};assert max(errors.values())<1e-6
    result=dict(dataset=job['dataset'],model=job['name'],samples=s['samples'],reused=job['reuse'],
        checkpoint_sha256=s['checkpoint_sha256'],pair_sha256=s['pair_sha256'],ruler_sha256=s['ruler_sha256'],
        result_path=str(path),max_recompute_error=max(errors.values()),**metrics)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    jobs=build_jobs(out)
    if a.prepare_only:print('SOURCE_ASSETS_LOCKED',len(jobs),flush=True);return
    pending=queue.Queue()
    for job in jobs:
        if not (Path(job['output'])/'summary.json').exists():pending.put(job)
        else:verify(job);print('SOURCE_FID_REUSED_OR_COMPLETE',job['dataset'],job['name'],flush=True)
    def worker(gpu):
        while not pending.empty():
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=20).strip())
            if free<32768:time.sleep(30);continue
            try:job=pending.get_nowait()
            except queue.Empty:return
            label=f"{job['dataset']}_{job['name']}";print('SOURCE_FID_START',label,gpu,flush=True)
            path=Path(job['output']);log=out/'logs'/f'{label}.log';log.parent.mkdir(exist_ok=True)
            cmd=[sys.executable,'-u',str(ROOT/'tools/eval_unified_mixed.py'),'--assets',job['assets'],'--dataset',job['dataset'],
                 '--model',job['model'],'--output',job['output']]
            if path.exists():cmd+=['--resume']
            with log.open('a') as f:
                result=subprocess.run(cmd,cwd=ROOT,env=env(str(gpu)),stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:
                (out/f'failed_{label}.json').write_text(json.dumps(dict(job=job,returncode=result.returncode,log=str(log)),indent=2)+'\n')
                print('SOURCE_FID_FAILED',label,result.returncode,flush=True);continue
            verify(job);print('SOURCE_FID_COMPLETE',label,flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
    missing=[job for job in jobs if not (Path(job['output'])/'summary.json').exists()]
    if missing:raise RuntimeError(f'{len(missing)} source-encoder evaluations incomplete; see failure logs')
    rows=[verify(job) for job in jobs]
    (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    fields=['dataset','model','samples','FID','FID_P','FID_D','Diversity_O','Diversity_real','paired_embedding_L2','reused','max_recompute_error','checkpoint_sha256','pair_sha256','ruler_sha256','result_path']
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lock(out/'complete.json',dict(complete=True,unique_runs=18,new_runs=16,reused_runs=2,all_fid_recomputed=True,paper_modified=False,adapted_results_modified=False))
    print('ALL_SOURCE_ENCODER_FID_COMPLETE',flush=True)


if __name__=='__main__':main()