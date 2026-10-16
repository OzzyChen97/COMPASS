"""Lock accepted KIT assets and run smoke-gated native evaluations sequentially."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_motionx_random_retrieval import digest


def main():
    base=ROOT/'ablation/unified_retest_cfg13_20260909'
    kit=base/'remaining/kit'
    accepted=kit/'ruler_training/acceptance.json'
    gate=json.loads(accepted.read_text())
    assert gate['accepted'] and gate['frozen_verified'] and not gate['smoke']
    assert digest(gate['checkpoint'])==gate['checkpoint_sha256']
    manifest=json.loads((base/'assets/manifest.json').read_text())
    manifest['kit_ruler_acceptance']=str(accepted)
    for name in ('t2mgpt','satopp','motiongpt'):
        path=kit/f'formal/{name}/manifest.json'
        src=json.loads(path.read_text())
        assert src['steps']==3000 and src['frozen_verified'] and not src['smoke']
        assert digest(src['checkpoint'])==src['checkpoint_sha256']
        manifest['ready_models']['kit/'+name]=dict(checkpoint=src['checkpoint'],sha256=src['checkpoint_sha256'],
            config=src['source'].get('config'),adaptation_manifest=str(path),adaptation_manifest_sha256=digest(path))
    target=kit/'evaluation_assets.json'
    if target.exists():assert json.loads(target.read_text())==manifest
    else:target.write_text(json.dumps(manifest,indent=2)+'\n')
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONNOUSERSITE='1')
    for name in ('t2mgpt','satopp','motiongpt'):
        for smoke in (True,False):
            output=kit/('eval_smoke' if smoke else 'evaluations')/name
            if (output/'summary.json').exists():continue
            cmd=[sys.executable,str(ROOT/'tools/eval_unified_mixed.py'),'--assets',str(target),
                 '--dataset','kit','--model',name,'--output',str(output)]
            if smoke:cmd+=['--limit','32']
            if output.exists():cmd+=['--resume']
            log=kit/'logs'/f'eval_{name}_{"smoke" if smoke else "formal"}.log'
            with log.open('a') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            result=json.loads((output/'summary.json').read_text())
            assert result['samples']==(32 if smoke else 734) and result['cfg_scale']==1.3
            assert result['ruler_sha256']==gate['checkpoint_sha256']
            print('KIT_EVAL_COMPLETE',name,'smoke' if smoke else 'formal',flush=True)
    (kit/'evaluations_complete.json').write_text(json.dumps(dict(complete=True,models=3,samples=734,cfg_scale=1.3),indent=2)+'\n')


if __name__=='__main__':main()