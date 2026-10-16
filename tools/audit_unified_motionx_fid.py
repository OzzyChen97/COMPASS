"""Independent numerical/provenance audit of saved Motion-X FID embeddings."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.linalg import eigh

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.eval_unified_mixed import distribution_metrics
from tools import train_motionx_text_adaptation_pilot as pilot


def psd_fid(a,b):
    a=a.astype(np.float64);b=b.astype(np.float64)
    ma=a.mean(0);mb=b.mean(0);ca=np.cov(a,rowvar=False);cb=np.cov(b,rowvar=False)
    w,v=eigh(ca);root=(v*np.sqrt(np.maximum(w,0)))@v.T
    eig=eigh(root@cb@root,eigvals_only=True)
    return float((ma-mb)@(ma-mb)+np.trace(ca)+np.trace(cb)-2*np.sqrt(np.maximum(eig,0)).sum())


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    root=ROOT/'ablation/unified_retest_cfg13_20260909';report={};all_real=[];ids=[]
    for name in ['t2mgpt','satopp','motiongpt','motiongpt_satopp']:
        folder=root/'runs/motionx'/name;z=np.load(folder/'embeddings.npz');stored=json.loads((folder/'summary.json').read_text())
        computed=distribution_metrics(z['real'],z['original'],z['perturbed'])
        stable={key:psd_fid(z[x],z[y]) for key,x,y in [('FID','real','original'),('FID_P','real','perturbed'),('FID_D','original','perturbed')]}
        report[name]={'stored':{k:stored[k] for k in stable},'recomputed':{k:computed[k] for k in stable},'independent_float64_psd':stable,
            'max_recompute_error':max(abs(computed[k]-stored[k]) for k in stable),
            'max_psd_difference':max(abs(stable[k]-stored[k]) for k in stable)}
        all_real.append(z['real']);ids.append(z['motion_id'])
    assert all(np.array_equal(ids[0],v) for v in ids)
    report['reference_embeddings_max_difference']=max(float(np.max(np.abs(all_real[0]-v))) for v in all_real)
    standard=torch.load(ROOT/'checkpoints/motionx322/text_mot_match/model/finest.tar',map_location='cpu',weights_only=False)
    aware=torch.load(ROOT/'checkpoints/motionx322/text_mot_match_para_aware/model/finest.tar',map_location='cpu',weights_only=False)
    report['motion_ruler_tensor_changes']={}
    for key in ['movement_encoder','motion_encoder']:
        report['motion_ruler_tensor_changes'][key]=[n for n,v in standard[key].items() if not torch.equal(v,aware[key][n])]
    source=json.loads((ROOT/'ablation/motiongpt_full_t5_epochs3_20260906/source_models.json').read_text())
    for name,original in [('t2mgpt',pilot.BASELINE_TRANS),('satopp',Path(source['satopp']['checkpoint']))]:
        current=json.loads((root/f'runs/motionx/{name}/run.json').read_text())['source']['checkpoint']
        old=torch.load(original,map_location='cpu',weights_only=False);new=torch.load(current,map_location='cpu',weights_only=False)
        report[name]['changed_generator_tensors']=[k for k,v in old['trans'].items() if not torch.equal(v,new['trans'][k])]
        del old,new
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()