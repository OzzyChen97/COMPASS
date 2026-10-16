"""Recompute every unified metric and verify manuscript tables against sources."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.eval_unified_mixed import distribution_metrics
from tools.audit_motionx_random_retrieval import GROUP_SEEDS,retrieval,digest
from tools.render_independent_cfg_paper_tables import render
BASE=ROOT/'ablation/unified_retest_cfg13_20260909'


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda')
    rows=json.loads((BASE/'consolidated/results.json').read_text())
    paths=sorted({row['result_path'] for row in rows});assert len(paths)==18
    reports=[]
    for raw in paths:
        path=Path(raw);s=json.loads((path/'summary.json').read_text())
        z=np.load(path/'embeddings.npz',allow_pickle=False)
        measured=distribution_metrics(z.get('real'),z['original'],z['perturbed'])
        for condition,suffix in [('original','O'),('perturbed','P')]:
            t=torch.from_numpy(z['text_'+condition]).to(device);m=torch.from_numpy(z[condition]).to(device)
            vals=[retrieval(t,m,np.random.default_rng(seed).permutation(len(t))) for seed in GROUP_SEEDS]
            for k,v in enumerate(np.mean(vals,axis=0)):measured[f'R{k+1}_{suffix}']=float(v)
            measured['Matching_'+suffix]=float(np.linalg.norm(z['text_'+condition]-z[condition],axis=1).mean())
        error={key:abs(value-s[key]) for key,value in measured.items()}
        assert all(v<1e-6 for v in error.values()),(path,error)
        reports.append(dict(path=raw,samples=s['samples'],max_abs_error=max(error.values()),errors=error))
        print('METRICS_RECOMPUTED',raw,max(error.values()),flush=True)
    for dataset in ('humanml','motionx'):
        assert (ROOT/f'paper_tip/tables/{dataset}_independent_cfg13.tex').read_text()==render(dataset)
    result=dict(complete=True,unique_runs=18,level_values_verified=72,runs=reports)
    (BASE/'consolidated/recomputation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ALL_METRICS_AND_LEVEL_TABLES_VERIFIED',flush=True)


if __name__=='__main__':main()