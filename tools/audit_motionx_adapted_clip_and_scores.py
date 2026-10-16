"""Read-only weight and cached-embedding diagnosis of the adapted Motion-X test."""
import argparse
import csv
import gc
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import train_motionx_text_adaptation_pilot as pilot
from tools.audit_motionx_random_retrieval import GROUP_SEEDS,canonical_order,digest,retrieval,cache_path
RUN=ROOT/'ablation/motionx_four_model_text_adapted_test_20260905'
OLD=ROOT/'ablation/motionx322_random32_audit_20260905'


def comparison(a,b,names):
    names=list(names); changed=[]; maxabs=0.; change2=0.; ref2=0.
    for n in names:
        x,y=a[n].detach().float(),b[n].detach().float()
        assert x.shape==y.shape
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        delta=y-x
        if not torch.equal(x,y):changed.append(n)
        maxabs=max(maxabs,float(delta.abs().max()))
        change2+=float(delta.double().square().sum());ref2+=float(x.double().square().sum())
    return {'tensors':len(names),'changed_tensors':len(changed),'changed_names':changed,
            'max_abs_delta':maxabs,'relative_l2_delta':(change2/max(ref2,1e-30))**.5}


def weights(out):
    result={}
    for model in ['t2mgpt','satopp']:
        manifest=json.loads((RUN/'test'/model/'P1/manifest.json').read_text())
        path=Path(manifest['checkpoint']); h=digest(path)
        assert h==manifest['checkpoint_sha256']
        for level in ['P2','P3']:
            assert json.loads((RUN/'test'/model/level/'manifest.json').read_text())['checkpoint_sha256']==h
        adapted=torch.load(path,map_location='cpu',weights_only=False)
        if model=='t2mgpt':
            clip=pilot.load_clip_model(torch.device('cpu'))
            before=clip.state_dict()
        else:
            spec=json.loads((RUN/'source_models.json').read_text())[model]
            original=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
            before=original['clip']
        result[model]={'checkpoint_hash_matches_all_tests':True,'checkpoint_sha256':h,
                       'text':comparison(before,adapted['clip'],[k for k in before if pilot.is_clip_text_parameter(k)]),
                       'visual':comparison(before,adapted['clip'],[k for k in before if pilot.is_clip_visual_parameter(k)])}
        if model=='t2mgpt':
            del before,clip;gc.collect()
            original=torch.load(pilot.BASELINE_TRANS,map_location='cpu',weights_only=False)
        else:del before
        result[model]['generator']=comparison(original['trans'],adapted['trans'],original['trans'].keys())
        del original,adapted;gc.collect()
        print(model,json.dumps({k:v for k,v in result[model].items() if k not in ['text','visual','generator']}),flush=True)
        for k in ['text','visual','generator']:print(k,{n:v for n,v in result[model][k].items() if n!='changed_names'},flush=True)
    paths={'t2mgpt':ROOT/'ablation/motionx_text_adaptation_extended_3000_20260905/main/text/train_losses.csv',
           **{m:RUN/'main'/m/'losses.csv' for m in ['satopp','motiongpt','motiongpt_satopp']}}
    losses={}
    for m,p in paths.items():
        rr=list(csv.DictReader(p.open()))
        keys=[k for k in ['loss','clean_loss','paraphrase_loss','grad_norm'] if k in rr[0]]
        losses[m]={'steps':len(rr),'pair_draws':len(rr)*16,
                   'first100':{k:float(np.mean([float(r[k]) for r in rr[:100]])) for k in keys},
                   'last100':{k:float(np.mean([float(r[k]) for r in rr[-100:]])) for k in keys}}
    result['losses']=losses
    print('LOSSES',json.dumps(losses),flush=True)
    (out/'weight_audit.json').write_text(json.dumps(result,indent=2)+'\n')


def calibrate(out):
    gt=np.load(OLD/'gt_para.npz')
    prov=json.loads((OLD/'gt_para_provenance.json').read_text())
    rows=[];checks={}
    def score(t,m):
        t=torch.from_numpy(t);m=torch.from_numpy(m)
        return np.mean([retrieval(t,m,np.random.default_rng(gs).permutation(len(t))) for gs in GROUP_SEEDS],axis=0)
    for level in ['P1','P2','P3']:
        ref=np.load(RUN/'test/t2mgpt'/level/'embeddings.npz')
        assert np.array_equal(gt['motion_id'],ref['motion_id'])
        man=json.loads((RUN/'test/t2mgpt'/level/'manifest.json').read_text())
        assert man['ruler_sha256']==prov['checkpoint_sha256']
        checks[level]={'cached_current_text_max_abs_difference':float(np.max(np.abs(gt[level]-ref['text_current'])))}
        for mode in ['current','fallback_lower']:
            values=score(ref['text_'+mode],gt['gt'])
            rows.append({'model':'ground_truth','level':level,'lookup':mode,**{f'R{k+1}':float(v) for k,v in enumerate(values)}})
        for model in ['t2mgpt','satopp','motiongpt','motiongpt_satopp']:
            z=np.load(RUN/'test'/model/level/'embeddings.npz')
            assert np.array_equal(z['motion_id'],ref['motion_id'])
            checks[level][model]={'length_mean':float(np.mean(z['length'])),
                                 'length_le4_fraction':float(np.mean(z['length']<=4)),
                                 'length_300_fraction':float(np.mean(z['length']==300))}
            for mode in ['current','fallback_lower']:
                error=float(np.max(np.abs(z['text_'+mode]-ref['text_'+mode])))
                checks[level][model]['text_difference_'+mode]=error
                values=score(z['text_'+mode],z['motion'])
                rows.append({'model':model,'level':level,'lookup':mode,**{f'R{k+1}':float(v) for k,v in enumerate(values)}})
        print('CALIBRATION',level,[r for r in rows if r['level']==level and r['model']=='ground_truth'],flush=True)
    with (out/'gt_generated_calibration.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'embedding_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('EMBEDDING_CHECKS',json.dumps(checks),flush=True)


def shared_text(out):
    """One validation-environment text cache for all models; no motion regeneration."""
    rows=[];sources=[]
    for level in ['P1','P2','P3']:
        reference_path=RUN/'test/t2mgpt'/level/'embeddings.npz'
        reference=np.load(reference_path)
        reference_manifest=json.loads((reference_path.parent/'manifest.json').read_text())
        for model in ['t2mgpt','satopp','motiongpt','motiongpt_satopp']:
            path=RUN/'test'/model/level/'embeddings.npz'
            z=np.load(path);manifest=json.loads((path.parent/'manifest.json').read_text())
            assert np.array_equal(z['motion_id'],reference['motion_id'])
            for key in ['pair_file_sha256','ruler_sha256','group_seeds','generation_seed','samples']:
                assert manifest[key]==reference_manifest[key],key
            # Confirm raw text identity, not only motion IDs, before replacing evaluator features.
            audit={r['motion_id']:r for r in (json.loads(l) for l in (path.parent/'generation_audit.jsonl').read_text().splitlines())}
            ref_audit={r['motion_id']:r for r in (json.loads(l) for l in (reference_path.parent/'generation_audit.jsonl').read_text().splitlines())}
            for mid in z['motion_id']:
                assert audit[mid]['generation_text']==audit[mid]['retrieval_text']==ref_audit[mid]['retrieval_text']
            sources.append({'model':model,'level':level,'motion_cache':str(path),'motion_cache_sha256':digest(path),
                            'shared_text_cache':str(reference_path),'shared_text_cache_sha256':digest(reference_path)})
            old={r['lookup']:r for r in json.loads((path.parent/'summary.json').read_text())}
            m=torch.from_numpy(z['motion'])
            for mode in ['current','fallback_lower']:
                t=torch.from_numpy(reference['text_'+mode])
                values=np.mean([retrieval(t,m,np.random.default_rng(gs).permutation(len(t))) for gs in GROUP_SEEDS],axis=0)
                row={'model':model,'level':level,'lookup':mode,'samples':len(t),'generation_seed':123}
                row.update({f'R{k+1}':float(v) for k,v in enumerate(values)})
                row.update({f'R{k}_delta_vs_previous':row[f'R{k}']-old[mode][f'R{k}'] for k in [1,2,3]})
                rows.append(row)
                if mode=='fallback_lower':print('SHARED_TEXT',json.dumps(row),flush=True)
    with (out/'shared_text_summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'shared_text_provenance.json').write_text(json.dumps({'source_environment':'T2M-GPT; spaCy 3.7.5, en_core_web_sm 3.7.1',
        'reason':'Same evaluator text preprocessing as validation-selected protocol; fix cross-environment spaCy drift',
        'unchanged_generated_motion_embeddings':True,'changed_model_weights':False,'sources':sources,
        'old_summary_retained':str(RUN/'four_model_summary.csv')},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['weights','calibrate','shared']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    {'weights':weights,'calibrate':calibrate,'shared':shared_text}[a.mode](a.output)

if __name__=='__main__':main()