"""Validation-only MotionGPT adaptation comparison with one immutable text cache.

Loads LM/VAE directly, avoiding MotionGPTDataModule's implicit tiny test split.
All candidates share evaluation text embeddings from the original validation
pipeline; local spaCy versions cannot affect checkpoint selection.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import train_motionx_text_adaptation_pilot as pilot
from tools.train_motionx_four_model_text_adaptation import load_lm,sequence_ce
from tools.audit_motionx_random_retrieval import GROUP_SEEDS,canonical_order,digest,retrieval
ASSETS=ROOT/'ablation/motionx_text_adaptation_pilot_20260905/assets'
CACHE=ROOT/'ablation/motionx_text_adaptation_pilot_20260905/validation/baseline/seed123.npz'
QUERIES=['clean','P1','P2','P3']


def load_inference(config,checkpoint,device):
    lm,source,cfg=load_lm(config,checkpoint)
    from mGPT.config import instantiate_from_config
    before=Path.cwd()
    try:
        os.chdir(ROOT/'MotionGPT')
        vae=instantiate_from_config(cfg.vq.default)
    finally:os.chdir(before)
    vae.load_state_dict({k[4:]:v for k,v in source['state_dict'].items() if k.startswith('vae.')},strict=True)
    lm.float().to(device).eval().requires_grad_(False)
    vae.float().to(device).eval().requires_grad_(False)
    model=SimpleNamespace(lm=lm,vae=vae,device=device,hparams=SimpleNamespace(cfg=cfg,codebook_size=lm.m_codebook_size))
    data=SimpleNamespace(nfeats=322,renorm4t2m=lambda x:x)
    return model,data


def validation_pairs():
    asset_manifest=json.loads((ASSETS/'manifest.json').read_text())
    for q in QUERIES[1:]:
        filename=f'val_{q.lower()}.json'
        assert digest(ASSETS/filename)==asset_manifest['asset_hashes'][filename], 'Validation pair asset changed'
    pairs={q:json.loads((ASSETS/f'val_{q.lower()}.json').read_text()) for q in QUERIES[1:]}
    expected=sorted(r['motion_id'] for r in pairs['P1'])
    assert len(expected)==661 and len(set(expected))==661
    val=pilot.load_split_ids(pilot.DATASET_ROOT,'val')
    forbidden=pilot.load_split_ids(pilot.DATASET_ROOT,'train')|pilot.load_split_ids(pilot.DATASET_ROOT,'test')
    assert set(expected)<=val and not set(expected)&forbidden
    for q,rows in pairs.items():assert sorted(r['motion_id'] for r in rows)==expected
    return pairs,expected


@torch.no_grad()
def condition_diagnostics(model,pairs,expected,limit=128):
    token_dir=pilot.DATASET_ROOT/str(model.hparams.cfg.DATASET.CODE_PATH)
    lm=model.lm; device=model.device; nb=lm.m_codebook_size
    maps={q:{r['motion_id']:r for r in rows} for q,rows in pairs.items()}
    # Native exported tokens cover training, not necessarily validation. Encode
    # validation motions with this checkpoint's frozen VAE rather than borrowing
    # training tokens or silently dropping all held-out examples.
    eligible=[];tokens=[]
    mean=np.load(pilot.DATASET_ROOT/'Mean.npy').astype(np.float32)
    std=np.maximum(np.load(pilot.DATASET_ROOT/'Std.npy').astype(np.float32),1e-8)
    for mid in expected:
        raw=np.load(pilot.DATASET_ROOT/'new_joint_vecs'/f'{mid}.npy',allow_pickle=False).astype(np.float32)
        frames=(len(raw)//4)*4
        if frames<4:continue
        native=torch.from_numpy((raw[:frames]-mean)/std).unsqueeze(0).to(device)
        code,_=model.vae.encode(native)
        values=code[0].cpu().numpy()
        assert len(values)==frames//4 and np.all((values>=0)&(values<nb))
        eligible.append(mid);tokens.append(values)
        if len(eligible)>=limit:break
    if len(eligible)<2:raise RuntimeError('Insufficient validation motion-token targets')
    targets=[f'<motion_id_{nb}>'+''.join(f'<motion_id_{int(t)}>' for t in row)+f'<motion_id_{nb+1}>' for row in tokens]
    rows=[]
    for q in QUERIES:
        texts=[maps['P1' if q=='clean' else q][mid]['text_orig' if q=='clean' else 'text_pair'] for mid in eligible]
        for corrupt in [False,True]:
            prompts=texts[1:]+texts[:1] if corrupt else texts
            total=0.;early_total=0.
            for start in range(0,len(prompts),4):
                end=min(start+4,len(prompts))
                enc=lm.tokenizer(['Generate motion: '+s for s in prompts[start:end]],padding=True,truncation=True,
                                 max_length=lm.max_length,return_tensors='pt').to(device)
                target=lm.tokenizer(targets[start:end],padding=True,truncation=False,return_tensors='pt').to(device)
                labels=target.input_ids.masked_fill(target.attention_mask.eq(0),-100)
                pred=lm.language_model(**enc,labels=labels,decoder_attention_mask=target.attention_mask,use_cache=False)
                total+=float(sequence_ce(pred.logits,labels))*(end-start)
                early=labels.clone();early[:,16:]=-100
                early_total+=float(sequence_ce(pred.logits,early))*(end-start)
            rows.append({'query':q,'shuffled_text':corrupt,'samples':len(prompts),'ce':total/len(prompts),'first16_ce':early_total/len(prompts)})
    return {'rows':rows,'eligible_ids':eligible,'missing_native_token_count':sum(not (token_dir/f'{m}.npy').exists() for m in expected),
            'target_source':'validation motions encoded by frozen native VAE; floor frame length to multiple of four',
            'target_sha256':pilot.canonical_sha256([v.tolist() for v in tokens]),
            'interpretation':'Shuffled minus correct CE measures text conditioning; comparisons use identical fixed validation motion targets.'}


@torch.no_grad()
def run(args):
    from dataset.config import configure_evaluator_source,evaluator_opt_path
    from dataset.dataset_TM_eval import CaptionLockedDATALoader
    from models.evaluator_wrapper import EvaluatorModelWrapper
    from options.get_eval_option import get_opt
    from utils.word_vectorizer import WordVectorizer
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda');pilot.seed_all(123)
    pairs,expected=validation_pairs()
    cache=np.load(CACHE)
    assert np.array_equal(cache['motion_id'],np.asarray(expected))
    cache_manifest=json.loads((CACHE.parent/'manifest.json').read_text())
    assert cache_manifest['asset_manifest_sha256']==digest(ASSETS/'manifest.json')
    if args.limit:
        keep=set(expected[:args.limit]);pairs={q:[r for r in rows if r['motion_id'] in keep] for q,rows in pairs.items()};expected=expected[:args.limit]
    cache_ix=canonical_order(cache['motion_id'],sorted(cache['motion_id']))
    assert np.array_equal(cache_ix,np.arange(len(cache_ix)))
    maps={q:{r['motion_id']:r for r in rows} for q,rows in pairs.items()}
    vectorizer=WordVectorizer(str(ROOT/'glove'),'our_vab')
    loader=CaptionLockedDATALoader('motionx322_75',False,32,vectorizer,pair_data=pairs['P1'],num_workers=0,length_policy='strict')
    assert len(loader.dataset)==len(expected)
    ruler=ROOT/'checkpoints/motionx322/text_mot_match_para_aware/model/finest.tar'
    assert digest(ruler)==cache_manifest['ruler_sha256']
    opt=configure_evaluator_source(get_opt(evaluator_opt_path('motionx322_75'),device),'motionx322_75')
    opt.evaluator_matcher_checkpoint=str(ruler);evaluator=EvaluatorModelWrapper(opt)
    model,data=load_inference(args.config,args.checkpoint,device)
    from MotionGPT.eval_loader_robustness import generate_features,get_motion_embeddings_in_input_order
    diag=condition_diagnostics(model,pairs,expected,min(args.diagnostic_limit,len(expected)))
    (out/'conditioning.json').write_text(json.dumps(diag,indent=2)+'\n')
    print('CONDITIONING_COMPLETE',args.name,flush=True)
    ids=[];motion={q:[] for q in QUERIES};lengths={q:[] for q in QUERIES}
    with (out/'prompts.jsonl').open('w') as audit:
        for bi,batch in enumerate(loader):
            _,_,captions,_,_,mlens,_,names=batch
            ids.extend(names)
            for qi,q in enumerate(QUERIES):
                prompts=list(captions) if q=='clean' else [maps[q][mid]['text_pair'] for mid in names]
                seed=123+bi*1000003+qi*100003
                random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
                pred,plen=generate_features(model,data,prompts,'categorical',300,reference_lengths=mlens,
                                           temperature=1.,top_k=0,top_p=1.,prompt_mode='official-default')
                em=get_motion_embeddings_in_input_order(evaluator,pred,plen)
                motion[q].append(em.cpu().numpy());lengths[q].extend(plen.cpu().tolist())
                for mid,text in zip(names,prompts):audit.write(json.dumps({'motion_id':mid,'query':q,'generation_text':text,'retrieval_text':text})+'\n')
            audit.flush()
            if bi%5==0:print('VAL_PROGRESS',args.name,bi,len(loader),flush=True)
    ix=canonical_order(ids,expected)
    lookup={str(mid):i for i,mid in enumerate(cache['motion_id'])};ci=[lookup[m] for m in expected]
    arrays={'motion_id':np.asarray(expected)};summary=[];runs=[]
    for q in QUERIES:
        values=np.concatenate(motion[q])[ix];arrays['motion_'+q]=values
        arrays['length_'+q]=np.asarray(lengths[q])[ix]
        m=torch.from_numpy(values).to(device)
        for mode in ['current','fallback_lower']:
            t=torch.from_numpy(cache[f'text_{mode}_{q}'][ci]).to(device)
            vv=[]
            for gs in GROUP_SEEDS:
                v=retrieval(t,m,np.random.default_rng(gs).permutation(len(expected)));vv.append(v)
                runs.append({'name':args.name,'query':q,'lookup':mode,'group_seed':gs,**{f'R{k+1}':float(v[k]) for k in range(3)}})
            v=np.mean(vv,axis=0)
            summary.append({'name':args.name,'query':q,'lookup':mode,'samples':len(expected),**{f'R{k+1}':float(v[k]) for k in range(3)}})
    np.savez_compressed(out/'generated_embeddings.npz',**arrays)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(runs[0]));w.writeheader();w.writerows(runs)
    (out/'manifest.json').write_text(json.dumps({'name':args.name,'checkpoint':str(args.checkpoint),'checkpoint_sha256':digest(args.checkpoint),
        'samples':len(expected),'split':'val','test_content_read':False,'generation_seed':123,'group_seeds':GROUP_SEEDS,
        'asset_manifest_sha256':digest(ASSETS/'manifest.json'),'shared_text_cache_sha256':digest(CACHE),
        'ruler_sha256':digest(ruler),'script_sha256':digest(__file__),'config_sha256':digest(args.config),
        'checkpoint_load':'strict LM and VAE; auxiliary heads unused by generation','smoke':bool(args.limit)},indent=2)+'\n')
    print('VALIDATION_COMPLETE',args.name,flush=True)


def select_candidate(results):
    """Predeclared conservative validation rule, never consumes test scores."""
    def metrics(rows):
        values={r['query']:r['R3'] for r in rows if r['lookup']=='fallback_lower'}
        if set(values)!=set(QUERIES) or not all(np.isfinite(v) and 0<=v<=1 for v in values.values()):
            raise ValueError('Incomplete or invalid validation results')
        return {'clean_R3':values['clean'],'macro_R3':sum(values[q] for q in QUERIES[1:])/3,
                'levels':{q:values[q] for q in QUERIES[1:]}}
    scores={k:metrics(v) for k,v in results.items()}
    original=scores['original']
    for name,s in scores.items():
        s['eligible']=s['clean_R3']>=original['clean_R3']-.02 and all(
            s['levels'][q]>=original['levels'][q]-.01 for q in QUERIES[1:])
    controls=[n for n in ['original','previous_1e6'] if scores[n]['eligible']]
    control=max(controls,key=lambda n:(scores[n]['macro_R3'],scores[n]['clean_R3']))
    candidates=[n for n in scores if n not in ['original','previous_1e6'] and scores[n]['eligible']
                and scores[n]['macro_R3']>=scores[control]['macro_R3']+.005]
    winner=max(candidates,key=lambda n:(scores[n]['macro_R3'],scores[n]['clean_R3'])) if candidates else control
    return {'selected':winner,'improvement_qualified':bool(candidates),'best_control':control,'scores':scores,
            'rule':'macro P1/P2/P3 R3 improves >=0.5 percentage points over best eligible control; clean drop <=2pp and each perturbation drop <=1pp versus original',
            'split':'val','statistical_significance_claimed':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--diagnostic-limit',type=int,default=128)
    args=p.parse_args()
    if args.diagnostic_limit<2 or args.limit not in (0,) and args.limit<3:p.error('Invalid diagnostic or smoke sample limit')
    run(args)

if __name__=='__main__':main()