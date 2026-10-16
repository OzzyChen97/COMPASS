"""Unified fixed-mixture generation and retrieval; resumable per-batch artifacts."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import train_motionx_text_adaptation_pilot as pilot
from tools.audit_motionx_random_retrieval import digest,GROUP_SEEDS,canonical_order,retrieval
from tools.humanml_independent_assets import load_native,model_args
from tools.eval_motiongpt_text_adaptation_validation import load_inference
from models.motiongpt_cfg import install_motiongpt_cfg


def distribution_metrics(real,original,perturbed,seed=123):
    from utils.sato_pp_eval import calculate_activation_statistics,calculate_frechet_distance
    def fid(a,b):
        ma,ca=calculate_activation_statistics(a);mb,cb=calculate_activation_statistics(b)
        return float(calculate_frechet_distance(ma,ca,mb,cb))
    def diversity(x):
        n=min(300,len(x)-1);rng=np.random.default_rng(seed)
        i=rng.choice(len(x),n,replace=False);j=rng.choice(len(x),n,replace=False)
        return float(np.linalg.norm(x[i]-x[j],axis=1).mean())
    result={'FID_D':fid(original,perturbed),'Diversity_O':diversity(original),
            'paired_embedding_L2':float(np.linalg.norm(original-perturbed,axis=1).mean())}
    if real is not None:result.update(FID=fid(real,original),FID_P=fid(real,perturbed),Diversity_real=diversity(real))
    if not all(np.isfinite(v) for v in result.values()):raise ValueError('Nonfinite metrics')
    return result


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--dataset',choices=['humanml','motionx','transfer','kit'],required=True)
    p.add_argument('--model',choices=['t2mgpt','satopp','motiongpt','motiongpt_satopp'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--resume',action='store_true')
    a=p.parse_args();manifest=json.loads(a.assets.read_text());ds=manifest['datasets'][a.dataset]
    key=('humanml' if a.dataset=='transfer' else a.dataset)+'/'+a.model
    source=manifest['ready_models'][key]
    assert digest(source['checkpoint'])==source['sha256'] and digest(ds['pair_file'])==ds['pair_sha256']
    if a.dataset=='transfer':assert a.model in ['t2mgpt','satopp']
    run=dict(dataset=a.dataset,model=a.model,source=source,asset_sha256=digest(a.assets),
        cfg_scale=1.3,seed=123,group_seeds=GROUP_SEEDS,limit=a.limit,script_sha256=digest(__file__),
        crop='caption-locked deterministic center crop; repair-all for HumanML/MotionX, strict KIT resolution',
        condition_rng='same per-batch seed reset before original and perturbed generation',
        scope='whole fixed mixture FID; randomized within-32 retrieval')
    a.output.mkdir(parents=True,exist_ok=a.resume);shards=a.output/'shards';shards.mkdir(exist_ok=a.resume)
    if a.resume:assert json.loads((a.output/'run.json').read_text())==run
    else:(a.output/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;pilot.seed_all(123);device=torch.device('cuda')
    from dataset.dataset_TM_eval import CaptionLockedDATALoader
    from dataset.config import configure_evaluator_source,evaluator_opt_path
    from models.evaluator_wrapper import EvaluatorModelWrapper
    from options.get_eval_option import get_opt
    from utils.word_vectorizer import WordVectorizer
    from utils.sato_pp_eval import TM2TTextProcessor,generate_motion_batch
    from tools.audit_motionx_text_coverage import LookupVectorizer
    from MotionGPT.eval_loader_robustness import generate_features,get_motion_embeddings_in_input_order
    pairs=json.loads(Path(ds['pair_file']).read_text());pairs=pairs[:a.limit] if a.limit else pairs
    expected=sorted(r['motion_id'] for r in pairs);lookup={r['motion_id']:r for r in pairs}
    data_name={'humanml':'t2m','motionx':'motionx322_75','transfer':'kit','kit':'kit'}[a.dataset]
    ruler_name='motionx322' if a.dataset=='motionx' else 't2m'
    vec=WordVectorizer(str(ROOT/'glove'),'our_vab')
    loader=CaptionLockedDATALoader(data_name,True,32,vec,pair_data=pairs,num_workers=0,
        length_policy='strict' if a.dataset in ('transfer','kit') else 'repair-all')
    assert len(loader.dataset)==len(expected)
    ruler=ROOT/f'checkpoints/{ruler_name}/text_mot_match_para_aware/model/finest.tar'
    evaluator_dataset='motionx322_75' if a.dataset=='motionx' else 't2m'
    if a.dataset=='kit':
        acceptance=json.loads(Path(manifest['kit_ruler_acceptance']).read_text())
        assert acceptance['accepted'] and acceptance['frozen_verified'] and not acceptance['smoke']
        ruler=Path(acceptance['checkpoint'])
        assert digest(ruler)==acceptance['checkpoint_sha256']
        evaluator_dataset='kit'
    opt=configure_evaluator_source(get_opt(evaluator_opt_path(evaluator_dataset),device),evaluator_dataset)
    opt.evaluator_matcher_checkpoint=str(ruler);evaluator=EvaluatorModelWrapper(opt)
    processor=TM2TTextProcessor(LookupVectorizer(vec,'fallback_lower'),max_text_len=20,alphabetic_only=a.dataset!='motionx')
    native_loader=load_native;native_args=model_args
    if a.dataset=='kit':
        from tools.kit_independent_native import load_native as native_loader,model_args as native_args
    mgpt=a.model.startswith('motiongpt');dim=322 if a.dataset=='motionx' else (251 if a.dataset=='kit' else 263);maxframes=300 if dim==322 else 196
    if mgpt:
        if a.dataset=='motionx':model,data=load_inference(source['config'],source['checkpoint'],device)
        else:
            lm,vae,_,cfg=native_loader(source['config'],source['checkpoint'],device,True)
            model=SimpleNamespace(lm=lm,vae=vae,device=device,hparams=SimpleNamespace(cfg=cfg,codebook_size=lm.m_codebook_size))
            data=SimpleNamespace(nfeats=dim,renorm4t2m=lambda x:x)
        install_motiongpt_cfg(model.lm,1.3)
    else:
        from ablation.repeat20_perturb_fast_batch import load_vqvae
        ma=pilot.make_model_args() if a.dataset=='motionx' else native_args()
        payload=torch.load(source['checkpoint'],map_location='cpu',weights_only=False)
        clip=pilot.load_clip_model(device)
        if 'clip' in payload:
            clip.load_state_dict(payload['clip'],strict=True)
        elif not source.get('official_clip',False):
            raise ValueError('Checkpoint lacks CLIP weights without an explicit official-CLIP source declaration')
        clip.eval().requires_grad_(False)
        trans=pilot.load_transformer(ma,device);trans.load_state_dict(payload['trans'],strict=True);trans.eval().requires_grad_(False)
        net=load_vqvae(ma,strict=True).eval().requires_grad_(False);del payload
    for bi,batch in enumerate(loader):
        dest=shards/f'batch_{bi:04d}.npz';auditpath=shards/f'batch_{bi:04d}.json'
        if dest.exists() and auditpath.exists():continue
        word,pos,captions,lens,real,mlens,_,names=batch
        assert all(' '.join(c.split()).lower()==' '.join(lookup[mid]['text_orig'].split()).lower() for c,mid in zip(captions,names))
        arrays={'motion_id':np.asarray(names),'reference_length':mlens.cpu().numpy()};audits={}
        if a.dataset!='transfer':arrays['real']=get_motion_embeddings_in_input_order(evaluator,real.to(device).float(),mlens).cpu().numpy()
        for condition,field in [('original','text_orig'),('perturbed','text_pair')]:
            prompts=[lookup[mid][field] for mid in names];pilot.seed_all(123+bi*1000003);generated=[]
            if mgpt:
                pred,plen=generate_features(model,data,prompts,'categorical',maxframes,reference_lengths=mlens,
                    temperature=1.,top_k=0,top_p=1.,audit_rows=generated,prompt_mode='official-default')
                codes=[r['motion_tokens'] for r in generated]
            else:
                feature=clip.encode_text(pilot.tokenize_texts(prompts,device)).float()
                pred,plen,codes=generate_motion_batch(net,trans,feature,maxframes,dim,use_fast_batch=True,
                    cfg_scale=1.3,uncond_feat=torch.zeros_like(feature),return_codes=True,strict_fast_path=True)
            arrays[condition]=get_motion_embeddings_in_input_order(evaluator,pred,plen).cpu().numpy()
            arrays['text_'+condition]=processor.encode(evaluator,prompts).cpu().numpy()
            arrays['length_'+condition]=plen.cpu().numpy()
            audits[condition]=[dict(motion_id=mid,generation_text=t,retrieval_text=t,tokens=code) for mid,t,code in zip(names,prompts,codes)]
        temp=dest.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(dest)
        auditpath.write_text(json.dumps(audits)+'\n')
        if bi%5==0:print('UNIFIED_EVAL_PROGRESS',a.dataset,a.model,bi,len(loader),flush=True)
    values=[np.load(shards/f'batch_{bi:04d}.npz',allow_pickle=False) for bi in range(len(loader))]
    ids=np.concatenate([v['motion_id'] for v in values]);ix=canonical_order(ids,expected)
    arrays={k:np.concatenate([v[k] for v in values])[ix] for k in values[0].files}
    result=distribution_metrics(arrays.get('real'),arrays['original'],arrays['perturbed'])
    partitions=[]
    for condition,suffix in [('original','O'),('perturbed','P')]:
        t=arrays['text_'+condition];m=arrays[condition]
        result['Matching_'+suffix]=float(np.linalg.norm(t-m,axis=1).mean())
        scores=[]
        for gs in GROUP_SEEDS:
            r=retrieval(torch.from_numpy(t).to(device),torch.from_numpy(m).to(device),np.random.default_rng(gs).permutation(len(expected)))
            scores.append(r);partitions.append(dict(condition=condition,group_seed=gs,R=[float(x) for x in r]))
        for k,v in enumerate(np.mean(scores,axis=0)):result[f'R{k+1}_{suffix}']=float(v)
    tokens=[]
    for bi in range(len(loader)):
        d=json.loads((shards/f'batch_{bi:04d}.json').read_text())
        tokens.extend((o['tokens'],q['tokens']) for o,q in zip(d['original'],d['perturbed']))
    result['first_token_agreement']=float(np.mean([bool(o and q and o[0]==q[0]) for o,q in tokens]))
    result['exact_sequence_agreement']=float(np.mean([o==q for o,q in tokens]))
    result.update(dataset=a.dataset,model=a.model,samples=len(expected),smoke=bool(a.limit),cfg_scale=1.3,
        checkpoint_sha256=source['sha256'],ruler_sha256=digest(ruler),pair_sha256=ds['pair_sha256'])
    np.savez_compressed(a.output/'embeddings.npz',**arrays)
    (a.output/'partitions.json').write_text(json.dumps(partitions,indent=2)+'\n')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('UNIFIED_EVAL_COMPLETE',a.dataset,a.model,flush=True)


if __name__=='__main__':main()