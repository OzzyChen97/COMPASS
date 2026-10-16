"""HumanML3D native independent CLIP/T5 perturbation tests, CFG=1.3."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.humanml_independent_assets import DATA,MODELS,model_args,load_native,pilot
from tools.audit_motionx_random_retrieval import GROUP_SEEDS,canonical_order,digest,retrieval
from models.motiongpt_cfg import install_motiongpt_cfg


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--level',choices=['P1','P2','P3'],required=True);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);pilot.seed_all(123);device=torch.device('cuda')
    spec=json.loads(a.spec.read_text())[a.model]
    from dataset.config import configure_evaluator_source,evaluator_opt_path
    from dataset.dataset_TM_eval import CaptionLockedDATALoader
    from options.get_eval_option import get_opt
    from models.evaluator_wrapper import EvaluatorModelWrapper
    from utils.word_vectorizer import WordVectorizer
    from utils.sato_pp_eval import TM2TTextProcessor,generate_motion_batch
    from tools.audit_motionx_text_coverage import LookupVectorizer
    from MotionGPT.eval_loader_robustness import generate_features,get_motion_embeddings_in_input_order
    pairs=json.loads((DATA/f'test_perb_table1_level_{a.level.lower()}.json').read_text())
    assert len(pairs)==4380 and len({r['motion_id'] for r in pairs})==4380
    test=pilot.load_split_ids(DATA,'test');forbidden=pilot.load_split_ids(DATA,'train')|pilot.load_split_ids(DATA,'val')
    assert {r['motion_id'] for r in pairs}<=test and not {r['motion_id'] for r in pairs}&forbidden
    if a.limit:pairs=pairs[:a.limit]
    expected=sorted(r['motion_id'] for r in pairs);lookup={r['motion_id']:r for r in pairs}
    vec=WordVectorizer(str(ROOT/'glove'),'our_vab')
    loader=CaptionLockedDATALoader('t2m',True,32,vec,pair_data=pairs,num_workers=0,length_policy='repair-all')
    assert len(loader.dataset)==len(expected)
    ruler=ROOT/'checkpoints/t2m/text_mot_match_para_aware/model/finest.tar'
    opt=configure_evaluator_source(get_opt(evaluator_opt_path('t2m'),device),'t2m');opt.evaluator_matcher_checkpoint=str(ruler)
    evaluator=EvaluatorModelWrapper(opt)
    processors={mode:TM2TTextProcessor(LookupVectorizer(vec,mode),max_text_len=20,alphabetic_only=True) for mode in ['current','fallback_lower']}
    mgpt=a.model.startswith('motiongpt')
    if mgpt:
        lm,vae,payload,cfg=load_native(spec['config'],a.checkpoint,device,True)
        install_motiongpt_cfg(lm,1.3)
        model=SimpleNamespace(lm=lm,vae=vae,device=device,hparams=SimpleNamespace(cfg=cfg,codebook_size=lm.m_codebook_size))
        data=SimpleNamespace(nfeats=263,renorm4t2m=lambda x:x)
    else:
        from ablation.repeat20_perturb_fast_batch import load_vqvae
        payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        clip=pilot.load_clip_model(device);clip.load_state_dict(payload['clip'],strict=True);clip.requires_grad_(False).eval()
        trans=pilot.load_transformer(model_args(),device);trans.load_state_dict(payload['trans'],strict=True);trans.requires_grad_(False).eval()
        net=load_vqvae(model_args(),strict=True).eval().requires_grad_(False)
    meta=payload['text_adaptation'];assert meta['dataset']=='humanml3d' and meta['text_only'] and meta['frozen_verified']
    assert meta['source']['sha256']==spec['sha256']
    if not a.limit:assert not meta['smoke'] and meta['steps']==3000
    del payload
    ids=[];motion=[];texts={mode:[] for mode in processors}
    with (a.output/'prompts.jsonl').open('x') as audit:
        for bi,batch in enumerate(loader):
            word,pos,captions,lens,real,mlens,_,names=batch
            prompts=[lookup[mid]['text_pair'] for mid in names]
            assert all(' '.join(str(c).split()).lower()==' '.join(lookup[mid]['text_orig'].split()).lower() for c,mid in zip(captions,names))
            seed=123+bi*1000003+['P1','P2','P3'].index(a.level)*100003;pilot.seed_all(seed)
            generated=[]
            if mgpt:
                pred,plen=generate_features(model,data,prompts,'categorical',196,reference_lengths=mlens,
                    temperature=1.,top_k=0,top_p=1.,audit_rows=generated,prompt_mode='official-default')
            else:
                feature=clip.encode_text(pilot.tokenize_texts(prompts,device)).float()
                pred,plen=generate_motion_batch(net,trans,feature,196,263,use_fast_batch=True,cfg_scale=1.3,
                    uncond_feat=torch.zeros_like(feature),strict_fast_path=True)
            em=get_motion_embeddings_in_input_order(evaluator,pred,plen)
            ids.extend(names);motion.append(em.cpu().numpy())
            for mode,processor in processors.items():texts[mode].append(processor.encode(evaluator,prompts).cpu().numpy())
            for i,mid in enumerate(names):
                row=dict(motion_id=mid,generation_text=prompts[i],retrieval_text=prompts[i],level=a.level,length=int(plen[i]))
                if generated:row['generation']=generated[i]
                audit.write(json.dumps(row)+'\n')
            audit.flush()
            if bi%10==0:print('HUMAN_CFG_TEST_PROGRESS',a.model,a.level,bi,len(loader),flush=True)
    ix=canonical_order(ids,expected);mv=np.concatenate(motion)[ix];rows=[];arrays={'motion_id':np.asarray(expected),'motion':mv}
    for mode in processors:
        tv=np.concatenate(texts[mode])[ix];arrays['text_'+mode]=tv
        values=[retrieval(torch.from_numpy(tv).to(device),torch.from_numpy(mv).to(device),np.random.default_rng(seed).permutation(len(expected))) for seed in GROUP_SEEDS]
        result=np.mean(values,axis=0)
        rows.append(dict(model=a.model,level=a.level,lookup=mode,samples=len(expected),**{f'R{i+1}':float(result[i]) for i in range(3)}))
    np.savez_compressed(a.output/'embeddings.npz',**arrays)
    (a.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    result=dict(dataset='humanml3d',samples=len(expected),smoke=bool(a.limit),cfg_scale=1.3,
        cfg=lm.cfg_audit if mgpt else {'scale':1.3,'unconditional_features':'zero'},checkpoint_sha256=digest(a.checkpoint),
        ruler_sha256=digest(ruler),group_seeds=GROUP_SEEDS,generation_seed=123,independent_text_encoder=True)
    if mgpt:assert lm.cfg_audit['calls']>0
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('HUMAN_CFG_TEST_COMPLETE',a.model,a.level,flush=True)


if __name__=='__main__':main()