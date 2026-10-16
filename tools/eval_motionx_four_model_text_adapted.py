"""Single-generation-seed Motion-X P1/P2/P3 retrieval for text-adapted models."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import random

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_motionx_random_retrieval import GROUP_SEEDS,canonical_order,digest,retrieval
from tools.audit_motionx_text_coverage import LookupVectorizer
from tools import train_motionx_text_adaptation_pilot as pilot


def load_mgpt(config,checkpoint,device):
    sys.path.insert(0,str(ROOT/'MotionGPT'))
    from MotionGPT.eval_loader_robustness import load_motiongpt_config
    from mGPT.data.build_data import build_data
    from mGPT.models.build_model import build_model
    before=Path.cwd()
    try:
        os.chdir(ROOT/'MotionGPT')
        cfg=load_motiongpt_config(config,ROOT/'MotionGPT/configs/assets_motionx322.yaml')
        cfg.TEST.CHECKPOINTS=str(checkpoint)
        cfg.TRAIN.STRICT_LOAD=False
        cfg.DATASET.ALLOW_MISSING_WORD_VECTORIZER=True
        cfg.DEBUG=False
        data=build_data(cfg,phase='test')
        model=build_model(cfg,data)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        # Fail rather than accepting a partially loaded adapted model.
        model.load_state_dict(payload['state_dict'],strict=True)
        model.to(device).eval().requires_grad_(False)
    finally:
        os.chdir(before)
    return model,data


@torch.no_grad()
def run(args):
    from dataset.config import configure_evaluator_source,evaluator_opt_path
    from dataset.dataset_TM_eval import CaptionLockedDATALoader
    from models.evaluator_wrapper import EvaluatorModelWrapper
    from options.get_eval_option import get_opt
    from utils.word_vectorizer import WordVectorizer
    from utils.sato_pp_eval import TM2TTextProcessor,generate_motion_batch
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    pilot.seed_all(args.seed)
    pair_path=ROOT/f'dataset/Motion-X-322/test_perb_gemini31_max75_{args.level.lower()}.json'
    pairs=json.loads(pair_path.read_text())
    if isinstance(pairs,dict): pairs=pairs['pairs']
    if len(pairs)!=7679: raise ValueError(f'Expected 7679 test pairs; got {len(pairs)}')
    ids=[p['motion_id'] for p in pairs]
    assert len(set(ids))==7679
    train_ids=pilot.load_split_ids(pilot.DATASET_ROOT,'train')
    val_ids=pilot.load_split_ids(pilot.DATASET_ROOT,'val')
    test_ids=pilot.load_split_ids(pilot.DATASET_ROOT,'test')
    assert not set(ids)&(train_ids|val_ids) and set(ids)<=test_ids
    if args.limit: pairs=pairs[:args.limit]
    expected=sorted(p['motion_id'] for p in pairs)
    lookup={p['motion_id']:p for p in pairs}
    base=WordVectorizer(str(ROOT/'glove'),'our_vab')
    processors={k:TM2TTextProcessor(LookupVectorizer(base,k),max_text_len=20,alphabetic_only=False)
                for k in ('current','fallback_lower')}
    loader=CaptionLockedDATALoader('motionx322_75',True,32,base,pair_data=pairs,num_workers=0,length_policy='repair-all')
    assert len(loader.dataset)==len(expected), 'Test loader dropped paired samples'
    ckpt=ROOT/'checkpoints/motionx322/text_mot_match_para_aware/model/finest.tar'
    opt=configure_evaluator_source(get_opt(evaluator_opt_path('motionx322_75'),device),'motionx322_75')
    opt.evaluator_matcher_checkpoint=str(ckpt)
    evaluator=EvaluatorModelWrapper(opt)
    mgpt=args.model.startswith('motiongpt')
    if mgpt:
        model,data=load_mgpt(args.config,args.checkpoint,device)
        from MotionGPT.eval_loader_robustness import generate_features,get_motion_embeddings_in_input_order
    else:
        from ablation.repeat20_perturb_fast_batch import load_vqvae
        ma=pilot.make_model_args()
        clip=pilot.load_clip_model(device)
        trans=pilot.load_transformer(ma,device)
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        clip.load_state_dict(payload['clip'],strict=True)
        trans.load_state_dict(payload['trans'],strict=True)
        clip.eval().requires_grad_(False); trans.eval().requires_grad_(False)
        net=load_vqvae(ma,strict=True)
        del payload
    collected=[]; motion=[]; texts={k:[] for k in processors}; lengths=[]
    audit_file=(out/'generation_audit.jsonl').open('w')
    try:
        for bi,batch in enumerate(loader):
            word,pos,captions,lens,real,mlens,_,names=batch
            prompts=[lookup[mid]['text_pair'] for mid in names]
            assert all(' '.join(str(c).split()).lower()==' '.join(str(lookup[n]['text_orig']).split()).lower()
                       for c,n in zip(captions,names)), 'Caption mismatch'
            rng_seed=args.seed+bi*1000003+('P1','P2','P3').index(args.level)*100003
            random.seed(rng_seed); np.random.seed(rng_seed%(2**32)); torch.manual_seed(rng_seed)
            generated=[]
            if mgpt:
                pred,plen=generate_features(model,data,prompts,'categorical',300,reference_lengths=mlens,
                                            temperature=1.0,top_k=0,top_p=1.0,audit_rows=generated,
                                            prompt_mode='official-default')
                em=get_motion_embeddings_in_input_order(evaluator,pred,plen)
            else:
                feature=clip.encode_text(pilot.tokenize_texts(prompts,device)).float()
                pred,plen=generate_motion_batch(net,trans,feature,300,322,use_fast_batch=True,cfg_scale=1.3,
                                               uncond_feat=torch.zeros_like(feature),strict_fast_path=True)
                _,em=evaluator.get_co_embeddings(word,pos,lens,pred,plen)
            collected.extend(names); motion.append(em.cpu().numpy()); lengths.extend(plen.cpu().tolist())
            for k,processor in processors.items(): texts[k].append(processor.encode(evaluator,prompts).cpu().numpy())
            for i,(name,prompt) in enumerate(zip(names,prompts)):
                row={'motion_id':name,'generation_text':prompt,'retrieval_text':prompt,
                     'length':int(plen[i]),'batch':bi,'rng_seed':rng_seed}
                if mgpt: row['generation']=generated[i]
                audit_file.write(json.dumps(row)+'\n')
            audit_file.flush()
            if bi%10==0: print(f'TEST_PROGRESS {args.model}/{args.level} {bi}/{len(loader)}',flush=True)
    finally:
        audit_file.close()
    ix=canonical_order(collected,expected)
    m=torch.as_tensor(np.concatenate(motion)[ix],device=device)
    arrays={'motion_id':np.asarray(expected),'motion':m.cpu().numpy(),'length':np.asarray(lengths)[ix]}
    rows=[]; summary=[]
    for mode in processors:
        t=torch.as_tensor(np.concatenate(texts[mode])[ix],device=device)
        arrays['text_'+mode]=t.cpu().numpy()
        for gs in GROUP_SEEDS:
            v=retrieval(t,m,np.random.default_rng(gs).permutation(len(expected)))
            rows.append({'model':args.model,'level':args.level,'lookup':mode,'generation_seed':args.seed,
                         'group_seed':gs,'samples':len(expected),**{f'R{k+1}':float(v[k]) for k in range(3)}})
        rr=[r for r in rows if r['lookup']==mode]
        summary.append({'model':args.model,'level':args.level,'lookup':mode,'generation_seed':args.seed,
                        'samples':len(expected),**{k:float(np.mean([r[k] for r in rr])) for k in ('R1','R2','R3')}})
    np.savez_compressed(out/'embeddings.npz',**arrays)
    with (out/'runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest={'model':args.model,'level':args.level,'checkpoint':str(args.checkpoint),
              'checkpoint_sha256':digest(args.checkpoint),'ruler_sha256':digest(ckpt),
              'pair_file_sha256':digest(pair_path),'sample_ids_sha256':__import__('hashlib').sha256('\n'.join(expected).encode()).hexdigest(),
              'samples':len(expected),'smoke':bool(args.limit),'generation_seed':args.seed,'group_seeds':GROUP_SEEDS,
              'lookup_modes':list(processors),'generation_and_retrieval_use_identical_perturbation':True,
              'script_sha256':digest(__file__),'selection':'fixed validation-chosen protocol; no test configuration selection',
              'mgpt_prompt':'official-default','t2m_cfg_scale':1.3,'checkpoint_load':'strict'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'TEST_COMPLETE {args.model}/{args.level}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['t2mgpt','satopp','motiongpt','motiongpt_satopp'],required=True)
    p.add_argument('--level',choices=['P1','P2','P3'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    if args.model.startswith('motiongpt') and args.config is None: p.error('MotionGPT requires --config')
    run(args)

if __name__=='__main__':main()