"""Fixed 3000-step HumanML3D independent CLIP/T5 text-only adaptation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.humanml_independent_assets import DATA,MODELS,model_args,load_native,pilot
from tools.train_motionx_four_model_text_adaptation import freeze_t5_text_only,sequence_ce


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--train-pairs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);pilot.seed_all(123);device=torch.device('cuda')
    spec=json.loads(a.spec.read_text())[a.model];assert pilot.sha256_file(Path(spec['checkpoint']))==spec['sha256']
    mgpt=a.model.startswith('motiongpt')
    if mgpt:
        lm,_,source,cfg=load_native(spec['config'],spec['checkpoint'],device)
        freeze_t5_text_only(lm);module=lm
        token_dir=DATA/str(cfg.DATASET.CODE_PATH);nb=lm.m_codebook_size
        token_cap=None
    else:
        source=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
        trans=pilot.load_transformer(model_args(),device);trans.load_state_dict(source['trans'],strict=True)
        clip=pilot.load_clip_model(device)
        if 'clip' in source:clip.load_state_dict(source['clip'],strict=True)
        pilot.configure_trainability(trans,clip,'text');trans.eval();clip.eval()
        module=torch.nn.ModuleDict({'clip':clip,'trans':trans});token_dir=DATA/'VQVAE';nb=512;token_cap=50
    # Preserve each dataset's native export length bounds.
    min_native_frames = 24 if DATA.name == 'KIT-ML' else 40
    rows=pilot.read_jsonl(a.train_pairs);eligible=[];missing={}
    for row in rows:
        path=token_dir/(row['motion_id']+'.npy')
        if not path.exists():
            raw=np.load(DATA/'new_joint_vecs'/(row['motion_id']+'.npy'),mmap_mode='r')
            if len(raw)<min_native_frames or len(raw)>=200:
                missing[row['motion_id']]=len(raw);continue
            raise FileNotFoundError(path)
        eligible.append(row)
    effective=a.output/'eligible.jsonl'
    with effective.open('x') as f:
        for row in eligible:f.write(json.dumps(row)+'\n')
    examples,report=pilot.load_training_pairs(effective,dataset_root=DATA,token_dir=token_dir,nb_code=nb,dataset_limit=a.limit or None)
    if token_cap is not None:
        # Full-caption supervision only: reject overlength rows rather than crop
        # away a potentially caption-relevant action without an aligned label.
        long=[e.motion_id for e in examples if e.token_rows.shape[1]>token_cap]
        examples=[e for e in examples if e.token_rows.shape[1]<=token_cap]
    else:long=[]
    assert examples
    frozen=[(n,v) for n,v in module.named_parameters() if not v.requires_grad]
    before=pilot.named_parameter_sha256(frozen)
    params=[v for v in module.parameters() if v.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-6,weight_decay=0.)
    manifest=dict(dataset='kit' if DATA.name=='KIT-ML' else 'humanml3d',model=a.model,source=spec,steps=a.steps,learning_rate=1e-6,
        warmup_steps=min(100,a.steps),seed=123,split='train',text_only=True,smoke=bool(a.limit or a.steps!=3000),
        objective='equal original/paraphrase action prediction CE; same short text-only adaptation protocol',
        frozen_initial=before,filtering=report,native_length_excluded=missing,
        overlength_filtered=len(long),accepted=len(examples),selection='fixed endpoint; no test selection',
        train_pairs_sha256=pilot.sha256_file(a.train_pairs),test_content_read=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rng=np.random.default_rng(123)
    with (a.output/'loss.jsonl').open('x') as log:
        for step in range(1,a.steps+1):
            opt.zero_grad(set_to_none=True);total=0.
            for _ in range(4):
                selected=[examples[int(i)] for i in rng.integers(0,len(examples),4)]
                tokens=[e.token_rows[int(rng.integers(e.token_rows.shape[0]))] for e in selected]
                texts=[e.text_orig for e in selected]+[e.text_pair for e in selected]
                if mgpt:
                    targets=[f'<motion_id_{nb}>'+''.join(f'<motion_id_{int(t)}>' for t in row)+f'<motion_id_{nb+1}>' for row in tokens]
                    enc=lm.tokenizer(['Generate motion: '+s for s in texts],padding=True,truncation=True,max_length=lm.max_length,return_tensors='pt').to(device)
                    tar=lm.tokenizer(targets+targets,padding=True,truncation=False,return_tensors='pt').to(device)
                    assert tar.input_ids.shape[1]<=lm.max_length
                    labels=tar.input_ids.masked_fill(tar.attention_mask.eq(0),-100)
                    pred=lm.language_model(**enc,labels=labels,decoder_attention_mask=tar.attention_mask,use_cache=False)
                    loss=sequence_ce(pred.logits,labels)
                else:
                    lengths=torch.tensor([len(t)+1 for t in tokens],device=device)
                    targets=torch.full((4,int(lengths.max())),513,device=device,dtype=torch.long)
                    for i,t in enumerate(tokens):targets[i,:len(t)]=torch.as_tensor(t,device=device);targets[i,len(t)]=512
                    inputs=targets[:,:-1]
                    corrupted=torch.where(torch.rand(inputs.shape,device=device)<.5,inputs,torch.randint(512,inputs.shape,device=device))
                    loss,_=pilot.compute_paired_supervised_loss(clip,trans,pilot.tokenize_texts(texts,device),
                        torch.cat([corrupted]*2),torch.cat([targets]*2),torch.cat([lengths]*2),4)
                assert torch.isfinite(loss);(loss/4).backward();total+=float(loss.detach())/4
            if step==1:
                assert all(v.grad is None for _,v in frozen)
                assert any(v.grad is not None and v.grad.abs().sum()>0 for v in params)
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            for group in opt.param_groups:group['lr']=1e-6*min(1.,step/max(1,min(100,a.steps)))
            opt.step()
            row=dict(step=step,loss=total,grad_norm=float(norm));log.write(json.dumps(row)+'\n');log.flush()
            if step==1 or step%50==0:print('HUMAN_TEXT_TRAIN',a.model,json.dumps(row),flush=True)
    assert before==pilot.named_parameter_sha256(frozen)
    if mgpt:
        for k,v in lm.state_dict().items():source['state_dict']['lm.'+k]=v.detach().cpu()
        for key in ['optimizer_states','lr_schedulers','loops']:source.pop(key,None)
    else:
        source['clip']={k:v.detach().cpu() for k,v in clip.state_dict().items()}
        for key in ['optimizer','scheduler']:source.pop(key,None)
    manifest['frozen_verified']=True;source['text_adaptation']=manifest
    path=a.output/('adapted.ckpt' if mgpt else 'adapted.pth');torch.save(source,path)
    manifest.update(checkpoint=str(path),checkpoint_sha256=pilot.sha256_file(path))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('HUMAN_TEXT_TRAIN_COMPLETE',a.model,flush=True)


if __name__=='__main__':main()