"""Full-T5, complete-epoch MotionGPT fine-tuning with synchronized data parallelism.

MotionGPT uses paired supervised generation. MotionGPT+SATO++ reuses its native
preserve-only objective, including differentiable attribution and a frozen
teacher. VAE tensors are never trained. Only complete-epoch checkpoints resume.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import train_motionx_text_adaptation_pilot as pilot
from tools.train_motionx_four_model_text_adaptation import load_lm,filter_native_short_pairs


def epoch_indices(size,world,rank,epoch,seed=123):
    sampler=DistributedSampler(range(size),num_replicas=world,rank=rank,shuffle=True,seed=seed,drop_last=False)
    sampler.set_epoch(epoch)
    return list(sampler)


def lr_at(step,total,warmup,peak):
    if step<=warmup:return peak*step/max(warmup,1)
    fraction=min(1.,(step-warmup)/max(total-warmup,1))
    return peak*(.1+.9*.5*(1+math.cos(math.pi*fraction)))


def synchronize_gradients(params,world):
    """Explicit mean reduction supports native higher-order autograd.grad losses."""
    if world==1:return
    flags=torch.tensor([p.grad is not None for p in params],dtype=torch.int32,device=params[0].device)
    dist.all_reduce(flags)
    if not torch.all((flags==0)|(flags==world)):
        raise RuntimeError('Gradient reachability differs between ranks')
    active=[p for p,n in zip(params,flags.tolist()) if n==world]
    bucket=[];elements=0
    def flush():
        flat=torch.cat([p.grad.reshape(-1) for p in bucket])
        dist.all_reduce(flat);flat.div_(world)
        offset=0
        for p in bucket:
            p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
    for p in active:
        if bucket and elements+p.numel()>4*1024*1024:
            flush();bucket=[];elements=0
        bucket.append(p);elements+=p.numel()
    if bucket:flush()


class FullObjective(torch.nn.Module):
    def __init__(self,lm,source,cfg,sato):
        super().__init__()
        self.lm=lm
        self.lm.requires_grad_(True)
        self.hparams=SimpleNamespace(cfg=cfg)
        self.sato=sato
        self.metrics={}
        if sato:
            from mGPT.models.mgpt_sato_pp import AttributionPooling,PredictorHead
            self.attr_pooling=AttributionPooling(lm.language_model.config.d_model,cfg.SATO.PROJ_DIM,cfg.SATO.PROJ_TYPE)
            self.predictor_head=PredictorHead(cfg.SATO.PROJ_DIM)
            for prefix,module in [('attr_pooling',self.attr_pooling),('predictor_head',self.predictor_head)]:
                module.load_state_dict({k[len(prefix)+1:]:v for k,v in source['state_dict'].items() if k.startswith(prefix+'.')},strict=True)
        # Teacher deliberately excluded from optimizer/state_dict/gradient reduction.
        object.__setattr__(self,'_sato_teacher',None)

    def initialize_teacher(self):
        if self.sato:
            teacher=copy.deepcopy(self.lm).eval().requires_grad_(False)
            object.__setattr__(self,'_sato_teacher',teacher)

    def log(self,name,value,**kwargs):
        self.metrics[name]=value.detach()

    def _prompt_edit_mask(self,*args):
        from mGPT.models.mgpt_sato_pp import MotionGPTSATOPP
        return MotionGPTSATOPP._prompt_edit_mask(self,*args)

    @staticmethod
    def _pad_prompt_values(*args):
        from mGPT.models.mgpt_sato_pp import MotionGPTSATOPP
        return MotionGPTSATOPP._pad_prompt_values(*args)

    def forward(self,batch):
        if self.sato:
            from mGPT.models.mgpt_sato_pp import MotionGPTSATOPP
            result=MotionGPTSATOPP.train_lm_forward(self,batch)
            return result['outputs'].loss
        choices=self.lm.sample_template_choices(batch['tasks'])
        clean=self.lm.sato_forward_encdec(batch['text'],batch['motion'],batch['length'],batch['tasks'],template_choices=choices)
        pair=self.lm.sato_forward_encdec(batch['text_pair'],batch['motion'],batch['length'],batch['tasks'],template_choices=choices)
        # Same supervised coefficients as native SATO++; auxiliary losses differ.
        loss=clean['outputs'].loss+pair['outputs'].loss
        self.metrics={'sato/base':clean['outputs'].loss.detach(),'sato/pair_gen':pair['outputs'].loss.detach(),
                      'sato/total':loss.detach()}
        return loss


def make_batch(examples,indices,device):
    selected=[examples[i] for i in indices]
    lengths=[len(e.token_rows[0]) for e in selected]
    tokens=torch.zeros(len(selected),max(lengths),dtype=torch.long,device=device)
    for i,e in enumerate(selected):tokens[i,:lengths[i]]=torch.as_tensor(e.token_rows[0],device=device)
    task={'input':['Generate motion: <Caption_Placeholder>'],'output':['<Motion_Placeholder>']}
    return {'motion':tokens,'length':lengths,'text':[e.text_orig for e in selected],
            'text_pair':[e.text_pair for e in selected],'relation':['preserve']*len(selected),'tasks':[task]*len(selected)}


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def state_hash(state,prefix):
    return pilot.named_parameter_sha256((k,v) for k,v in state.items() if k.startswith(prefix))


def run(a):
    rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl')
    torch.set_num_threads(2)
    pilot.seed_all(a.seed+rank)
    torch.backends.cuda.matmul.allow_tf32=False
    # Native SATO++ differentiates through Grad x Input; fused attention cannot.
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    if rank==0:
        if a.resume:
            if not a.output.is_dir():raise FileNotFoundError(a.output)
        else:pilot.ensure_empty_output_dir(a.output)
    if world>1:dist.barrier()
    source_spec=json.loads(a.spec.read_text())[a.model]
    assert pilot.sha256_file(Path(source_spec['checkpoint']))==source_spec['sha256']
    lm,source,cfg=load_lm(source_spec['config'],source_spec['checkpoint'])
    sato=a.model=='motiongpt_satopp'
    objective=FullObjective(lm,source,cfg,sato).float().to(device)
    objective.initialize_teacher()
    teacher_hash=pilot.named_parameter_sha256(objective._sato_teacher.named_parameters()) if sato else None
    vae_hash=state_hash(source['state_dict'],'vae.')
    # Native auxiliary code uses differentiable gathers and reductions per rank.
    # Synchronize every trainable student tensor once, then gradients every update.
    if world>1:
        for value in objective.state_dict().values():dist.broadcast(value,0)
    effective=a.output/'eligible_train.jsonl'
    token_dir=pilot.DATASET_ROOT/str(cfg.DATASET.CODE_PATH)
    if rank==0 and not a.resume:
        report=filter_native_short_pairs(a.train_pairs,token_dir,pilot.DATASET_ROOT,effective)
        atomic_json(a.output/'native_filter.json',report)
    if world>1:dist.barrier()
    examples,filtering=pilot.load_training_pairs(effective,token_dir=token_dir,nb_code=lm.m_codebook_size,dataset_limit=a.limit or None)
    assert a.limit or filtering['accepted']==201335
    params=[p for p in objective.parameters() if p.requires_grad]
    assert all(p.requires_grad for p in objective.lm.parameters()), 'T5 was partially frozen'
    groups={k:[p for n,p in objective.lm.language_model.named_parameters() if n.startswith(prefix)]
            for k,prefix in [('encoder','encoder.'),('decoder','decoder.'),('shared','shared.'),('output','lm_head.')]}
    optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01)
    microsteps=math.ceil(math.ceil(len(examples)/world)/a.microbatch)
    steps_epoch=math.ceil(microsteps/a.accumulate)
    total_steps=steps_epoch*a.epochs
    meta={'model':a.model,'source':source_spec,'epochs':a.epochs,'seed':a.seed,'world_size':world,
          'local_microbatch':a.microbatch,'accumulate':a.accumulate,'effective_pairs_per_step':world*a.microbatch*a.accumulate,
          'peak_lr':a.lr,'warmup_steps':min(a.warmup,total_steps),'scheduler':'linear warmup then cosine to 10% peak',
          'trainable_parameters':sum(p.numel() for p in params),'full_t5_trainable':True,
          'native_sato_objective':sato,'teacher_frozen':sato,'teacher_initial_hash':teacher_hash,
          'vae_initial_hash':vae_hash,'filtering':filtering,'steps_per_epoch':steps_epoch,
          'train_pairs_sha256':pilot.sha256_file(a.train_pairs),'native_eligible_pairs_sha256':pilot.sha256_file(effective),
          'test_content_read':False,'sampling':'DistributedSampler shuffle without replacement each epoch; at most world_size-1 padding repeats, no dropped rows',
          'initial_student_hash':pilot.named_parameter_sha256(objective.named_parameters()),
          'script_sha256':pilot.sha256_file(Path(__file__))}
    if sato:
        from omegaconf import OmegaConf
        meta['native_sato_config']=OmegaConf.to_container(cfg.SATO,resolve=True)
    start_epoch=0;global_step=0
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        for k in ['model','source','epochs','seed','world_size','local_microbatch','accumulate','peak_lr','train_pairs_sha256']:
            assert saved['metadata'][k]==meta[k],f'Resume configuration mismatch: {k}'
        objective.load_state_dict(saved['student'],strict=True)
        optimizer.load_state_dict(saved['optimizer']);start_epoch=saved['epoch'];global_step=saved['step']
    if rank==0:atomic_json(a.output/'manifest.json',meta)
    fields=['epoch','step','microstep','loss','base','pair_gen','teach','align','var','cov','grad_norm','lr','pairs_seen_rank']
    handle=(a.output/f'loss_rank{rank}.csv').open('a' if a.resume else 'x',newline='')
    writer=csv.DictWriter(handle,fieldnames=fields)
    if not a.resume:writer.writeheader()
    try:
        for epoch in range(start_epoch,a.epochs):
            pilot.seed_all(a.seed+rank+epoch*100003)
            objective.train()
            indices=epoch_indices(len(examples),world,rank,epoch,a.seed)
            chunks=[indices[i:i+a.microbatch] for i in range(0,len(indices),a.microbatch)]
            coverage={'epoch':epoch+1,'rank':rank,'samples':len(indices),'unique':len(set(indices)),
                      'indices_sha256':pilot.canonical_sha256(indices)}
            atomic_json(a.output/f'coverage_epoch{epoch+1}_rank{rank}.json',coverage)
            if rank==0:print(f'EPOCH_START model={a.model} epoch={epoch+1} unique_pool={len(examples)} microsteps={len(chunks)}',flush=True)
            seen=0
            for window in range(0,len(chunks),a.accumulate):
                optimizer.zero_grad(set_to_none=True)
                window_chunks=chunks[window:window+a.accumulate]
                denom=sum(len(c) for c in window_chunks)
                totals={};total_loss=0.
                for offset,indices_batch in enumerate(window_chunks):
                    batch=make_batch(examples,indices_batch,device)
                    loss=objective(batch)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-T5 loss')
                    weight=len(indices_batch)/denom
                    (loss*weight).backward()
                    total_loss+=float(loss.detach())*weight
                    for k,v in objective.metrics.items():totals[k]=totals.get(k,0.)+float(v)*weight
                    seen+=len(indices_batch)
                synchronize_gradients(params,world)
                if global_step==0:
                    reach={k:float(sum((p.grad.float().square().sum() for p in ps if p.grad is not None),torch.zeros((),device=device)).sqrt()) for k,ps in groups.items()}
                    assert all(reach[k]>0 for k in ['encoder','decoder','shared']),reach
                    assert not sato or all(p.grad is None for p in objective._sato_teacher.parameters())
                    atomic_json(a.output/f'gradient_reachability_rank{rank}.json',reach)
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                global_step+=1
                lr=lr_at(global_step,total_steps,min(a.warmup,total_steps),a.lr)
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.step()
                row={'epoch':epoch+1,'step':global_step,'microstep':window+len(window_chunks),
                     'loss':total_loss,'grad_norm':float(norm),'lr':lr,'pairs_seen_rank':seen,
                     **{k:totals.get('sato/'+k,0.) for k in ['base','pair_gen','teach','align','var','cov']}}
                writer.writerow(row);handle.flush()
                if rank==0 and (global_step==1 or global_step%50==0):print('TRAIN',json.dumps(row),flush=True)
            if world>1:dist.barrier()
            student_hash=pilot.named_parameter_sha256(objective.named_parameters())
            if world>1:
                hashes=[None]*world;dist.all_gather_object(hashes,student_hash)
                assert len(set(hashes))==1, 'Student weights diverged across ranks'
            assert state_hash(source['state_dict'],'vae.')==vae_hash
            if sato:assert pilot.named_parameter_sha256(objective._sato_teacher.named_parameters())==teacher_hash
            if rank==0:
                # Preserve every unused source tensor. Student includes LM and
                # native SATO heads; VAE is copied verbatim from source bundle.
                state=dict(source['state_dict'])
                state.update({k:v.detach().cpu() for k,v in objective.state_dict().items()})
                payload={'state_dict':state,'full_t5_training':{**meta,'epoch':epoch+1,'global_step':global_step,
                    'vae_frozen_verified':True,'teacher_frozen_verified':sato,'student_sha256':student_hash}}
                checkpoint=a.output/f'epoch_{epoch+1:02d}.ckpt'
                temp=checkpoint.with_suffix('.tmp');torch.save(payload,temp);temp.replace(checkpoint)
                resume={'student':{k:v.detach().cpu() for k,v in objective.state_dict().items()},'optimizer':optimizer.state_dict(),
                        'epoch':epoch+1,'step':global_step,'metadata':meta}
                temp=a.output/'resume_latest.tmp';torch.save(resume,temp);temp.replace(a.output/'resume_latest.pth')
                atomic_json(a.output/f'epoch_{epoch+1:02d}.ready.json',{'epoch':epoch+1,'checkpoint':str(checkpoint),
                    'checkpoint_sha256':pilot.sha256_file(checkpoint),'global_step':global_step,'student_sha256':student_hash,
                    'unique_pairs_per_epoch':len(examples),'padding_repeats_per_epoch':math.ceil(len(examples)/world)*world-len(examples),
                    'vae_frozen_verified':True,'teacher_frozen_verified':sato})
                print('EPOCH_COMPLETE',a.model,epoch+1,flush=True)
            if world>1:dist.barrier()
        if rank==0:
            atomic_json(a.output/'completed.json',{'complete':True,'epochs':a.epochs,'steps':global_step,'full_unique_coverage_each_epoch':True})
            print('FULL_T5_TRAINING_COMPLETE',a.model,flush=True)
    finally:
        handle.close()
        if world>1:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['motiongpt','motiongpt_satopp'],required=True)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--train-pairs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--microbatch',type=int,default=8);p.add_argument('--accumulate',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--warmup',type=int,default=500)
    p.add_argument('--seed',type=int,default=123);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--resume',type=Path)
    a=p.parse_args()
    if min(a.epochs,a.microbatch,a.accumulate)<1 or a.lr<=0 or not math.isfinite(a.lr) or a.warmup<0:p.error('Invalid training budget')
    run(a)

if __name__=='__main__':main()