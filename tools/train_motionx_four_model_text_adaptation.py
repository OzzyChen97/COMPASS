"""Fixed-endpoint text-only adaptation of the remaining Motion-X model families.

Keeps each source bundle and its generator/decoder weights; never transfers a
text encoder between families. T5 shared embeddings are frozen because the
motion decoder also uses them. Only train/validation assets enter training.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_motionx_text_adaptation_pilot as pilot


def freeze_t5_text_only(lm):
    lm.requires_grad_(False)
    # Enabling encoder.requires_grad_ would also enable decoder-shared embeddings.
    names = []
    for name, param in lm.language_model.named_parameters():
        if name.startswith('encoder.') and not name.startswith('encoder.embed_tokens.'):
            param.requires_grad_(True)
            names.append(name)
    if not names:
        raise RuntimeError('No T5 encoder parameters found')
    decoder_ids = {id(p) for p in lm.language_model.decoder.parameters()}
    assert not any(p.requires_grad for p in lm.language_model.decoder.parameters())
    assert not lm.language_model.shared.weight.requires_grad
    assert not lm.language_model.lm_head.weight.requires_grad
    assert not any(id(p) in decoder_ids for p in lm.parameters() if p.requires_grad)
    return names


def load_lm(config_path, checkpoint):
    sys.path.insert(0, str(ROOT / 'MotionGPT'))
    from MotionGPT.eval_loader_robustness import load_motiongpt_config
    from mGPT.config import instantiate_from_config
    before = Path.cwd()
    try:
        os.chdir(ROOT / 'MotionGPT')
        cfg = load_motiongpt_config(Path(config_path), ROOT/'MotionGPT/configs/assets_motionx322.yaml')
        lm = instantiate_from_config(cfg.lm.default)
    finally:
        os.chdir(before)
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weights = {k[len('lm.'):]: v for k,v in source['state_dict'].items() if k.startswith('lm.')}
    lm.load_state_dict(weights, strict=True)
    return lm, source, cfg


def sequence_ce(logits, labels):
    mask = labels.ne(-100)
    loss = F.cross_entropy(logits.transpose(1,2), labels, reduction='none', ignore_index=-100)
    return ((loss * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()


def filter_native_short_pairs(pair_path, token_dir, dataset_root, output):
    """Match MotionGPT's native <4-frame exclusion; reject other missing assets."""
    rows=[json.loads(line) for line in Path(pair_path).read_text().splitlines() if line.strip()]
    train=pilot.load_split_ids(dataset_root,'train')
    forbidden=pilot.load_split_ids(dataset_root,'val') | pilot.load_split_ids(dataset_root,'test')
    missing={}; kept=[]; removed=0
    for row in rows:
        mid=row['motion_id']
        if mid not in train or mid in forbidden:
            raise pilot.TrainingDataError(f'Nontrain motion in adaptation pool: {mid}')
        if not (token_dir/f'{mid}.npy').exists():
            if mid not in missing:
                motion=np.load(dataset_root/'new_joint_vecs'/f'{mid}.npy',mmap_mode='r',allow_pickle=False)
                if motion.ndim!=2 or motion.shape[1]!=322 or not 0 < len(motion) < 4:
                    raise FileNotFoundError(f'Missing tokens for eligible MotionGPT motion: {mid}')
                missing[mid]=len(motion)
            removed+=1
        else:
            kept.append(row)
    with Path(output).open('x') as f:
        for row in kept:f.write(json.dumps(row)+'\n')
    return {'native_short_motions':missing,'native_short_pair_rows_removed':removed,
            'original_pair_rows':len(rows),'remaining_pair_rows':len(kept),
            'rule':'MotionGPT native tokenization requires at least 4 frames; no padding or foreign VQ tokens'}


def run(args):
    out = pilot.ensure_empty_output_dir(args.output)
    log = pilot.make_logger(out)
    pilot.seed_all(args.seed)
    torch.set_num_threads(2)
    spec = json.loads(args.spec.read_text())[args.model]
    checkpoint = Path(spec['checkpoint'])
    assert pilot.sha256_file(checkpoint) == spec['sha256'], 'Source checkpoint changed'
    mgpt = args.model.startswith('motiongpt')
    device = torch.device('cuda')
    source = None
    if mgpt:
        lm, source, cfg = load_lm(spec['config'], checkpoint)
        token_dir = pilot.DATASET_ROOT / str(cfg.DATASET.CODE_PATH)
        nb_code = int(lm.m_codebook_size)
        train_names = freeze_t5_text_only(lm)
        lm = lm.float().to(device).eval()
        modules = {'lm':lm}
    else:
        source = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model_args = pilot.make_model_args()
        trans = pilot.load_transformer(model_args, device)
        clip = pilot.load_clip_model(device)
        trans.load_state_dict(source['trans'], strict=True)
        clip.load_state_dict(source['clip'], strict=True)
        pilot.configure_trainability(trans, clip, 'text')
        trans.eval(); clip.eval()
        modules = {'trans': trans, 'clip':clip}
        token_dir, nb_code = pilot.TOKEN_DIR, 512
        train_names = [n for n,p in clip.named_parameters() if p.requires_grad]
    effective_pairs=args.train_pairs
    native_filter={}
    if mgpt:
        effective_pairs=out/'native_eligible_train_pairs.jsonl'
        native_filter=filter_native_short_pairs(args.train_pairs,token_dir,pilot.DATASET_ROOT,effective_pairs)
        (out/'native_short_filter.json').write_text(json.dumps(native_filter,indent=2)+'\n')
    examples, filtering = pilot.load_training_pairs(effective_pairs, token_dir=token_dir,
                                                    nb_code=nb_code, dataset_limit=args.limit or None)
    filtering['native_eligibility']=native_filter
    log.info('Filtering report: %s', json.dumps(filtering))
    frozen = [(f'{m}.{n}',p) for m,mod in modules.items() for n,p in mod.named_parameters() if not p.requires_grad]
    params = [p for mod in modules.values() for p in mod.parameters() if p.requires_grad]
    before_frozen = pilot.named_parameter_sha256(frozen)
    learning_rate=getattr(args,'learning_rate',1e-6)
    warmup_steps=getattr(args,'warmup_steps',0)
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)
    meta = {'model':args.model, 'source':spec, 'steps':args.steps, 'seed':args.seed,
            'batch_pairs':16, 'microbatch_pairs':args.microbatch, 'learning_rate':learning_rate,
            'warmup_steps':warmup_steps, 'learning_rate_schedule':'linear warmup then constant',
            'train_pairs_sha256':pilot.sha256_file(args.train_pairs), 'filtering':filtering,
            'trainable_names':train_names, 'trainable_parameters':sum(p.numel() for p in params),
            'frozen_hash_initial':before_frozen, 'text_only':True, 'test_used':False,
            'objective':'equal clean/paraphrase sequence-mean supervised CE; no SATO auxiliary losses',
            'token_dir':str(token_dir), 't5_shared_embeddings_frozen':mgpt,
            'checkpoint_selection':'fixed endpoint; not selected using test',
            'script_sha256':pilot.sha256_file(Path(__file__))}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    rng = np.random.default_rng(args.seed)
    schedule = hashlib.sha256()
    log.info('TRAIN_START trainable=%s frozen_hash=%s',meta['trainable_parameters'],before_frozen)
    with (out/'losses.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['step','loss','clean_loss','paraphrase_loss','grad_norm','learning_rate','batch_sha256']); writer.writeheader()
        for step in range(1,args.steps+1):
            lr=learning_rate * min(1.0, step/max(warmup_steps,1))
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            clean_total=0.0; pair_total=0.0
            # Reuse pilot sampling/corruption for matched T2M-family controls.
            batch = pilot.make_training_batch(examples,16,rng,device,nb_code=nb_code,block_size=76,pkeep=0.5)
            schedule.update(batch.plan_sha256.encode())
            if mgpt:
                total=0.0
                for start in range(0,16,args.microbatch):
                    stop=min(start+args.microbatch,16)
                    indices=list(range(start,stop))+list(range(16+start,16+stop))
                    texts=[batch.texts[i] for i in indices]
                    targets=[]
                    for i in range(start,stop):
                        e=examples[batch.pair_indices[i]]
                        tokens=e.token_rows[batch.token_row_indices[i]].tolist()
                        targets.append(f'<motion_id_{nb_code}>'+''.join(f'<motion_id_{int(t)}>' for t in tokens)+f'<motion_id_{nb_code+1}>')
                    targets=targets+targets
                    enc=lm.tokenizer(['Generate motion: '+t for t in texts],padding=True,truncation=True,
                                     max_length=lm.max_length,return_tensors='pt').to(device)
                    target=lm.tokenizer(targets,padding=True,truncation=False,return_tensors='pt').to(device)
                    if target.input_ids.shape[1] > lm.max_length:
                        raise RuntimeError('Motion target would be truncated')
                    labels=target.input_ids.masked_fill(target.attention_mask.eq(0),-100)
                    result=lm.language_model(**enc,labels=labels,decoder_attention_mask=target.attention_mask,use_cache=False)
                    n=stop-start
                    clean_loss=sequence_ce(result.logits[:n],labels[:n])
                    pair_loss=sequence_ce(result.logits[n:],labels[n:])
                    loss=0.5*(clean_loss+pair_loss)*(n/16)
                    clean_total+=float(clean_loss.detach())*(n/16)
                    pair_total+=float(pair_loss.detach())*(n/16)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                    loss.backward(); total+=float(loss.detach())
            else:
                tokens=pilot.tokenize_texts(batch.texts,device)
                loss,metrics=pilot.compute_paired_supervised_loss(clip,trans,tokens,batch.input_indices,
                                                           batch.targets,batch.lengths,clean_batch_size=16)
                clean_total=float(metrics['clean_loss']); pair_total=float(metrics['paraphrase_loss'])
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                loss.backward(); total=float(loss.detach())
            if step==1:
                assert all(p.grad is None for _,p in frozen), 'Frozen parameter has gradient'
                assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in params), 'No text gradient'
                if mgpt:
                    meta['first_step_encoder_block_grad_norms']={str(i):float(sum(
                        p.grad.detach().float().square().sum() for p in block.parameters() if p.grad is not None).sqrt())
                        for i,block in enumerate(lm.language_model.encoder.block)}
                    (out/'gradient_audit.json').write_text(json.dumps(meta['first_step_encoder_block_grad_norms'],indent=2)+'\n')
            norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            optimizer.step()
            row={'step':step,'loss':total,'clean_loss':clean_total,'paraphrase_loss':pair_total,
                 'grad_norm':float(norm),'learning_rate':lr,'batch_sha256':batch.plan_sha256}
            writer.writerow(row); f.flush()
            if step==1 or step%50==0 or step==args.steps: log.info('step=%d loss=%.6f grad_norm=%.4f',step,total,float(norm))
    after_frozen=pilot.named_parameter_sha256(frozen)
    assert before_frozen==after_frozen, 'Frozen parameters changed'
    meta.update(frozen_verified=True,frozen_hash_final=after_frozen,schedule_sha256=schedule.hexdigest(),last_step=args.steps)
    # Preserve every unused auxiliary/teacher/VAE tensor in its original bundle.
    if mgpt:
        for k,v in lm.state_dict().items(): source['state_dict']['lm.'+k]=v.detach().cpu()
        for key in ('optimizer_states','lr_schedulers','loops'): source.pop(key,None)
    else:
        source['clip']={k:v.detach().cpu() for k,v in clip.state_dict().items()}
        source['trans']={k:v.detach().cpu() for k,v in trans.state_dict().items()}
        for key in ('optimizer','scheduler'): source.pop(key,None)
    source['text_adaptation']=meta
    destination=out/('adapted.ckpt' if mgpt else 'adapted.pth')
    torch.save(source,destination)
    meta['checkpoint']=str(destination); meta['checkpoint_sha256']=pilot.sha256_file(destination)
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    log.info('ADAPTATION_COMPLETE %s',destination)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['satopp','motiongpt','motiongpt_satopp'],required=True)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--train-pairs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--microbatch',type=int,default=4)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--learning-rate',type=float,default=1e-6)
    p.add_argument('--warmup-steps',type=int,default=0)
    a=p.parse_args()
    if a.steps<1 or a.microbatch<1: p.error('Steps and microbatch must be positive')
    if not np.isfinite(a.learning_rate) or a.learning_rate<=0 or not 0<=a.warmup_steps<=a.steps:
        p.error('Learning rate must be finite and positive; warmup must fit the step budget')
    run(a)

if __name__=='__main__': main()