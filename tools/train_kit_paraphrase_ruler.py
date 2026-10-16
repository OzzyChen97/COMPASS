"""KIT text-only retrieval-ruler adaptation with official-validation acceptance.

Frozen standard motion embeddings are cached from KIT 251D motions. No test
captions or test motions are read by this training program.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import train_motionx_text_adaptation_pilot as pilot
from tools.audit_motionx_random_retrieval import GROUP_SEEDS,digest,retrieval
from utils.sato_pp_eval import TM2TTextProcessor
from tools.audit_motionx_text_coverage import LookupVectorizer
from tools.train_paraphrase_aware_text_encoder import contrastive_losses

LEVELS=('clean','P1','P2','P3')


def acceptance(baseline,current):
    macro=lambda d:sum(d[q]['R1'] for q in LEVELS[1:])/3
    return (current['clean']['R1']>=baseline['clean']['R1']-.01 and
            macro(current)>=macro(baseline)+.005 and current['P3']['R1']>=baseline['P3']['R1'])


def encode_grad(model,values,indices,device):
    word,pos,lens=values
    ix=torch.as_tensor(indices,dtype=torch.long)
    lengths=lens[ix];order=torch.argsort(lengths,descending=True,stable=True)
    out=model(word[ix][order].to(device),pos[ix][order].to(device),lengths[order].to(device))
    inverse=torch.argsort(order).to(device)
    return out[inverse]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-pairs',type=Path,required=True);p.add_argument('--val-pairs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--steps-per-epoch',type=int,default=200);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);pilot.seed_all(123);device=torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32=False
    from dataset.config import configure_evaluator_source,evaluator_opt_path,dataset_mean_std_paths
    from models.evaluator_wrapper import EvaluatorModelWrapper
    from options.get_eval_option import get_opt
    from utils.word_vectorizer import WordVectorizer
    from MotionGPT.eval_loader_robustness import get_motion_embeddings_in_input_order
    data=ROOT/'dataset/KIT-ML';train=pilot.load_split_ids(data,'train');val=pilot.load_split_ids(data,'val');test=pilot.load_split_ids(data,'test')
    canonical=lambda mid:mid[1:] if mid.startswith('M') else mid
    assert not {canonical(x) for x in train}&{canonical(x) for x in val|test}
    assert not {canonical(x) for x in val}&{canonical(x) for x in test}
    vm=json.loads((a.val_pairs.parent/'manifest.json').read_text())
    assert vm['complete'] and not vm['smoke'] and vm['output_sha256']==digest(a.val_pairs)
    validation=json.loads(a.val_pairs.read_text());assert len(validation)==864
    val_map={q:{} for q in LEVELS[1:]}
    for row in validation:
        assert row['motion_id'] in val and row['motion_id'] not in train|test and row['relation']=='preserve'
        assert row['motion_id'] not in val_map[row['level']]
        val_map[row['level']][row['motion_id']]=row
    val_ids=sorted(val_map['P1']);assert len(val_ids)==288
    assert all(sorted(v)==val_ids for v in val_map.values())
    assert all(len({val_map[q][mid]['text_orig'] for q in LEVELS[1:]})==1 for mid in val_ids)
    for mid in val_ids:
        assert val_map['P1'][mid]['text_orig'] in pilot.full_motion_captions(data/'texts'/f'{mid}.txt'), mid
    train_rows=pilot.read_jsonl(a.train_pairs);by_motion={};skips={};caps={}
    for row in train_rows:
        mid=row['motion_id'];assert mid in train and mid not in val|test
        if row.get('relation')!='preserve' or row.get('level') not in LEVELS[1:]:continue
        if mid not in caps:
            raw_path=data/'new_joint_vecs'/f'{mid}.npy'
            if not raw_path.exists():skips[mid]='missing training motion';caps[mid]=set();continue
            raw=np.load(raw_path,mmap_mode='r')
            if not 24<=len(raw)<200:skips[mid]='native training length exclusion';caps[mid]=set();continue
            caps[mid]=pilot.full_motion_captions(data/'texts'/f'{mid}.txt')
        if row['text_orig'] not in caps[mid]:continue
        groups=by_motion.setdefault(mid,{q:[] for q in LEVELS})
        groups['clean'].append(row['text_orig']);groups[row['level']].append(row['text_pair'])
    train_ids=sorted(mid for mid,g in by_motion.items() if all(g[q] for q in LEVELS))
    assert len(train_ids)>=32
    if a.smoke:train_ids=train_ids[:32];val_ids=val_ids[:32]
    ruler=ROOT/'checkpoints/kit/text_mot_match/model/finest.tar'
    opt=configure_evaluator_source(get_opt(evaluator_opt_path('kit'),device),'kit')
    opt.evaluator_matcher_checkpoint=str(ruler);wrapper=EvaluatorModelWrapper(opt)
    wrapper.movement_encoder.requires_grad_(False).eval();wrapper.motion_encoder.requires_grad_(False).eval()
    wrapper.text_encoder.requires_grad_(True).eval()
    frozen_before={k:pilot.named_parameter_sha256(getattr(wrapper,k).named_parameters()) for k in ['movement_encoder','motion_encoder']}
    mean_path,std_path=dataset_mean_std_paths('kit');mean=np.load(mean_path);std=np.load(std_path)
    motion_cache={}
    with torch.no_grad():
        all_ids=train_ids+val_ids
        for start in range(0,len(all_ids),32):
            batch=all_ids[start:start+32];motions=[];lengths=[]
            for mid in batch:
                raw=np.load(data/'new_joint_vecs'/f'{mid}.npy').astype(np.float32)
                assert raw.shape[1]==251 and 24<=len(raw)<200
                length=len(raw)//4*4;begin=(len(raw)-length)//2
                motion=(raw[begin:begin+length]-mean)/std
                padded=np.zeros((196,251),np.float32);padded[:length]=motion
                motions.append(padded);lengths.append(length)
            emb=get_motion_embeddings_in_input_order(wrapper,torch.from_numpy(np.stack(motions)).to(device),torch.tensor(lengths))
            for mid,value in zip(batch,emb.cpu()):motion_cache[mid]=value
    print('KIT_RULER_MOTION_CACHE_READY',len(train_ids),len(val_ids),flush=True)
    vec=LookupVectorizer(WordVectorizer(str(ROOT/'glove'),'our_vab'),'fallback_lower')
    processor=TM2TTextProcessor(vec,max_text_len=20,alphabetic_only=True)
    training_texts=sorted({text for mid in train_ids for q in LEVELS for text in by_motion[mid][q]})
    index={text:i for i,text in enumerate(training_texts)}
    training_vectors=processor.vectorize(training_texts)
    validation_vectors={q:processor.vectorize([val_map['P1' if q=='clean' else q][mid]['text_orig' if q=='clean' else 'text_pair'] for mid in val_ids]) for q in LEVELS}
    val_motion=torch.stack([motion_cache[mid] for mid in val_ids]).to(device)
    @torch.no_grad()
    def evaluate():
        wrapper.text_encoder.eval();result={}
        for q in LEVELS:
            text=torch.cat([encode_grad(wrapper.text_encoder,validation_vectors[q],list(range(i,min(i+32,len(val_ids)))),device) for i in range(0,len(val_ids),32)])
            scores=[retrieval(text,val_motion,np.random.default_rng(seed).permutation(len(val_ids))) for seed in GROUP_SEEDS]
            avg=np.mean(scores,axis=0);result[q]={f'R{k+1}':float(avg[k]) for k in range(3)}
        return result
    baseline=evaluate();baseline_macro=sum(baseline[q]['R1'] for q in LEVELS[1:])/3
    protocol=dict(dataset='kit',motion_dim=251,train_motion_count=len(train_ids),validation_motion_count=len(val_ids),
        train_ids_sha256=pilot.canonical_sha256(train_ids),val_ids_sha256=pilot.canonical_sha256(val_ids),
        train_pairs_sha256=digest(a.train_pairs),val_pairs_sha256=digest(a.val_pairs),standard_ruler_sha256=digest(ruler),
        epochs=a.epochs,steps_per_epoch=a.steps_per_epoch,lr=1e-4,batch_size=32,seed=123,smoke=a.smoke,
        loss='margin-10 paired contrastive, rolled negatives; 8 unique motions x clean/P1/P2/P3',
        frozen=frozen_before,group_seeds=GROUP_SEEDS,lookup='fallback_lower',alphabetic_only=True,
        gate='clean R1 drop <=1pp; macro perturbation R1 gain >=0.5pp; P3 R1 nonregression',
        training_exclusions=skips,test_content_read=False,baseline=baseline)
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('KIT_RULER_BASELINE',json.dumps(baseline),flush=True)
    optimizer=torch.optim.Adam(wrapper.text_encoder.parameters(),lr=1e-4)
    rng=np.random.default_rng(123);history=[];best=None;best_score=-float('inf')
    for epoch in range(1,a.epochs+1):
        wrapper.text_encoder.train();losses=[]
        for step in range(a.steps_per_epoch):
            chosen=[train_ids[int(i)] for i in rng.choice(len(train_ids),8,replace=False)]
            texts=[];motions=[]
            for q in LEVELS:
                for mid in chosen:
                    choices=by_motion[mid][q];texts.append(index[choices[int(rng.integers(len(choices)))]]);motions.append(motion_cache[mid])
            text=encode_grad(wrapper.text_encoder,training_vectors,texts,device)
            motion=torch.stack(motions).to(device)
            loss=contrastive_losses(text,motion,10.,shift=1)[0]
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward()
            assert all(p.grad is None for module in [wrapper.movement_encoder,wrapper.motion_encoder] for p in module.parameters())
            torch.nn.utils.clip_grad_norm_(wrapper.text_encoder.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        metrics=evaluate();qualified=acceptance(baseline,metrics);score=sum(metrics[q]['R1'] for q in LEVELS[1:])/3
        record=dict(epoch=epoch,loss=float(np.mean(losses)),metrics=metrics,qualified=qualified)
        history.append(record);(a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        checkpoint={'epoch':epoch,'text_encoder':wrapper.text_encoder.state_dict(),'motion_encoder':wrapper.motion_encoder.state_dict(),
                    'movement_encoder':wrapper.movement_encoder.state_dict()}
        torch.save(checkpoint,a.output/f'epoch_{epoch:02d}.tar')
        if a.smoke:
            reloaded=torch.load(a.output/f'epoch_{epoch:02d}.tar',map_location=device,weights_only=False)
            wrapper.text_encoder.load_state_dict(reloaded['text_encoder'],strict=True)
            assert evaluate()==metrics, 'Checkpoint reload changed validation scores'
            for q in LEVELS:
                direct=processor.encode(wrapper,[val_map['P1' if q=='clean' else q][mid]['text_orig' if q=='clean' else 'text_pair'] for mid in val_ids])
                with torch.no_grad():
                    cached=encode_grad(wrapper.text_encoder,validation_vectors[q],list(range(len(val_ids))),device)
                torch.testing.assert_close(direct,cached,rtol=1e-5,atol=1e-6)
        if qualified and score>best_score:best=copy.deepcopy(record);best_score=score
        print('KIT_RULER_EPOCH',json.dumps(record),flush=True)
    assert all(frozen_before[k]==pilot.named_parameter_sha256(getattr(wrapper,k).named_parameters()) for k in frozen_before)
    result=dict(accepted=best is not None,selected=best,baseline=baseline,baseline_macro_R1=baseline_macro,
                smoke=a.smoke,frozen_verified=True,validation_samples=len(val_ids),test_content_read=False)
    if best is not None and not a.smoke:
        import shutil
        target=a.output/'finest.tar';shutil.copy2(a.output/f"epoch_{best['epoch']:02d}.tar",target)
        result.update(checkpoint=str(target.resolve()),checkpoint_sha256=digest(target))
    (a.output/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    if a.smoke:print('KIT_RULER_SMOKE_COMPLETE',flush=True)
    elif best is None:raise RuntimeError('KIT ruler failed official-validation gate; formal tests are blocked')
    else:print('KIT_RULER_ACCEPTED',best['epoch'],flush=True)


if __name__=='__main__':main()