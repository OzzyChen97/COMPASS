"""HumanML3D-specific native model loading and train-only adaptation assets."""
import json
import os
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import train_motionx_text_adaptation_pilot as pilot
from ablation.repeat20_perturb_fast_batch import build_baseline_args
DATA=ROOT/'dataset/HumanML3D'
ASSETS=ROOT/'MotionGPT/configs/assets_humanml3d_official_local.yaml'
MODELS=('t2mgpt','satopp','motiongpt','motiongpt_satopp')


def human_sources():
    cfg=ROOT/'ablation/humanml3d_table1_levels/motiongpt_sources/p1/configs/motiongpt_humanml3d_seed123.yaml'
    spec={
        't2mgpt':dict(checkpoint=str(ROOT/'pretrained/VQTransformer_corruption05/net_best_fid.pth')),
        'satopp':dict(checkpoint=str(ROOT/'output_GPT_Final/sato_pp_allloss_gpu0_teachl1_05_eval5k_resume104k_200k_single/sato_pp_iter_130000.pth')),
        'motiongpt':dict(checkpoint=str(ROOT/'MotionGPT/experiments/mgpt/Instruct_H3D_OFFICIAL_FRESH_SEED123_20260818/checkpoints/last.ckpt'),config=str(cfg)),
        'motiongpt_satopp':dict(checkpoint=str(ROOT/'MotionGPT/experiments/mgpt_sato_pp/SATO_PP_H3D_MOTIONX_RECIPE_NOEARLY_100K_SEED123_20260906/checkpoints/best-validation.ckpt'),config=str(ROOT/'MotionGPT/configs/config_h3d_sato_pp.yaml'))}
    for row in spec.values():row['sha256']=pilot.sha256_file(Path(row['checkpoint']))
    return spec


def model_args():
    return build_baseline_args(dict(resume_pth='pretrained/VQVAE/net_best_fid.pth',
                                   resume_trans='pretrained/VQTransformer_corruption05/net_best_fid.pth'))


def load_native(config,checkpoint,device,with_vae=False):
    sys.path.insert(0,str(ROOT/'MotionGPT'))
    from MotionGPT.eval_loader_robustness import load_motiongpt_config
    from mGPT.config import instantiate_from_config
    old=Path.cwd()
    try:
        os.chdir(ROOT/'MotionGPT')
        cfg=load_motiongpt_config(Path(config),ASSETS)
        lm=instantiate_from_config(cfg.lm.default)
        vae=instantiate_from_config(cfg.vq.default) if with_vae else None
    finally:os.chdir(old)
    source=torch.load(checkpoint,map_location='cpu',weights_only=False)
    lm.load_state_dict({k[3:]:v for k,v in source['state_dict'].items() if k.startswith('lm.')},strict=True)
    lm.float().to(device).eval().requires_grad_(False)
    if vae is not None:
        vae.load_state_dict({k[4:]:v for k,v in source['state_dict'].items() if k.startswith('vae.')},strict=True)
        vae.float().to(device).eval().requires_grad_(False)
    return lm,vae,source,cfg


def prepare_train(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    path=DATA/'preserving_pairs_filtered.jsonl'
    train=pilot.load_split_ids(DATA,'train');forbidden=pilot.load_split_ids(DATA,'val')|pilot.load_split_ids(DATA,'test')
    canon=lambda x:x[1:] if x.startswith('M') else x
    forbidden_base={canon(x) for x in forbidden}
    kept=[];removed=0
    for row in pilot.read_jsonl(path):
        mid=str(row['motion_id'])
        if row.get('relation')!='preserve' or mid not in train or mid in forbidden or canon(mid) in forbidden_base:
            removed+=1;continue
        kept.append(row)
    assert kept
    target=output/'train_preserve.jsonl'
    with target.open('x') as f:
        for row in kept:f.write(json.dumps(row)+'\n')
    report=dict(source=str(path),source_sha256=pilot.sha256_file(path),accepted=len(kept),filtered=removed,
                split='train',mirror_overlap_excluded=True,output_sha256=pilot.sha256_file(target))
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return target