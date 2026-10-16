"""KIT-native model interfaces for independent text adaptation and evaluation."""
import os
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ablation.repeat20_perturb_fast_batch import build_baseline_args


def model_args():
    return build_baseline_args(dict(dataname='kit',resume_pth='output_kit/VQVAE_KIT/net_best_fid.pth',
                                    resume_trans='output_kit/GPT_KIT/net_best_fid.pth',args={'down_t':2,'block_size':51}))


def load_native(config,checkpoint,device,with_vae=False):
    sys.path.insert(0,str(ROOT/'MotionGPT'))
    from MotionGPT.eval_loader_robustness import load_motiongpt_config
    from mGPT.config import instantiate_from_config
    old=Path.cwd()
    try:
        os.chdir(ROOT/'MotionGPT')
        cfg=load_motiongpt_config(Path(config),ROOT/'MotionGPT/configs/assets_kit_stage2.yaml')
        assert int(cfg.DATASET.NFEATS)==251
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


if __name__=='__main__':
    from tools import train_humanml_independent_text as training
    training.DATA=ROOT/'dataset/KIT-ML';training.MODELS=('t2mgpt','satopp','motiongpt')
    training.model_args=model_args;training.load_native=load_native
    training.main()