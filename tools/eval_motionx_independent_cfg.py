"""Motion-X independent CLIP/T5 evaluation with real CFG=1.3 for both families."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.motiongpt_cfg import install_motiongpt_cfg
from tools import eval_motionx_four_model_text_adapted as evaluation
from tools.eval_motiongpt_text_adaptation_validation import load_inference
from tools.audit_motionx_random_retrieval import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['t2mgpt','satopp','motiongpt','motiongpt_satopp'],required=True)
    p.add_argument('--level',choices=['P1','P2','P3'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--seed',type=int,default=123)
    a=p.parse_args(); loaded=[]
    if a.model.startswith('motiongpt') and a.config is None:p.error('Native T5 config required')
    def load(config,checkpoint,device):
        model,data=load_inference(config,checkpoint,device)
        install_motiongpt_cfg(model.lm,1.3);loaded.append(model)
        return model,data
    evaluation.load_mgpt=load
    evaluation.run(a)
    path=a.output/'manifest.json';manifest=json.loads(path.read_text())
    manifest.update(cfg_scale=1.3,independent_text_encoder=True,
                    text_encoder='T5' if loaded else 'CLIP',
                    cfg_script_sha256=digest(ROOT/'models/motiongpt_cfg.py'),
                    entry_script_sha256=digest(__file__))
    if loaded:
        assert loaded[0].lm.cfg_audit['calls']>0
        manifest['cfg']=loaded[0].lm.cfg_audit
    else:
        assert manifest['t2m_cfg_scale']==1.3
        manifest['cfg']={'scale':1.3,'unconditional_features':'zero CLIP features; existing T2M implementation'}
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('INDEPENDENT_CFG_TEST_COMPLETE',a.model,a.level,flush=True)


if __name__=='__main__':main()