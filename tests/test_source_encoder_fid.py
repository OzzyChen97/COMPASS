import json
import pytest
from tools.launch_source_encoder_fid import lock,verify


def test_lock_preserves_existing_protocol(tmp_path):
    path=tmp_path/'assets.json';lock(path,{'steps':0});lock(path,{'steps':0})
    with pytest.raises(AssertionError):lock(path,{'steps':3000})
    assert json.loads(path.read_text())=={'steps':0}


def test_verify_rejects_adapted_checkpoint_hash(tmp_path):
    output=tmp_path/'run';output.mkdir()
    (output/'summary.json').write_text(json.dumps(dict(samples=734,smoke=False,cfg_scale=1.3,checkpoint_sha256='adapted',pair_sha256='pairs')))
    from tools.audit_motionx_random_retrieval import GROUP_SEEDS
    (output/'run.json').write_text(json.dumps(dict(seed=123,group_seeds=GROUP_SEEDS)))
    assets=tmp_path/'assets.json';assets.write_text(json.dumps(dict(datasets={'kit':dict(samples=734,pair_sha256='pairs')})))
    with pytest.raises(AssertionError):
        verify(dict(output=str(output),assets=str(assets),dataset='kit',source={'sha256':'source'}))