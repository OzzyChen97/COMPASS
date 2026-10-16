import json
import pytest
from tools import humanml_independent_assets as assets


def test_training_pool_excludes_validation_test_and_mirrors(tmp_path,monkeypatch):
    data=tmp_path/'data';data.mkdir()
    (data/'train.txt').write_text('001\nM002\n003\n')
    (data/'val.txt').write_text('002\n')
    (data/'test.txt').write_text('003\n')
    rows=[dict(motion_id=mid,relation='preserve',text_orig='walk',text_pair='walking') for mid in ['001','M002','003']]
    rows.append(dict(motion_id='001',relation='alter',text_orig='walk',text_pair='run'))
    (data/'preserving_pairs_filtered.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(assets,'DATA',data)
    path=assets.prepare_train(tmp_path/'output')
    kept=[json.loads(l) for l in path.read_text().splitlines()]
    assert len(kept)==1 and kept[0]['motion_id']=='001'
    with pytest.raises(FileExistsError):assets.prepare_train(tmp_path/'output')