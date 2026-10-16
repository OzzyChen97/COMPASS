import numpy as np
import pytest
from tools.prepare_unified_retest_assets import balanced_mix
from tools.eval_unified_mixed import distribution_metrics


def triple(n=11):
    return {q:[dict(motion_id=str(i),text_orig='walk',text_pair=q+str(i),level=q,relation='preserve') for i in range(n)] for q in ['P1','P2','P3']}


def test_mix_balanced_unique_and_independent_of_input_order():
    rows=triple();a=balanced_mix(rows)
    b=balanced_mix({k:list(reversed(v)) for k,v in rows.items()})
    assert a==b and len({r['motion_id'] for r in a})==11
    counts=[sum(r['level']==q for r in a) for q in rows]
    assert max(counts)-min(counts)<=1


def test_mix_rejects_incomplete_mismatched_or_duplicate_pairs():
    rows=triple();rows['P2'].pop()
    with pytest.raises(ValueError):balanced_mix(rows)
    rows=triple();rows['P3'][0]['text_orig']='run'
    with pytest.raises(ValueError):balanced_mix(rows)
    rows=triple();rows['P1'].append(rows['P1'][0])
    with pytest.raises(ValueError):balanced_mix(rows)


def test_fid_semantics_and_transfer_has_no_reference_fid():
    real=np.random.default_rng(123).normal(size=(40,4))
    values=distribution_metrics(real,real,real)
    assert abs(values['FID'])<1e-8 and abs(values['FID_P'])<1e-8 and abs(values['FID_D'])<1e-8
    assert values['paired_embedding_L2']==0
    transfer=distribution_metrics(None,real,real+1)
    assert 'FID' not in transfer and 'FID_P' not in transfer
    assert transfer['FID_D']==pytest.approx(4.)
    assert distribution_metrics(real,real,real)==values