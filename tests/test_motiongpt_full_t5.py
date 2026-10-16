import math
import torch
from tools.train_motiongpt_full_t5 import epoch_indices,lr_at,synchronize_gradients,make_batch
from tools.train_motionx_text_adaptation_pilot import PairExample
import numpy as np


def test_complete_distributed_epoch_without_dropped_rows():
    for size in [31,32,33,201335]:
        for world in [2,5]:
            ranks=[epoch_indices(size,world,r,0) for r in range(world)]
            all_ids=[i for shard in ranks for i in shard]
            assert set(all_ids)==set(range(size))
            assert len(set(map(len,ranks)))==1
            assert len(all_ids)-size < world
            assert len(all_ids)==math.ceil(size/world)*world


def test_epoch_shuffling_is_reproducible_and_changes_each_epoch():
    assert epoch_indices(100,2,0,1)==epoch_indices(100,2,0,1)
    assert epoch_indices(100,2,0,1)!=epoch_indices(100,2,0,0)


def test_warmup_and_cosine_schedule():
    assert math.isclose(lr_at(1,1000,100,1e-5),1e-7,rel_tol=1e-12)
    assert lr_at(100,1000,100,1e-5)==1e-5
    assert abs(lr_at(1000,1000,100,1e-5)-1e-6)<1e-15
    assert lr_at(500,1000,100,1e-5)<lr_at(200,1000,100,1e-5)


def test_single_rank_gradient_sync_preserves_gradients():
    p=torch.nn.Parameter(torch.ones(4));p.sum().backward()
    synchronize_gradients([p],1)
    assert torch.equal(p.grad,torch.ones(4))


def test_full_batch_keeps_native_motion_targets_and_pair_identity():
    e=[PairExample(i,str(i),f'm{i}','original '+str(i),'rewrite '+str(i),np.array([list(range(i+2))]),'unused') for i in range(2)]
    batch=make_batch(e,[1,0],torch.device('cpu'))
    assert batch['length']==[3,2]
    assert batch['text']==['original 1','original 0']
    assert batch['text_pair']==['rewrite 1','rewrite 0']
    assert batch['relation']==['preserve','preserve']
    assert batch['motion'].tolist()==[[0,1,2],[0,1,0]]
    assert batch['tasks'][0]['output']==['<Motion_Placeholder>']