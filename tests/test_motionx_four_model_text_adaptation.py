import torch
from torch import nn
from tools.train_motionx_four_model_text_adaptation import freeze_t5_text_only, sequence_ce


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model=nn.Module()
        m=self.language_model
        m.shared=nn.Embedding(9,4)
        m.encoder=nn.Module()
        m.encoder.embed_tokens=m.shared
        m.encoder.block=nn.Linear(4,4)
        m.decoder=nn.Module()
        m.decoder.embed_tokens=m.shared
        m.decoder.block=nn.Linear(4,4)
        m.lm_head=nn.Linear(4,9,bias=False)
        m.lm_head.weight=m.shared.weight


def test_t5_freeze_preserves_shared_decoder_and_output_weights():
    lm=TinyLM()
    names=freeze_t5_text_only(lm)
    assert names==['encoder.block.weight','encoder.block.bias']
    m=lm.language_model
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01)
    features=m.encoder.block(m.shared(torch.tensor([[1,2]])))
    logits=m.lm_head(m.decoder.block(features))
    sequence_ce(logits,torch.tensor([[3,4]])).backward()
    assert m.encoder.block.weight.grad.abs().sum()>0
    assert m.shared.weight.grad is None and m.decoder.block.weight.grad is None
    opt.step()
    for n,p in m.named_parameters():
        if n not in names: assert torch.equal(p,before[n])
    assert not torch.equal(m.encoder.block.weight,before['encoder.block.weight'])


def test_sequence_loss_ignores_padding_and_weights_sequences_equally():
    torch.manual_seed(1)
    logits=torch.randn(2,3,5)
    labels=torch.tensor([[1,-100,-100],[1,2,3]])
    expected=(torch.nn.functional.cross_entropy(logits[0,:1],labels[0,:1])+
              torch.nn.functional.cross_entropy(logits[1],labels[1]))/2
    assert torch.allclose(sequence_ce(logits,labels),expected)


def test_microbatch_loss_matches_full_batch():
    torch.manual_seed(2)
    logits=torch.randn(6,4,7,requires_grad=True)
    labels=torch.randint(0,7,(6,4)); labels[0,2:]=-100
    full=sequence_ce(logits,labels)
    split=sum(sequence_ce(logits[i:i+2],labels[i:i+2])/3 for i in range(0,6,2))
    assert torch.allclose(full,split)
    g1=torch.autograd.grad(full,logits,retain_graph=True)[0]
    g2=torch.autograd.grad(split,logits)[0]
    assert torch.allclose(g1,g2,atol=1e-7)


def test_native_filter_only_removes_unsupported_short_training_motions(tmp_path):
    import json
    import numpy as np
    from tools.train_motionx_four_model_text_adaptation import filter_native_short_pairs
    for split,ids in [('train','short\nvalid\n'),('val','heldout\n'),('test','test\n')]:
        (tmp_path/f'{split}.txt').write_text(ids)
    tokens=tmp_path/'tokens'; tokens.mkdir()
    motions=tmp_path/'new_joint_vecs'; motions.mkdir()
    np.save(tokens/'valid.npy',np.array([1,2]))
    np.save(motions/'short.npy',np.zeros((2,322)))
    pairs=tmp_path/'pairs.jsonl'
    pairs.write_text('\n'.join(json.dumps({'motion_id':m}) for m in ['short','valid','short']))
    output=tmp_path/'filtered.jsonl'
    result=filter_native_short_pairs(pairs,tokens,tmp_path,output)
    assert result['native_short_pair_rows_removed']==2
    assert result['native_short_motions']=={'short':2}
    assert json.loads(output.read_text())['motion_id']=='valid'


def test_native_filter_rejects_missing_eligible_motion_and_split_leak(tmp_path):
    import json
    import numpy as np
    import pytest
    from tools.train_motionx_four_model_text_adaptation import filter_native_short_pairs
    from tools.train_motionx_text_adaptation_pilot import TrainingDataError
    for split,ids in [('train','valid\n'),('val','heldout\n'),('test','test\n')]:
        (tmp_path/f'{split}.txt').write_text(ids)
    tokens=tmp_path/'tokens';tokens.mkdir()
    motions=tmp_path/'new_joint_vecs';motions.mkdir()
    np.save(motions/'valid.npy',np.zeros((8,322)))
    pairs=tmp_path/'pairs.jsonl';pairs.write_text(json.dumps({'motion_id':'valid'}))
    with pytest.raises(FileNotFoundError):filter_native_short_pairs(pairs,tokens,tmp_path,tmp_path/'out.jsonl')
    pairs.write_text(json.dumps({'motion_id':'heldout'}))
    with pytest.raises(TrainingDataError):filter_native_short_pairs(pairs,tokens,tmp_path,tmp_path/'out.jsonl')