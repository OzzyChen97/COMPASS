import copy
import torch
from tools.train_kit_paraphrase_ruler import acceptance, encode_grad


def scores(clean=.4,p1=.2,p2=.3,p3=.2):
    return {key:{'R1':value} for key,value in zip(('clean','P1','P2','P3'),(clean,p1,p2,p3))}


def test_acceptance_requires_all_predeclared_conditions():
    base=scores()
    assert acceptance(base,scores(p1=.3,p2=.4,p3=.3))
    assert not acceptance(base,base)
    assert not acceptance(base,scores(clean=.38,p1=.3,p2=.4,p3=.3))
    assert not acceptance(base,scores(p1=.4,p2=.4,p3=.19))
    original=copy.deepcopy(base)
    acceptance(base,scores())
    assert base==original


def test_sorted_encoding_restores_input_order_and_gradients():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.scale=torch.nn.Parameter(torch.tensor(2.))
        def forward(self,word,pos,lengths):
            assert all(lengths[i]>=lengths[i+1] for i in range(len(lengths)-1))
            return word[:,0]*self.scale
    model=Encoder();word=torch.arange(12.).reshape(4,3,1)
    indices=[2,0,3,1];values=(word,torch.zeros_like(word),torch.tensor([1,3,2,3]))
    out=encode_grad(model,values,indices,torch.device('cpu'))
    torch.testing.assert_close(out,word[indices,0]*2)
    out.sum().backward();assert model.scale.grad.item()!=0


def test_rolled_negatives_never_pair_rewrites_of_same_motion():
    ids=torch.arange(8).repeat(4)
    assert not torch.any(ids==ids.roll(1))