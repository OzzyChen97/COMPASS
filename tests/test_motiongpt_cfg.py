import pytest
import torch
from types import SimpleNamespace
from unittest.mock import Mock
from transformers import T5Config, T5ForConditionalGeneration
from models.motiongpt_cfg import T5CaptionCFG, install_motiongpt_cfg


def tiny():
    torch.manual_seed(123)
    cfg = T5Config(vocab_size=16,d_model=8,d_ff=16,num_layers=1,num_decoder_layers=1,
                   num_heads=2,d_kv=4,dropout_rate=0.,decoder_start_token_id=0,eos_token_id=1,pad_token_id=0)
    return T5ForConditionalGeneration(cfg).eval()


def test_guidance_formula_and_same_decoder_prefix():
    model=tiny(); ids=torch.tensor([[5,1],[5,1]]); mask=torch.ones_like(ids)
    prefix=torch.tensor([[0,2,3],[0,4,2]])
    scores=torch.randn(2,16)
    with torch.no_grad():
        processor=T5CaptionCFG(model,ids,mask,1.3)
        unconditional=model(encoder_outputs=processor.encoded,attention_mask=mask,
                            decoder_input_ids=prefix,use_cache=False).logits[:,-1]
        guided=processor(prefix,scores)
    torch.testing.assert_close(guided,unconditional+1.3*(scores-unconditional))
    assert not torch.equal(guided,scores)
    assert processor.calls==1
    processor.scale=1.
    assert processor(prefix,scores) is scores
    processor.scale=0.
    torch.testing.assert_close(processor(prefix,scores),unconditional)


def test_cfg_one_delegates_exact_original_generation():
    original=Mock(return_value='native')
    lm=SimpleNamespace(lm_type='encdec',generate_direct=original)
    install_motiongpt_cfg(lm,1.)
    assert lm.generate_direct(['Generate motion: walk'],max_length=17,do_sample=False)=='native'
    assert original.call_count==1
    assert original.call_args.kwargs['max_length']==17
    assert lm.cfg_audit['calls']==0


def test_invalid_guidance_and_double_install_rejected():
    for scale in [-1.,float('nan'),float('inf')]:
        with pytest.raises(ValueError):install_motiongpt_cfg(SimpleNamespace(lm_type='encdec'),scale)
    lm=SimpleNamespace(lm_type='encdec',generate_direct=Mock())
    install_motiongpt_cfg(lm)
    with pytest.raises(ValueError):install_motiongpt_cfg(lm)