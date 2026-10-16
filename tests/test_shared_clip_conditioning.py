"""CPU-only interface checks using tiny real CLIP/T5 architectures."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from models.shared_clip_conditioning import SharedCLIPText, CLIPMotionGPT


@pytest.fixture
def shared():
    from clip.model import CLIP
    torch.manual_seed(123)
    clip = CLIP(embed_dim=8, image_resolution=16, vision_layers=1, vision_width=64,
                vision_patch_size=8, context_length=6, vocab_size=32,
                transformer_width=8, transformer_heads=2, transformer_layers=1).float().eval()
    return clip, SharedCLIPText(clip)


def test_official_parity_mask_and_visual_exclusion(shared):
    official, text = shared
    ids = torch.tensor([[30, 0, 3, 31, 0, 0], [30, 2, 31, 0, 0, 0]])
    out = text(ids)
    # Exact parity under the same kernel; fused kernels can round differently.
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        expected = official.encode_text(ids)
    torch.testing.assert_close(out.sentence_features, expected, rtol=0, atol=0)
    torch.testing.assert_close(out.sentence_features, official.encode_text(ids), rtol=1e-5, atol=1e-6)
    assert out.attention_mask.tolist() == [[True]*4 + [False]*2, [True]*3 + [False]*3]
    assert not any('visual' in k for k in text.state_dict())
    with pytest.raises(ValueError):
        text(ids[:, :3])


class Tokenizer:
    def __call__(self, texts, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[3, 4, 1]] * len(texts)),
                               attention_mask=torch.ones(len(texts), 3, dtype=torch.long))

    def batch_decode(self, ids, **kwargs):
        return [str(row.tolist()) for row in ids]


class Native(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import T5Config, T5ForConditionalGeneration
        cfg = T5Config(vocab_size=32, d_model=8, d_ff=16, num_layers=1,
                       num_decoder_layers=1, num_heads=2, d_kv=4, dropout_rate=0.0,
                       decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
        self.language_model = T5ForConditionalGeneration(cfg)
        self.tokenizer = Tokenizer()
        self.max_length = 16

    def motion_string_to_token(self, texts):
        return texts, texts


def test_decoder_bypasses_encoder_and_reaches_shared_clip(shared):
    _, text = shared
    native = Native()
    model = CLIPMotionGPT(native, text).train()
    assert not native.training
    ids = torch.tensor([[30, 2, 31, 0, 0, 0]])
    before = {k: v.clone() for k, v in native.state_dict().items()}
    with patch.object(text, 'tokenize', return_value=(ids, (False,))), \
         patch.object(native.language_model.encoder, 'forward', side_effect=AssertionError('T5 encoder called')):
        out = model.supervised(['a walk'], ['motion'], attribution=True)
        grad, = torch.autograd.grad(out['outputs'].loss, out['input_embeddings'], create_graph=True, retain_graph=True)
        loss = out['outputs'].loss + (grad * out['input_embeddings']).square().sum()
        loss.backward()
        assert text.token_embedding.weight.grad.abs().sum() > 0
        assert model.adapter[1].weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in native.parameters())
        model.generate_direct(['a walk'], max_length=4, do_sample=False)
    for k, v in native.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert not any('_shared_clip' in k for k in model.state_dict())


def test_adapter_warmup_keeps_clip_frozen_and_reload(shared):
    _, text = shared
    text.requires_grad_(False)
    model = CLIPMotionGPT(Native(), text)
    ids = torch.tensor([[30, 2, 31, 0, 0, 0]])
    with patch.object(text, 'tokenize', return_value=(ids, (False,))):
        result = model.supervised(['walk'], ['motion'])
        result['outputs'].loss.backward()
        assert all(p.grad is None for p in text.parameters())
        assert model.adapter[1].weight.grad.abs().sum() > 0
        clone = CLIPMotionGPT(Native(), text)
        clone.load_state_dict(model.state_dict(), strict=True)
        actual = clone.supervised(['walk'], ['motion'])
        torch.testing.assert_close(result['outputs'].logits, actual['outputs'].logits)


def test_teacher_rejects_mismatched_action_targets():
    from models.shared_clip_sato import FrozenTeacher
    native = Native()
    native.sato_forward_encdec = lambda: {'labels': torch.tensor([[3, 4, 1]])}
    teacher = FrozenTeacher(native)
    with pytest.raises(RuntimeError, match='action targets differ'):
        teacher.sato_forward_encdec()
    teacher.expected_labels = torch.tensor([[3, 4, 1]])
    teacher.sato_forward_encdec()
    teacher.expected_labels = torch.tensor([[4, 3, 1]])
    with pytest.raises(RuntimeError, match='action targets differ'):
        teacher.sato_forward_encdec()


def test_conditional_generation_rejects_unsupported_tasks(shared):
    model = CLIPMotionGPT(Native(), shared[1])
    with pytest.raises(NotImplementedError):
        model.generate_conditional(['walk'], task='m2t')
    with pytest.raises(NotImplementedError):
        model.generate_conditional(['walk'], with_len=True)
    with patch.object(model, 'generate_direct', return_value=(['tokens'], ['text'])) as generate:
        assert model.generate_conditional(['walk']) == ['tokens']
        assert generate.call_args.args[0] == ['walk']