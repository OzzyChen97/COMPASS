import unittest

import torch
from torch import nn

from models.clip_attribution import compute_grad_input_attribution
from train_sato_pp_ddp import compute_grad_input_attribution_batched


class TinyClip(nn.Module):
    def __init__(self, vocab_size=16, width=4, context_length=5):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.randn(context_length, width))
        self.transformer = nn.Identity()
        self.ln_final = nn.Identity()
        self.text_projection = nn.Parameter(torch.eye(width))

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    def forward(self, text_tokens, mode='attribution'):
        if mode != 'attribution':
            raise ValueError(mode)
        h = self.token_embedding(text_tokens).type(self.dtype)
        h = h.detach().requires_grad_(True)
        x = self.ln_final(h + self.positional_embedding.type(self.dtype))
        eos_idx = text_tokens.argmax(dim=-1)
        pooled = x[torch.arange(x.shape[0]), eos_idx]
        return h, pooled @ self.text_projection


class TinyTransformer(nn.Module):
    def __init__(self, width=4, vocab_size=5):
        super().__init__()
        self.proj = nn.Linear(width, vocab_size, bias=False)

    def forward(self, input_index, text_feature):
        logits = self.proj(text_feature)
        return logits.unsqueeze(1).expand(-1, input_index.shape[1], -1)


class AttributionGradientTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.clip = TinyClip()
        self.trans = TinyTransformer()
        self.text = torch.tensor([[1, 2, 3, 15, 0]], dtype=torch.long)
        self.motion = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        self.valid_len = torch.tensor([3], dtype=torch.long)

    def test_detached_attribution_has_no_backbone_graph(self):
        attribution = compute_grad_input_attribution(
            self.clip,
            self.trans,
            self.text,
            self.motion,
            K=3,
            valid_len=self.valid_len,
            create_graph=False,
        )
        self.assertFalse(attribution.requires_grad)

    def test_full_attribution_updates_backbone(self):
        attribution = compute_grad_input_attribution(
            self.clip,
            self.trans,
            self.text,
            self.motion,
            K=3,
            valid_len=self.valid_len,
            create_graph=True,
        )
        self.assertTrue(attribution.requires_grad)

        attribution.square().mean().backward()
        grad = self.trans.proj.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_batched_full_attribution_updates_backbone(self):
        attribution = compute_grad_input_attribution_batched(
            self.clip,
            self.trans,
            self.text,
            self.motion,
            K=3,
            valid_len=self.valid_len,
            create_graph=True,
        )
        attribution.square().mean().backward()
        grad = self.trans.proj.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
