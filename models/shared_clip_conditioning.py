"""Opt-in shared CLIP text conditioning; existing model paths are unchanged.

Only the experiment owner registers/optimizes SharedCLIPText. MotionGPT branches
hold non-owning references, so saving a branch never duplicates shared weights.
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CLIPFeatures:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    input_embeddings: torch.Tensor
    token_features: torch.Tensor
    sentence_features: torch.Tensor
    truncated: tuple[bool, ...]


class SharedCLIPText(nn.Module):
    """Text-only view of an OpenAI CLIP model, retaining its exact forward math."""

    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.context_length = int(self.positional_embedding.shape[0])
        self.width = int(self.token_embedding.weight.shape[1])
        self.output_dim = int(self.text_projection.shape[1])

    @property
    def device(self):
        return self.token_embedding.weight.device

    def tokenize(self, texts):
        import clip
        from clip.simple_tokenizer import SimpleTokenizer

        # Lazy cache avoids rebuilding the BPE vocabulary on every minibatch.
        if not hasattr(self, '_bpe_tokenizer'):
            self._bpe_tokenizer = SimpleTokenizer()
        truncated = tuple(len(self._bpe_tokenizer.encode(text)) + 2 > self.context_length for text in texts)
        ids = clip.tokenize(texts, context_length=self.context_length, truncate=True).to(self.device)
        return ids, truncated

    def forward(self, input_ids, *, truncated=None, require_input_grad=False):
        if input_ids.ndim != 2 or input_ids.shape[1] != self.context_length:
            raise ValueError('CLIP inputs must have the full positional context width')
        # OpenAI CLIP EOT has the largest token ID. Positions through EOT are
        # valid; token ID 0 can also occur in content and is NOT a padding mask.
        eot = input_ids.argmax(dim=-1)
        mask = torch.arange(self.context_length, device=input_ids.device)[None, :] <= eot[:, None]
        embeddings = self.token_embedding(input_ids)
        if require_input_grad and not embeddings.requires_grad:
            embeddings = embeddings.detach().requires_grad_(True)
        x = embeddings + self.positional_embedding.to(embeddings.dtype)
        # SATO++ needs double backward, unsupported by fused SDP kernels.
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x)
        sentence = x[torch.arange(x.shape[0], device=x.device), eot] @ self.text_projection
        return CLIPFeatures(input_ids, mask, embeddings, x, sentence,
                            tuple(truncated) if truncated is not None else (False,) * x.shape[0])

    def encode_texts(self, texts, *, require_input_grad=False):
        ids, truncated = self.tokenize(texts)
        return self(ids, truncated=truncated, require_input_grad=require_input_grad)


class CLIPMotionGPT(nn.Module):
    """CLIP sequence -> independent adapter -> frozen native T5 decoder.

    `native_lm` is the existing MLM object. Its encoder is retained for checkpoint
    compatibility but never called by this branch. Text-to-motion only.
    """

    def __init__(self, native_lm, shared_clip):
        super().__init__()
        self.native_lm = native_lm.requires_grad_(False).eval()
        object.__setattr__(self, '_shared_clip', shared_clip)
        dim = int(native_lm.language_model.config.d_model)
        self.adapter = nn.Sequential(nn.LayerNorm(shared_clip.width),
                                     nn.Linear(shared_clip.width, dim), nn.GELU(), nn.Linear(dim, dim))

    @property
    def language_model(self):
        return self.native_lm.language_model

    @property
    def tokenizer(self):
        return self.native_lm.tokenizer

    def train(self, mode=True):
        super().train(mode)
        self.native_lm.eval()  # Freeze dropout as well as parameters.
        return self

    def encode_condition(self, texts, *, attribution=False):
        from transformers.modeling_outputs import BaseModelOutput

        features = self._shared_clip.encode_texts(texts, require_input_grad=attribution)
        hidden = self.adapter(features.token_features)
        return features, BaseModelOutput(last_hidden_state=hidden)

    def supervised(self, texts, targets, *, attribution=False):
        features, encoded = self.encode_condition(texts, attribution=attribution)
        target = self.tokenizer(targets, padding=True, truncation=False, return_tensors='pt')
        if target.input_ids.shape[1] > self.native_lm.max_length:
            raise ValueError('Native motion target exceeds maximum length; refusing truncation')
        ids = target.input_ids.to(features.input_ids.device)
        target_mask = target.attention_mask.to(ids.device)
        labels = ids.masked_fill(target_mask.eq(0), -100)
        outputs = self.language_model(encoder_outputs=encoded,
                                      attention_mask=features.attention_mask,
                                      labels=labels, decoder_attention_mask=target_mask,
                                      use_cache=False, return_dict=True)
        token_features = encoded.last_hidden_state
        text_feature = torch.nn.functional.normalize(
            (torch.nn.functional.normalize(token_features, dim=-1) * features.attention_mask[..., None]).sum(1)
            / features.attention_mask.sum(1, keepdim=True).clamp_min(1), dim=-1)
        return dict(outputs=outputs, input_ids=features.input_ids,
                    attention_mask=features.attention_mask, labels=labels,
                    input_embeddings=features.input_embeddings, token_features=token_features,
                    text_feature=text_feature, inputs=list(texts), targets=list(targets),
                    truncated=features.truncated)

    def template_fulfill(self, tasks, lengths, motion_strings, texts, stage='test', **kwargs):
        if stage != 'test' or any(motion_strings):
            raise NotImplementedError('Only raw-caption text-to-motion inference is supported')
        return list(texts), [''] * len(texts)

    @property
    def m_codebook_size(self):
        return self.native_lm.m_codebook_size

    @property
    def max_length(self):
        return self.native_lm.max_length

    def sample_template_choices(self, tasks):
        return self.native_lm.sample_template_choices(tasks)

    def sato_forward_encdec(self, texts, motion_tokens, lengths, tasks, template_choices=None):
        motion_strings = self.native_lm.motion_token_to_string(motion_tokens, lengths)
        inputs, targets = self.native_lm.template_fulfill(
            tasks, lengths, motion_strings, texts, template_choices=template_choices)
        # CLIP always sees raw descriptions, not T5-specific task prefixes.
        # Only supported text-to-motion templates may reach this wrapper.
        if any('<motion_id_' in text for text in inputs):
            raise ValueError('CLIP branch supports text-to-motion only')
        return self.supervised(texts, targets, attribution=torch.is_grad_enabled())

    @torch.no_grad()
    def generate_direct(self, texts, max_length=256, num_beams=1, do_sample=True,
                        bad_words_ids=None, temperature=None, top_k=None, top_p=None):
        features, encoded = self.encode_condition(texts)
        sampling = {k: v for k, v in dict(temperature=temperature, top_k=top_k, top_p=top_p).items()
                    if v is not None}
        ids = self.language_model.generate(
            encoder_outputs=encoded, attention_mask=features.attention_mask,
            max_length=max_length, num_beams=num_beams, do_sample=do_sample,
            bad_words_ids=bad_words_ids, **sampling)
        strings = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
        self.native_lm.device = ids.device
        return self.native_lm.motion_string_to_token(strings)

    def generate_conditional(self, texts=None, motion_tokens=None, lengths=None,
                             task='t2m', with_len=False, stage='train', tasks=None):
        if task != 't2m' or with_len or motion_tokens is not None or tasks is not None:
            raise NotImplementedError('Shared CLIP supports raw-text t2m without length or custom templates')
        if texts is None:
            raise ValueError('Text descriptions are required')
        tokens, _ = self.generate_direct(texts, max_length=128, num_beams=1, do_sample=True)
        return tokens
