"""Opt-in T5 text-to-motion CFG; independent of shared-CLIP experiments.

The unconditional prompt retains the task prefix and removes the caption.
This is inference-time guidance, not a claim of unconditional training.
"""
import math
from types import MethodType

import torch
from transformers import LogitsProcessor, LogitsProcessorList


class T5CaptionCFG(LogitsProcessor):
    def __init__(self, language_model, unconditional_ids, attention_mask, scale):
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('CFG scale must be finite and nonnegative')
        self.model = language_model
        self.scale = float(scale)
        self.mask = attention_mask
        self.encoded = language_model.get_encoder()(input_ids=unconditional_ids,
                                                    attention_mask=attention_mask, return_dict=True)
        self.calls = 0

    @torch.no_grad()
    def __call__(self, input_ids, scores):
        if self.scale == 1.:
            return scores
        if input_ids.shape[0] != self.mask.shape[0]:
            raise ValueError('CFG requires one beam and one returned sequence per prompt')
        uncond = self.model(encoder_outputs=self.encoded, attention_mask=self.mask,
                            decoder_input_ids=input_ids, use_cache=False, return_dict=True).logits[:, -1]
        self.calls += 1
        return uncond + self.scale * (scores - uncond)


def install_motiongpt_cfg(lm, scale=1.3):
    """Patch only this loaded LM instance, preserving the original CFG=1 path."""
    if hasattr(lm, '_independent_cfg_original'):
        raise ValueError('CFG already installed on this LM')
    if lm.lm_type != 'encdec':
        raise ValueError('T5 encoder-decoder required')
    if not math.isfinite(scale) or scale < 0:
        raise ValueError('Invalid CFG scale')
    original = lm.generate_direct
    object.__setattr__(lm, '_independent_cfg_original', original)
    lm.cfg_audit = {'scale': float(scale), 'unconditional_prompt': 'Generate motion: ',
                    'mix': 'unconditional_logits + scale * (conditional_logits - unconditional_logits)',
                    'inference_extension': True, 'calls': 0}

    @torch.no_grad()
    def generate(self, texts, max_length=256, num_beams=1, do_sample=True,
                 bad_words_ids=None, temperature=None, top_k=None, top_p=None):
        if scale == 1.:
            return original(texts, max_length=max_length, num_beams=num_beams, do_sample=do_sample,
                            bad_words_ids=bad_words_ids, temperature=temperature, top_k=top_k, top_p=top_p)
        if num_beams != 1 or bad_words_ids:
            raise ValueError('Guidance is tested for single-beam generation without bad-word processing')
        self.device = self.language_model.device
        cond = self.tokenizer(texts, padding='max_length', max_length=self.max_length,
            truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors='pt').to(self.device)
        uncond = self.tokenizer(['Generate motion: ']*len(texts), padding=True,
            truncation=True, max_length=self.max_length, return_tensors='pt').to(self.device)
        processor = T5CaptionCFG(self.language_model, uncond.input_ids, uncond.attention_mask, scale)
        sampling = {k:v for k,v in {'temperature':temperature,'top_k':top_k,'top_p':top_p}.items() if v is not None}
        outputs = self.language_model.generate(input_ids=cond.input_ids, attention_mask=cond.attention_mask,
            max_length=max_length, num_beams=1, do_sample=do_sample,
            logits_processor=LogitsProcessorList([processor]), **sampling)
        self.cfg_audit['calls'] += processor.calls
        assert processor.calls > 0
        strings = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return self.motion_string_to_token(strings)

    lm.generate_direct = MethodType(generate, lm)
    return lm