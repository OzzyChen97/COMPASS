"""Native MotionGPT SATO++ objective with an opt-in CLIP-conditioned student."""
from types import SimpleNamespace

import torch
from torch import nn


class FrozenTeacher:
    """Check exact action targets before native logit distillation.

    Source text tokenizers differ, but the action target tokenizer must not.
    This object is deliberately not a registered student submodule.
    """
    def __init__(self, lm):
        self.lm = lm.requires_grad_(False).eval()
        self.expected_labels = None

    def sato_forward_encdec(self, *args, **kwargs):
        result = self.lm.sato_forward_encdec(*args, **kwargs)
        if self.expected_labels is None or not torch.equal(result['labels'], self.expected_labels):
            raise RuntimeError('Teacher/student action targets differ; distillation refused')
        return result


class CLIPMotionGPTSATOObjective(nn.Module):
    def __init__(self, branch, native_teacher, source, cfg):
        super().__init__()
        from mGPT.models.mgpt_sato_pp import AttributionPooling, PredictorHead
        self.lm = branch
        self.hparams = SimpleNamespace(cfg=cfg)
        dim = branch.language_model.config.d_model
        self.attr_pooling = AttributionPooling(dim, cfg.SATO.PROJ_DIM, cfg.SATO.PROJ_TYPE)
        self.predictor_head = PredictorHead(cfg.SATO.PROJ_DIM)
        for prefix, module in [('attr_pooling', self.attr_pooling), ('predictor_head', self.predictor_head)]:
            module.load_state_dict({k[len(prefix)+1:]: v for k, v in source['state_dict'].items()
                                    if k.startswith(prefix + '.')}, strict=True)
            module.requires_grad_(False).eval()
        object.__setattr__(self, '_sato_teacher', FrozenTeacher(native_teacher))
        self.metrics = {}

    def train(self, mode=True):
        super().train(mode)
        self.attr_pooling.eval()
        self.predictor_head.eval()
        return self

    def log(self, name, value, **kwargs):
        self.metrics[name] = value.detach()

    def _prompt_edit_mask(self, *args):
        from mGPT.models.mgpt_sato_pp import MotionGPTSATOPP
        return MotionGPTSATOPP._prompt_edit_mask(self, *args)

    @staticmethod
    def _pad_prompt_values(*args):
        from mGPT.models.mgpt_sato_pp import MotionGPTSATOPP
        return MotionGPTSATOPP._pad_prompt_values(*args)

    def forward(self, batch):
        from mGPT.models.mgpt_sato_pp import MotionGPTSATOPP
        if any(relation != 'preserve' for relation in batch['relation']):
            raise ValueError('Shared CLIP experiment accepts preserve-only training pairs')
        # Intercept the already computed student labels without an extra model
        # forward or a second random template selection.
        original_forward = self.lm.sato_forward_encdec
        def checked_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            expected = self._sato_teacher.expected_labels
            if expected is not None and not torch.equal(expected, result['labels']):
                raise RuntimeError('Preserve pair has different action targets')
            self._sato_teacher.expected_labels = result['labels']
            return result
        self.lm.sato_forward_encdec = checked_forward
        try:
            self.metrics = {}
            return MotionGPTSATOPP.train_lm_forward(self, batch)['outputs'].loss
        finally:
            del self.lm.sato_forward_encdec
            self._sato_teacher.expected_labels = None