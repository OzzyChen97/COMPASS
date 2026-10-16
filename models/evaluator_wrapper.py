
import torch
from os.path import join as pjoin
import numpy as np
from models.modules import MovementConvEncoder, TextEncoderBiGRUCo, MotionEncoderBiGRUCo
from utils.word_vectorizer import POS_enumerator
from dataset.config import dataset_max_motion_length, get_dataset_spec
from dataset.variable_motion import downsample_lengths

def build_models(opt):
    movement_input_dim = int(getattr(
        opt, 'motion_input_dim', opt.dim_pose - int(getattr(opt, 'motion_drop_tail', 4))))
    movement_enc = MovementConvEncoder(
        movement_input_dim, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
                                  pos_size=opt.dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)

    if getattr(opt, "evaluator_matcher_checkpoint", None):
        checkpoint_path = opt.evaluator_matcher_checkpoint
    else:
        checkpoint_path = pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar')
    checkpoint = torch.load(checkpoint_path, map_location=opt.device, weights_only=False)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!! from %s' % (checkpoint['epoch'], checkpoint_path))
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, opt):

        spec = get_dataset_spec(opt.dataset_name)
        opt.dim_pose = spec.dim_pose
        opt.motion_drop_tail = int(getattr(
            opt, 'motion_drop_tail', spec.evaluator_drop_tail))
        opt.motion_input_dim = int(getattr(
            opt, 'motion_input_dim', opt.dim_pose - opt.motion_drop_tail))

        opt.dim_word = 300
        opt.max_motion_length = dataset_max_motion_length(opt.dataset_name)
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = int(getattr(opt, 'max_text_len', 20))
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        # print(opt)

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    def _movement_lengths(self, motion_lengths):
        return downsample_lengths(
            motion_lengths,
            self.opt.unit_length,
            include_partial=get_dataset_spec(
                self.opt.dataset_name).variable_length,
        ).to(motion_lengths.device)

    def _movement_input(self, motions):
        drop_tail = int(self.opt.motion_drop_tail)
        return motions if drop_tail == 0 else motions[..., :-drop_tail]

    def _renormalize(self, motions):
        if not hasattr(self.opt, 'source_mean'):
            return motions
        source_mean = torch.as_tensor(
            self.opt.source_mean, device=motions.device, dtype=motions.dtype)
        source_std = torch.as_tensor(
            self.opt.source_std, device=motions.device, dtype=motions.dtype)
        evaluator_mean = torch.as_tensor(
            self.opt.evaluator_mean, device=motions.device, dtype=motions.dtype)
        evaluator_std = torch.as_tensor(
            self.opt.evaluator_std, device=motions.device, dtype=motions.dtype)
        return (motions * source_std + source_mean - evaluator_mean) / evaluator_std

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()
            motions = self._renormalize(motions)

            '''Movement Encoding'''
            movements = self.movement_encoder(self._movement_input(motions)).detach()
            m_lens = self._movement_lengths(m_lens)
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()
            motions = self._renormalize(motions)

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(self._movement_input(motions)).detach()
            m_lens = self._movement_lengths(m_lens)
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding
