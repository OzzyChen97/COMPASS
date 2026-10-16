"""
SATO++ Training Script

基于 T2M-GPT 的 preserving/altering 混合训练。

用法:
    python train_sato_pp.py \
        --dataname t2m \
        --resume-pth pretrained/vq_best.pth \
        --resume-trans pretrained/net_best_fid.pth \
        --pair-data-path dataset/KIT-ML/preserving_pairs_filtered.jsonl \
        --exp-name sato_pp_kit
"""

import os
import sys
import copy
import json
import random
import time
from functools import partial
from collections import Counter
from datetime import timedelta
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from os.path import join as pjoin
from torch.distributions import Categorical

# 项目内模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clip
from clip_local import model as clip_model_patch  # noqa: F401 — 确保 patch 可用
from options import option_sato_pp as option_sato_pp
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
import utils.sato_pp_eval as sato_pp_eval
from utils.plateau_control import PlateauController
from dataset import dataset_sato_pp
from dataset import dataset_TM_eval
from dataset import dataset_tokenize
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
from dataset.config import (
    configure_evaluator_source,
    dataset_max_motion_length,
    evaluator_opt_path,
    get_dataset_spec,
)
from dataset.variable_motion import DistributedLengthBucketBatchSampler

# SATO++ 模块
from models.clip_attribution import (
    compute_grad_input_attribution,
    compute_ar_score,
    compute_attention_attn,
    normalize_attribution,
    configure_second_order_attention,
)
from models.sato_pp_modules import AttributionPooling, PredictorHead
from models.sato_pp_losses import (
    loss_align, loss_variance, loss_covariance,
    loss_attribution_diff, loss_locality,
    loss_teacher, loss_topk_attention
)

import warnings
warnings.filterwarnings('ignore')


class NoOpLogger:
    def info(self, *args, **kwargs):
        pass


def init_wandb(args, logger):
    if getattr(args, 'disable_wandb', False):
        logger.info("W&B disabled by --disable-wandb")
        return None

    try:
        import wandb
    except Exception as exc:
        logger.warning(f"W&B unavailable: {exc}")
        return None

    mode = args.wandb_mode
    if mode is None and not os.environ.get('WANDB_API_KEY'):
        mode = 'offline'

    tags = [tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()]
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.wandb_run_id,
            name=args.wandb_name or args.exp_name,
            group=args.wandb_group,
            tags=tags,
            config=vars(args),
            dir=args.out_dir,
            mode=mode,
            resume='allow')
        run.config.update(vars(args), allow_val_change=True)
        run.define_metric('optimization_step')
        run.define_metric('*', step_metric='optimization_step')
        logger.info(
            f"W&B initialized: entity={args.wandb_entity}, "
            f"project={args.wandb_project}, "
            f"name={args.wandb_name or args.exp_name}, "
            f"run_id={run.id}, mode={mode or 'default'}")
        return run
    except Exception as exc:
        logger.warning(f"W&B init failed, continuing without W&B: {exc}")
        return None


class SATOClipTextWrapper(torch.nn.Module):
    """Expose CLIP text paths through forward so DDP can hook gradients."""

    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, text_tokens, mode='features'):
        if mode == 'features':
            return self.clip.encode_text_with_features(text_tokens)
        if mode == 'pooled':
            return self.clip.encode_text(text_tokens)
        if mode == 'attribution':
            dtype = self.clip.dtype
            h = self.clip.token_embedding(text_tokens).type(dtype)
            h = h.detach().requires_grad_(True)

            x = h + self.clip.positional_embedding.type(dtype)
            x = x.permute(1, 0, 2)
            x = self.clip.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.clip.ln_final(x).type(dtype)

            eos_idx = text_tokens.argmax(dim=-1)
            pooled = x[torch.arange(x.shape[0], device=x.device), eos_idx]
            if self.clip.text_projection is not None:
                pooled = pooled @ self.clip.text_projection
            return h, pooled
        raise ValueError(f'unknown CLIP forward mode: {mode}')

    def encode_text(self, text_tokens):
        return self.clip.encode_text(text_tokens)

    def encode_text_with_features(self, text_tokens):
        return self.clip.encode_text_with_features(text_tokens)


def setup_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world_size > 1

    if not torch.cuda.is_available():
        raise RuntimeError('SATO++ training requires CUDA')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    if distributed:
        timeout_seconds = int(os.environ.get('TORCH_DISTRIBUTED_TIMEOUT_SECONDS', '7200'))
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(seconds=timeout_seconds))

    return distributed, rank, local_rank, world_size, device


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def unwrap_module(module):
    return module.module if hasattr(module, 'module') else module


def unwrap_clip_model(module):
    module = unwrap_module(module)
    return module.clip if isinstance(module, SATOClipTextWrapper) else module


def average_gradients(modules, world_size):
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            if not param.requires_grad:
                continue
            # Conditional preserve/alter branches can leave a parameter unused
            # on one rank but used on another. Every rank must still execute the
            # same collective sequence, and the unused rank contributes zero.
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size


def compute_grad_input_attribution_batched(clip_model, trans_encoder,
                                           text_tokens, target_tokens, K=16,
                                           valid_len=None,
                                           create_graph=False):
    """
    Batched Grad x Input attribution.

    This keeps each sample's score independent, then sums the per-sample
    scores before autograd. Gradients for each text row therefore match the
    single-sample computation, without running one backward per sample.
    """
    h, pooled = clip_model(text_tokens, mode='attribution')

    input_index = target_tokens[:, :-1]
    cls_pred = trans_encoder(input_index, pooled.float())

    max_steps = min(K, input_index.shape[1])
    log_probs = F.log_softmax(cls_pred[:, :max_steps, :], dim=-1)
    # Padded motion tokens can be pad_idx (= vocab size) beyond each sample's
    # valid length. Clamp before gather, then remove them with valid_mask below.
    target_slice = target_tokens[:, :max_steps].clamp(
        min=0, max=cls_pred.shape[-1] - 1)
    token_log_probs = log_probs.gather(
        2, target_slice.unsqueeze(-1)).squeeze(-1)

    if valid_len is None:
        score_per_sample = token_log_probs.mean(dim=1)
    else:
        valid_steps = valid_len.to(token_log_probs.device).long()
        valid_steps = valid_steps.clamp(min=1, max=max_steps)
        step_ids = torch.arange(max_steps, device=token_log_probs.device)
        valid_mask = step_ids.unsqueeze(0) < valid_steps.unsqueeze(1)
        score_per_sample = (
            token_log_probs * valid_mask.float()).sum(dim=1)
        score_per_sample = score_per_sample / valid_steps.float()

    grad_h, = torch.autograd.grad(
        score_per_sample.sum(),
        h,
        retain_graph=create_graph,
        create_graph=create_graph,
    )
    return (grad_h.float() * h.detach().float()).abs().sum(dim=-1)


def save_sato_pp_checkpoint(path, nb_iter, args, clip_model, trans_encoder,
                            attr_pooling, predictor_head, optimizer, scheduler,
                            best_fid, best_iter, best_div, best_top1, best_top2,
                            best_top3, best_matching, plateau=None,
                            best_robust_score=float('inf'),
                            robust_baseline=None):
    ckpt = {
        'iter': nb_iter,
        'args': vars(args),
        'clip': unwrap_clip_model(clip_model).state_dict(),
        'trans': unwrap_module(trans_encoder).state_dict(),
        'attr_pooling': unwrap_module(attr_pooling).state_dict(),
        'predictor_head': unwrap_module(predictor_head).state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'plateau': plateau.state_dict() if plateau is not None else None,
        'best_fid': best_fid,
        'best_iter': best_iter,
        'best_div': best_div,
        'best_top1': best_top1,
        'best_top2': best_top2,
        'best_top3': best_top3,
        'best_matching': best_matching,
        'best_robust_score': best_robust_score,
        'robust_baseline': robust_baseline,
    }
    torch.save(ckpt, path)


def load_pair_rows(path):
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f'Expected JSON list in {path}')
    return payload


def evaluate_robust_levels(args, net, trans_encoder, clip_model,
                           eval_wrapper, w_vectorizer, logger, writer,
                           nb_iter, seed_offset=0):
    """Evaluate caption-locked validation P1/P2/P3 with fixed seeds."""
    if not args.robust_val_pair_paths:
        return None
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    trans_was_training = trans_encoder.training
    clip_was_training = clip_model.training
    trans_encoder.eval()
    clip_model.eval()
    rows = []
    try:
        for level_index, pair_path in enumerate(args.robust_val_pair_paths):
            pair_data = load_pair_rows(pair_path)
            val_loader = dataset_TM_eval.CaptionLockedDATALoader(
                args.dataname, False, 32, w_vectorizer, pair_data)
            level_name = os.path.basename(pair_path)
            for repeat in range(args.robust_eval_repeats):
                seed = args.seed + seed_offset + level_index * 100 + repeat
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                metrics, _, _, _, _ = sato_pp_eval.evaluation_sato_pp(
                    args.out_dir, val_loader, net, trans_encoder,
                    clip_model, eval_wrapper, logger, writer, nb_iter,
                    pair_data=pair_data, sample_categorical=True,
                    use_fast_batch=True, cfg_scale=args.cfg_scale,
                    cfg_uncond='zero', sampling_seed=seed,
                    caption_locked=True,
                )
                rows.append({
                    'level': level_name,
                    'repeat': repeat,
                    'seed': seed,
                    'fid': float(metrics['fid']),
                    'fid_p': float(metrics['fid_p']),
                    'fid_d': float(metrics['fid_d']),
                })
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_state)
        trans_encoder.train(trans_was_training)
        clip_model.train(clip_was_training)
    aggregate = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ('fid', 'fid_p', 'fid_d')
    }
    aggregate['rows'] = rows
    return aggregate


def robust_score(current, baseline, fid_tolerance):
    fid_ratio = current['fid'] / max(baseline['fid'], 1e-12)
    fid_p_ratio = current['fid_p'] / max(baseline['fid_p'], 1e-12)
    fid_d_ratio = current['fid_d'] / max(baseline['fid_d'], 1e-12)
    score = 0.5 * (fid_p_ratio + fid_d_ratio)
    eligible = fid_ratio <= 1.0 + fid_tolerance
    controlled_score = score + max(0.0, fid_ratio - 1.0 - fid_tolerance) * 10.0
    return {
        'score': score,
        'controlled_score': controlled_score,
        'eligible': eligible,
        'fid_ratio': fid_ratio,
        'fid_p_ratio': fid_p_ratio,
        'fid_d_ratio': fid_d_ratio,
    }


def main():
    # ======== Args ========
    args = option_sato_pp.get_sato_pp_args()
    configure_second_order_attention(args.attribution_create_graph)
    distributed, rank, local_rank, world_size, device = setup_distributed()
    # Every rank must start from identical parameters because this script uses
    # manual gradient all-reduce rather than DDP parameter broadcasting.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.world_size = world_size
    spec = get_dataset_spec(args.dataname)
    if spec.variable_length:
        args.sequence_mode = "full_variable"
        args.seq_len = None
        args.max_motion_length = dataset_max_motion_length(args.dataname)
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be positive")
    args.global_batch_size = args.batch_size * world_size * args.grad_accum_steps

    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    data_root = str(get_dataset_spec(args.dataname).root)
    args.vq_dir = os.path.join(data_root, args.vq_name)
    if is_main_process(rank):
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(args.vq_dir, exist_ok=True)
    if distributed:
        dist.barrier(device_ids=[local_rank])

    # ======== Logger ========
    if is_main_process(rank):
        logger = utils_model.get_logger(args.out_dir)
        writer = SummaryWriter(args.out_dir)
        logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
        logger.info(
            f"DDP enabled={distributed}, world_size={world_size}, "
            f"local_batch_size={args.batch_size}, "
            f"global_batch_size={args.global_batch_size}")
        wandb_run = init_wandb(args, logger)
    else:
        logger = NoOpLogger()
        writer = None
        wandb_run = None

    # ======== Dataloaders ========
    if is_main_process(rank):
        from utils.word_vectorizer import WordVectorizer
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        val_loader = dataset_TM_eval.DATALoader(
            args.dataname, False, 32, w_vectorizer)

        dataset_opt_path = evaluator_opt_path(args.dataname)
        wrapper_opt = get_opt(dataset_opt_path, device)
        wrapper_opt = configure_evaluator_source(wrapper_opt, args.dataname)
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    else:
        val_loader = None
        eval_wrapper = None

    # ======== CLIP (解冻) ========
    clip_model, clip_preprocess = clip.load(
        "ViT-B/32", device=device, jit=False)
    clip.model.convert_weights(clip_model)
    clip_model.float()  # SATO++: keep float32 to avoid fp16 overflow during fine-tuning

    if args.unfreeze_clip:
        for p in clip_model.parameters():
            p.requires_grad = True
        logger.info("CLIP unfrozen for SATO++ fine-tuning (float32)")
    else:
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        logger.info("CLIP frozen (original T2M-GPT mode)")
    clip_model = SATOClipTextWrapper(clip_model).to(device)

    # ======== VQ-VAE (冻结) ========
    net = vqvae.HumanVQVAE(args, args.nb_code, args.code_dim,
                            args.output_emb_width, args.down_t, args.stride_t,
                            args.width, args.depth, args.dilation_growth_rate)
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=False)
    net.eval()
    net.to(device)

    # ======== Transformer (训练) ========
    trans_encoder = trans.Text2Motion_Transformer(
        num_vq=args.nb_code, embed_dim=args.embed_dim_gpt,
        clip_dim=args.clip_dim, block_size=args.block_size,
        num_layers=args.num_layers, n_head=args.n_head_gpt,
        drop_out_rate=args.drop_out_rate, fc_rate=args.ff_rate)

    if args.resume_trans is not None:
        ckpt_trans = torch.load(args.resume_trans, map_location='cpu')
        trans_encoder.load_state_dict(ckpt_trans['trans'], strict=True)
    trans_encoder.train()
    trans_encoder.to(device)

    # ======== Teacher (冻结的原始 T2M-GPT) ========
    teacher_trans = copy.deepcopy(trans_encoder)
    for p in teacher_trans.parameters():
        p.requires_grad = False
    teacher_trans.eval()
    logger.info("Teacher model created (frozen copy of original T2M-GPT)")

    # ======== SATO++ 新模块 ========
    attr_pooling = AttributionPooling(
        clip_dim=args.clip_dim_attr, proj_dim=args.proj_dim,
        proj_type=args.proj_type).to(device)
    predictor_head = PredictorHead(dim=args.proj_dim).to(device)

    logger.info(f"AttributionPooling: clip_dim_attr={args.clip_dim_attr} → proj_dim={args.proj_dim} ({args.proj_type})")
    logger.info(f"Attribution: method={args.attribution_method}, K={args.attribution_K}, tau={args.tau}")

    # ======== Optimizer ========
    param_groups = [
        {'params': trans_encoder.parameters(), 'lr': args.lr,
         'min_lr': args.plateau_min_lr},
        {'params': attr_pooling.parameters(), 'lr': args.lr,
         'min_lr': args.plateau_min_lr},
        {'params': predictor_head.parameters(), 'lr': args.lr,
         'min_lr': args.plateau_min_lr},
    ]
    if args.unfreeze_clip:
        param_groups.append({
            'params': clip_model.parameters(), 'lr': args.clip_lr,
            'min_lr': args.clip_min_lr})

    optim_kwargs = {
        'weight_decay': args.weight_decay,
        # Match utils_model.initial_optim used by the original T2M-GPT
        # transformer training script when decay_option == 'all'.
        'betas': (0.5, 0.9),
    }
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, **optim_kwargs)
    else:
        optimizer = torch.optim.Adam(param_groups, **optim_kwargs)
    logger.info("Optimizer: %s betas=%s weight_decay=%s",
                args.optimizer, optim_kwargs['betas'], args.weight_decay)

    scheduler = None
    plateau = None
    if args.lr_policy == 'fixed':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
    else:
        plateau = PlateauController(
            mode='min', factor=args.plateau_factor,
            lr_patience=args.plateau_patience,
            stop_patience=args.early_stop_patience,
            min_index=args.min_train_iter,
            min_lr=args.plateau_min_lr)

    # ======== Optional full SATO++ resume ========
    resume_iter = 0
    resume_state = None
    if args.resume_sato_pp is not None:
        logger.info(f"loading full SATO++ checkpoint from {args.resume_sato_pp}")
        resume_state = torch.load(args.resume_sato_pp, map_location='cpu')
        unwrap_clip_model(clip_model).load_state_dict(
            resume_state['clip'], strict=True)
        trans_encoder.load_state_dict(resume_state['trans'], strict=True)
        attr_pooling.load_state_dict(resume_state['attr_pooling'], strict=True)
        predictor_head.load_state_dict(resume_state['predictor_head'], strict=True)
        if 'optimizer' in resume_state:
            optimizer.load_state_dict(resume_state['optimizer'])
        if scheduler is not None and resume_state.get('scheduler') is not None:
            scheduler.load_state_dict(resume_state['scheduler'])
            resumed_milestones = Counter(scheduler.milestones)
            requested_milestones = Counter(args.lr_scheduler)
            if resumed_milestones != requested_milestones:
                missed_decay_count = sum(
                    max(0, count - resumed_milestones[milestone])
                    for milestone, count in requested_milestones.items()
                    if milestone <= scheduler.last_epoch)
                if missed_decay_count:
                    decay = args.gamma ** missed_decay_count
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= decay
                    scheduler._last_lr = [
                        param_group['lr']
                        for param_group in optimizer.param_groups
                    ]
                    logger.info(
                        "Applied %d newly requested resumed LR decay(s); lr=%s",
                        missed_decay_count,
                        [group['lr'] for group in optimizer.param_groups])
                logger.info(
                    "Overriding resumed LR milestones: %s -> %s",
                    sorted(resumed_milestones.elements()),
                    sorted(requested_milestones.elements()))
                scheduler.milestones = requested_milestones
        if plateau is not None and resume_state.get('plateau') is not None:
            plateau.load_state_dict(resume_state['plateau'])
        resume_iter = int(resume_state.get('iter', 0))

    # Parameters must be initialized identically across ranks, while training
    # randomness should remain rank-local as in the HumanML3D DDP runs.
    train_seed = args.seed + rank
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)

    if distributed:
        logger.info(
            "Using manual gradient all-reduce data parallelism. "
            "This avoids DDP reducer conflicts from repeated CLIP forwards.")

    # ======== Loss ========
    loss_ce = torch.nn.CrossEntropyLoss()

    # ======== Tokenize motions ========
    # Skip: tokens already exist in dataset/HumanML3D/VQVAE/
    # (VQ-VAE architecture has depth mismatch with pretrained weights)
    logger.info(f"Skipping tokenization, using existing tokens in {pjoin(data_root, args.vq_name)}")

    # ======== SATO++ DataLoader ========
    train_dataset = dataset_sato_pp.SATOPPDataset(
        args.dataname, pair_data_path=args.pair_data_path,
        codebook_size=args.nb_code, tokenizer_name=args.vq_name,
        unit_length=2 ** args.down_t,
        preserve_ratio=args.preserve_ratio,
        limit=args.dataset_limit)
    train_batch_sampler = None
    if train_dataset.variable_length:
        train_sampler = None
        train_batch_sampler = DistributedLengthBucketBatchSampler(
            train_dataset.lengths, batch_size=args.batch_size,
            num_replicas=world_size, rank=rank, seed=args.seed,
            shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_batch_sampler,
            num_workers=args.num_workers,
            collate_fn=partial(
                dataset_sato_pp.collate_fn_sato_pp_variable,
                pad_idx=train_dataset.mot_pad_idx),
            pin_memory=True,
            persistent_workers=args.num_workers > 0)
    else:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True) if distributed else None
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=(train_sampler is None), sampler=train_sampler,
            num_workers=args.num_workers, drop_last=True,
            collate_fn=dataset_sato_pp.collate_fn_sato_pp)
    train_epoch_sampler = train_batch_sampler or train_sampler
    train_epoch = 0
    if train_epoch_sampler is not None:
        train_epoch_sampler.set_epoch(train_epoch)
    train_loader_iter = iter(train_loader)

    # ======== Initial Eval ========
    best_fid, best_iter, best_div = 1000, 0, 100
    best_top1, best_top2, best_top3, best_matching = 0, 0, 0, 100
    best_robust_score = float('inf')
    robust_baseline = None
    if resume_state is not None:
        best_fid = resume_state.get('best_fid', best_fid)
        best_iter = resume_state.get('best_iter', best_iter)
        best_div = resume_state.get('best_div', best_div)
        best_top1 = resume_state.get('best_top1', best_top1)
        best_top2 = resume_state.get('best_top2', best_top2)
        best_top3 = resume_state.get('best_top3', best_top3)
        best_matching = resume_state.get('best_matching', best_matching)
        best_robust_score = resume_state.get(
            'best_robust_score', best_robust_score)
        robust_baseline = resume_state.get('robust_baseline')

    if (is_main_process(rank) and args.eval_perturbed and
            args.robust_val_pair_paths and robust_baseline is None):
        logger.info('Evaluating frozen T2M-GPT robustness baseline on validation pairs')
        robust_baseline = evaluate_robust_levels(
            args, net, teacher_trans, unwrap_clip_model(clip_model),
            eval_wrapper, w_vectorizer, logger, writer, 0)
        teacher_trans.eval()
        baseline_path = os.path.join(args.out_dir, 'robust_baseline.json')
        with open(baseline_path, 'w', encoding='utf-8') as handle:
            json.dump(robust_baseline, handle, indent=2)
        logger.info('Robust baseline: %s', json.dumps(robust_baseline))
    if distributed:
        dist.barrier(device_ids=[local_rank])

    # Skip initial eval — too slow, start training directly
    # best_fid, best_iter, best_div, best_top1, best_top2, best_top3, \
    #     best_matching, writer, logger = eval_trans.evaluation_transformer(
    #         args.out_dir, val_loader, net, trans_encoder, logger, writer, 0,
    #         best_fid=best_fid, best_iter=0, best_div=100,
    #         best_top1=0, best_top2=0, best_top3=0, best_matching=100,
    #         clip_model=clip_model, eval_wrapper=eval_wrapper)
    logger.info("Skipped initial eval, starting training directly.")

    # ======== Training Loop ========
    nb_iter = resume_iter
    avg_loss = {'gen': 0., 'teach': 0., 'align': 0., 'var': 0.,
                'cov': 0., 'diff': 0., 'loc': 0., 'total': 0.,
                'u_std': 0., 'u_norm': 0., 'cos_u': 0.,
                'alpha_entropy': 0., 'mask_zero': 0., 'mask_count': 0.}
    log_started_at = time.perf_counter()
    accumulation_index = 0
    accumulated_preserve = 0
    accumulated_alter = 0
    optimizer.zero_grad(set_to_none=True)

    while nb_iter <= args.total_iter:
        try:
            batch = next(train_loader_iter)
        except StopIteration:
            train_epoch += 1
            if train_epoch_sampler is not None:
                train_epoch_sampler.set_epoch(train_epoch)
            train_loader_iter = iter(train_loader)
            batch = next(train_loader_iter)
        (cap_orig, cap_pair, m_tokens, m_tokens_len,
         relation, edit_type, slot_mask) = batch

        m_tokens = m_tokens.cuda()
        m_tokens_len = m_tokens_len.cuda()
        slot_mask = slot_mask.cuda()  # (bs, 77) bool
        bs = m_tokens.shape[0]
        target = m_tokens.cuda()

        # ---- Encode text ----
        text_orig = clip.tokenize(list(cap_orig), truncate=True).cuda()
        text_pair = clip.tokenize(list(cap_pair), truncate=True).cuda()

        with torch.cuda.amp.autocast(enabled=False):
            # 获取 token features + pooled features
            H_orig, feat_orig = clip_model(text_orig, mode='features')
            H_pair, feat_pair = clip_model(text_pair, mode='features')
            feat_orig = feat_orig.float()
            feat_pair = feat_pair.float()
            H_orig = H_orig.float()
            H_pair = H_pair.float()

        # ---- Generation loss (original CE) ----
        input_index = target[:, :-1]

        if args.pkeep == -1:
            proba = np.random.rand(1)[0]
            mask = torch.bernoulli(proba * torch.ones(input_index.shape,
                                                       device=input_index.device))
        else:
            mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape,
                                                             device=input_index.device))
        mask = mask.round().to(dtype=torch.int64)
        r_indices = torch.randint_like(input_index, args.nb_code)
        a_indices = mask * input_index + (1 - mask) * r_indices

        cls_pred = trans_encoder(a_indices, feat_orig)
        cls_pred = cls_pred.contiguous()

        loss_gen = 0.0
        for i in range(bs):
            loss_gen += loss_ce(
                cls_pred[i][:m_tokens_len[i] + 1],
                target[i][:m_tokens_len[i] + 1]) / bs

        # ---- Teacher loss ----
        with torch.no_grad():
            teacher_pred = teacher_trans(a_indices, feat_orig)
        loss_teach = loss_teacher(
            cls_pred, teacher_pred, m_tokens_len + 1,
            loss_type=args.teacher_loss_type)

        total_loss = loss_gen + args.lambda_teach * loss_teach

        # ---- Relation-conditioned losses ----
        loss_align_val = torch.tensor(0., device='cuda')
        loss_diff_val = torch.tensor(0., device='cuda')
        loss_loc_val = torch.tensor(0., device='cuda')

        preserve_mask = torch.tensor([r == 'preserve' for r in relation],
                                     device=target.device, dtype=torch.bool)
        alter_mask = torch.tensor([r == 'alter' for r in relation],
                                  device=target.device, dtype=torch.bool)
        n_preserve = int(preserve_mask.sum().item())
        n_alter = int(alter_mask.sum().item())

        alpha_orig = compute_grad_input_attribution_batched(
            clip_model, trans_encoder, text_orig, target,
            K=args.attribution_K, valid_len=m_tokens_len,
            create_graph=args.attribution_create_graph)
        alpha_orig = normalize_attribution(alpha_orig, args.tau)

        alpha_pair = compute_grad_input_attribution_batched(
            clip_model, trans_encoder, text_pair, target,
            K=args.attribution_K, valid_len=m_tokens_len,
            create_graph=args.attribution_create_graph)
        alpha_pair = normalize_attribution(alpha_pair, args.tau)

        alpha_entropy_vals = [
            -torch.cat([alpha_orig, alpha_pair], dim=0).mul(
                torch.cat([alpha_orig, alpha_pair], dim=0).add(1e-8).log()
            ).sum(dim=-1).mean()
        ]

        # Batch-level var/cov (VICReg requires >= 2 samples)
        loss_var_val = torch.tensor(0., device='cuda')
        loss_cov_val = torch.tensor(0., device='cuda')
        u_std_mean = torch.tensor(0., device='cuda')
        u_norm_mean = torch.tensor(0., device='cuda')
        cos_u_mean = torch.tensor(0., device='cuda')
        if n_preserve > 0:
            u_orig_cat = attr_pooling(H_orig[preserve_mask],
                                      alpha_orig[preserve_mask])
            u_pair_cat = attr_pooling(H_pair[preserve_mask],
                                      alpha_pair[preserve_mask])
            z_orig = predictor_head(u_orig_cat)
            z_pair = predictor_head(u_pair_cat)
            loss_align_val = loss_align(z_orig, z_pair)

        if n_preserve >= 2:
            u_all = torch.cat([u_orig_cat, u_pair_cat], dim=0)
            loss_var_val = loss_variance(u_all, args.var_threshold)
            loss_cov_val = loss_covariance(u_all)
            u_std_mean = u_all.std(dim=0).mean()
            u_norm_mean = u_all.norm(dim=-1).mean()
            cos_u_mean = F.cosine_similarity(u_orig_cat, u_pair_cat,
                                             dim=-1).mean()

        # 归一化
        if n_preserve > 0:
            total_loss = total_loss + (
                args.lambda_align * loss_align_val +
                args.lambda_var * loss_var_val +
                args.lambda_cov * loss_cov_val)

        if n_alter > 0:
            loss_diff_val = loss_attribution_diff(
                alpha_orig[alter_mask], alpha_pair[alter_mask],
                slot_mask[alter_mask])
            loss_loc_val = loss_locality(
                alpha_orig[alter_mask], alpha_pair[alter_mask],
                slot_mask[alter_mask])
            total_loss = total_loss + (
                args.lambda_diff * loss_diff_val +
                args.lambda_loc * loss_loc_val)

        # ---- Backward ----
        (total_loss / args.grad_accum_steps).backward()

        # ---- Logging ----
        scale = 1.0 / args.grad_accum_steps
        avg_loss['gen'] += loss_gen.item() * scale
        avg_loss['teach'] += loss_teach.item() * scale
        avg_loss['align'] += loss_align_val.item() * scale
        avg_loss['var'] += loss_var_val.item() * scale
        avg_loss['cov'] += loss_cov_val.item() * scale
        avg_loss['diff'] += loss_diff_val.item() * scale
        avg_loss['loc'] += loss_loc_val.item() * scale
        avg_loss['total'] += total_loss.item() * scale
        avg_loss['u_std'] += u_std_mean.item() * scale
        avg_loss['u_norm'] += u_norm_mean.item() * scale
        avg_loss['cos_u'] += cos_u_mean.item() * scale
        if alpha_entropy_vals:
            alpha_entropy = torch.stack(alpha_entropy_vals).mean()
            avg_loss['alpha_entropy'] += alpha_entropy.item() * scale
        edited_count = slot_mask.sum(dim=-1).float()
        avg_loss['mask_zero'] += (edited_count == 0).float().mean().item() * scale
        avg_loss['mask_count'] += edited_count.mean().item() * scale

        accumulation_index += 1
        accumulated_preserve += n_preserve
        accumulated_alter += n_alter
        if accumulation_index < args.grad_accum_steps:
            continue

        if distributed:
            sync_modules = [trans_encoder, attr_pooling, predictor_head]
            if args.unfreeze_clip:
                sync_modules.append(clip_model)
            average_gradients(sync_modules, world_size)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        n_preserve = accumulated_preserve
        n_alter = accumulated_alter
        accumulation_index = 0
        accumulated_preserve = 0
        accumulated_alter = 0

        nb_iter += 1

        if nb_iter % args.print_iter == 0:
            for k in avg_loss:
                avg_loss[k] /= args.print_iter

            metric_keys = list(avg_loss.keys())
            metric_vals = torch.tensor(
                [avg_loss[k] for k in metric_keys], device=device)
            count_vals = torch.tensor(
                [float(n_preserve), float(n_alter)], device=device)
            if distributed:
                dist.all_reduce(metric_vals, op=dist.ReduceOp.SUM)
                metric_vals /= world_size
                dist.all_reduce(count_vals, op=dist.ReduceOp.SUM)

            avg_print = {
                k: metric_vals[i].item() for i, k in enumerate(metric_keys)
            }
            n_preserve_print = int(count_vals[0].item())
            n_alter_print = int(count_vals[1].item())

            if is_main_process(rank):
                log_elapsed = max(time.perf_counter() - log_started_at, 1e-6)
                samples_per_second = (
                    args.print_iter * args.global_batch_size / log_elapsed)
                writer.add_scalar('./Loss/gen', avg_print['gen'], nb_iter)
                writer.add_scalar('./Loss/teach', avg_print['teach'], nb_iter)
                writer.add_scalar('./Loss/align', avg_print['align'], nb_iter)
                writer.add_scalar('./Loss/var', avg_print['var'], nb_iter)
                writer.add_scalar('./Loss/cov', avg_print['cov'], nb_iter)
                writer.add_scalar('./Loss/diff', avg_print['diff'], nb_iter)
                writer.add_scalar('./Loss/loc', avg_print['loc'], nb_iter)
                writer.add_scalar('./Loss/total', avg_print['total'], nb_iter)
                writer.add_scalar('./Diag/u_std_mean', avg_print['u_std'], nb_iter)
                writer.add_scalar('./Diag/u_norm_mean', avg_print['u_norm'], nb_iter)
                writer.add_scalar('./Diag/cos_u_orig_pair', avg_print['cos_u'], nb_iter)
                writer.add_scalar('./Diag/alpha_entropy', avg_print['alpha_entropy'], nb_iter)
                writer.add_scalar('./Diag/edited_mask_zero_ratio', avg_print['mask_zero'], nb_iter)
                writer.add_scalar('./Diag/edited_mask_count', avg_print['mask_count'], nb_iter)
                writer.add_scalar('./Stats/n_preserve', n_preserve_print, nb_iter)
                writer.add_scalar('./Stats/n_alter', n_alter_print, nb_iter)
                writer.add_scalar(
                    './Perf/samples_per_second', samples_per_second, nb_iter)
                if wandb_run is not None:
                    wandb_run.log({
                        'train/loss_gen': avg_print['gen'],
                        'train/loss_teach': avg_print['teach'],
                        'train/loss_align': avg_print['align'],
                        'train/loss_var': avg_print['var'],
                        'train/loss_cov': avg_print['cov'],
                        'train/loss_diff': avg_print['diff'],
                        'train/loss_loc': avg_print['loc'],
                        'train/loss_total': avg_print['total'],
                        'diag/u_std_mean': avg_print['u_std'],
                        'diag/u_norm_mean': avg_print['u_norm'],
                        'diag/cos_u_orig_pair': avg_print['cos_u'],
                        'diag/alpha_entropy': avg_print['alpha_entropy'],
                        'diag/edited_mask_zero_ratio': avg_print['mask_zero'],
                        'diag/edited_mask_count': avg_print['mask_count'],
                        'stats/n_preserve': n_preserve_print,
                        'stats/n_alter': n_alter_print,
                        'stats/world_size': world_size,
                        'stats/local_batch_size': args.batch_size,
                        'stats/global_batch_size': args.global_batch_size,
                        'perf/samples_per_second': samples_per_second,
                        'train/samples_seen': nb_iter * args.global_batch_size,
                        'lr/trans': optimizer.param_groups[0]['lr'],
                        'optimization_step': nb_iter,
                    })

                msg = (f"Train. Iter {nb_iter}: "
                       f"Total {avg_print['total']:.4f} | "
                       f"Gen {avg_print['gen']:.4f} | "
                       f"Teach {avg_print['teach']:.4f} | "
                       f"Align {avg_print['align']:.4f} | "
                       f"Var {avg_print['var']:.4f} | "
                       f"Cov {avg_print['cov']:.4f} | "
                       f"Diff {avg_print['diff']:.4f} | "
                       f"Loc {avg_print['loc']:.4f} | "
                       f"UStd {avg_print['u_std']:.4f} | "
                       f"UNorm {avg_print['u_norm']:.4f} | "
                       f"CosU {avg_print['cos_u']:.4f} | "
                       f"AlphaH {avg_print['alpha_entropy']:.4f} | "
                       f"Mask0 {avg_print['mask_zero']:.3f} | "
                       f"MaskCnt {avg_print['mask_count']:.2f} | "
                       f"p/a={n_preserve_print}/{n_alter_print}")
                logger.info(msg)
                log_started_at = time.perf_counter()

            for k in avg_loss:
                avg_loss[k] = 0.

        if args.save_iter > 0 and nb_iter % args.save_iter == 0:
            if is_main_process(rank):
                last_path = os.path.join(args.out_dir, 'sato_pp_last.pth')
                save_sato_pp_checkpoint(
                    last_path, nb_iter, args, clip_model, trans_encoder,
                    attr_pooling, predictor_head, optimizer, scheduler,
                    best_fid, best_iter, best_div, best_top1, best_top2,
                    best_top3, best_matching, plateau,
                    best_robust_score, robust_baseline)
                logger.info(f"Saved full SATO++ checkpoint: {last_path}")

            if (is_main_process(rank) and args.milestone_save_iter > 0 and
                    nb_iter % args.milestone_save_iter == 0):
                milestone_path = os.path.join(
                    args.out_dir, f'sato_pp_iter_{nb_iter}.pth')
                save_sato_pp_checkpoint(
                    milestone_path, nb_iter, args, clip_model, trans_encoder,
                    attr_pooling, predictor_head, optimizer, scheduler,
                    best_fid, best_iter, best_div, best_top1, best_top2,
                    best_top3, best_matching, plateau,
                    best_robust_score, robust_baseline)
                logger.info(f"Saved full SATO++ milestone: {milestone_path}")
            if distributed:
                dist.barrier(device_ids=[local_rank])

        if nb_iter % args.eval_iter == 0:
            if distributed:
                dist.barrier(device_ids=[local_rank])
            if is_main_process(rank):
                prev_best_fid = best_fid
                best_fid, best_iter, best_div, best_top1, best_top2, best_top3, \
                    best_matching, writer, logger, current_metrics = eval_trans.evaluation_transformer(
                        args.out_dir, val_loader, net, unwrap_module(trans_encoder),
                        logger, writer, nb_iter, best_fid, best_iter, best_div,
                        best_top1, best_top2, best_top3, best_matching,
                        clip_model=unwrap_clip_model(clip_model),
                        eval_wrapper=eval_wrapper, draw=args.eval_draw,
                        save=False, wandb_run=wandb_run,
                        wandb_step=nb_iter, return_metrics=True)
                current_fid = current_metrics['fid']

                if best_fid < prev_best_fid:
                    best_path = os.path.join(args.out_dir, 'sato_pp_best_fid.pth')
                    save_sato_pp_checkpoint(
                        best_path, nb_iter, args, clip_model, trans_encoder,
                        attr_pooling, predictor_head, optimizer, scheduler,
                        best_fid, best_iter, best_div, best_top1, best_top2,
                        best_top3, best_matching, plateau,
                        best_robust_score, robust_baseline)
                    logger.info(
                        f"Saved full SATO++ best-FID checkpoint: {best_path}")
            else:
                current_fid = float('nan')
            if distributed:
                best_vals = torch.tensor(
                    [best_fid, best_iter, best_div, best_top1, best_top2,
                     best_top3, best_matching],
                    dtype=torch.float64, device=device)
                dist.broadcast(best_vals, src=0)
                best_fid = float(best_vals[0].item())
                best_iter = int(best_vals[1].item())
                best_div = float(best_vals[2].item())
                best_top1 = float(best_vals[3].item())
                best_top2 = float(best_vals[4].item())
                best_top3 = float(best_vals[5].item())
                best_matching = float(best_vals[6].item())
                dist.barrier(device_ids=[local_rank])

            control_metric = current_fid
            robust_summary = None
            if (nb_iter % args.robust_eval_iter == 0 and
                    args.eval_perturbed and args.robust_val_pair_paths):
                if is_main_process(rank):
                    robust_summary = evaluate_robust_levels(
                        args, net, unwrap_module(trans_encoder),
                        unwrap_clip_model(clip_model), eval_wrapper,
                        w_vectorizer, logger, writer, nb_iter)
                    comparison = robust_score(
                        robust_summary, robust_baseline,
                        args.robust_fid_tolerance)
                    control_metric = comparison['controlled_score']
                    logger.info(
                        'Robust validation iter %d summary=%s comparison=%s',
                        nb_iter, json.dumps(robust_summary),
                        json.dumps(comparison))
                    if comparison['eligible'] and comparison['score'] < best_robust_score:
                        best_robust_score = comparison['score']
                        balanced_path = os.path.join(
                            args.out_dir, 'sato_pp_best_balanced.pth')
                        save_sato_pp_checkpoint(
                            balanced_path, nb_iter, args, clip_model,
                            trans_encoder, attr_pooling, predictor_head,
                            optimizer, scheduler, best_fid, best_iter,
                            best_div, best_top1, best_top2, best_top3,
                            best_matching, plateau, best_robust_score,
                            robust_baseline)
                        logger.info(
                            'Saved balanced SATO++ checkpoint: %s score %.6f',
                            balanced_path, best_robust_score)
                    if wandb_run is not None:
                        wandb_run.log({
                            'robust/fid': robust_summary['fid'],
                            'robust/fid_p': robust_summary['fid_p'],
                            'robust/fid_d': robust_summary['fid_d'],
                            'robust/score': comparison['score'],
                            'robust/controlled_score': comparison['controlled_score'],
                            'robust/fid_ratio': comparison['fid_ratio'],
                            'robust/fid_p_ratio': comparison['fid_p_ratio'],
                            'robust/fid_d_ratio': comparison['fid_d_ratio'],
                            'robust/eligible': int(comparison['eligible']),
                            'best/robust_score': best_robust_score,
                            'optimization_step': nb_iter,
                        })

            metric_tensor = torch.tensor(
                float(control_metric), dtype=torch.float64, device=device)
            if distributed:
                dist.broadcast(metric_tensor, src=0)
            stop_training = False
            if plateau is not None and (
                    not args.eval_perturbed or not args.robust_val_pair_paths or
                    nb_iter % args.robust_eval_iter == 0):
                decision = plateau.step(
                    float(metric_tensor.item()), nb_iter, [optimizer])
                stop_training = decision.stop
                if is_main_process(rank):
                    logger.info(
                        'Plateau control iter %d metric %.6f best %.6f bad %d lr %.3g reduce=%s stop=%s',
                        nb_iter, float(metric_tensor.item()),
                        decision.best_metric, decision.bad_evals,
                        decision.current_lr, decision.reduce_lr,
                        decision.stop)
                    if wandb_run is not None:
                        wandb_run.log({
                            'control/metric': float(metric_tensor.item()),
                            'control/best_metric': decision.best_metric,
                            'control/bad_evals': decision.bad_evals,
                            'control/lr_reduced': int(decision.reduce_lr),
                            'control/early_stop': int(decision.stop),
                            'optimization_step': nb_iter,
                        })
            if distributed:
                dist.barrier(device_ids=[local_rank])
            if stop_training:
                if is_main_process(rank):
                    final_path = os.path.join(args.out_dir, 'sato_pp_final.pth')
                    save_sato_pp_checkpoint(
                        final_path, nb_iter, args, clip_model, trans_encoder,
                        attr_pooling, predictor_head, optimizer, scheduler,
                        best_fid, best_iter, best_div, best_top1, best_top2,
                        best_top3, best_matching, plateau,
                        best_robust_score, robust_baseline)
                    logger.info('Early stop saved final checkpoint: %s', final_path)
                if distributed:
                    dist.barrier(device_ids=[local_rank])
                break

        if nb_iter == args.total_iter:
            if is_main_process(rank):
                final_path = os.path.join(args.out_dir, 'sato_pp_final.pth')
                save_sato_pp_checkpoint(
                    final_path, nb_iter, args, clip_model, trans_encoder,
                    attr_pooling, predictor_head, optimizer, scheduler,
                    best_fid, best_iter, best_div, best_top1, best_top2,
                    best_top3, best_matching, plateau,
                    best_robust_score, robust_baseline)
                logger.info(
                    f"Saved exact final SATO++ checkpoint: {final_path}")
            if distributed:
                dist.barrier(device_ids=[local_rank])
            msg_final = (f"Best. Iter {best_iter}: "
                         f"FID {best_fid:.5f}, Div {best_div:.4f}, "
                         f"Top1 {best_top1:.4f}, Top2 {best_top2:.4f}, "
                         f"Top3 {best_top3:.4f}")
            if is_main_process(rank):
                logger.info(msg_final)
            break

    if writer is not None:
        writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed(distributed)


if __name__ == '__main__':
    main()
