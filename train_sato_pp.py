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
import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
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
from dataset import dataset_sato_pp
from dataset import dataset_TM_eval
from dataset import dataset_tokenize
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
from dataset.config import (
    configure_evaluator_source,
    evaluator_opt_path,
    get_dataset_spec,
)

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
        # Avoid crashing unattended runs on machines that are not logged in.
        mode = 'offline'

    tags = [t.strip() for t in args.wandb_tags.split(',') if t.strip()]
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or args.exp_name,
            group=args.wandb_group,
            tags=tags,
            config=vars(args),
            dir=args.out_dir,
            mode=mode,
            resume='allow')
        logger.info(
            f"W&B initialized: project={args.wandb_project}, "
            f"name={args.wandb_name or args.exp_name}, mode={mode or 'default'}")
        return run
    except Exception as exc:
        logger.warning(f"W&B init failed, continuing without W&B: {exc}")
        return None


def save_sato_pp_checkpoint(path, nb_iter, args, clip_model, trans_encoder,
                            attr_pooling, predictor_head, optimizer, scheduler,
                            best_fid, best_iter, best_div, best_top1, best_top2,
                            best_top3, best_matching):
    ckpt = {
        'iter': nb_iter,
        'args': vars(args),
        'clip': clip_model.state_dict(),
        'trans': trans_encoder.state_dict(),
        'attr_pooling': attr_pooling.state_dict(),
        'predictor_head': predictor_head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_fid': best_fid,
        'best_iter': best_iter,
        'best_div': best_div,
        'best_top1': best_top1,
        'best_top2': best_top2,
        'best_top3': best_top3,
        'best_matching': best_matching,
    }
    torch.save(ckpt, path)


def main():
    # ======== Args ========
    args = option_sato_pp.get_sato_pp_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    configure_second_order_attention(args.attribution_create_graph)

    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    data_root = str(get_dataset_spec(args.dataname).root)
    args.vq_dir = os.path.join(data_root, args.vq_name)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.vq_dir, exist_ok=True)

    # ======== Logger ========
    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    wandb_run = init_wandb(args, logger)

    # ======== Dataloaders ========
    from utils.word_vectorizer import WordVectorizer
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer)

    dataset_opt_path = evaluator_opt_path(args.dataname)
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    wrapper_opt = configure_evaluator_source(wrapper_opt, args.dataname)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # ======== CLIP (解冻) ========
    clip_model, clip_preprocess = clip.load(
        "ViT-B/32", device=torch.device('cuda'), jit=False)
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

    # ======== VQ-VAE (冻结) ========
    net = vqvae.HumanVQVAE(args, args.nb_code, args.code_dim,
                            args.output_emb_width, args.down_t, args.stride_t,
                            args.width, args.depth, args.dilation_growth_rate)
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=False)
    net.eval()
    net.cuda()

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
    trans_encoder.cuda()

    # ======== Teacher (冻结的原始 T2M-GPT) ========
    teacher_trans = copy.deepcopy(trans_encoder)
    for p in teacher_trans.parameters():
        p.requires_grad = False
    teacher_trans.eval()
    logger.info("Teacher model created (frozen copy of original T2M-GPT)")

    # ======== SATO++ 新模块 ========
    attr_pooling = AttributionPooling(
        clip_dim=args.clip_dim_attr, proj_dim=args.proj_dim,
        proj_type=args.proj_type).cuda()
    predictor_head = PredictorHead(dim=args.proj_dim).cuda()

    logger.info(f"AttributionPooling: clip_dim_attr={args.clip_dim_attr} → proj_dim={args.proj_dim} ({args.proj_type})")
    logger.info(f"Attribution: method={args.attribution_method}, K={args.attribution_K}, tau={args.tau}")

    # ======== Optimizer ========
    param_groups = [
        {'params': trans_encoder.parameters(), 'lr': args.lr},
        {'params': attr_pooling.parameters(), 'lr': args.lr},
        {'params': predictor_head.parameters(), 'lr': args.lr},
    ]
    if args.unfreeze_clip:
        param_groups.append({'params': clip_model.parameters(), 'lr': args.clip_lr})

    optim_kwargs = {
        'weight_decay': args.weight_decay,
        'betas': (0.5, 0.9),
    }
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, **optim_kwargs)
    else:
        optimizer = torch.optim.Adam(param_groups, **optim_kwargs)
    logger.info("Optimizer: %s betas=%s weight_decay=%s",
                args.optimizer, optim_kwargs['betas'], args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    # ======== Optional full SATO++ resume ========
    resume_iter = 0
    resume_state = None
    if args.resume_sato_pp is not None:
        logger.info(f"loading full SATO++ checkpoint from {args.resume_sato_pp}")
        resume_state = torch.load(args.resume_sato_pp, map_location='cpu')
        clip_model.load_state_dict(resume_state['clip'], strict=True)
        trans_encoder.load_state_dict(resume_state['trans'], strict=True)
        attr_pooling.load_state_dict(resume_state['attr_pooling'], strict=True)
        predictor_head.load_state_dict(resume_state['predictor_head'], strict=True)
        if 'optimizer' in resume_state:
            optimizer.load_state_dict(resume_state['optimizer'])
        if 'scheduler' in resume_state:
            scheduler.load_state_dict(resume_state['scheduler'])
        resume_iter = int(resume_state.get('iter', 0))

    # ======== Loss ========
    loss_ce = torch.nn.CrossEntropyLoss()

    # ======== Tokenize motions ========
    # Skip: tokens already exist in dataset/HumanML3D/VQVAE/
    # (VQ-VAE architecture has depth mismatch with pretrained weights)
    logger.info(f"Skipping tokenization, using existing tokens in {pjoin(data_root, args.vq_name)}")

    # ======== SATO++ DataLoader ========
    train_loader = dataset_sato_pp.DATALoader(
        args.dataname, args.batch_size, args.nb_code, args.vq_name,
        pair_data_path=args.pair_data_path,
        unit_length=2 ** args.down_t,
        preserve_ratio=args.preserve_ratio,
        num_workers=0,
        limit=args.dataset_limit)
    train_loader_iter = dataset_sato_pp.cycle(train_loader)

    # ======== Initial Eval ========
    best_fid, best_iter, best_div = 1000, 0, 100
    best_top1, best_top2, best_top3, best_matching = 0, 0, 0, 100
    if resume_state is not None:
        best_fid = resume_state.get('best_fid', best_fid)
        best_iter = resume_state.get('best_iter', best_iter)
        best_div = resume_state.get('best_div', best_div)
        best_top1 = resume_state.get('best_top1', best_top1)
        best_top2 = resume_state.get('best_top2', best_top2)
        best_top3 = resume_state.get('best_top3', best_top3)
        best_matching = resume_state.get('best_matching', best_matching)

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

    while nb_iter <= args.total_iter:
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

        use_teacher_loss = args.lambda_teach != 0
        use_preserve_losses = (
            args.lambda_align != 0 or args.lambda_var != 0 or args.lambda_cov != 0)
        use_alter_losses = (args.lambda_diff != 0 or args.lambda_loc != 0)
        use_attribution_losses = use_preserve_losses or use_alter_losses

        with torch.cuda.amp.autocast(enabled=False):
            # 获取 token features + pooled features. For ablations without
            # attribution losses, avoid the paired-text feature pass.
            if use_attribution_losses:
                H_orig, feat_orig = clip_model.encode_text_with_features(text_orig)
                H_pair, feat_pair = clip_model.encode_text_with_features(text_pair)
                H_orig = H_orig.float()
                H_pair = H_pair.float()
            else:
                H_orig = H_pair = None
                feat_orig = clip_model.encode_text(text_orig)
                feat_pair = None
            feat_orig = feat_orig.float()
            if feat_pair is not None:
                feat_pair = feat_pair.float()

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
        if use_teacher_loss:
            with torch.no_grad():
                teacher_pred = teacher_trans(a_indices, feat_orig)
            loss_teach = loss_teacher(
                cls_pred, teacher_pred, m_tokens_len + 1,
                loss_type=args.teacher_loss_type)
        else:
            loss_teach = torch.tensor(0., device='cuda')

        total_loss = loss_gen + args.lambda_teach * loss_teach

        # ---- Relation-conditioned losses ----
        loss_align_val = torch.tensor(0., device='cuda')
        loss_diff_val = torch.tensor(0., device='cuda')
        loss_loc_val = torch.tensor(0., device='cuda')

        n_preserve = 0
        n_alter = 0

        # Accumulate u vectors for batch-level var/cov
        u_orig_list = []
        u_pair_list = []
        alpha_entropy_vals = []

        for i in range(bs):
            rel = relation[i]

            if rel == 'preserve':
                n_preserve += 1
                if not use_preserve_losses:
                    continue

                # 原文归因
                alpha_orig = compute_grad_input_attribution(
                    clip_model, trans_encoder,
                    text_orig[i:i+1], target[i:i+1], K=args.attribution_K,
                    valid_len=m_tokens_len[i:i+1],
                    create_graph=args.attribution_create_graph)
                alpha_orig = normalize_attribution(alpha_orig, args.tau)
                alpha_entropy_vals.append(
                    -(alpha_orig * (alpha_orig + 1e-8).log()).sum(dim=-1).mean())

                # pair 归因
                alpha_pair = compute_grad_input_attribution(
                    clip_model, trans_encoder,
                    text_pair[i:i+1], target[i:i+1], K=args.attribution_K,
                    valid_len=m_tokens_len[i:i+1],
                    create_graph=args.attribution_create_graph)
                alpha_pair = normalize_attribution(alpha_pair, args.tau)
                alpha_entropy_vals.append(
                    -(alpha_pair * (alpha_pair + 1e-8).log()).sum(dim=-1).mean())

                # 表示
                u_orig = attr_pooling(H_orig[i:i+1], alpha_orig)
                u_pair = attr_pooling(H_pair[i:i+1], alpha_pair)

                if args.lambda_align != 0:
                    # SimSiam: predictor on one side, stop-grad on other
                    z_orig = predictor_head(u_orig)
                    z_pair = predictor_head(u_pair)
                    loss_align_val = loss_align_val + loss_align(z_orig, z_pair)
                u_orig_list.append(u_orig)
                u_pair_list.append(u_pair)

            elif rel == 'alter':
                n_alter += 1
                if not use_alter_losses:
                    continue

                alpha_orig = compute_grad_input_attribution(
                    clip_model, trans_encoder,
                    text_orig[i:i+1], target[i:i+1], K=args.attribution_K,
                    valid_len=m_tokens_len[i:i+1],
                    create_graph=args.attribution_create_graph)
                alpha_orig = normalize_attribution(alpha_orig, args.tau)
                alpha_entropy_vals.append(
                    -(alpha_orig * (alpha_orig + 1e-8).log()).sum(dim=-1).mean())

                alpha_pair = compute_grad_input_attribution(
                    clip_model, trans_encoder,
                    text_pair[i:i+1], target[i:i+1], K=args.attribution_K,
                    valid_len=m_tokens_len[i:i+1],
                    create_graph=args.attribution_create_graph)
                alpha_pair = normalize_attribution(alpha_pair, args.tau)
                alpha_entropy_vals.append(
                    -(alpha_pair * (alpha_pair + 1e-8).log()).sum(dim=-1).mean())

                mask_i = slot_mask[i].unsqueeze(0)  # (1, 77) bool

                if args.lambda_diff != 0:
                    loss_diff_val = loss_diff_val + loss_attribution_diff(
                        alpha_orig, alpha_pair, mask_i)
                if args.lambda_loc != 0:
                    loss_loc_val = loss_loc_val + loss_locality(
                        alpha_orig, alpha_pair, mask_i)

        # Batch-level var/cov (VICReg requires >= 2 samples)
        loss_var_val = torch.tensor(0., device='cuda')
        loss_cov_val = torch.tensor(0., device='cuda')
        u_std_mean = torch.tensor(0., device='cuda')
        u_norm_mean = torch.tensor(0., device='cuda')
        cos_u_mean = torch.tensor(0., device='cuda')
        if n_preserve >= 2 and len(u_orig_list) >= 2:
            u_orig_cat = torch.cat(u_orig_list, dim=0)
            u_pair_cat = torch.cat(u_pair_list, dim=0)
            u_all = torch.cat([u_orig_cat, u_pair_cat], dim=0)
            loss_var_val = loss_variance(u_all, args.var_threshold)
            loss_cov_val = loss_covariance(u_all)
            u_std_mean = u_all.std(dim=0).mean()
            u_norm_mean = u_all.norm(dim=-1).mean()
            cos_u_mean = F.cosine_similarity(u_orig_cat, u_pair_cat,
                                             dim=-1).mean()

        # 归一化
        if n_preserve > 0:
            loss_align_val = loss_align_val / n_preserve
            total_loss = total_loss + (
                args.lambda_align * loss_align_val +
                args.lambda_var * loss_var_val +
                args.lambda_cov * loss_cov_val)

        if n_alter > 0:
            loss_diff_val = loss_diff_val / n_alter
            loss_loc_val = loss_loc_val / n_alter
            total_loss = total_loss + (
                args.lambda_diff * loss_diff_val +
                args.lambda_loc * loss_loc_val)

        # ---- Backward ----
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # ---- Logging ----
        avg_loss['gen'] += loss_gen.item()
        avg_loss['teach'] += loss_teach.item()
        avg_loss['align'] += loss_align_val.item()
        avg_loss['var'] += loss_var_val.item()
        avg_loss['cov'] += loss_cov_val.item()
        avg_loss['diff'] += loss_diff_val.item()
        avg_loss['loc'] += loss_loc_val.item()
        avg_loss['total'] += total_loss.item()
        avg_loss['u_std'] += u_std_mean.item()
        avg_loss['u_norm'] += u_norm_mean.item()
        avg_loss['cos_u'] += cos_u_mean.item()
        if alpha_entropy_vals:
            alpha_entropy = torch.stack(alpha_entropy_vals).mean()
            avg_loss['alpha_entropy'] += alpha_entropy.item()
        edited_count = slot_mask.sum(dim=-1).float()
        avg_loss['mask_zero'] += (edited_count == 0).float().mean().item()
        avg_loss['mask_count'] += edited_count.mean().item()

        nb_iter += 1

        if nb_iter % args.print_iter == 0:
            for k in avg_loss:
                avg_loss[k] /= args.print_iter

            writer.add_scalar('./Loss/gen', avg_loss['gen'], nb_iter)
            writer.add_scalar('./Loss/teach', avg_loss['teach'], nb_iter)
            writer.add_scalar('./Loss/align', avg_loss['align'], nb_iter)
            writer.add_scalar('./Loss/var', avg_loss['var'], nb_iter)
            writer.add_scalar('./Loss/cov', avg_loss['cov'], nb_iter)
            writer.add_scalar('./Loss/diff', avg_loss['diff'], nb_iter)
            writer.add_scalar('./Loss/loc', avg_loss['loc'], nb_iter)
            writer.add_scalar('./Loss/total', avg_loss['total'], nb_iter)
            writer.add_scalar('./Diag/u_std_mean', avg_loss['u_std'], nb_iter)
            writer.add_scalar('./Diag/u_norm_mean', avg_loss['u_norm'], nb_iter)
            writer.add_scalar('./Diag/cos_u_orig_pair', avg_loss['cos_u'], nb_iter)
            writer.add_scalar('./Diag/alpha_entropy', avg_loss['alpha_entropy'], nb_iter)
            writer.add_scalar('./Diag/edited_mask_zero_ratio', avg_loss['mask_zero'], nb_iter)
            writer.add_scalar('./Diag/edited_mask_count', avg_loss['mask_count'], nb_iter)
            writer.add_scalar('./Stats/n_preserve', n_preserve, nb_iter)
            writer.add_scalar('./Stats/n_alter', n_alter, nb_iter)
            if wandb_run is not None:
                wandb_run.log({
                    'train/loss_gen': avg_loss['gen'],
                    'train/loss_teach': avg_loss['teach'],
                    'train/loss_align': avg_loss['align'],
                    'train/loss_var': avg_loss['var'],
                    'train/loss_cov': avg_loss['cov'],
                    'train/loss_diff': avg_loss['diff'],
                    'train/loss_loc': avg_loss['loc'],
                    'train/loss_total': avg_loss['total'],
                    'diag/u_std_mean': avg_loss['u_std'],
                    'diag/u_norm_mean': avg_loss['u_norm'],
                    'diag/cos_u_orig_pair': avg_loss['cos_u'],
                    'diag/alpha_entropy': avg_loss['alpha_entropy'],
                    'diag/edited_mask_zero_ratio': avg_loss['mask_zero'],
                    'diag/edited_mask_count': avg_loss['mask_count'],
                    'stats/n_preserve': n_preserve,
                    'stats/n_alter': n_alter,
                    'lr/trans': optimizer.param_groups[0]['lr'],
                }, step=nb_iter)

            msg = (f"Train. Iter {nb_iter}: "
                   f"Total {avg_loss['total']:.4f} | "
                   f"Gen {avg_loss['gen']:.4f} | "
                   f"Teach {avg_loss['teach']:.4f} | "
                   f"Align {avg_loss['align']:.4f} | "
                   f"Var {avg_loss['var']:.4f} | "
                   f"Cov {avg_loss['cov']:.4f} | "
                   f"Diff {avg_loss['diff']:.4f} | "
                   f"Loc {avg_loss['loc']:.4f} | "
                   f"UStd {avg_loss['u_std']:.4f} | "
                   f"UNorm {avg_loss['u_norm']:.4f} | "
                   f"CosU {avg_loss['cos_u']:.4f} | "
                   f"AlphaH {avg_loss['alpha_entropy']:.4f} | "
                   f"Mask0 {avg_loss['mask_zero']:.3f} | "
                   f"MaskCnt {avg_loss['mask_count']:.2f} | "
                   f"p/a={n_preserve}/{n_alter}")
            logger.info(msg)

            for k in avg_loss:
                avg_loss[k] = 0.

        if args.save_iter > 0 and nb_iter % args.save_iter == 0:
            last_path = os.path.join(args.out_dir, 'sato_pp_last.pth')
            save_sato_pp_checkpoint(
                last_path, nb_iter, args, clip_model, trans_encoder,
                attr_pooling, predictor_head, optimizer, scheduler,
                best_fid, best_iter, best_div, best_top1, best_top2,
                best_top3, best_matching)
            logger.info(f"Saved full SATO++ checkpoint: {last_path}")

            if (args.milestone_save_iter > 0 and
                    nb_iter % args.milestone_save_iter == 0):
                milestone_path = os.path.join(
                    args.out_dir, f'sato_pp_iter_{nb_iter}.pth')
                save_sato_pp_checkpoint(
                    milestone_path, nb_iter, args, clip_model, trans_encoder,
                    attr_pooling, predictor_head, optimizer, scheduler,
                    best_fid, best_iter, best_div, best_top1, best_top2,
                    best_top3, best_matching)
                logger.info(f"Saved full SATO++ milestone: {milestone_path}")

        if nb_iter % args.eval_iter == 0:
            prev_best_fid = best_fid
            best_fid, best_iter, best_div, best_top1, best_top2, best_top3, \
                best_matching, writer, logger = eval_trans.evaluation_transformer(
                    args.out_dir, val_loader, net, trans_encoder, logger, writer,
                    nb_iter, best_fid, best_iter, best_div,
                    best_top1, best_top2, best_top3, best_matching,
                    clip_model=clip_model, eval_wrapper=eval_wrapper,
                    save=False)
            if wandb_run is not None:
                wandb_run.log({
                    'eval/best_fid': best_fid,
                    'eval/best_iter': best_iter,
                    'eval/best_diversity': best_div,
                    'eval/best_top1': best_top1,
                    'eval/best_top2': best_top2,
                    'eval/best_top3': best_top3,
                    'eval/best_matching': best_matching,
                }, step=nb_iter)

            if best_fid < prev_best_fid:
                best_path = os.path.join(args.out_dir, 'sato_pp_best_fid.pth')
                save_sato_pp_checkpoint(
                    best_path, nb_iter, args, clip_model, trans_encoder,
                    attr_pooling, predictor_head, optimizer, scheduler,
                    best_fid, best_iter, best_div, best_top1, best_top2,
                    best_top3, best_matching)
                logger.info(
                    f"Saved full SATO++ best-FID checkpoint: {best_path}")

        if nb_iter == args.total_iter:
            final_path = os.path.join(args.out_dir, 'sato_pp_final.pth')
            save_sato_pp_checkpoint(
                final_path, nb_iter, args, clip_model, trans_encoder,
                attr_pooling, predictor_head, optimizer, scheduler,
                best_fid, best_iter, best_div, best_top1, best_top2,
                best_top3, best_matching)
            logger.info(f"Saved exact final SATO++ checkpoint: {final_path}")
            msg_final = (f"Best. Iter {best_iter}: "
                         f"FID {best_fid:.5f}, Div {best_div:.4f}, "
                         f"Top1 {best_top1:.4f}, Top2 {best_top2:.4f}, "
                         f"Top3 {best_top3:.4f}")
            logger.info(msg_final)
            break

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
