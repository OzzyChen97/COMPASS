"""
SATO++ Pipeline Quick Test (v2)

跳过 motion tokenization（数据维度问题），用已有 token 或 dummy 数据。
只跑 3 个 iteration 验证整个 SATO++ 训练流程。
"""

import os
import sys
import copy
import json
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clip
from options import option_sato_pp as option_sato_pp
import models.vqvae as vqvae
import models.t2m_trans as trans

from models.clip_attribution import (
    compute_grad_input_attribution,
    normalize_attribution
)
from models.sato_pp_modules import AttributionPooling, PredictorHead
from models.sato_pp_losses import (
    loss_align, loss_variance, loss_covariance,
    loss_attribution_diff, loss_locality,
    loss_teacher
)

import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    # ======== Args ========
    test_args = [
        '--dataname', 't2m',
        '--resume-pth', 'pretrained/VQVAE/net_best_fid.pth',
        '--resume-trans', 'pretrained/VQTransformer_corruption05/net_best_fid.pth',
        '--pair-data-path', 'dataset/HumanML3D/preserving_pairs_filtered.jsonl',
        '--exp-name', 'test_sato_pp_v2',
        '--batch-size', '4',
        '--total-iter', '3',
        '--print-iter', '1',
        '--eval-iter', '100',
        '--vq-name', 'VQVAE',
        '--out-dir', 'output_GPT_Final/',
        '--down-t', '2',
        '--block-size', '51',
        '--embed-dim-gpt', '1024',
        '--num-layers', '9',
        '--n-head-gpt', '8',
    ]
    sys.argv = [sys.argv[0]] + test_args

    args = option_sato_pp.get_sato_pp_args()
    torch.manual_seed(args.seed)
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== SATO++ Pipeline Quick Test (v2) ===")
    print(f"clip_dim={args.clip_dim}, clip_dim_attr={args.clip_dim_attr}, embed_dim_gpt={args.embed_dim_gpt}")

    # ======== CLIP ========
    print("Loading CLIP...")
    clip_model, _ = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)
    clip.model.convert_weights(clip_model)
    for p in clip_model.parameters():
        p.requires_grad = True
    print("CLIP loaded & unfrozen")

    # ======== VQ-VAE (skip loading — data mismatch, not needed for pipeline test) ========
    print("Skipping VQ-VAE (data dim mismatch, not needed for pipeline test)")

    # ======== Transformer ========
    print("Loading Transformer...")
    trans_encoder = trans.Text2Motion_Transformer(
        num_vq=args.nb_code, embed_dim=args.embed_dim_gpt,
        clip_dim=args.clip_dim, block_size=args.block_size,
        num_layers=args.num_layers, n_head=args.n_head_gpt,
        drop_out_rate=args.drop_out_rate, fc_rate=args.ff_rate)

    ckpt_trans = torch.load(args.resume_trans, map_location='cpu', weights_only=False)
    trans_encoder.load_state_dict(ckpt_trans['trans'], strict=True)
    trans_encoder.train()
    trans_encoder.cuda()
    print("Transformer loaded")

    # ======== Teacher ========
    teacher_trans = copy.deepcopy(trans_encoder)
    for p in teacher_trans.parameters():
        p.requires_grad = False
    teacher_trans.eval()
    print("Teacher created")

    # ======== SATO++ modules ========
    attr_pooling = AttributionPooling(
        clip_dim=args.clip_dim_attr, proj_dim=args.proj_dim,
        proj_type=args.proj_type).cuda()
    predictor_head = PredictorHead(dim=args.proj_dim).cuda()
    print(f"AttributionPooling(clip_dim={args.clip_dim_attr}→{args.proj_dim}), PredictorHead created")

    # ======== Optimizer ========
    param_groups = [
        {'params': trans_encoder.parameters(), 'lr': args.lr},
        {'params': attr_pooling.parameters(), 'lr': args.lr},
        {'params': predictor_head.parameters(), 'lr': args.lr},
        {'params': clip_model.parameters(), 'lr': args.clip_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    loss_ce = torch.nn.CrossEntropyLoss()
    print("Optimizer ready")

    # ======== Load pair data ========
    pair_data = []
    with open(args.pair_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            pair_data.append(json.loads(line.strip()))
    print(f"Loaded {len(pair_data)} pairs")

    # Build caption lookup
    caption_pairs = {}
    for p in pair_data:
        orig = p.get('text_orig', '')
        if orig not in caption_pairs:
            caption_pairs[orig] = []
        caption_pairs[orig].append(p)

    # ======== Training Loop with synthetic data ========
    print("\n=== Starting training (3 iterations) ===")

    # Use some real captions from pair data
    captions = list(caption_pairs.keys())[:20]

    nb_iter = 0
    while nb_iter < 3:
        # Build synthetic batch
        batch_captions_orig = []
        batch_captions_pair = []
        batch_relations = []

        for i in range(args.batch_size):
            cap = captions[i % len(captions)]
            pairs = caption_pairs[cap]
            pair = pairs[i % len(pairs)]
            batch_captions_orig.append(cap)
            batch_captions_pair.append(pair['text_pair'])
            batch_relations.append(pair.get('relation', 'preserve'))

        bs = args.batch_size

        # Create synthetic motion tokens
        m_tokens_len = torch.randint(10, 30, (bs,)).cuda()
        max_len = m_tokens_len.max().item() + 1
        target = torch.randint(0, args.nb_code, (bs, max_len)).cuda()
        for i in range(bs):
            target[i, m_tokens_len[i]] = args.nb_code  # end token

        print(f"\n--- Iter {nb_iter+1} ---")
        print(f"  relations: {batch_relations}")
        print(f"  orig[0]: {batch_captions_orig[0][:60]}...")
        print(f"  pair[0]: {batch_captions_pair[0][:60]}...")

        # Encode text
        text_orig = clip.tokenize(batch_captions_orig, truncate=True).cuda()
        text_pair = clip.tokenize(batch_captions_pair, truncate=True).cuda()

        H_orig, feat_orig = clip_model.encode_text_with_features(text_orig)
        H_pair, feat_pair = clip_model.encode_text_with_features(text_pair)
        feat_orig = feat_orig.float()
        feat_pair = feat_pair.float()
        H_orig = H_orig.float()
        H_pair = H_pair.float()

        print(f"  H_orig: {H_orig.shape}, feat_orig: {feat_orig.shape}")

        # Generation loss
        input_index = target[:, :-1]
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

        # Teacher loss
        with torch.no_grad():
            teacher_pred = teacher_trans(a_indices, feat_orig)
        loss_teach = loss_teacher(cls_pred, teacher_pred)

        total_loss = loss_gen + args.lambda_teach * loss_teach
        print(f"  loss_gen={loss_gen.item():.4f}, loss_teach={loss_teach.item():.4f}")

        # Relation-conditioned losses
        loss_align_val = torch.tensor(0., device='cuda')
        loss_var_val = torch.tensor(0., device='cuda')
        loss_cov_val = torch.tensor(0., device='cuda')
        loss_diff_val = torch.tensor(0., device='cuda')
        loss_loc_val = torch.tensor(0., device='cuda')

        n_preserve = 0
        n_alter = 0

        for i in range(bs):
            rel = batch_relations[i]

            if rel == 'preserve':
                n_preserve += 1

                alpha_orig = compute_grad_input_attribution(
                    clip_model, trans_encoder,
                    text_orig[i:i+1], target[i:i+1], K=args.attribution_K)
                alpha_orig = normalize_attribution(alpha_orig, args.tau)

                alpha_pair = compute_grad_input_attribution(
                    clip_model, trans_encoder,
                    text_pair[i:i+1], target[i:i+1], K=args.attribution_K)
                alpha_pair = normalize_attribution(alpha_pair, args.tau)

                print(f"    [{i}] preserve: alpha shape={alpha_orig.shape}, sum={alpha_orig.sum().item():.4f}")

                u_orig = attr_pooling(H_orig[i:i+1], alpha_orig)
                u_pair = attr_pooling(H_pair[i:i+1], alpha_pair)

                z_orig = predictor_head(u_orig)
                z_pair = predictor_head(u_pair)

                la = loss_align(z_orig, z_pair)
                lv = (loss_variance(u_orig, args.var_threshold) +
                       loss_variance(u_pair, args.var_threshold)) / 2
                lc = (loss_covariance(u_orig) + loss_covariance(u_pair)) / 2

                loss_align_val = loss_align_val + la
                loss_var_val = loss_var_val + lv
                loss_cov_val = loss_cov_val + lc

                print(f"    [{i}] align={la.item():.4f}, var={lv.item():.4f}, cov={lc.item():.4f}")

            elif rel == 'alter':
                n_alter += 1

                alpha_orig = compute_grad_input_attribution(
                    clip_model, trans_encoder,
                    text_orig[i:i+1], target[i:i+1], K=args.attribution_K)
                alpha_orig = normalize_attribution(alpha_orig, args.tau)

                alpha_pair = compute_grad_input_attribution(
                    clip_model, trans_encoder,
                    text_pair[i:i+1], target[i:i+1], K=args.attribution_K)
                alpha_pair = normalize_attribution(alpha_pair, args.tau)

                # Use first 10 tokens as "edited" for synthetic test
                slot_mask = torch.zeros(77, dtype=torch.bool, device='cuda')
                slot_mask[:10] = True

                ld = loss_attribution_diff(alpha_orig, alpha_pair, slot_mask)
                ll = loss_locality(alpha_orig, alpha_pair, slot_mask)

                loss_diff_val = loss_diff_val + ld
                loss_loc_val = loss_loc_val + ll

                print(f"    [{i}] alter: diff={ld.item():.4f}, loc={ll.item():.4f}")

        # Aggregate
        if n_preserve > 0:
            loss_align_val = loss_align_val / n_preserve
            loss_var_val = loss_var_val / n_preserve
            loss_cov_val = loss_cov_val / n_preserve
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

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"  TOTAL loss={total_loss.item():.4f} (preserve={n_preserve}, alter={n_alter})")
        print(f"  Iter {nb_iter+1} PASSED ✓")

        nb_iter += 1

    print("\n=== ALL 3 ITERATIONS COMPLETED ===")
    print("SATO++ pipeline test PASSED!")


if __name__ == '__main__':
    main()
