"""
SATO++ End-to-End Pipeline Debug Script

Steps:
1. Tokenize motions with VQVAE
2. Load pair data
3. Run SATOPPDataset
4. Run a few training iterations
5. Verify all losses flow

Usage:
    python debug_pipeline.py
"""

import os
import sys
import copy
import json
import torch
import numpy as np
import traceback

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Patch: use clip_local before importing clip
import clip

from options import option_sato_pp as option_sato_pp
import models.vqvae as vqvae
import models.t2m_trans as trans
import utils.utils_model as utils_model
from dataset import dataset_sato_pp
from dataset import dataset_TM_eval
from dataset import dataset_tokenize

from models.clip_attribution import (
    compute_grad_input_attribution,
    compute_ar_score,
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

def debug_step(name, fn):
    """Run a debug step with clear output."""
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    try:
        result = fn()
        print(f"  ✓ {name} PASSED")
        return result
    except Exception as e:
        print(f"  ✗ {name} FAILED: {e}")
        traceback.print_exc()
        return None


def step1_tokenize_motions(args, net):
    """Tokenize motions using VQVAE."""
    os.makedirs(args.vq_dir, exist_ok=True)

    train_loader_token = dataset_tokenize.DATALoader(
        args.dataname, 1, unit_length=2 ** args.down_t)

    count = 0
    for batch in train_loader_token:
        pose, name = batch
        pose = pose.cuda().float()
        target = net.encode(pose)
        target = target.cpu().numpy()
        np.save(pjoin(args.vq_dir, name[0] + '.npy'), target)
        count += 1
        if count % 5000 == 0:
            print(f"  Tokenized {count} motions...")

    print(f"  Total tokenized: {count}")
    return count


def step2_load_data(args):
    """Test dataset loading."""
    train_loader = dataset_sato_pp.DATALoader(
        args.dataname, args.batch_size, args.nb_code, args.vq_name,
        pair_data_path=args.pair_data_path,
        unit_length=2 ** args.down_t,
        preserve_ratio=0.7)

    batch = next(iter(train_loader))
    cap_orig, cap_pair, m_tokens, m_tokens_len, relation, edit_type, slot_mask = batch

    print(f"  Batch size: {len(cap_orig)}")
    print(f"  m_tokens shape: {m_tokens.shape}, dtype: {m_tokens.dtype}")
    print(f"  m_tokens_len: {m_tokens_len}")
    print(f"  relations: {relation}")
    print(f"  slot_mask shape: {slot_mask.shape}, dtype: {slot_mask.dtype}")
    print(f"  sample orig: {cap_orig[0][:80]}")
    print(f"  sample pair: {cap_pair[0][:80]}")

    return train_loader


def step3_clip_features(clip_model, cap_orig, cap_pair):
    """Test CLIP feature extraction."""
    text_orig = clip.tokenize(list(cap_orig), truncate=True).cuda()
    text_pair = clip.tokenize(list(cap_pair), truncate=True).cuda()

    print(f"  text_orig shape: {text_orig.shape}")

    H_orig, feat_orig = clip_model.encode_text_with_features(text_orig)
    H_pair, feat_pair = clip_model.encode_text_with_features(text_pair)

    feat_orig = feat_orig.float()
    feat_pair = feat_pair.float()
    H_orig = H_orig.float()
    H_pair = H_pair.float()

    print(f"  H_orig: {H_orig.shape}, feat_orig: {feat_orig.shape}")
    print(f"  H_orig range: [{H_orig.min():.4f}, {H_orig.max():.4f}]")
    print(f"  feat_orig range: [{feat_orig.min():.4f}, {feat_orig.max():.4f}]")

    return text_orig, text_pair, H_orig, feat_orig, H_pair, feat_pair


def step4_generation_loss(trans_encoder, feat_orig, m_tokens, m_tokens_len, args, loss_ce):
    """Test generation loss."""
    target = m_tokens.cuda()
    bs = target.shape[0]
    input_index = target[:, :-1]

    mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape,
                                                     device=input_index.device))
    mask = mask.round().to(dtype=torch.int64)
    r_indices = torch.randint_like(input_index, args.nb_code)
    a_indices = mask * input_index + (1 - mask) * r_indices

    cls_pred = trans_encoder(a_indices, feat_orig)
    cls_pred = cls_pred.contiguous()

    print(f"  cls_pred shape: {cls_pred.shape}")
    print(f"  a_indices shape: {a_indices.shape}")

    loss_gen = 0.0
    for i in range(bs):
        loss_gen += loss_ce(
            cls_pred[i][:m_tokens_len[i] + 1],
            target[i][:m_tokens_len[i] + 1]) / bs

    print(f"  loss_gen: {loss_gen.item():.4f}")
    return loss_gen, cls_pred, a_indices


def step5_teacher_loss(cls_pred, teacher_trans, a_indices, feat_orig):
    """Test teacher loss."""
    with torch.no_grad():
        teacher_pred = teacher_trans(a_indices, feat_orig)
    loss_teach = loss_teacher(cls_pred, teacher_pred)
    print(f"  loss_teach: {loss_teach.item():.4f}")
    return loss_teach


def step6_attribution(clip_model, trans_encoder, text_tokens, m_tokens, K):
    """Test attribution computation."""
    target = m_tokens.cuda()
    alpha = compute_grad_input_attribution(
        clip_model, trans_encoder, text_tokens, target, K=K,
        valid_len=m_tokens_len)
    alpha = normalize_attribution(alpha, temperature=1.0)

    print(f"  alpha shape: {alpha.shape}")
    print(f"  alpha sum: {alpha.sum(dim=-1)}")
    print(f"  alpha range: [{alpha.min():.6f}, {alpha.max():.6f}]")
    print(f"  alpha top-5 tokens: {alpha[0].topk(5)}")
    return alpha


def step7_preserving_loss(attr_pooling, predictor_head,
                          H_orig, alpha_orig, H_pair, alpha_pair):
    """Test preserving losses."""
    u_orig = attr_pooling(H_orig, alpha_orig)
    u_pair = attr_pooling(H_pair, alpha_pair)
    z_orig = predictor_head(u_orig)
    z_pair = predictor_head(u_pair)

    print(f"  u_orig: {u_orig.shape}, range: [{u_orig.min():.4f}, {u_orig.max():.4f}]")
    print(f"  z_orig: {z_orig.shape}")

    la = loss_align(z_orig, z_pair)
    lv = (loss_variance(u_orig, 1.0) + loss_variance(u_pair, 1.0)) / 2
    lc = (loss_covariance(u_orig) + loss_covariance(u_pair)) / 2

    print(f"  loss_align: {la.item():.4f}")
    print(f"  loss_variance: {lv.item():.4f}")
    print(f"  loss_covariance: {lc.item():.4f}")
    return la, lv, lc


def step8_altering_loss(alpha_orig, alpha_pair, slot_mask):
    """Test altering losses."""
    ld = loss_attribution_diff(alpha_orig, alpha_pair, slot_mask)
    ll = loss_locality(alpha_orig, alpha_pair, slot_mask)

    print(f"  loss_attr_diff: {ld.item():.4f}")
    print(f"  loss_locality: {ll.item():.4f}")
    return ld, ll


def step9_full_training_iteration(clip_model, trans_encoder, teacher_trans,
                                   attr_pooling, predictor_head, optimizer,
                                   train_loader_iter, args, loss_ce):
    """Run 3 full training iterations."""
    for it in range(3):
        print(f"\n  --- Iteration {it+1} ---")
        batch = next(train_loader_iter)
        (cap_orig, cap_pair, m_tokens, m_tokens_len,
         relation, edit_type, slot_mask) = batch

        m_tokens = m_tokens.cuda()
        m_tokens_len = m_tokens_len.cuda()
        slot_mask = slot_mask.cuda()
        bs = m_tokens.shape[0]
        target = m_tokens.cuda()

        print(f"  relations: {relation}")

        # Encode text
        text_orig = clip.tokenize(list(cap_orig), truncate=True).cuda()
        text_pair = clip.tokenize(list(cap_pair), truncate=True).cuda()

        H_orig, feat_orig = clip_model.encode_text_with_features(text_orig)
        H_pair, feat_pair = clip_model.encode_text_with_features(text_pair)
        feat_orig = feat_orig.float()
        feat_pair = feat_pair.float()
        H_orig = H_orig.float()
        H_pair = H_pair.float()

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

        # Relation-conditioned
        n_preserve = 0
        n_alter = 0
        loss_align_v = torch.tensor(0., device='cuda')
        loss_diff_v = torch.tensor(0., device='cuda')
        loss_loc_v = torch.tensor(0., device='cuda')

        # Accumulate u vectors for batch-level var/cov
        u_orig_list = []
        u_pair_list = []

        for i in range(bs):
            rel = relation[i]

            if rel == 'preserve':
                n_preserve += 1
                alpha_orig = compute_grad_input_attribution(
                    clip_model, trans_encoder,
                    text_orig[i:i+1], target[i:i+1], K=args.attribution_K,
                    valid_len=m_tokens_len[i:i+1])
                alpha_orig = normalize_attribution(alpha_orig, args.tau)

                alpha_pair = compute_grad_input_attribution(
                    clip_model, trans_encoder,
                    text_pair[i:i+1], target[i:i+1], K=args.attribution_K,
                    valid_len=m_tokens_len[i:i+1])
                alpha_pair = normalize_attribution(alpha_pair, args.tau)

                u_orig = attr_pooling(H_orig[i:i+1], alpha_orig)
                u_pair = attr_pooling(H_pair[i:i+1], alpha_pair)
                z_orig = predictor_head(u_orig)
                z_pair = predictor_head(u_pair)

                loss_align_v = loss_align_v + loss_align(z_orig, z_pair)
                u_orig_list.append(u_orig)
                u_pair_list.append(u_pair)

            elif rel == 'alter':
                n_alter += 1
                alpha_orig = compute_grad_input_attribution(
                    clip_model, trans_encoder,
                    text_orig[i:i+1], target[i:i+1], K=args.attribution_K,
                    valid_len=m_tokens_len[i:i+1])
                alpha_orig = normalize_attribution(alpha_orig, args.tau)

                alpha_pair = compute_grad_input_attribution(
                    clip_model, trans_encoder,
                    text_pair[i:i+1], target[i:i+1], K=args.attribution_K,
                    valid_len=m_tokens_len[i:i+1])
                alpha_pair = normalize_attribution(alpha_pair, args.tau)

                mask_i = slot_mask[i]
                loss_diff_v = loss_diff_v + loss_attribution_diff(
                    alpha_orig, alpha_pair, mask_i)
                loss_loc_v = loss_loc_v + loss_locality(
                    alpha_orig, alpha_pair, mask_i)

        # Batch-level var/cov (need >= 2 samples)
        loss_var_v = torch.tensor(0., device='cuda')
        loss_cov_v = torch.tensor(0., device='cuda')
        if n_preserve >= 2 and len(u_orig_list) >= 2:
            u_all = torch.cat(u_orig_list + u_pair_list, dim=0)
            loss_var_v = loss_variance(u_all, args.var_threshold)
            loss_cov_v = loss_covariance(u_all)

        if n_preserve > 0:
            loss_align_v = loss_align_v / n_preserve
            total_loss = total_loss + (
                args.lambda_align * loss_align_v +
                args.lambda_var * loss_var_v +
                args.lambda_cov * loss_cov_v)

        if n_alter > 0:
            loss_diff_v = loss_diff_v / n_alter
            loss_loc_v = loss_loc_v / n_alter
            total_loss = total_loss + (
                args.lambda_diff * loss_diff_v +
                args.lambda_loc * loss_loc_v)

        # Backward
        optimizer.zero_grad()

        # Check for NaN before backward
        if torch.isnan(total_loss):
            print(f"  !!! total_loss is NaN before backward !!!")
            print(f"  loss_gen={loss_gen.item():.4f}, loss_teach={loss_teach.item():.4f}")
            break

        total_loss.backward()

        # Check gradient norms
        for name, p in [('trans', trans_encoder),
                         ('attr_pool', attr_pooling),
                         ('pred_head', predictor_head)]:
            total_norm = 0.0
            has_nan = False
            for p_ in p.parameters():
                if p_.grad is not None:
                    total_norm += p_.grad.norm().item() ** 2
                    if torch.isnan(p_.grad).any():
                        has_nan = True
            total_norm = total_norm ** 0.5
            print(f"  grad_norm_{name}={total_norm:.4f}{' !!!NaN!!!' if has_nan else ''}")

        # Clip ALL gradients
        torch.nn.utils.clip_grad_norm_(trans_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(attr_pooling.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor_head.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(clip_model.parameters(), 1.0)
        optimizer.step()

        print(f"  loss_gen={loss_gen.item():.4f}, "
              f"loss_teach={loss_teach.item():.4f}, "
              f"total={total_loss.item():.4f}")
        print(f"  preserve={n_preserve}, alter={n_alter}")
        if n_preserve > 0:
            print(f"  align={loss_align_v.item():.4f}, "
                  f"var={loss_var_v.item():.4f}, "
                  f"cov={loss_cov_v.item():.4f}")
        if n_alter > 0:
            print(f"  diff={loss_diff_v.item():.4f}, "
                  f"loc={loss_loc_v.item():.4f}")

    print("\n  ✓ ALL 3 TRAINING ITERATIONS PASSED")


def main():
    # ======== Args ========
    test_args = [
        '--dataname', 't2m',
        '--resume-pth', 'pretrained/VQVAE/net_best_fid.pth',
        '--resume-trans', 'pretrained/VQTransformer_corruption05/net_best_fid.pth',
        '--pair-data-path', 'dataset/HumanML3D/preserving_pairs_debug.jsonl',
        '--exp-name', 'debug_sato_pp',
        '--batch-size', '4',
        '--total-iter', '100',
        '--print-iter', '10',
        '--eval-iter', '99999',
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
    data_root = "./dataset/HumanML3D"
    args.vq_dir = os.path.join(data_root, args.vq_name)
    os.makedirs(args.out_dir, exist_ok=True)

    print("="*60)
    print("  SATO++ END-TO-END DEBUG")
    print("="*60)
    print(f"  embed_dim_gpt={args.embed_dim_gpt}, clip_dim={args.clip_dim}")
    print(f"  clip_dim_attr={args.clip_dim_attr}, proj_dim={args.proj_dim}")
    print(f"  vq_dir={args.vq_dir}")
    print(f"  pair_data={args.pair_data_path}")

    # ======== Step 1: VQVAE tokenize ========
    print(f"\n{'='*60}")
    print(f"  STEP 1: VQVAE Tokenize")
    print(f"{'='*60}")

    net = vqvae.HumanVQVAE(args, args.nb_code, args.code_dim,
                            args.output_emb_width, args.down_t, args.stride_t,
                            args.width, args.depth, args.dilation_growth_rate)
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net.cuda()
    print("  VQVAE loaded")

    if os.path.exists(args.vq_dir) and len(os.listdir(args.vq_dir)) > 100:
        print(f"  VQ tokens already exist ({len(os.listdir(args.vq_dir))} files), skipping")
    else:
        step1_tokenize_motions(args, net)
        print("  ✓ Tokenization complete")

    # ======== Step 2: Load CLIP ========
    print(f"\n{'='*60}")
    print(f"  STEP 2: Load CLIP")
    print(f"{'='*60}")

    clip_model, _ = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)
    clip.model.convert_weights(clip_model)
    clip_model.float()  # SATO++ fine-tuning: keep float32 to avoid fp16 overflow
    for p in clip_model.parameters():
        p.requires_grad = True
    print("  CLIP loaded, unfrozen & float32")

    # Quick test of encode_text_with_features
    test_text = clip.tokenize(["a person walks forward"], truncate=True).cuda()
    H_test, feat_test = clip_model.encode_text_with_features(test_text)
    print(f"  encode_text_with_features test: H={H_test.shape}, feat={feat_test.shape}")

    # ======== Step 3: Load Transformer ========
    print(f"\n{'='*60}")
    print(f"  STEP 3: Load Transformer")
    print(f"{'='*60}")

    trans_encoder = trans.Text2Motion_Transformer(
        num_vq=args.nb_code, embed_dim=args.embed_dim_gpt,
        clip_dim=args.clip_dim, block_size=args.block_size,
        num_layers=args.num_layers, n_head=args.n_head_gpt,
        drop_out_rate=args.drop_out_rate, fc_rate=args.ff_rate)

    ckpt_trans = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt_trans['trans'], strict=True)
    trans_encoder.train()
    trans_encoder.cuda()
    print("  Transformer loaded")

    # ======== Step 4: Teacher ========
    teacher_trans = copy.deepcopy(trans_encoder)
    for p in teacher_trans.parameters():
        p.requires_grad = False
    teacher_trans.eval()
    print("  Teacher created")

    # ======== Step 5: SATO++ modules ========
    print(f"\n{'='*60}")
    print(f"  STEP 5: SATO++ Modules")
    print(f"{'='*60}")

    attr_pooling = AttributionPooling(
        clip_dim=args.clip_dim_attr, proj_dim=args.proj_dim,
        proj_type=args.proj_type).cuda()
    predictor_head = PredictorHead(dim=args.proj_dim).cuda()
    print(f"  AttributionPooling({args.clip_dim_attr}→{args.proj_dim})")
    print(f"  PredictorHead({args.proj_dim})")

    # ======== Step 6: Dataset ========
    print(f"\n{'='*60}")
    print(f"  STEP 6: Load Dataset")
    print(f"{'='*60}")

    train_loader = step2_load_data(args)
    train_loader_iter = dataset_sato_pp.cycle(train_loader)

    # ======== Step 7: Optimizer ========
    loss_ce = torch.nn.CrossEntropyLoss()
    param_groups = [
        {'params': trans_encoder.parameters(), 'lr': args.lr},
        {'params': attr_pooling.parameters(), 'lr': args.lr},
        {'params': predictor_head.parameters(), 'lr': args.lr},
        {'params': clip_model.parameters(), 'lr': args.clip_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # ======== Step 8: Component tests ========
    print(f"\n{'='*60}")
    print(f"  STEP 8: Component Tests")
    print(f"{'='*60}")

    batch = next(train_loader_iter)
    cap_orig, cap_pair, m_tokens, m_tokens_len, relation, edit_type, slot_mask = batch
    m_tokens = m_tokens.cuda()
    m_tokens_len = m_tokens_len.cuda()
    slot_mask = slot_mask.cuda()
    bs = m_tokens.shape[0]
    target = m_tokens.cuda()

    # 8a. CLIP features
    print("\n  [8a] CLIP features...")
    text_orig = clip.tokenize(list(cap_orig), truncate=True).cuda()
    text_pair = clip.tokenize(list(cap_pair), truncate=True).cuda()
    H_orig, feat_orig = clip_model.encode_text_with_features(text_orig)
    H_pair, feat_pair = clip_model.encode_text_with_features(text_pair)
    feat_orig = feat_orig.float()
    feat_pair = feat_pair.float()
    H_orig = H_orig.float()
    H_pair = H_pair.float()
    print(f"  H_orig: {H_orig.shape}, feat_orig: {feat_orig.shape}")

    # 8b. Generation loss
    print("\n  [8b] Generation loss...")
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
    print(f"  loss_gen: {loss_gen.item():.4f}")

    # 8c. Teacher loss
    print("\n  [8c] Teacher loss...")
    with torch.no_grad():
        teacher_pred = teacher_trans(a_indices, feat_orig)
    loss_teach = loss_teacher(cls_pred, teacher_pred)
    print(f"  loss_teach: {loss_teach.item():.4f}")

    # 8d. Attribution (sample 1 to save time)
    print("\n  [8d] Attribution (sample 0)...")
    alpha_orig_0 = compute_grad_input_attribution(
        clip_model, trans_encoder,
        text_orig[0:1], target[0:1], K=args.attribution_K,
        valid_len=m_tokens_len[0:1])
    alpha_orig_0 = normalize_attribution(alpha_orig_0, args.tau)
    print(f"  alpha shape: {alpha_orig_0.shape}, sum: {alpha_orig_0.sum().item():.4f}")

    # 8e. Attribution pooling
    print("\n  [8e] Attribution pooling...")
    u_orig_0 = attr_pooling(H_orig[0:1], alpha_orig_0)
    print(f"  u shape: {u_orig_0.shape}, norm: {u_orig_0.norm().item():.4f}")

    # 8f. Predictor head
    print("\n  [8f] Predictor head...")
    z_orig_0 = predictor_head(u_orig_0)
    print(f"  z shape: {z_orig_0.shape}")

    print("\n  ✓ ALL COMPONENT TESTS PASSED")

    # ======== Step 9: Full training iterations ========
    print(f"\n{'='*60}")
    print(f"  STEP 9: Full Training Iterations")
    print(f"{'='*60}")

    # Re-init optimizer after component tests consumed gradients
    param_groups = [
        {'params': trans_encoder.parameters(), 'lr': args.lr},
        {'params': attr_pooling.parameters(), 'lr': args.lr},
        {'params': predictor_head.parameters(), 'lr': args.lr},
        {'params': clip_model.parameters(), 'lr': args.clip_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    step9_full_training_iteration(
        clip_model, trans_encoder, teacher_trans,
        attr_pooling, predictor_head, optimizer,
        train_loader_iter, args, loss_ce)

    print(f"\n{'='*60}")
    print(f"  ALL DEBUG STEPS PASSED ✓")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
