"""
SATO++ CLIP Attribution Module

提供:
- AR score 计算 (T2M-GPT 专用)
- Grad×Input 归因
- Attention raw 归因
- 归因归一化
"""

import torch
import torch.nn.functional as F


def configure_second_order_attention(enabled):
    """Use the CUDA attention backend that supports double backward."""
    if not enabled or not torch.cuda.is_available():
        return
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def compute_ar_score(trans_encoder, clip_feat, target_tokens, K=16,
                     valid_len=None):
    """
    T2M-GPT 的 motion-conditioned AR score.

    s_AR(x, y) = (1/K) * sum_{t=1}^{K} log p(y_t | y_{<t}, x)

    Args:
        trans_encoder: Text2Motion_Transformer
        clip_feat: (bs, clip_dim) CLIP pooled text feature
        target_tokens: (bs, L) 离散 motion token 序列（含 end+pad）
        K: 取前 K 步计算 score
        valid_len: (bs,) 每个 sample 的有效 token 长度（含 end token），
                   超出部分是 pad，不参与计算

    Returns:
        score: scalar，前 K 步平均 log-likelihood
    """
    input_index = target_tokens[:, :-1]
    cls_pred = trans_encoder(input_index, clip_feat)  # (bs, L-1, num_vq+1)

    # K 不能超过有效 token 数
    if valid_len is not None:
        max_valid = valid_len.min().item()
        max_K = min(K, max_valid, input_index.shape[1])
    else:
        max_K = min(K, input_index.shape[1])

    total_log_prob = 0.0
    for t in range(max_K):
        log_probs = F.log_softmax(cls_pred[:, t, :], dim=-1)  # (bs, num_vq+1)
        total_log_prob += log_probs.gather(
            1, target_tokens[:, t:t + 1]).squeeze(-1)  # (bs,)

    return total_log_prob.mean() / max_K


def compute_grad_input_attribution(clip_model, trans_encoder,
                                   text_tokens, target_tokens, K=16,
                                   valid_len=None, create_graph=False):
    """
    Grad×Input 归因: a_i = |∂s/∂h_i ⊙ h_i|_1

    对 CLIP 的 token_embedding h_i 求梯度，手动拼接 CLIP forward 后半段。

    Args:
        clip_model: 修改后的 CLIP 模型 (clip_local.model.CLIP)
        trans_encoder: Text2Motion_Transformer
        text_tokens: (bs, L) tokenize 后的文本 token ids
        target_tokens: (bs, max_motion_len) 离散 motion tokens
        K: AR score 步数
        valid_len: (bs,) 有效 token 长度（含 end token）

    Returns:
        attribution: (bs, L) 每个 token 的归因值
    """
    dtype = clip_model.dtype

    # 1. 获取 token embeddings，设置 requires_grad
    h = clip_model.token_embedding(text_tokens).type(dtype)  # (bs, L, width)
    h = h.detach().requires_grad_(True)

    # 2. 手动过 CLIP transformer（与 encode_text_with_features 一致）
    x = h + clip_model.positional_embedding.type(dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(dtype)  # (bs, L, width)

    # 3. EOS pooling + text_projection
    eos_idx = text_tokens.argmax(dim=-1)
    pooled = x[torch.arange(x.shape[0]), eos_idx]  # (bs, width)
    if clip_model.text_projection is not None:
        pooled = pooled @ clip_model.text_projection  # (bs, embed_dim)

    # 4. 过 trans_encoder → 计算 AR score
    score = compute_ar_score(trans_encoder, pooled.float(), target_tokens, K,
                             valid_len=valid_len)

    # 5. 只对 h 求梯度，不污染模型参数的 .grad
    grad_h, = torch.autograd.grad(
        score,
        h,
        retain_graph=create_graph,
        create_graph=create_graph,
    )

    # 6. Grad × Input
    attribution = (grad_h.float() * h.detach().float()).abs().sum(dim=-1)  # (bs, L)
    return attribution


def compute_attention_attn(clip_model, text_tokens):
    """
    用 CLIP 最后一层的 attention weights 做归因（SATO baseline 用）。

    Args:
        clip_model: CLIP 模型
        text_tokens: (bs, L)

    Returns:
        attn_weights: (bs, L) 最后一层对每个 token 的平均 attention
    """
    token_features, pooled, all_attn = clip_model.encode_text_with_attn(text_tokens)

    # 取最后一层 attention
    # all_attn[-1] shape: (bs, L, L) 或 (bs, n_head, L, L)
    last_attn = all_attn[-1]  # (bs, n_head, L, L) 或 (bs, L, L)

    if last_attn.dim() == 4:
        # (bs, n_head, L, L) → 对 head 维度平均
        last_attn = last_attn.mean(dim=1)  # (bs, L, L)

    # EOS token 对所有 token 的 attention
    eos_idx = text_tokens.argmax(dim=-1)
    # 对每个样本，取 eos_idx 行
    bs = text_tokens.shape[0]
    eos_attn = last_attn[torch.arange(bs), eos_idx, :]  # (bs, L)

    return eos_attn.float()


def normalize_attribution(attribution, temperature=1.0):
    """
    归因归一化: ReLU → softmax → 温度 τ

    Args:
        attribution: (bs, L) 原始归因值
        temperature: softmax 温度

    Returns:
        alpha: (bs, L) 归一化后的权重，sum=1
    """
    attr = F.relu(attribution)
    alpha = F.softmax(attr / temperature, dim=-1)
    return alpha
