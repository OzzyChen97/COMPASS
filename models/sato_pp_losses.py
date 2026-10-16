"""
SATO++ Loss Functions

Preserving 分支:
- loss_align: SimSiam stop-gradient 对齐
- loss_variance: VICReg 方差项 (anti-collapse)
- loss_covariance: VICReg 协方差项 (去冗余)

Altering 分支:
- loss_attribution_diff: 被编辑 slot 归因差异 (应增大)
- loss_locality: 未编辑 slot 归因稳定性 (应不变)

Teacher:
- loss_teacher: KL(student || teacher)

SATO Original (对比基线):
- loss_topk_attention: SATO 的 L_Topk
"""

import torch
import torch.nn.functional as F


# ========== Preserving Losses ==========

def loss_align(z1, z2):
    """
    SimSiam 风格负余弦相似度。
    z2 侧 stop-gradient，防止 collapse loop。

    Args:
        z1: (bs, dim) predictor 输出
        z2: (bs, dim) 另一侧表示 (会被 detach)

    Returns:
        scalar loss (越小越好)
    """
    z2 = z2.detach()
    return -F.cosine_similarity(z1, z2, dim=-1).mean()


def loss_variance(z, threshold=1.0):
    """
    VICReg variance term: batch 内每个维度的标准差必须 >= threshold。
    防止所有样本映射到同一个点 (collapse)。

    Args:
        z: (bs, dim)
        threshold: 标准差下界

    Returns:
        scalar loss
    """
    std_z = torch.sqrt(z.var(dim=0) + 1e-6)  # (dim,)
    return F.relu(threshold - std_z).mean()


def loss_covariance(z):
    """
    VICReg / Barlow Twins covariance term:
    去冗余，非对角线元素趋近 0。

    Args:
        z: (bs, dim)

    Returns:
        scalar loss
    """
    bs = z.shape[0]
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (bs - 1)  # (dim, dim)

    # 取 off-diagonal
    d = cov.shape[0]
    off_diag = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
    return (off_diag ** 2).sum() / d


# ========== Altering Losses ==========

def loss_attribution_diff(alpha_orig, alpha_pert, edited_mask):
    """
    被编辑 slot 的归因应该变化。最大化差异 → 取负。

    Args:
        alpha_orig: (bs, L) 原句归因
        alpha_pert: (bs, L) 扰动句归因
        edited_mask: (bs, L) bool，哪些 token 属于被编辑 slot

    Returns:
        scalar loss (越小 = 差异越大)
    """
    diff = (alpha_orig - alpha_pert).abs()
    edited_diff = diff[edited_mask]
    if edited_diff.numel() == 0:
        return torch.tensor(0.0, device=diff.device)
    return -edited_diff.mean()


def loss_locality(alpha_orig, alpha_pert, edited_mask):
    """
    未编辑 slot 的归因应该保持不变。最小化差异。

    Args:
        alpha_orig: (bs, L)
        alpha_pert: (bs, L)
        edited_mask: (bs, L) bool

    Returns:
        scalar loss
    """
    diff = (alpha_orig - alpha_pert).abs()
    non_edited_diff = diff[~edited_mask]
    if non_edited_diff.numel() == 0:
        return torch.tensor(0.0, device=diff.device)
    return non_edited_diff.mean()


# ========== Teacher Loss ==========

def loss_teacher(student_logits, teacher_logits, valid_len=None, loss_type='kl'):
    """
    KL divergence: KL(teacher || student)，teacher 是冻结的。

    Args:
        student_logits: (bs, L, vocab) 学生模型输出
        teacher_logits: (bs, L, vocab) 教师模型输出 (已 detach)
        valid_len: (bs,) 每条序列参与 distill 的有效长度；None 保持旧行为
        loss_type: 'kl' 或 'l1'

    Returns:
        scalar loss
    """
    teacher_logits = teacher_logits.detach()
    if loss_type == 'kl':
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        log_student = F.log_softmax(student_logits, dim=-1)
        loss_by_token = F.kl_div(log_student, teacher_probs,
                                 reduction='none').sum(dim=-1)
    elif loss_type == 'l1':
        loss_by_token = F.l1_loss(student_logits, teacher_logits,
                                  reduction='none').mean(dim=-1)
    else:
        raise ValueError(f'Unsupported teacher loss type: {loss_type}')

    if valid_len is None:
        return loss_by_token.sum() / loss_by_token.shape[0]

    seq_len = loss_by_token.shape[1]
    valid_len = valid_len.to(device=loss_by_token.device).long().clamp(min=0, max=seq_len)
    mask = torch.arange(seq_len, device=loss_by_token.device).unsqueeze(0) < valid_len.unsqueeze(1)
    return (loss_by_token * mask).sum() / mask.sum().clamp_min(1)


# ========== SATO Original (对比基线) ==========

def loss_topk_attention(omega_orig, omega_pert, k=5):
    """
    SATO 的 L_Topk (Eq.3): 交叉距离。

    Args:
        omega_orig: (bs, n) 原始注意力权重
        omega_pert: (bs, n) 扰动后注意力权重
        k: top-k

    Returns:
        scalar loss
    """
    _, idx_o = omega_orig.topk(k, dim=-1)  # (bs, k)
    _, idx_p = omega_pert.topk(k, dim=-1)  # (bs, k)

    # 交叉距离
    val_o_at_o = omega_orig.gather(1, idx_o)
    val_p_at_o = omega_pert.gather(1, idx_o)
    val_p_at_p = omega_pert.gather(1, idx_p)
    val_o_at_p = omega_orig.gather(1, idx_p)

    d1 = (val_o_at_o - val_p_at_o).abs().sum()
    d2 = (val_p_at_p - val_o_at_p).abs().sum()

    return (d1 + d2) / (2 * k)
