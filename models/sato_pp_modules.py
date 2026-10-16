"""
SATO++ Representation Modules

- AttributionPooling: u(x) = Σ α_i φ(h_i)
- PredictorHead: z(x) = q(u(x))，仅用于 preserving 对齐 (SimSiam 风格)
"""

import torch
import torch.nn as nn


class AttributionPooling(nn.Module):
    """
    u(x) = Σ_i α_i · φ(h_i)

    将 token-level features 通过归因权重加权求和，得到主语义表示。

    Args:
        clip_dim: CLIP transformer width (ViT-B/32 为 768)
        proj_dim: 输出维度
        proj_type: 'linear' 或 'mlp'
    """

    def __init__(self, clip_dim=768, proj_dim=512, proj_type='linear'):
        super().__init__()
        self.clip_dim = clip_dim
        self.proj_dim = proj_dim

        if proj_type == 'linear':
            self.phi = nn.Linear(clip_dim, proj_dim)
        elif proj_type == 'mlp':
            self.phi = nn.Sequential(
                nn.Linear(clip_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim)
            )
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

    def forward(self, token_features, alpha):
        """
        Args:
            token_features: (bs, L, clip_dim) CLIP token-level features
            alpha: (bs, L) 归因权重 (已归一化)

        Returns:
            u: (bs, proj_dim) 主语义表示
        """
        projected = self.phi(token_features)  # (bs, L, proj_dim)
        alpha = alpha.unsqueeze(-1)  # (bs, L, 1)
        u = (alpha * projected).sum(dim=1)  # (bs, proj_dim)
        return u


class PredictorHead(nn.Module):
    """
    z(x) = q(u(x))

    仅用于 preserving 对齐的 predictor head。
    SimSiam 风格：一边过 predictor，另一边 stop-gradient。

    Args:
        dim: 输入输出维度
    """

    def __init__(self, dim=512):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, u):
        """
        Args:
            u: (bs, dim) 主语义表示

        Returns:
            z: (bs, dim) predictor 输出
        """
        return self.q(u)
