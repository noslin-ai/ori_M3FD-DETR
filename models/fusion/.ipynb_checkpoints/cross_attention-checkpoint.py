"""Cross Modal Attention — 跨模态注意力机制。

核心创新: RGB 作为 Query，IR + Depth 作为 Key/Value。
让 RGB 语义主干主动查询互补模态信息。
"""

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """多头交叉注意力。

    Query: RGB 特征
    Key / Value: IR + Depth 融合特征
    """

    def __init__(self, dim, heads=8):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, query, kv):
        """
        Args:
            query: (B, N, dim) RGB 特征作为 Query
            kv:    (B, M, dim) IR + Depth 特征作为 Key/Value

        Returns:
            (B, N, dim) 交叉注意力增强后的特征
        """
        out, _ = self.attn(query, kv, kv)
        return out
