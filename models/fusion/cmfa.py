"""CMFA — Cross Modal Feature Aggregation 三模态融合模块。

将 IR 和 Depth 特征拼接后作为 Key/Value，
RGB 特征作为 Query，通过交叉注意力融合多模态信息。
"""

import torch
import torch.nn as nn

from .cross_attention import CrossAttention


class CMFA(nn.Module):
    """三模态交叉融合模块。

    流程:
        1. RGB / IR / Depth 各自 flatten 并转置为 (B, N, C) 序列格式
        2. IR + Depth 拼接作为 Key/Value
        3. RGB 作为 Query 做交叉注意力
        4. 残差连接 + LayerNorm
        5. 恢复为 (B, C, H, W) 空间格式
    """

    def __init__(self, dim):
        super().__init__()

        self.cross = CrossAttention(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, rgb, ir, depth):
        """
        Args:
            rgb:   (B, C, H, W) RGB 特征
            ir:    (B, C, H, W) IR 特征
            depth: (B, C, H, W) Depth 特征

        Returns:
            fused: (B, C, H, W) 融合后的特征
        """
        B, C, H, W = rgb.shape

        # (B, C, H, W) -> (B, N, C)  where N = H * W
        rgb_flat = rgb.flatten(2).transpose(1, 2)
        ir_flat = ir.flatten(2).transpose(1, 2)
        depth_flat = depth.flatten(2).transpose(1, 2)

        # IR + Depth 拼接作为 Key/Value
        kv = torch.cat([ir_flat, depth_flat], dim=1)

        # 交叉注意力
        fused = self.cross(rgb_flat, kv)

        # 残差 + Norm
        fused = self.norm(fused + rgb_flat)

        # 恢复空间格式
        fused = fused.transpose(1, 2).reshape(B, C, H, W)

        return fused
