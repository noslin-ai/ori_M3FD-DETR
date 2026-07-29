"""CMFA — Cross Modal Feature Aggregation 三模态融合模块。

将 IR 和 Depth 特征拼接后作为 Key/Value，
RGB 特征作为 Query，通过交叉注意力融合多模态信息。

支持两种模式:
    - 标准模式 (use_downsample=False):  全分辨率 P2 做全局注意力，显存高
    - 降采样模式 (use_downsample=True): 2×2 pool → 注意力 → upsample，显存降低 ~16×
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attention import CrossAttention


class CMFA(nn.Module):
    """三模态交叉融合模块。

    流程 (降采样模式):
        1. RGB / IR / Depth 各自 2×2 avg pool 降采样
        2. flatten 并转置为 (B, N, C) 序列格式
        3. IR + Depth 拼接作为 Key/Value
        4. RGB 作为 Query 做交叉注意力
        5. 残差连接 + LayerNorm
        6. 恢复空间格式 → upsample 回原始分辨率

    Args:
        dim: 特征通道数
        use_downsample: 是否启用降采样，True 时可大幅降低显存 (推荐训练)
    """

    def __init__(self, dim, use_downsample=True):
        super().__init__()

        self.dim = dim
        self.use_downsample = use_downsample
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
        orig_h, orig_w = H, W

        # ---- 降采样: 2×2 pool, token 减少 4×, attention 降低 ~16× ----
        if self.use_downsample and H > 32 and W > 32:
            rgb   = F.avg_pool2d(rgb,   2)
            ir    = F.avg_pool2d(ir,    2)
            depth = F.avg_pool2d(depth, 2)
            H, W = rgb.shape[2:]

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

        # ---- 上采样回原始分辨率 ----
        if self.use_downsample and (H != orig_h or W != orig_w):
            fused = F.interpolate(fused, size=(orig_h, orig_w),
                                  mode="bilinear", align_corners=False)

        return fused
