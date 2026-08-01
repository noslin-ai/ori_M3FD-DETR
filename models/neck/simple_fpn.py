"""Simple FPN — 轻量级多尺度特征金字塔。

将 backbone 输出的 4 层多尺度特征进行融合，为检测头提供统一的多尺度表示。
（注：当前主要使用 `MultiScaleFusion` 作为 FPN，`SimpleFPN` 保留作为备选实现。）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFPN(nn.Module):
    """简易 FPN 多尺度特征融合。

    输入: backbone 输出的多尺度特征列表 [P2, P3, P4, P5]
    输出: 融合后的多尺度特征列表
    """

    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: 各层输入通道数（由 backbone 决定，需显式传入）
            out_channels: 输出通道数（统一对齐）
        """
        super().__init__()

        # 横向连接: 1x1 卷积对齐通道
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, 1)
            for in_c in in_channels_list
        ])

        # 平滑卷积: 融合后去混叠
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

        self.out_channels = out_channels

    def forward(self, feats):
        """
        Args:
            feats: list of (B, C_i, H_i, W_i)，自底向上（stride 递增）

        Returns:
            list of (B, out_channels, H_i, W_i) FPN 输出
        """
        # 横向连接
        laterals = [
            conv(feat)
            for conv, feat in zip(self.lateral_convs, feats)
        ]

        # 自顶向下融合
        for i in range(len(laterals) - 1, 0, -1):
            target_size = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(laterals[i], size=target_size, mode="nearest")
            laterals[i - 1] = laterals[i - 1] + upsampled

        # 平滑
        outs = [
            smooth(lat)
            for smooth, lat in zip(self.smooth_convs, laterals)
        ]

        return outs
