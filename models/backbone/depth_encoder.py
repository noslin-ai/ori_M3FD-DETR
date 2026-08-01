"""Depth Encoder — 深度几何特征编码器。

深度输入为 3 通道: depth + gx + gy（几何梯度信息）
使用 stride=2 的首层下采样编码几何信息。
"""

import torch
import torch.nn as nn


class DepthEncoder(nn.Module):
    """深度图编码器。

    首层使用 7x7 kernel + stride=2 下采样，捕获几何结构，
    第二层 3x3 精调，输出与指定通道数对齐。
    """

    def __init__(self, out_channels):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) 深度 3 通道 (depth, gx, gy)

        Returns:
            (B, out_channels, H/4, W/4) 与 RGB P2 对齐分辨率
        """
        return self.encoder(x)
