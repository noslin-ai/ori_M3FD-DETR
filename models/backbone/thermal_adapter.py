"""IR (Infrared) 适配器 — 轻量级卷积编码器。

IR 三通道实际上是灰度堆叠，信息量有限，不需要大型网络。
用一个轻量卷积适配器将 3 通道 IR 映射到指定通道数。
"""

import torch
import torch.nn as nn


class ThermalAdapter(nn.Module):
    """红外热成像适配器。

    两层卷积 + BN + ReLU，将 IR 三通道映射到与 RGB 早年层对齐的通道数。
    """

    def __init__(self, out_channels):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) IR 三通道灰度堆叠

        Returns:
            (B, out_channels, H/4, W/4) 与 RGB P2 对齐分辨率
        """
        return self.encoder(x)
