"""跨模态注意力融合模块（v0.9.0 双分支方案）。

对 P3/P4/P5 每个尺度，将 RGB 分支特征与 IR+Depth 辅助分支特征融合:
    1. 通道注意力: 由两个分支拼接后的全局统计生成通道权重，重标定 RGB 特征；
    2. 空间注意力: 由拼接特征生成空间门控，突出两分支共同关注的区域；
    3. 残差融合: 1x1 卷积学一个跨模态残差，稳定训练并保留 RGB 主干信息。

设计动机: 早期融合（5ch 拼接）在 1600 张小数据上未跑赢 RGB-only，
双分支让每个模态先独立提特征，再用注意力决定"在哪里、看哪个通道"，
能更充分利用红外（温差目标）与深度（几何结构）的特性。
"""

import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    """P3-P5 跨模态注意力融合单元（单尺度）。"""

    def __init__(self, in_channels, hidden=None, bias=False):
        """
        Args:
            in_channels: 该尺度特征通道数（yolo11 P3/P4/P5 = 64/128/256）
            hidden: 通道注意力隐藏维度
            bias: 注意力投影是否带偏置
        """
        super().__init__()
        hidden = hidden or max(in_channels // 4, 8)

        # 通道注意力: 拼接两分支后做全局池化 -> MLP -> sigmoid
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels * 2, hidden, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=bias),
            nn.Sigmoid(),
        )

        # 空间注意力: 拼接两分支 -> 1x1 conv -> sigmoid
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, kernel_size=1, bias=bias),
            nn.Sigmoid(),
        )

        # 跨模态残差: 拼接 -> 1x1 投影 -> 与注意力加权的 RGB 特征相加
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, rgb, aux):
        """融合 RGB 分支与辅助分支的同尺度特征。

        Args:
            rgb: (B, C, H, W) RGB 分支特征
            aux: (B, C, H, W) IR+Depth 分支特征

        Returns:
            fused: (B, C, H, W)
        """
        cat = nn.functional.relu(torch.cat([rgb, aux], dim=1))
        ca = self.channel_att(cat).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        sa = self.spatial_att(cat)                                # (B, 1, H, W)
        gated = rgb * ca * sa
        return gated + self.residual(cat)



class ZeroConv2d(nn.Module):
    """零初始化卷积（MCF 核心：初始输出为 0，融合 = 主分支特征）。"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)




class ZeroConv2d(nn.Module):
    """零初始化卷积（MCF 核心：初始输出为 0，融合 = 主分支特征）。"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)

