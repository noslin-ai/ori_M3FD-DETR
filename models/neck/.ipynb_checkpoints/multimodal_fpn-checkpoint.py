"""Multi-scale Feature Pyramid Network — 多模态多尺度FPN。

替换 SimpleFPN，添加:
    1. 通道对齐 (1x1 conv)
    2. 自顶向下路径
    3. 3×3 平滑卷积
    4. 多尺度输出 (stride 8, 16, 32, 64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    """多尺度特征融合 FPN。

    输入: backbone 多尺度特征 [P2(192), P3(384), P4(768), P5(1536)]
    输出: 融合后多尺度特征 [256, 256, 256, 256]

    P5 → 1×1 → 256 → upsample → + P4 → 256 → upsample → + P3 → 256 → upsample → + P2 → 256
    """

    def __init__(
        self,
        in_channels_list=None,
        out_channels=256,
        use_deform=False,
    ):
        """
        Args:
            in_channels_list: 各层输入通道数，默认 [192, 384, 768, 1536] (SwinV2-L)
            out_channels: 输出通道数
            use_deform: 是否使用可变形卷积
        """
        super().__init__()

        if in_channels_list is None:
            in_channels_list = [96, 192, 384, 768]

        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels

        # 横向连接: 1×1 卷积对齐通道
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, 1)
            for in_c in in_channels_list
        ])

        # 输出平滑: 3×3 卷积去混叠
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Args:
            features: list of (B, C_i, H_i, W_i)，自底向上

        Returns:
            list of (B, out_channels, H_i, W_i)
        """
        # 横向连接
        laterals = [
            conv(feat) for conv, feat in zip(self.lateral_convs, features)
        ]

        # 自顶向下融合
        for i in range(self.num_levels - 1, 0, -1):
            target_h, target_w = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(
                laterals[i], size=(target_h, target_w), mode="nearest"
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # 输出平滑
        outputs = [
            conv(lat) for conv, lat in zip(self.output_convs, laterals)
        ]

        return outputs
