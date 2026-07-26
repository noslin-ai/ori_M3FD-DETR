"""DINO Detection Head — 第一版检测头占位。

第三阶段将实现完整的 DINO Detector:
    - Deformable Transformer Encoder
    - Object Query
    - Hungarian Matcher
    - 分类 + 回归 + GIoU Loss
    - bbox 输出转换

当前为占位模块，用于验证模型管线可运行。
"""

import torch
import torch.nn as nn


class DINOHead(nn.Module):
    """DINO 检测头占位模块。

    第三阶段将替换为完整实现。
    """

    def __init__(self, in_channels, hidden_dim=256, num_classes=12):
        super().__init__()

        self.num_classes = num_classes

        # 占位：简单分类 + 回归头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        self.bbox_regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 融合特征

        Returns:
            cls:   (B, num_classes) 分类 logits
            bbox:  (B, 4) 回归坐标 (cx, cy, w, h)
        """
        cls_logits = self.classifier(x)
        bbox = self.bbox_regressor(x).sigmoid()
        return cls_logits, bbox
