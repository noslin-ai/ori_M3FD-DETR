"""DINO Detection Head — 旧版检测头（已废弃）。

注：当前项目使用 ``dino_detector.py`` 中的 ``DINODetector`` 作为主检测头。
本模块 (``DINOHead``) 为早期占位实现，仅保留用于参考，不参与实际推理或训练。
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
