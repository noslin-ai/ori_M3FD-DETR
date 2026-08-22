"""Focal Loss — Sigmoid Focal Loss。

用于解决类别不平衡问题，DETR/DINO 中的标准分类损失。

公式:
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

相比标准 CrossEntropy:
    - 对易分类样本降权 (1-p_t)^gamma
    - 通过 alpha 平衡正负样本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Sigmoid Focal Loss。

    Args:
        alpha: 正负样本平衡因子（默认 0.25）
        gamma: 聚焦因子（默认 2.0）
    """

    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer(
                "class_weights",
                torch.as_tensor(class_weights, dtype=torch.float32),
            )

    def forward(self, logits, targets, normalizer=None):
        """
        Args:
            logits:  (N, C) 分类 logits（未经 sigmoid/softmax）
            targets: (N, C) one-hot 标签
            normalizer: 可选归一化因子，推荐使用 batch 内 GT 数量。

        Returns:
            loss: scalar focal loss
        """
        # sigmoid 概率
        p = logits.sigmoid()

        # p_t: 正样本取 p，负样本取 1-p
        p_t = p * targets + (1 - p) * (1 - targets)

        # focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # alpha 平衡
        alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # BCE with logits
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        loss = alpha * focal_weight * bce   # (N, C)

        if self.class_weights is not None:
            weights = self.class_weights.to(device=loss.device, dtype=loss.dtype)
            if weights.numel() != loss.shape[-1]:
                raise ValueError(
                    f"class_weights 长度 {weights.numel()} != 类别数 {loss.shape[-1]}"
                )
            # 只放大匹配到的正样本类别。
            # 若同时放大负样本项，少数类在大量 unmatched query 上会承受更强负梯度，
            # 反而更难从 class 0/2 塌缩中抬头。
            pos_weights = 1.0 + (weights.view(1, -1) - 1.0) * targets
            loss = loss * pos_weights

        # 先对类别维度取平均，再按 GT 数量归一化。
        # 直接对 N*C 全部求和会让分类项随类别数线性放大，训练日志常被抬到 5+，
        # 但这不是更强监督，只是量纲偏大。
        loss = loss.mean(-1)

        if normalizer is not None:
            return loss.sum() / max(float(normalizer), 1.0)

        return loss.mean()
