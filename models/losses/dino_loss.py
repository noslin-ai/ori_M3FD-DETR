"""DINO Loss — 完整训练损失。

整合:
    1. Hungarian Matching: 预测框与真实框一对一匹配
    2. 分类损失: Focal Loss
    3. 回归损失: L1 + GIoU

总损失 = cost_class * focal_loss + cost_bbox * l1_loss + cost_giou * giou_loss
"""

import torch
import torch.nn as nn

from ..detector.matcher import HungarianMatcher
from .focal_loss import FocalLoss
from .box_loss import box_l1_loss, giou_loss


class DINOLoss(nn.Module):
    """DINO 训练损失。

    Args:
        num_classes: 检测类别数（不含背景）
        cost_class: 分类损失权重
        cost_bbox: L1 损失权重
        cost_giou: GIoU 损失权重
        focal_alpha: Focal Loss alpha
        focal_gamma: Focal Loss gamma
    """

    def __init__(
        self,
        num_classes,
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        # 匹配器
        self.matcher = HungarianMatcher(
            cost_class=cost_class,
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
        )

        # 分类损失
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict
                pred_logits: (B, Q, num_classes+1)
                pred_boxes:  (B, Q, 4)
            targets: list of dict
                labels: (N_i,) 类别索引
                boxes:  (N_i, 4) cxcywh 归一化

        Returns:
            dict: {loss_class, loss_bbox, loss_giou, loss}
        """
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        B, Q = pred_logits.shape[:2]

        # 1. Hungarian 匹配
        target_labels = [t["labels"] for t in targets]
        target_boxes = [t["boxes"] for t in targets]

        indices = self.matcher(pred_logits, pred_boxes, target_labels, target_boxes)

        # 2. 构造目标 label 和 box
        # 分类: (B*Q, num_classes+1) one-hot
        target_classes = torch.full(
            (B, Q), self.num_classes,  # 默认背景
            dtype=torch.long,
            device=pred_logits.device,
        )

        # 填入匹配到的真实标签
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() > 0:
                target_classes[i, src_idx] = target_labels[i][tgt_idx]

        # 转 one-hot
        target_onehot = torch.zeros(
            B * Q, self.num_classes + 1,
            device=pred_logits.device,
        )
        target_onehot.scatter_(1, target_classes.reshape(-1, 1), 1)

        # 分类损失
        loss_class = self.focal_loss(
            pred_logits.reshape(-1, self.num_classes + 1),
            target_onehot,
        )

        # 3. 回归损失（仅对匹配到的 query）
        matched_boxes = []
        matched_targets = []

        for i, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() > 0:
                matched_boxes.append(pred_boxes[i, src_idx])
                matched_targets.append(target_boxes[i][tgt_idx])

        if matched_boxes:
            matched_boxes = torch.cat(matched_boxes)
            matched_targets = torch.cat(matched_targets)

            loss_bbox = box_l1_loss(matched_boxes, matched_targets)
            loss_giou = giou_loss(matched_boxes, matched_targets)
        else:
            loss_bbox = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0

        # 4. 总损失
        total = (
            self.cost_class * loss_class
            + self.cost_bbox * loss_bbox
            + self.cost_giou * loss_giou
        )

        return {
            "loss_class": loss_class.item(),
            "loss_bbox": loss_bbox.item(),
            "loss_giou": loss_giou.item(),
            "loss": total,
        }
