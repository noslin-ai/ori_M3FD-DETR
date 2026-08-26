"""DINO Loss — 完整训练损失。

整合:
    1. Hungarian Matching: 预测框与真实框一对一匹配
    2. 分类损失: Focal Loss
    3. 回归损失: L1 + GIoU

总损失 = cost_class * focal_loss + cost_bbox * l1_loss + cost_giou * giou_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..detector.matcher import HungarianMatcher
from .focal_loss import FocalLoss
from .box_loss import box_l1_loss, giou_loss


def _box_iou_diag_cxcywh(boxes1, boxes2):
    """Return diagonal IoU for matched cxcywh boxes."""
    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))

    x11 = boxes1[:, 0] - boxes1[:, 2] / 2
    y11 = boxes1[:, 1] - boxes1[:, 3] / 2
    x12 = boxes1[:, 0] + boxes1[:, 2] / 2
    y12 = boxes1[:, 1] + boxes1[:, 3] / 2

    x21 = boxes2[:, 0] - boxes2[:, 2] / 2
    y21 = boxes2[:, 1] - boxes2[:, 3] / 2
    x22 = boxes2[:, 0] + boxes2[:, 2] / 2
    y22 = boxes2[:, 1] + boxes2[:, 3] / 2

    lt_x = torch.maximum(x11, x21)
    lt_y = torch.maximum(y11, y21)
    rb_x = torch.minimum(x12, x22)
    rb_y = torch.minimum(y12, y22)

    inter = (rb_x - lt_x).clamp(min=0) * (rb_y - lt_y).clamp(min=0)
    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)
    return inter / (area1 + area2 - inter).clamp(min=1e-7)


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
        class_weights=None,
        cost_ce=0.0,
        aux_loss_weight=0.5,
        quality_class_targets=False,
        quality_floor=0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_ce = cost_ce
        self.aux_loss_weight = aux_loss_weight
        self.quality_class_targets = quality_class_targets
        self.quality_floor = quality_floor

        # 匹配器
        self.matcher = HungarianMatcher(
            cost_class=cost_class,
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
        )

        # 分类损失
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            class_weights=class_weights,
        )

    def _loss_single(self, pred_logits, pred_boxes, targets):
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
        B, Q = pred_logits.shape[:2]
        num_boxes = sum(t["boxes"].shape[0] for t in targets)

        # 1. Hungarian 匹配
        target_labels = [t["labels"] for t in targets]
        target_boxes = [t["boxes"] for t in targets]

        indices = self.matcher(pred_logits, pred_boxes, target_labels, target_boxes)

        # 2. 构造前景类别 one-hot。
        # Sigmoid focal 不需要显式背景类：未匹配 query 的所有前景目标均为 0。
        # 若继续监督 background=1，会和 matcher/evaluator 的“忽略背景类”目标冲突，
        # 并把模型推向背景 logit 长期最高的全背景吸引子。
        target_onehot = torch.zeros(
            B, Q, self.num_classes,
            device=pred_logits.device,
        )
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() > 0:
                labels = target_labels[i][tgt_idx].long()
                if self.quality_class_targets:
                    ious = _box_iou_diag_cxcywh(
                        pred_boxes[i, src_idx].detach(),
                        target_boxes[i][tgt_idx],
                    )
                    quality = ious.clamp(min=self.quality_floor, max=1.0)
                    target_onehot[i, src_idx, labels] = quality
                else:
                    target_onehot[i, src_idx, labels] = 1

        # 分类损失：只使用前景 logits，保留模型最后一维背景 logit 以兼容旧 checkpoint。
        loss_class = self.focal_loss(
            pred_logits[:, :, :self.num_classes].reshape(-1, self.num_classes),
            target_onehot.reshape(-1, self.num_classes),
            normalizer=num_boxes,
        )

        # 3. 回归损失（仅对匹配到的 query）
        matched_boxes = []
        matched_targets = []
        matched_logits = []
        matched_labels = []

        for i, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() > 0:
                matched_boxes.append(pred_boxes[i, src_idx])
                matched_targets.append(target_boxes[i][tgt_idx])
                matched_logits.append(pred_logits[i, src_idx, :self.num_classes])
                matched_labels.append(target_labels[i][tgt_idx].long())

        if matched_boxes:
            matched_boxes = torch.cat(matched_boxes)
            matched_targets = torch.cat(matched_targets)
            matched_logits = torch.cat(matched_logits)
            matched_labels = torch.cat(matched_labels)

            loss_bbox = box_l1_loss(matched_boxes, matched_targets)
            loss_giou = giou_loss(matched_boxes, matched_targets)
            loss_ce = F.cross_entropy(matched_logits, matched_labels)
        else:
            loss_bbox = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0
            loss_ce = pred_logits.sum() * 0.0

        # 4. 总损失
        total = (
            self.cost_class * loss_class
            + self.cost_ce * loss_ce
            + self.cost_bbox * loss_bbox
            + self.cost_giou * loss_giou
        )

        return loss_class, loss_ce, loss_bbox, loss_giou, total

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict
                pred_logits: (B, Q, num_classes+1)
                pred_boxes:  (B, Q, 4)
                aux_outputs: optional list of intermediate decoder outputs
            targets: list of dict
                labels: (N_i,) 类别索引
                boxes:  (N_i, 4) cxcywh 归一化

        Returns:
            dict: {loss_class, loss_bbox, loss_giou, loss}
        """
        loss_class, loss_ce, loss_bbox, loss_giou, total = self._loss_single(
            outputs["pred_logits"],
            outputs["pred_boxes"],
            targets,
        )

        aux_total = outputs["pred_logits"].sum() * 0.0
        aux_outputs = outputs.get("aux_outputs") or []
        for aux in aux_outputs:
            _, _, _, _, aux_loss = self._loss_single(
                aux["pred_logits"],
                aux["pred_boxes"],
                targets,
            )
            aux_total = aux_total + aux_loss

        if aux_outputs:
            total = total + self.aux_loss_weight * aux_total / len(aux_outputs)

        return {
            "loss_class": loss_class.item(),
            "loss_ce": loss_ce.item(),
            "loss_bbox": loss_bbox.item(),
            "loss_giou": loss_giou.item(),
            "loss_aux": aux_total.item() / max(len(aux_outputs), 1),
            "loss": total,
        }
