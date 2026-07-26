"""Box Loss — 边界框损失函数。

包含:
    1. L1 Loss: 预测框与真实框的 L1 距离
    2. GIoU Loss: Generalized IoU 损失（尺度不变）

DETR/DINO 标准配置: L1 + GIoU 联合优化。
"""

import torch
from ..detector.matcher import box_cxcywh_to_xyxy, generalized_box_iou


def box_l1_loss(pred_boxes, target_boxes):
    """L1 损失。

    Args:
        pred_boxes:   (N, 4) cxcywh 归一化
        target_boxes: (N, 4) cxcywh 归一化

    Returns:
        loss: scalar
    """
    return torch.abs(pred_boxes - target_boxes).mean()


def giou_loss(pred_boxes, target_boxes):
    """GIoU 损失。

    Args:
        pred_boxes:   (N, 4) cxcywh
        target_boxes: (N, 4) cxcywh

    Returns:
        loss: scalar (1 - mean_giou)
    """
    if pred_boxes.numel() == 0:
        return pred_boxes.sum() * 0.0

    giou = generalized_box_iou(pred_boxes, target_boxes)
    # 取对角线（匹配后的对应对）
    giou_diag = torch.diag(giou)
    return (1 - giou_diag).mean()
