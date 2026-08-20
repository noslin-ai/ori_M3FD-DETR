"""Hungarian Matcher — 匈牙利匹配。

DINO/DETR 训练的核心：将 Nq 个预测框与 N 个真实框进行一对一最优匹配。

匹配代价 = 分类代价 + L1 框代价 + GIoU 代价
使用 scipy 的 linear_sum_assignment 求解二分图最小权匹配。
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes):
    """(cx, cy, w, h) -> (x1, y1, x2, y2)"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """计算 Generalized IoU (GIoU)。

    Args:
        boxes1: (N, 4) cxcywh 格式
        boxes2: (M, 4) cxcywh 格式

    Returns:
        giou: (N, M) GIoU 矩阵
    """
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # 交集
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # 并集
    union = area1[:, None] + area2[None] - inter

    # 最小外接矩形
    lt_enclose = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_enclose = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_enclose = (rb_enclose - lt_enclose).clamp(min=0)
    enclose = wh_enclose[:, :, 0] * wh_enclose[:, :, 1]

    eps = 1e-7
    iou = inter / union.clamp(min=eps)
    giou = iou - (enclose - union) / enclose.clamp(min=eps)
    return giou


class HungarianMatcher(nn.Module):
    """匈牙利匹配器。

    Args:
        cost_class: 分类代价权重
        cost_bbox: L1 框代价权重
        cost_giou: GIoU 代价权重
    """

    def __init__(self, cost_class=1.0, cost_bbox=1.0, cost_giou=1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, pred_logits, pred_boxes, target_labels, target_boxes):
        """
        Args:
            pred_logits: (B, num_queries, num_classes+1)
            pred_boxes:  (B, num_queries, 4) cxcywh
            target_labels: list of (N_i,) 每张图的标签
            target_boxes:  list of (N_i, 4) 每张图的框

        Returns:
            indices: list of (row_idx, col_idx) 每张图的匹配对
        """
        B = pred_logits.shape[0]
        indices = []

        for i in range(B):
            tgt_labels = target_labels[i]
            tgt_boxes = target_boxes[i]

            if tgt_labels.numel() == 0:
                indices.append((torch.tensor([], dtype=torch.long),
                                torch.tensor([], dtype=torch.long)))
                continue

            # 分类代价: 与 sigmoid Focal Loss 对齐。
            # 旧版 softmax 会让背景类和目标类强制竞争，和当前 sigmoid 训练目标不一致，
            # 容易在类别不平衡时把匹配推向低质量 query。
            out_prob = pred_logits[i].sigmoid()[:, :-1]  # (Q, C), drop background
            cost_class = -out_prob[:, tgt_labels.long()]  # (Q, N)

            # L1 框代价
            cost_bbox = torch.cdist(
                pred_boxes[i], tgt_boxes, p=1
            )  # (Q, N)

            # GIoU 代价
            cost_giou = -generalized_box_iou(
                pred_boxes[i], tgt_boxes
            )  # (Q, N)

            # 总代价
            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )

            # 匈牙利算法求解；防止偶发退化框造成 NaN/Inf 代价。
            cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=-1e6)
            row, col = linear_sum_assignment(cost.cpu())
            indices.append(
                (torch.as_tensor(row, dtype=torch.long),
                 torch.as_tensor(col, dtype=torch.long))
            )

        return indices


# 便捷函数：简单 L1 匹配（教学版，与 ChatGPT 对话中一致）
def match(pred_boxes, target_boxes):
    """简单 L1 距离匹配。

    Args:
        pred_boxes: (Q, 4) 预测框
        target_boxes: (N, 4) 真实框

    Returns:
        row, col: 匹配索引
    """
    cost = (pred_boxes[:, None] - target_boxes[None]).abs().sum(-1)
    row, col = linear_sum_assignment(cost.detach().cpu())
    return row, col
