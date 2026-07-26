"""Metrics — 评估指标工具。

提供:
    1. IoU 计算
    2. AP@50 / AP@75 手动计算（不依赖 pycocotools 的备选方案）
    3. PR 曲线绘制数据
"""

import numpy as np
import torch


def compute_iou_matrix(boxes1, boxes2):
    """计算 IoU 矩阵。

    Args:
        boxes1: (N, 4) xyxy 格式
        boxes2: (M, 4) xyxy 格式

    Returns:
        iou: (N, M) IoU 矩阵
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None] - inter
    iou = inter / np.clip(union, 1e-6, None)
    return iou


def compute_ap(predictions, gts, iou_threshold=0.5):
    """计算单个类别在指定 IoU 阈值下的 AP。

    Args:
        predictions: list of (bbox, score) — bbox 为 xyxy 格式
        gts: list of bbox (xyxy 格式)
        iou_threshold: IoU 阈值

    Returns:
        ap: Average Precision
    """
    if len(gts) == 0:
        return 0.0 if len(predictions) > 0 else 1.0

    if len(predictions) == 0:
        return 0.0

    # 按 score 降序排列
    scores = np.array([p[1] for p in predictions])
    order = np.argsort(-scores)
    sorted_boxes = np.array([p[0] for p in predictions])[order]

    gt_boxes = np.array(gts)
    used = np.zeros(len(gt_boxes))

    tp = np.zeros(len(sorted_boxes))
    fp = np.zeros(len(sorted_boxes))

    for i, box in enumerate(sorted_boxes):
        if len(gt_boxes) > 0:
            ious = compute_iou_matrix(box[None], gt_boxes)[0]
            best_iou = ious.max()
            best_idx = ious.argmax()

            if best_iou >= iou_threshold and not used[best_idx]:
                tp[i] = 1
                used[best_idx] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    # 累积 PR
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / len(gt_boxes)
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-6, None)

    # 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        mask = recall >= t
        p = precision[mask].max() if mask.any() else 0
        ap += p / 11

    return ap


def compute_map_50_95_manual(predictions_by_class, gts_by_class):
    """手动计算 mAP@50-95（不依赖 pycocotools）。

    Args:
        predictions_by_class: dict {class_id: [(bbox_xyxy, score), ...]}
        gts_by_class: dict {class_id: [bbox_xyxy, ...]}

    Returns:
        dict: {mAP50, mAP75, mAP50-95}
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.50, 0.55, ..., 0.95
    all_classes = set(list(predictions_by_class.keys()) + list(gts_by_class.keys()))

    aps_per_iou = {t: [] for t in iou_thresholds}

    for cls in all_classes:
        preds = predictions_by_class.get(cls, [])
        gts = gts_by_class.get(cls, [])

        for t in iou_thresholds:
            ap = compute_ap(preds, gts, iou_threshold=t)
            aps_per_iou[t].append(ap)

    results = {}
    for t in iou_thresholds:
        aps = aps_per_iou[t]
        results[f"AP@{t:.2f}"] = np.mean(aps) if aps else 0.0

    results["mAP50"] = results.get("AP@0.50", 0.0)
    results["mAP75"] = results.get("AP@0.75", 0.0)
    results["mAP50-95"] = np.mean([results[f"AP@{t:.2f}"] for t in iou_thresholds])

    return results
