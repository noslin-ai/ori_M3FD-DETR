"""Evaluator — 验证与评估。

支持:
    1. 模型推理验证
    2. COCO-style mAP@50-95 计算
    3. 预测结果 → COCO JSON 转换
"""

import json
import os
import torch
from torch.cuda.amp import autocast

from models.detector.matcher import box_cxcywh_to_xyxy


def _box_iou_xyxy(box, boxes):
    """计算单个 xyxy 框与一组 xyxy 框的 IoU。"""
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area1 = ((box[2] - box[0]).clamp(min=0) *
             (box[3] - box[1]).clamp(min=0))
    area2 = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) *
             (boxes[:, 3] - boxes[:, 1]).clamp(min=0))
    return inter / (area1 + area2 - inter).clamp(min=1e-7)


def class_aware_nms(boxes_cxcywh, scores, labels, iou_threshold=0.6, max_dets=100):
    """按类别做 NMS，输入/输出均为归一化 cxcywh。"""
    if boxes_cxcywh.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes_cxcywh.device)

    keep_all = []
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh).clamp(0, 1)

    for cls_id in labels.unique():
        cls_idx = torch.nonzero(labels == cls_id, as_tuple=False).flatten()
        order = scores[cls_idx].argsort(descending=True)
        cls_idx = cls_idx[order]

        keep_cls = []
        while cls_idx.numel() > 0:
            current = cls_idx[0]
            keep_cls.append(current)
            if cls_idx.numel() == 1:
                break
            ious = _box_iou_xyxy(boxes_xyxy[current], boxes_xyxy[cls_idx[1:]])
            cls_idx = cls_idx[1:][ious <= iou_threshold]

        keep_all.extend(keep_cls)

    keep = torch.stack(keep_all) if keep_all else torch.empty(0, dtype=torch.long, device=boxes_cxcywh.device)
    keep = keep[scores[keep].argsort(descending=True)]
    if max_dets is not None:
        keep = keep[:max_dets]
    return keep


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    use_amp=True,
    conf_threshold=0.001,
    max_dets=100,
    nms_iou=0.6,
):
    """模型推理验证。

    Args:
        model: M3F-DETR 模型（或 EMA 模型）
        loader: 验证集 DataLoader
        device: cuda / cpu
        use_amp: 是否使用混合精度
        conf_threshold: 置信度阈值（sigmoid），低于该值的预测被过滤
        max_dets: 每张图最多保留预测框数，和 COCOeval maxDets=100 对齐
        nms_iou: 同类别 NMS IoU 阈值；小于等于 0 时关闭 NMS

    Returns:
        predictions: list of dict（COCO 格式预测）
        targets: list of dict（COCO 格式标注）
    """
    model.eval()
    predictions = []
    targets = []

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        ir = batch["ir"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        batch_targets = batch["target"]

        with autocast(enabled=use_amp):
            output = model(rgb, ir, depth)

        B = rgb.shape[0]
        pred_logits = output["pred_logits"]  # (B, Q, C+1)
        pred_boxes = output["pred_boxes"]    # (B, Q, 4)

        for i in range(B):
            image_id = batch_targets[i].get("image_id", torch.tensor([i])).item()
            img_h, img_w = rgb.shape[2], rgb.shape[3]

            # 预测: sigmoid 后只在前景类中取最大分数。
            # 背景类不参与推理筛选，否则背景 logit 偏高时会把所有候选提前丢掉。
            probs = pred_logits[i].sigmoid()[:, :-1]  # (Q, C)
            confs, labels = probs.max(-1)             # (Q,), (Q,)

            keep = confs > conf_threshold
            confs = confs[keep]
            labels = labels[keep]
            boxes = pred_boxes[i][keep].clamp(0, 1)

            if nms_iou and nms_iou > 0:
                keep_idx = class_aware_nms(boxes, confs, labels, nms_iou, max_dets)
                confs = confs[keep_idx]
                labels = labels[keep_idx]
                boxes = boxes[keep_idx]
            elif max_dets is not None and len(confs) > max_dets:
                topk = confs.argsort(descending=True)[:max_dets]
                confs = confs[topk]
                labels = labels[topk]
                boxes = boxes[topk]

            # cxcywh → xywh (COCO format, pixel coords)
            cx, cy, w, h = boxes.unbind(-1)
            coco_boxes = torch.stack([
                cx * img_w - w * img_w / 2,
                cy * img_h - h * img_h / 2,
                w * img_w,
                h * img_h,
            ], dim=-1)
            coco_boxes[:, 0] = coco_boxes[:, 0].clamp(0, img_w)
            coco_boxes[:, 1] = coco_boxes[:, 1].clamp(0, img_h)
            coco_boxes[:, 2] = coco_boxes[:, 2].clamp(min=1e-3, max=img_w)
            coco_boxes[:, 3] = coco_boxes[:, 3].clamp(min=1e-3, max=img_h)

            for j in range(len(confs)):
                predictions.append({
                    "image_id": image_id,
                    "category_id": labels[j].item(),
                    "bbox": coco_boxes[j].cpu().tolist(),
                    "score": confs[j].item(),
                })

            # GT
            gt_boxes = batch_targets[i]["boxes"]
            gt_labels = batch_targets[i]["labels"]
            for j in range(len(gt_labels)):
                cx, cy, w, h = gt_boxes[j].unbind(-1)
                targets.append({
                    "image_id": image_id,
                    "category_id": gt_labels[j].item(),
                    "bbox": [
                        (cx * img_w - w * img_w / 2).item(),
                        (cy * img_h - h * img_h / 2).item(),
                        (w * img_w).item(),
                        (h * img_h).item(),
                    ],
                    "area": (w * img_w * h * img_h).item(),
                    "iscrowd": 0,
                })

    return predictions, targets


def compute_map(predictions, targets, num_classes=12):
    """计算 COCO-style mAP@50-95。

    需要安装 pycocotools。

    Args:
        predictions: list of dict (COCO prediction format)
        targets: list of dict (COCO annotation format)
        num_classes: 类别数

    Returns:
        dict: {mAP50, mAP75, mAP50-95, per_class}
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("  ⚠ pycocotools 未安装，跳过 mAP 计算")
        print("  安装: pip install pycocotools")
        return {"mAP50": 0.0, "mAP75": 0.0, "mAP50-95": 0.0}

    if len(predictions) == 0 or len(targets) == 0:
        print("  ⚠ 无预测或无标注，跳过 mAP 计算")
        return {"mAP50": 0.0, "mAP75": 0.0, "mAP50-95": 0.0}

    # 构造 COCO GT JSON
    gt_json = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": f"class_{i}"} for i in range(num_classes)
        ],
    }

    image_ids = set()
    ann_id = 1
    for t in targets:
        image_ids.add(t["image_id"])
    for img_id in sorted(image_ids):
        gt_json["images"].append({"id": img_id})

    for t in targets:
        gt_json["annotations"].append({
            "id": ann_id,
            "image_id": t["image_id"],
            "category_id": t["category_id"],
            "bbox": t["bbox"],
            "area": t.get("area", t["bbox"][2] * t["bbox"][3]),
            "iscrowd": t.get("iscrowd", 0),
        })
        ann_id += 1

    # 保存临时 JSON
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_json, f)
        gt_path = f.name

    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 提取关键指标
    stats = coco_eval.stats
    results = {
        "mAP50": stats[1],       # IoU=0.50
        "mAP75": stats[2],        # IoU=0.75
        "mAP50-95": stats[0],     # mAP @[IoU=0.50:0.95]
        "per_class": {},
    }

    # 每个类别的 AP（某类无预测或无标注时 stats 可能为空，跳过避免崩溃）
    for cat_id in range(num_classes):
        coco_eval_cls = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval_cls.params.catIds = [cat_id]
        try:
            coco_eval_cls.evaluate()
            coco_eval_cls.accumulate()
            stats_cls = coco_eval_cls.stats
        except Exception:
            stats_cls = []
        results["per_class"][cat_id] = stats_cls[0] if len(stats_cls) > 0 else 0.0

    os.unlink(gt_path)
    return results


def validate(
    model,
    loader,
    device,
    num_classes=12,
    use_amp=True,
    conf_threshold=0.001,
    nms_iou=0.6,
):
    """完整验证流程：推理 + mAP 计算。

    Args:
        model: M3F-DETR 模型
        loader: 验证集 DataLoader
        device: cuda / cpu
        num_classes: 类别数
        use_amp: 是否使用混合精度

    Returns:
        dict: mAP 结果
    """
    print("  开始验证...")
    predictions, targets = evaluate_model(
        model, loader, device, use_amp, conf_threshold, nms_iou=nms_iou
    )
    print(f"  预测 {len(predictions)} 个框, GT {len(targets)} 个框")

    results = compute_map(predictions, targets, num_classes)

    print(f"  mAP@50-95: {results['mAP50-95']:.4f}")
    print(f"  mAP@50:     {results['mAP50']:.4f}")
    print(f"  mAP@75:     {results['mAP75']:.4f}")

    return results
