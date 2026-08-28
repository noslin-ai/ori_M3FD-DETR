"""YOLO 评估 — 推理解码 + COCO mAP@50-95。

复用 engine/evaluator.py 的 compute_map（pycocotools，101 点插值），
与官方评测口径一致；这里只负责把 YOLO 的原始输出解码成 COCO 预测/标注。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics.utils.nms import non_max_suppression

from engine.evaluator import compute_map


@torch.no_grad()
def decode_predictions(
    model,
    loader,
    device,
    num_classes=12,
    conf_thres=0.001,
    iou_thres=0.6,
    max_det=100,
    use_amp=True,
):
    """对验证集做推理并输出 COCO 格式预测与标注。

    Args:
        model: DetectionModel（训练态由调用方切换为 eval）
        loader: DataLoader（collate_fn 来自 yolo.dataset）
        device: cuda / cpu
        num_classes: 类别数
        conf_thres: NMS 置信度阈值（评估用低阈值保留完整排序）
        iou_thres: NMS IoU 阈值
        max_det: 每图最多保留框数（与 COCOeval maxDets=100 对齐）
        use_amp: 是否使用混合精度

    Returns:
        predictions: COCO 预测 list
        targets: COCO 标注 list
    """
    model.eval()
    predictions = []
    targets = []
    global_idx = 0

    for batch_idx, batch in enumerate(loader):
        img = batch["img"].to(device, non_blocking=True)
        B, _, H, W = img.shape

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(img)

        y = out[0] if isinstance(out, tuple) else out
        dets = non_max_suppression(y, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)

        # 该 batch 内每个样本的目标（collate 后按 batch_idx 聚合）
        batch_cls = batch["cls"]
        batch_boxes = batch["bboxes"]
        batch_idx = batch["batch_idx"]

        for i in range(B):
            image_id = global_idx
            global_idx += 1

            # ---- 预测: xyxy(像素) -> COCO xywh(像素) ----
            det = dets[i]
            if det is not None and det.shape[0] > 0:
                xyxy = det[:, :4].clone()
                xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, W)
                xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, H)
                conf = det[:, 4]
                cat = det[:, 5].long()
                x = xyxy[:, 0]
                y0 = xyxy[:, 1]
                bw = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=1e-3)
                bh = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=1e-3)
                for j in range(det.shape[0]):
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(cat[j]),
                        "bbox": [float(x[j]), float(y0[j]), float(bw[j]), float(bh[j])],
                        "score": float(conf[j]),
                    })

            # ---- GT: 归一化 xywh -> COCO 像素 xywh ----
            mask = batch_idx == i
            gt_cls = batch_cls[mask]
            gt_boxes = batch_boxes[mask]
            for j in range(gt_boxes.shape[0]):
                cx, cy, w, h = gt_boxes[j].tolist()
                targets.append({
                    "image_id": image_id,
                    "category_id": int(gt_cls[j]),
                    "bbox": [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H],
                    "area": w * W * h * H,
                    "iscrowd": 0,
                })

    return predictions, targets


def evaluate(
    model,
    loader,
    device,
    num_classes=12,
    conf_thres=0.001,
    iou_thres=0.6,
    max_det=100,
    use_amp=True,
):
    """完整验证流程：推理 + mAP 计算（对齐旧 validate() 的打印风格）。"""
    print("  开始验证...")
    predictions, targets = decode_predictions(
        model, loader, device, num_classes,
        conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det, use_amp=use_amp,
    )
    print(f"  预测 {len(predictions)} 个框, GT {len(targets)} 个框")

    results = compute_map(predictions, targets, num_classes)

    print(f"  mAP@50-95: {results['mAP50-95']:.4f}")
    print(f"  mAP@50:     {results['mAP50']:.4f}")
    print(f"  mAP@75:     {results['mAP75']:.4f}")
    return results
