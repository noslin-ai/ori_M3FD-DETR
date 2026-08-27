"""诊断预测框与 GT 的 IoU，定位 mAP=0 的框/类别问题。

用法:
    python tools/diagnose_iou.py --checkpoint checkpoints/debug/latest.pth
    python tools/diagnose_iou.py --checkpoint checkpoints/debug/latest.pth --conf-threshold 0.01
"""

import argparse
import os
import sys
from collections import Counter

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.rgb_ir_depth_dataset import RGBIRDepthDataset
from engine.trainer import collate_fn
from models.m3f_detr import M3F_DETR
from models.detector.matcher import box_cxcywh_to_xyxy
from utils.checkpoint import _safe_torch_load, strip_state_dict_prefixes


def box_iou_cxcywh(boxes1, boxes2):
    """Return IoU matrix for normalized cxcywh boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-7)


def summarize_tensor(name, value):
    if value.numel() == 0:
        print(f"  {name}: empty")
        return
    print(
        f"  {name}: mean={value.mean():.4f} min={value.min():.4f} "
        f"max={value.max():.4f}"
    )
    if value.ndim == 2 and value.shape[1] == 4:
        coord_names = ("cx", "cy", "w", "h")
        parts = []
        for idx, coord in enumerate(coord_names):
            v = value[:, idx]
            parts.append(
                f"{coord}=mean:{v.mean():.4f}/min:{v.min():.4f}/max:{v.max():.4f}"
            )
        print("    " + " | ".join(parts))


def cfg_detector_kwargs(cfg):
    return {
        "decoder_feature_level": cfg.get("decoder_feature_level", -1),
        "decoder_feature_levels": cfg.get("decoder_feature_levels"),
        "use_anchor_boxes": bool(cfg.get("use_anchor_boxes", False)),
        "anchor_box_size": tuple(cfg.get("anchor_box_size", (0.06, 0.12))),
    }


@torch.no_grad()
def diagnose(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    state = _safe_torch_load(args.checkpoint, map_location="cpu")
    cfg = state.get("cfg", {}) or {}
    image_size = tuple(cfg.get("image_size", (384, 640)))
    normalize_rgb = bool(cfg.get("normalize_rgb", False))
    print("Checkpoint cfg:", cfg)

    model = M3F_DETR(
        num_classes=cfg.get("num_classes", 12),
        hidden_dim=cfg.get("hidden_dim", 256),
        num_queries=cfg.get("num_queries", 300),
        backbone_name=cfg.get("backbone", "swin_tiny"),
        pretrained=False,
        use_dn=cfg.get("use_dn", False),
        input_size=image_size,
        **cfg_detector_kwargs(cfg),
    ).to(device).eval()
    model.load_state_dict(strip_state_dict_prefixes(state["model"]))

    dataset = RGBIRDepthDataset(
        args.data_root,
        train=True,
        size=image_size,
        normalize_rgb=normalize_rgb,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    top1_any = []
    top1_same_cls = []
    best_any = []
    best_same_cls = []
    gt_recall_any = {0.1: 0, 0.3: 0, 0.5: 0}
    gt_recall_cls = {0.1: 0, 0.3: 0, 0.5: 0}
    total_gt = 0
    total_pred = 0
    kept_pred = 0
    pred_box_samples = []
    gt_box_samples = []
    pred_labels = Counter()
    gt_labels = Counter()

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        rgb = batch["rgb"].to(device, non_blocking=True)
        ir = batch["ir"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        targets = batch["target"]

        out = model(rgb, ir, depth)
        probs = out["pred_logits"][:, :, :-1].sigmoid()
        boxes = out["pred_boxes"].clamp(0, 1)
        scores, labels = probs.max(-1)

        for i in range(rgb.shape[0]):
            gt_boxes = targets[i]["boxes"].to(device)
            gt_labs = targets[i]["labels"].to(device).long()
            total_gt += gt_boxes.shape[0]
            gt_labels.update(gt_labs.cpu().tolist())
            gt_box_samples.append(gt_boxes.detach().cpu())

            keep = scores[i] > args.conf_threshold
            total_pred += scores.shape[1]
            if keep.sum() == 0:
                continue

            p_scores = scores[i][keep]
            p_labels = labels[i][keep].long()
            p_boxes = boxes[i][keep]
            if p_scores.numel() > args.topk:
                idx = p_scores.argsort(descending=True)[:args.topk]
                p_scores = p_scores[idx]
                p_labels = p_labels[idx]
                p_boxes = p_boxes[idx]

            kept_pred += p_boxes.shape[0]
            pred_labels.update(p_labels.cpu().tolist())
            pred_box_samples.append(p_boxes.detach().cpu())

            if gt_boxes.numel() == 0:
                continue

            ious = box_iou_cxcywh(p_boxes, gt_boxes)
            same_cls = p_labels[:, None] == gt_labs[None, :]
            ious_same = ious.masked_fill(~same_cls, 0)

            top1_any.append(ious[0].max().item())
            top1_same_cls.append(ious_same[0].max().item())
            best_any.append(ious.max().item())
            best_same_cls.append(ious_same.max().item())

            gt_best_any = ious.max(dim=0).values
            gt_best_cls = ious_same.max(dim=0).values
            for thr in gt_recall_any:
                gt_recall_any[thr] += (gt_best_any >= thr).sum().item()
                gt_recall_cls[thr] += (gt_best_cls >= thr).sum().item()

    def avg(values):
        return sum(values) / max(len(values), 1)

    print("\n=== IoU 诊断 ===")
    print(f"扫描 batch: {min(args.max_batches, len(loader))}, conf_threshold={args.conf_threshold}, topk={args.topk}")
    print(f"原始预测数: {total_pred}, 保留预测数: {kept_pred}, GT 数: {total_gt}")
    print(f"Top1 最大 IoU(不看类别): {avg(top1_any):.4f}")
    print(f"Top1 最大 IoU(同类别):   {avg(top1_same_cls):.4f}")
    print(f"每图最佳 IoU(不看类别):  {avg(best_any):.4f}")
    print(f"每图最佳 IoU(同类别):    {avg(best_same_cls):.4f}")

    for thr in (0.1, 0.3, 0.5):
        denom = max(total_gt, 1)
        print(
            f"GT recall@IoU{thr:.1f}: any={gt_recall_any[thr] / denom:.4f} "
            f"same_cls={gt_recall_cls[thr] / denom:.4f}"
        )

    print("\n=== 框分布(cx, cy, w, h) ===")
    if pred_box_samples:
        summarize_tensor("pred", torch.cat(pred_box_samples, dim=0))
    else:
        print("  pred: empty")
    if gt_box_samples:
        summarize_tensor("gt", torch.cat(gt_box_samples, dim=0))
    else:
        print("  gt: empty")

    print("\n=== 类别分布(top 12) ===")
    print("  pred:", pred_labels.most_common(12))
    print("  gt:  ", gt_labels.most_common(12))


def main():
    parser = argparse.ArgumentParser(description="诊断预测框 IoU")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/train")
    parser.add_argument("--conf-threshold", type=float, default=0.01)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    diagnose(args)


if __name__ == "__main__":
    main()
