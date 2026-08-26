"""Diagnose whether prediction confidence is aligned with localization quality.

The script scans a validation split and compares each kept prediction's score
with its best same-class IoU. It is intended for debugging very low mAP where
reasonable boxes may exist but are ranked behind poor boxes.
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.rgb_ir_depth_dataset import RGBIRDepthDataset
from engine.evaluator import class_aware_nms
from engine.trainer import collate_fn
from models.detector.matcher import box_cxcywh_to_xyxy
from models.m3f_detr import M3F_DETR
from utils.checkpoint import _safe_torch_load, strip_state_dict_prefixes


def box_iou_cxcywh(boxes1, boxes2):
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


def build_model(checkpoint, device, use_ema=False):
    state = _safe_torch_load(checkpoint, map_location="cpu")
    cfg = state.get("cfg", {}) or {}
    image_size = tuple(cfg.get("image_size", (384, 640)))
    print("Checkpoint cfg:", cfg)

    model = M3F_DETR(
        num_classes=cfg.get("num_classes", 12),
        hidden_dim=cfg.get("hidden_dim", 256),
        num_queries=cfg.get("num_queries", 300),
        backbone_name=cfg.get("backbone", "swin_tiny"),
        pretrained=False,
        use_dn=cfg.get("use_dn", False),
        input_size=image_size,
        decoder_feature_level=cfg.get("decoder_feature_level", -1),
    ).to(device).eval()

    key = "ema" if use_ema and state.get("ema") else "model"
    print(f"Using weights: {key}")
    model.load_state_dict(strip_state_dict_prefixes(state[key]))
    return model


def make_loader(data_root, split_dir, fold, batch_size, num_workers, image_size):
    dataset = RGBIRDepthDataset(data_root, train=True, size=image_size)
    val_file = os.path.join(split_dir, f"fold{fold}_val.txt")
    with open(val_file) as f:
        val_stems = set(line.strip() for line in f if line.strip())

    val_indices = [
        idx for idx, name in enumerate(dataset.rgb_names)
        if os.path.splitext(name)[0] in val_stems
    ]
    loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"Validation samples: {len(val_indices)}")
    return loader


@torch.no_grad()
def diagnose(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    model = build_model(args.checkpoint, device, args.use_ema)
    state = _safe_torch_load(args.checkpoint, map_location="cpu")
    image_size = tuple((state.get("cfg", {}) or {}).get("image_size", (384, 640)))
    loader = make_loader(
        args.data_root,
        args.split_dir,
        args.fold,
        args.batch_size,
        args.num_workers,
        image_size,
    )

    all_scores = []
    all_ious = []
    top1_ious = []
    best_ious = []
    first_good_ranks = []
    good_scores = []
    poor_scores = []
    image_count = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        rgb = batch["rgb"].to(device, non_blocking=True)
        ir = batch["ir"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        targets = batch["target"]

        outputs = model(rgb, ir, depth)
        probs = outputs["pred_logits"][:, :, :-1].sigmoid()
        boxes = outputs["pred_boxes"].clamp(0, 1)
        scores, labels = probs.max(-1)

        for i in range(rgb.shape[0]):
            gt_boxes = targets[i]["boxes"].to(device)
            gt_labels = targets[i]["labels"].to(device).long()
            if gt_boxes.numel() == 0:
                continue

            keep = scores[i] > args.conf_threshold
            pred_scores = scores[i][keep]
            pred_labels = labels[i][keep].long()
            pred_boxes = boxes[i][keep]
            if pred_scores.numel() == 0:
                continue

            keep_idx = class_aware_nms(
                pred_boxes,
                pred_scores,
                pred_labels,
                iou_threshold=args.nms_iou,
                max_dets=args.max_dets,
            )
            pred_scores = pred_scores[keep_idx]
            pred_labels = pred_labels[keep_idx]
            pred_boxes = pred_boxes[keep_idx]

            order = pred_scores.argsort(descending=True)
            pred_scores = pred_scores[order]
            pred_labels = pred_labels[order]
            pred_boxes = pred_boxes[order]

            ious = box_iou_cxcywh(pred_boxes, gt_boxes)
            same_class = pred_labels[:, None] == gt_labels[None, :]
            same_class_ious = ious.masked_fill(~same_class, 0)
            pred_best_iou = same_class_ious.max(dim=1).values

            all_scores.append(pred_scores.detach().cpu())
            all_ious.append(pred_best_iou.detach().cpu())
            top1_ious.append(float(pred_best_iou[0]))
            best_ious.append(float(pred_best_iou.max()))

            good = pred_best_iou >= args.good_iou
            if good.any():
                rank = int(torch.nonzero(good, as_tuple=False).flatten()[0]) + 1
                first_good_ranks.append(rank)
                good_scores.append(float(pred_scores[good].max()))
            poor = pred_best_iou < args.poor_iou
            poor_scores.extend(pred_scores[poor][:10].detach().cpu().tolist())
            image_count += 1

    scores = torch.cat(all_scores) if all_scores else torch.empty(0)
    ious = torch.cat(all_ious) if all_ious else torch.empty(0)
    if scores.numel() > 1 and ious.std() > 0 and scores.std() > 0:
        corr = torch.corrcoef(torch.stack([scores.float(), ious.float()]))[0, 1].item()
    else:
        corr = 0.0

    def avg(values):
        return sum(values) / max(len(values), 1)

    print("\n=== Score-IoU diagnosis ===")
    print(f"Images scanned: {image_count}")
    print(f"Predictions kept: {scores.numel()}")
    if scores.numel() > 0:
        print(f"Score mean/max: {scores.mean():.4f}/{scores.max():.4f}")
        print(f"Same-class IoU mean/max: {ious.mean():.4f}/{ious.max():.4f}")
    print(f"Score-IoU corr: {corr:.4f}")
    print(f"Top1 same-class IoU avg: {avg(top1_ious):.4f}")
    print(f"Best same-class IoU per image avg: {avg(best_ious):.4f}")
    print(f"Images with IoU>={args.good_iou}: {len(first_good_ranks)}")
    print(f"First good-box rank avg: {avg(first_good_ranks):.2f}")
    if first_good_ranks:
        mid = len(first_good_ranks) // 2
        print(f"First good-box rank median: {sorted(first_good_ranks)[mid]}")
    print(f"Good-box score avg: {avg(good_scores):.4f}")
    print(f"Poor-box score avg: {avg(poor_scores):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose confidence vs IoU ranking")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/train")
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--conf-threshold", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.6)
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--good-iou", type=float, default=0.5)
    parser.add_argument("--poor-iou", type=float, default=0.1)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    diagnose(args)


if __name__ == "__main__":
    main()
