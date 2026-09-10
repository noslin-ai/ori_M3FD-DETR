"""Train-time custom bbox loss patch for small/tiny objects (compliant, single model).

Implements a blended CIoU + NWD (Normalized Gaussian Wasserstein Distance,
arXiv:2110.13389) bounding-box regression loss, targeting the observed
bottleneck: high mAP50 (detection) but lower mAP50-95 (localization precision at
high IoU) especially on small classes (ball, garbage can, sign, bicycle).

NWD models each box as a 2D Gaussian and measures Wasserstein distance, which is
far less sensitive to small localization deviations than IoU -> better for tiny
objects. We blend it with the default CIoU so large objects keep their proven
behaviour:
    sim = (1 - alpha) * CIoU + alpha * NWD        (both in [-1,1]-ish, higher=better)
and feed 1 - sim as the IoU loss. No multi-model ensembling; single detector.

Usage (monkeypatches ultralytics THEN trains):
    python tools/train_with_nwd.py --cfg configs/xxx.yaml --alpha 0.5 --C 16
"""
import argparse
import math
import os
import sys

import torch

# ---------------------------------------------------------------------------
# NWD similarity
# ---------------------------------------------------------------------------
def nwd_similarity(box1, box2, C: float = 16.0, eps: float = 1e-7):
    """boxes in xyxy (pixel units). Returns NWD in (0,1], higher = closer."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)
    w1, h1 = (b1_x2 - b1_x1).clamp(min=eps), (b1_y2 - b1_y1).clamp(min=eps)
    w2, h2 = (b2_x2 - b2_x1).clamp(min=eps), (b2_y2 - b2_y1).clamp(min=eps)
    cx1, cy1 = b1_x1 + w1 / 2, b1_y1 + h1 / 2
    cx2, cy2 = b2_x1 + w2 / 2, b2_y1 + h2 / 2
    # squared Wasserstein distance between the two Gaussian boxes
    w2dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4.0
    return torch.exp(-torch.sqrt(w2dist + eps) / C)


def apply_patch(alpha: float = 0.5, C: float = 16.0, verbose: bool = True):
    import ultralytics.utils.loss as L

    orig_bbox_iou = L.bbox_iou

    def blended_bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        base = orig_bbox_iou(box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, eps=eps)
        if not CIoU:
            return base
        # base is CIoU similarity in [-1,1]; convert to xyxy for NWD
        b1 = box1 if not xywh else torch.cat(
            (box1[..., 0:1] - box1[..., 2:3] / 2, box1[..., 1:2] - box1[..., 3:4] / 2,
             box1[..., 0:1] + box1[..., 2:3] / 2, box1[..., 1:2] + box1[..., 3:4] / 2), dim=-1)
        b2 = box2 if not xywh else torch.cat(
            (box2[..., 0:1] - box2[..., 2:3] / 2, box2[..., 1:2] - box2[..., 3:4] / 2,
             box2[..., 0:1] + box2[..., 2:3] / 2, box2[..., 1:2] + box2[..., 3:4] / 2), dim=-1)
        nwd = nwd_similarity(b1, b2, C=C)
        # NWD in (0,1]; map to [-1,1]-compatible similarity for blending
        nwd_sim = 2.0 * nwd - 1.0
        return (1.0 - alpha) * base + alpha * nwd_sim

    L.bbox_iou = blended_bbox_iou
    if verbose:
        print(f"[NWD-LOSS] patched ultralytics.utils.loss.bbox_iou  alpha={alpha} C={C}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="ultralytics train cfg yaml")
    ap.add_argument("--alpha", type=float, default=0.5, help="NWD weight in blend [0,1]")
    ap.add_argument("--C", type=float, default=16.0, help="NWD constant ~ avg object size (px)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--model", default=None, help="override model/weights")
    args = ap.parse_args()

    apply_patch(alpha=args.alpha, C=args.C)

    import yaml
    from ultralytics import YOLO

    cfg = yaml.safe_load(open(args.cfg, encoding="utf-8"))
    model_path = args.model or cfg.get("model")
    overrides = {k: v for k, v in cfg.items() if k != "model"}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.name is not None:
        overrides["name"] = args.name
    model = YOLO(model_path)
    model.train(**overrides)


if __name__ == "__main__":
    main()
