"""Sweep inference thresholds and NMS for validation mAP.

Usage:
    python tools/sweep_eval.py --checkpoint checkpoints/rush_v2/latest.pth --data-root data/train
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.rgb_ir_depth_dataset import RGBIRDepthDataset
from engine import collate_fn, compute_map, evaluate_model
from models.m3f_detr import M3F_DETR
from utils.checkpoint import _safe_torch_load, strip_state_dict_prefixes


def parse_float_list(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def load_model(checkpoint, device, backbone=None, num_classes=12):
    state = _safe_torch_load(checkpoint, map_location="cpu")
    cfg = state.get("cfg", {}) if isinstance(state, dict) else {}
    image_size = tuple(cfg.get("image_size", (384, 640)))

    backbone_name = backbone or cfg.get("backbone", "swin_large")
    model = M3F_DETR(
        num_classes=cfg.get("num_classes", num_classes),
        hidden_dim=cfg.get("hidden_dim", 256),
        num_queries=cfg.get("num_queries", 900),
        backbone_name=backbone_name,
        use_dn=cfg.get("use_dn", True),
        input_size=image_size,
    ).to(device)

    if device != "cpu":
        state = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in state.items()
            if k != "cfg"
        }

    if "model" in state:
        model.load_state_dict(strip_state_dict_prefixes(state["model"]))
    else:
        model.load_state_dict(strip_state_dict_prefixes({k: v for k, v in state.items() if k != "cfg"}))
    model.eval()
    return model, image_size


def main():
    parser = argparse.ArgumentParser(description="Sweep validation thresholds/NMS")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default=None)
    parser.add_argument(
        "--thresholds",
        default="0.001,0.005,0.01,0.03,0.05,0.08,0.10,0.15,0.20,0.30",
    )
    parser.add_argument("--nms-ious", default="0.4,0.5,0.6,0.7")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, image_size = load_model(args.checkpoint, device, args.backbone, args.num_classes)
    dataset = RGBIRDepthDataset(args.data_root, train=True, size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    rows = []
    for nms_iou in parse_float_list(args.nms_ious):
        for conf in parse_float_list(args.thresholds):
            predictions, targets = evaluate_model(
                model,
                loader,
                device,
                use_amp=True,
                conf_threshold=conf,
                max_dets=100,
                nms_iou=nms_iou,
            )
            result = compute_map(predictions, targets, args.num_classes)
            rows.append((result["mAP50-95"], result["mAP50"], conf, nms_iou, len(predictions)))
            print(
                f"conf={conf:.3f} nms={nms_iou:.2f} preds={len(predictions):6d} "
                f"mAP50-95={result['mAP50-95']:.4f} mAP50={result['mAP50']:.4f}"
            )

    rows.sort(reverse=True)
    print("\nBest settings:")
    for map5095, map50, conf, nms_iou, num_preds in rows[:10]:
        print(
            f"conf={conf:.3f} nms={nms_iou:.2f} preds={num_preds:6d} "
            f"mAP50-95={map5095:.4f} mAP50={map50:.4f}"
        )


if __name__ == "__main__":
    main()
