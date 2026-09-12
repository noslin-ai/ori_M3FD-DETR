"""Train only distribution-anchored IR/depth adapters on the 57.024 champion."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
import adapter_model  # noqa: F401 - required when loading adapter checkpoints
import distribution_aligned_adapter  # noqa: F401 - required by checkpoint pickle
from adapter_dataset import TriModalDetectionTrainer
from distribution_aligned_adapter import attach_distribution_aligned_adapters


def flatten_tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return sum((flatten_tensors(value[key]) for key in sorted(value)), [])
    if isinstance(value, (tuple, list)):
        return sum((flatten_tensors(item) for item in value), [])
    return []


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default=str(root / "runs/detect/runs/native_m_trimodal/full2000cont_1280_refine/weights/best.pt"),
    )
    parser.add_argument("--data", default=str(root / "data/yolo_full2000_soft/data.yaml"))
    parser.add_argument("--project", default=str(root / "runs/detect/runs/s5"))
    parser.add_argument("--name", default="champion_da_trimodal_p23")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--align-weight", type=float, default=0.05)
    parser.add_argument("--depth-valid-gate", action="store_true")
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()

    reference = YOLO(args.base).model.float().cuda().eval()
    yolo = YOLO(args.base)
    yolo.model = attach_distribution_aligned_adapters(
        yolo.model, align_weight=args.align_weight
    ).float().cuda().eval()
    yolo.model.depth_valid_gate = args.depth_valid_gate

    probe = torch.rand(1, 6, 128, 128, device="cuda")
    with torch.inference_mode():
        expected = flatten_tensors(reference(probe[:, :3]))
        actual = flatten_tensors(yolo.model(probe))
        if len(expected) != len(actual):
            raise RuntimeError(f"identity output count mismatch: {len(expected)} != {len(actual)}")
        max_diff = max((left - right).abs().max().item() for left, right in zip(expected, actual))
    print(f"CHAMPION_ZERO_IDENTITY_MAX_DIFF {max_diff}", flush=True)
    if max_diff != 0.0:
        raise RuntimeError("zero adapter changed the champion output")

    yolo.train(
        trainer=TriModalDetectionTrainer,
        data=args.data,
        project=args.project,
        name=args.name,
        epochs=args.epochs,
        fraction=args.fraction,
        batch=args.batch,
        imgsz=1280,
        cache=False,
        device=0,
        workers=8,
        freeze=24,
        optimizer="AdamW",
        lr0=5e-4,
        lrf=0.05,
        cos_lr=True,
        warmup_epochs=2.0,
        momentum=0.937,
        weight_decay=5e-4,
        seed=42,
        deterministic=True,
        patience=999,
        amp=True,
        plots=False,
        hsv_h=0.002,
        hsv_s=0.06,
        hsv_v=0.06,
        degrees=0.0,
        translate=0.03,
        scale=0.15,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment=None,
        erasing=0.0,
        multi_scale=0.0,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
