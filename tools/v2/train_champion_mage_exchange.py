"""Train zero-init MAGE/CSSA-style exchange refiners from the 57.507 model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path("/root/autodl-tmp/aic_race/M3F-DETR")
sys.path.insert(0, str(ROOT / "tools/v2"))

import adapter_model  # noqa: F401,E402 - checkpoint pickle dependency
import distribution_aligned_adapter  # noqa: F401,E402 - checkpoint pickle dependency
from adapter_dataset import TriModalDetectionTrainer  # noqa: E402
from distribution_aligned_adapter import DistributionAlignedAdapterModel  # noqa: E402
from mage_exchange_adapter import attach_exchange_refiners  # noqa: E402


class ExchangeOnlyTrainer(TriModalDetectionTrainer):
    def _setup_train(self):
        super()._setup_train()
        model = unwrap_model(self.model)
        trainable = []
        for name, parameter in model.named_parameters():
            keep = name.startswith("exchange_refiners.")
            parameter.requires_grad_(keep)
            if keep:
                trainable.append(name)
        if not trainable or any(not name.startswith("exchange_refiners.") for name in trainable):
            raise RuntimeError(f"invalid exchange-only trainable set: {trainable}")
        model.train()
        unfrozen_bn = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training
        ]
        if unfrozen_bn:
            raise RuntimeError(f"pretrained BN buffers are not locked: {unfrozen_bn}")
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"EXCHANGE_ONLY_TRAINABLE tensors={len(trainable)} params={count}", flush=True)


def flatten_tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return sum((flatten_tensors(value[key]) for key in sorted(value)), [])
    if isinstance(value, (tuple, list)):
        return sum((flatten_tensors(item) for item in value), [])
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default=str(ROOT / "runs/detect/runs/s5/champion_da_ir_p23/weights/best.pt"),
    )
    parser.add_argument("--data", default=str(ROOT / "data/yolo_full2000_soft/data.yaml"))
    parser.add_argument("--project", default=str(ROOT / "runs/detect/runs/s8"))
    parser.add_argument("--name", default="champion_mage_exchange_p23")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--align-weight", type=float, default=0.02)
    parser.add_argument("--lr0", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=999)
    args = parser.parse_args()

    reference = YOLO(args.base).model.float().cuda().eval()
    if not isinstance(reference, DistributionAlignedAdapterModel):
        raise TypeError("base must be the 57.507 distribution-aligned IR adapter")
    yolo = YOLO(args.base)
    yolo.model = attach_exchange_refiners(yolo.model, args.align_weight).float().cuda().eval()

    probe = torch.rand(1, 6, 128, 128, device="cuda")
    with torch.inference_mode():
        expected = flatten_tensors(reference(probe))
        actual = flatten_tensors(yolo.model(probe))
        if len(expected) != len(actual):
            raise RuntimeError(f"identity output count mismatch: {len(expected)} != {len(actual)}")
        max_diff = max((left - right).abs().max().item() for left, right in zip(expected, actual))
    print(f"MAGE_ZERO_IDENTITY_MAX_DIFF {max_diff}", flush=True)
    if max_diff != 0.0:
        raise RuntimeError("zero exchange refiner changed the 57.507 model output")

    yolo.train(
        trainer=ExchangeOnlyTrainer,
        data=args.data,
        project=args.project,
        name=args.name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=1280,
        fraction=1.0,
        cache=False,
        device=0,
        workers=8,
        freeze=24,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.05,
        cos_lr=True,
        warmup_epochs=2.0,
        momentum=0.937,
        weight_decay=5e-4,
        seed=46,
        deterministic=True,
        patience=args.patience,
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
