"""M3F-DETR 评估脚本。

功能:
    1. 加载 checkpoint，在验证集/测试集上推理
    2. 计算 mAP@50-95
    3. 导出预测结果为 COCO JSON

运行方式:
    cd M3F-DETR
    # 评估 checkpoint
    python evaluate.py --checkpoint checkpoints/best.pth --data-root data/train
    # 在测试集上推理（无标签）
    python evaluate.py --checkpoint checkpoints/best.pth --data-root data/test --no-labels
    # 导出 COCO JSON
    python evaluate.py --checkpoint checkpoints/best.pth --data-root data/test --no-labels --export-json results.json
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.rgb_ir_depth_dataset import RGBIRDepthDataset
from models.m3f_detr import M3F_DETR
from engine import evaluate_model, validate, collate_fn
from utils.checkpoint import _safe_torch_load


def main():
    parser = argparse.ArgumentParser(description="M3F-DETR 评估")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 路径")
    parser.add_argument("--data-root", required=True, help="数据根目录")
    parser.add_argument("--no-labels", action="store_true", help="无标签（测试集）")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--export-json", default=None, help="导出 COCO JSON 路径")
    parser.add_argument("--use-ema", action="store_true", help="使用 EMA 权重")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA 不可用，切换到 CPU")
        device = "cpu"

    print("=" * 70)
    print("  M3F-DETR Evaluation")
    print("=" * 70)

    # ---- 数据集 ----
    print(f"\n[1] 加载数据: {args.data_root}")
    dataset = RGBIRDepthDataset(args.data_root, train=not args.no_labels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"  样本数: {len(dataset)}")

    # ---- 模型 ----
    print("\n[2] 加载模型...")
    model = M3F_DETR(num_classes=args.num_classes).to(device)

    # 加载 checkpoint
    print(f"  Checkpoint: {args.checkpoint}")
    state = _safe_torch_load(args.checkpoint, map_location=device)

    # 使用 EMA 权重
    if args.use_ema and "ema" in state and state["ema"]:
        print("  使用 EMA 权重")
        model.load_state_dict(state["ema"])
    else:
        print("  使用普通权重")
        model.load_state_dict(state["model"] if "model" in state else state)

    model.eval()

    # ---- 推理 ----
    print("\n[3] 开始推理...")
    predictions, targets = evaluate_model(model, loader, device, use_amp=True)

    print(f"  预测框: {len(predictions)}")
    print(f"  真实框: {len(targets)}")

    # ---- 导出 JSON ----
    if args.export_json:
        output = {
            "predictions": predictions,
            "targets": targets if not args.no_labels else [],
        }
        with open(args.export_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  结果已导出: {args.export_json}")

    # ---- mAP 计算 ----
    if not args.no_labels and len(targets) > 0:
        print("\n[4] 计算 mAP...")
        from engine import compute_map
        results = compute_map(predictions, targets, args.num_classes)

        print(f"\n  {'指标':<15} {'值'}")
        print(f"  {'─' * 30}")
        print(f"  {'mAP@50-95':<15} {results['mAP50-95']:.4f}")
        print(f"  {'mAP@50':<15} {results['mAP50']:.4f}")
        print(f"  {'mAP@75':<15} {results['mAP75']:.4f}")

        if "per_class" in results:
            print(f"\n  {'─' * 30}")
            print("  各类别 AP@50-95:")
            for cls_id, ap in results["per_class"].items():
                print(f"    Class {cls_id}: {ap:.4f}")
    else:
        print("\n  ⚠ 无标签数据，跳过 mAP 计算")

    print("\n" + "=" * 70)
    print("  评估完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
