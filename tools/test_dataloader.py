"""Dataloader + Model 测试脚本。

测试内容:
    1. Dataloader 正常加载三模态数据
    2. M3F-DETR 模型前向推理（DINO 检测器）
    3. DINOLoss 计算损失

运行方式:
    cd M3F-DETR
    python tools/test_dataloader.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from datasets.rgb_ir_depth_dataset import RGBIRDepthDataset
from models.m3f_detr import M3F_DETR
from models.losses import DINOLoss


def test_dataloader():
    """测试 Dataloader 三模态数据加载"""
    print("=" * 60)
    print("[1] 测试 Dataloader")
    print("=" * 60)

    dataset = RGBIRDepthDataset("data/train", train=True)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x,
    )

    for batch in loader:
        sample = batch[0]
        print("  RGB:   ", sample["rgb"].shape)
        print("  IR:    ", sample["ir"].shape)
        print("  Depth: ", sample["depth"].shape)
        print("  Boxes: ", sample["target"]["boxes"])
        print("  Labels:", sample["target"]["labels"])
        break

    print("  ✓ Dataloader OK\n")


def test_model_forward():
    """测试 M3F-DETR 模型前向推理（DINO 检测器）"""
    print("=" * 60)
    print("[2] 测试 M3F-DETR 前向推理")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model = M3F_DETR(num_classes=12).to(device)
    model.eval()

    # 模拟输入 (B=2, 3, 768, 1280)
    B = 2
    rgb = torch.randn(B, 3, 768, 1280).to(device)
    ir = torch.randn(B, 3, 768, 1280).to(device)
    depth = torch.randn(B, 3, 768, 1280).to(device)

    with torch.no_grad():
        output = model(rgb, ir, depth)

    print(f"  pred_logits: {output['pred_logits'].shape}")
    # 期望: (B, num_queries, 13) — 12 类 + 背景
    print(f"  pred_boxes:  {output['pred_boxes'].shape}")
    # 期望: (B, num_queries, 4) — cx, cy, w, h 归一化

    print("  ✓ Model forward OK\n")
    return output


def test_loss(output):
    """测试 DINOLoss 计算"""
    print("=" * 60)
    print("[3] 测试 DINOLoss")
    print("=" * 60)

    device = output["pred_logits"].device
    B = output["pred_logits"].shape[0]

    # 模拟 target
    targets = []
    for _ in range(B):
        n_obj = 3  # 每张图 3 个目标
        labels = torch.randint(0, 12, (n_obj,), device=device)
        boxes = torch.rand(n_obj, 4, device=device)
        # 确保 w, h > 0
        boxes[:, 2] = boxes[:, 2].clamp(min=0.1)
        boxes[:, 3] = boxes[:, 3].clamp(min=0.1)
        targets.append({"labels": labels, "boxes": boxes})

    criterion = DINOLoss(num_classes=12)
    losses = criterion(output, targets)

    print(f"  loss_class: {losses['loss_class']:.4f}")
    print(f"  loss_bbox:  {losses['loss_bbox']:.4f}")
    print(f"  loss_giou:  {losses['loss_giou']:.4f}")
    print(f"  total:      {losses['loss'].item():.4f}")

    # 测试反向传播
    losses["loss"].backward()
    print("  ✓ Backward OK\n")


def main():
    print("\n🧪 M3F-DETR Pipeline Test\n")

    # 测试 1: Dataloader（需要解压数据后才能跑）
    try:
        test_dataloader()
    except Exception as e:
        print(f"  ⚠ Dataloader 跳过: {e}\n")

    # 测试 2: Model forward
    try:
        output = test_model_forward()
    except Exception as e:
        print(f"  ✗ Model forward 失败: {e}\n")
        return

    # 测试 3: Loss
    try:
        test_loss(output)
    except Exception as e:
        print(f"  ✗ Loss 失败: {e}\n")

    print("=" * 60)
    print("✅ 全部测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
