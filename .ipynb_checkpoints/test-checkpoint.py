"""M3F-DINO 测试脚本 — 快速验证模型 pipeline。

用于:
    1. 验证数据加载正常
    2. 验证模型前向通过
    3. 验证损失计算
    4. 验证推理流程

运行方式:
    cd M3F-DETR
    python test.py
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.rgb_ir_depth_dataset import RGBIRDepthDataset
from models.m3f_detr import M3F_DETR
from models.losses import DINOLoss
from engine.trainer import collate_fn


def test_dataloader(data_root="data/train"):
    """测试数据加载器。"""
    print("=" * 70)
    print("  Test 1: DataLoader")
    print("=" * 70)

    dataset = RGBIRDepthDataset(data_root, train=True)
    print(f"  Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("  ⚠ Dataset is empty! Check data path.")
        return False

    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=collate_fn,
    )

    batch = next(iter(loader))
    print(f"  RGB shape:   {batch['rgb'].shape}")
    print(f"  IR shape:    {batch['ir'].shape}")
    print(f"  Depth shape: {batch['depth'].shape}")
    print(f"  Targets:     {len(batch['target'])} samples")
    for i, t in enumerate(batch["target"][:2]):
        print(f"    [{i}] boxes={t['boxes'].shape}, labels={t['labels'].shape}")

    return True


def test_model_forward(data_root="data/train"):
    """测试模型前向传播。"""
    print("\n" + "=" * 70)
    print("  Test 2: Model Forward Pass")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    dataset = RGBIRDepthDataset(data_root, train=True)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))

    model = M3F_DETR(num_classes=12, use_dn=True).to(device)
    model.train()

    rgb = batch["rgb"].to(device)
    ir = batch["ir"].to(device)
    depth = batch["depth"].to(device)
    targets = batch["target"]
    for t in targets:
        t["boxes"] = t["boxes"].to(device)
        t["labels"] = t["labels"].to(device)
    with torch.no_grad():
        output = model(rgb, ir, depth, targets)

    print(f"  pred_logits: {output['pred_logits'].shape}")
    print(f"  pred_boxes:  {output['pred_boxes'].shape}")
    print(f"  Forward pass: OK")

    return True


def test_loss_computation(data_root="data/train"):
    """测试损失计算。"""
    print("\n" + "=" * 70)
    print("  Test 3: Loss Computation")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = RGBIRDepthDataset(data_root, train=True)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))

    model = M3F_DETR(num_classes=12, use_dn=False).to(device)
    model.train()

    rgb = batch["rgb"].to(device)
    ir = batch["ir"].to(device)
    depth = batch["depth"].to(device)
    targets = batch["target"]
    for t in targets:
        t["boxes"] = t["boxes"].to(device)
        t["labels"] = t["labels"].to(device)

    output = model(rgb, ir, depth, targets)

    criterion = DINOLoss(num_classes=12).to(device)
    loss_dict = criterion(output, targets)

    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")

    loss = loss_dict["loss"]
    loss.backward()

    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"\n  Total grad norm: {total_grad:.4f}")
    print(f"  Loss + Backward: OK")

    return True


def test_inference(data_root="data/train"):
    """测试推理流程。"""
    print("\n" + "=" * 70)
    print("  Test 4: Inference Pipeline")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = RGBIRDepthDataset(data_root, train=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))

    model = M3F_DETR(num_classes=12, use_dn=False).to(device)
    model.eval()

    rgb = batch["rgb"].to(device)
    ir = batch["ir"].to(device)
    depth = batch["depth"].to(device)

    with torch.no_grad():
        output = model(rgb, ir, depth)

    pred_logits = output["pred_logits"]
    pred_boxes = output["pred_boxes"]

    # Softmax + top-k
    scores = pred_logits[0].softmax(-1)
    max_scores, labels = scores.max(-1)
    valid = (labels < scores.shape[-1] - 1) & (max_scores > 0.3)

    print(f"  pred_logits: {pred_logits.shape}")
    print(f"  pred_boxes:  {pred_boxes.shape}")
    print(f"  Detections (conf > 0.3): {valid.sum().item()} boxes")
    print(f"  Inference: OK")

    return True


def main():
    print("M3F-DINO Pipeline Test")
    print("=" * 70)

    # 检查数据路径
    data_root = "data/train"
    if not os.path.isdir(data_root):
        print(f"\n  ⚠ Data directory not found: {data_root}")
        print("  Please unzip dataset first:")
        print("    unzip data/AIC2026_Train_2000.zip -d data/train/")
        return

    tests = [
        ("DataLoader", test_dataloader),
        ("Model Forward", test_model_forward),
        ("Loss Computation", test_loss_computation),
        ("Inference Pipeline", test_inference),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn(data_root):
                passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  ❌ Test '{name}' FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
