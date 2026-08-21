"""Trainer — 训练循环。

支持:
    1. AMP 混合精度训练
    2. EMA 模型更新
    3. 梯度裁剪
    4. 分阶段冻结/解冻参数
    5. Checkpoint 保存
"""

import torch
from torch.cuda.amp import autocast, GradScaler

from .ema import EMA


def collate_fn(batch):
    """将 Dataset 返回的 list of dict 整理为 batched 格式。

    Args:
        batch: list of {"rgb": (3,H,W), "ir": (3,H,W), "depth": (3,H,W),
                         "target": {"boxes": (N,4), "labels": (N,)}}

    Returns:
        dict:
            rgb:   (B, 3, H, W)
            ir:    (B, 3, H, W)
            depth: (B, 3, H, W)
            target: list of {"boxes": (N,4), "labels": (N,), "image_id": (1,)}
    """
    rgb = torch.stack([b["rgb"] for b in batch])
    ir = torch.stack([b["ir"] for b in batch])
    depth = torch.stack([b["depth"] for b in batch])
    targets = [b["target"] for b in batch]
    names = [b.get("name", "") for b in batch]
    return {"rgb": rgb, "ir": ir, "depth": depth, "target": targets, "name": names}


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    ema=None,
    max_norm=0.1,
    use_amp=True,
    log_interval=50,
):
    """训练一个 epoch。

    Args:
        model: M3F-DETR 模型
        loader: DataLoader
        optimizer: AdamW 优化器
        criterion: DINOLoss
        device: cuda / cpu
        scaler: AMP GradScaler
        ema: EMA 实例（可选）
        max_norm: 梯度裁剪阈值
        use_amp: 是否使用混合精度
        log_interval: 日志打印间隔

    Returns:
        avg_loss: 该 epoch 平均损失
    """
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_bbox = 0.0
    total_giou = 0.0
    n_batches = len(loader)

    for i, batch in enumerate(loader):
        rgb = batch["rgb"].to(device, non_blocking=True)
        ir = batch["ir"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        targets = batch["target"]

        # 将 target 移到 GPU
        for t in targets:
            t["boxes"] = t["boxes"].to(device, non_blocking=True)
            t["labels"] = t["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # AMP 前向 + 损失（BF16 优于 FP16: 同指数范围无梯度溢出，RTX 5090 原生支持）
        with autocast(enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(rgb, ir, depth, targets)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss"]

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度裁剪
        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

        # EMA 更新
        if ema is not None:
            ema.update(model)
            ema.update_buffers(model)

        total_loss += loss.item()
        total_cls += float(loss_dict.get("loss_class", 0.0))
        total_bbox += float(loss_dict.get("loss_bbox", 0.0))
        total_giou += float(loss_dict.get("loss_giou", 0.0))

        if (i + 1) % log_interval == 0:
            avg = total_loss / (i + 1)
            avg_cls = total_cls / (i + 1)
            avg_bbox = total_bbox / (i + 1)
            avg_giou = total_giou / (i + 1)
            print(
                f"    [{i+1}/{n_batches}] "
                f"loss={loss.item():.4f} avg={avg:.4f} "
                f"cls={avg_cls:.4f} bbox={avg_bbox:.4f} giou={avg_giou:.4f}"
            )

    return total_loss / max(n_batches, 1)
