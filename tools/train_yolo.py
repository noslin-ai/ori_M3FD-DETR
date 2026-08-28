"""YOLO 训练脚本 — v0.8.0 方向切换后的主训练入口。

与旧 train.py 的差异:
    - 使用 ultralytics YOLO11（官方 backbone/neck/head/损失），
      仅自研"数据加载 + 首层卷积通道扩展 + 训练循环胶水"；
    - mode=rgb 先跑对照，验证环境/数据/评估链路；
    - mode=fusion 输出 5 通道早期融合（RGB+IR+Depth）。

运行方式:
    # RGB-only 对照（必须先跑通）
    python tools/train_yolo.py --config configs/yolo_rgb.yaml --fold 1

    # 5ch 早期融合（主路径）
    python tools/train_yolo.py --config configs/yolo_fusion.yaml --fold 1

    # 小数据过拟合冒烟（CPU 或无卡模式验证代码正确性）
    python tools/train_yolo.py --config configs/yolo_rgb.yaml --epochs 1 \
        --device cpu --batch-size 2
"""

import os
import sys
import argparse

import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader, Subset

from yolo.dataset import YOLOFusionDataset, collate_fn, get_fold_indices
from yolo.model import build_yolo_model, apply_freeze, count_parameters
from yolo.evaluate import evaluate
from engine.ema import EMA
from utils.scheduler import build_scheduler
from utils.seed import set_seed
from utils.amp import get_scaler


def build_optimizer(model, config):
    """构建 AdamW 优化器（head 用完整 lr，其余层用 backbone_lr）。"""
    head_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Detect 头在 model.model[-1]（yolo11 为 index 23）
        if name.startswith("model.23"):
            head_params.append(p)
        else:
            other_params.append(p)

    lr = float(config["train"]["lr"])
    backbone_lr = float(config["train"].get("backbone_lr", lr * 0.1))
    weight_decay = float(config["train"].get("weight_decay", 0.0005))

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})
    if not param_groups:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def train_one_epoch(model, loader, optimizer, device, scaler, ema=None,
                    max_norm=10.0, use_amp=True, log_interval=20):
    """训练一个 epoch，返回平均损失与分项损失。"""
    model.train()
    total_loss = 0.0
    total_box = 0.0
    total_cls = 0.0
    total_dfl = 0.0
    n_batches = len(loader)

    for i, batch in enumerate(loader):
        batch = {
            "img": batch["img"].to(device, non_blocking=True),
            "cls": batch["cls"].to(device, non_blocking=True),
            "bboxes": batch["bboxes"].to(device, non_blocking=True),
            "batch_idx": batch["batch_idx"].to(device, non_blocking=True),
        }

        optimizer.zero_grad()

        # AMP 前向（BF16: 同指数范围无梯度溢出，RTX 5090 Blackwell 原生支持）
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss, loss_items = model(batch)

        # 官方 v8DetectionLoss 返回 (3,) 的 box/cls/dfl 分项，求和后反向
        loss_sum = loss.sum()
        scaler.scale(loss_sum).backward()

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)
            ema.update_buffers(model)

        total_loss += loss_sum.item()
        total_box += float(loss_items.get("box_loss", 0.0))
        total_cls += float(loss_items.get("cls_loss", 0.0))
        total_dfl += float(loss_items.get("dfl_loss", 0.0))

        if (i + 1) % log_interval == 0:
            print(
                f"    [{i+1}/{n_batches}] "
                f"loss={loss_sum.item():.4f} avg={total_loss / (i + 1):.4f} "
                f"box={total_box / (i + 1):.4f} "
                f"cls={total_cls / (i + 1):.4f} "
                f"dfl={total_dfl / (i + 1):.4f}"
            )

    n = max(n_batches, 1)
    return total_loss / n, {"box": total_box / n, "cls": total_cls / n, "dfl": total_dfl / n}


def save_yolo_checkpoint(path, model, ema, optimizer, epoch, best_metric, cfg):
    """保存 checkpoint（与旧 utils/checkpoint.py 的字段风格一致）。"""
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "cfg": cfg,
        "epoch": epoch,
        "best_metric": best_metric,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="YOLO Training (v0.8.0)")
    parser.add_argument("--config", default="configs/yolo_rgb.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, switching to CPU")
        device = "cpu"
    device = torch.device(device)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.epochs:
        config["train"]["epochs"] = args.epochs

    set_seed(config.get("seed", 42))
    mode = config["dataset"].get("mode", "rgb")
    ch = 5 if mode == "fusion" else 3
    nc = config["dataset"]["num_classes"]
    image_size = (config["input"]["height"], config["input"]["width"])

    print("=" * 70)
    print(f"  YOLO Training System (v0.8.0, mode={mode})")
    print("=" * 70)
    print(f"  Config: {args.config}")
    print(f"  Channels: {ch} | Classes: {nc} | Image: {image_size}")
    print(f"  Device: {device}")

    # ---- 数据集 ----
    print("\n[1] Loading dataset...")
    full_dataset = YOLOFusionDataset(
        config["dataset"]["root"], mode=mode, size=image_size, train=True, nc=nc
    )
    print(f"  Total samples: {len(full_dataset)}")

    if args.fold > 0:
        train_indices, val_indices = get_fold_indices(
            full_dataset, args.fold, config["cv"]["split_dir"]
        )
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices) if val_indices else None
        print(f"  Fold {args.fold}: train={len(train_indices)}, val={len(val_indices)}")
    else:
        train_dataset = full_dataset
        val_dataset = None

    num_workers = config.get("num_workers", 8)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0 and device.type == "cuda"),
    )
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )

    # ---- 模型 ----
    print("\n[2] Building model...")
    model = build_yolo_model(
        ch=ch,
        nc=nc,
        pretrained=config["model"].get("pretrained", "yolo11n.pt"),
        verbose=False,
    ).to(device)
    apply_freeze(model, config["model"].get("freeze", []))
    total, trainable = count_parameters(model)
    print(f"  Total params: {total / 1e6:.1f}M")
    print(f"  Trainable: {trainable / 1e6:.1f}M")

    # ---- 优化器 / 调度器 / AMP / EMA ----
    print("\n[3] Optimizer & EMA...")
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(
        optimizer,
        config["train"]["epochs"],
        scheduler_type=config["scheduler"].get("type", "warmup_cosine"),
        warmup_epochs=config["scheduler"].get("warmup_epochs", 3),
    )
    use_amp = config["train"].get("amp", True) and device.type == "cuda"
    scaler = get_scaler(enabled=use_amp)

    ema = None
    if config["train"].get("ema", True):
        ema = EMA(model, decay=config["train"].get("ema_decay", 0.9999))
        ema.model = ema.model.to(device)
        print(f"  EMA enabled (decay={config['train'].get('ema_decay', 0.9999)})")

    cfg_dict = {
        "mode": mode,
        "ch": ch,
        "nc": nc,
        "image_size": image_size,
        "pretrained": config["model"].get("pretrained", "yolo11n.pt"),
        "conf_thres": config.get("postprocess", {}).get("conf_thres", 0.001),
        "iou_thres": config.get("postprocess", {}).get("iou_thres", 0.6),
        "max_det": config.get("postprocess", {}).get("max_det", 100),
    }

    # ---- 恢复训练 ----
    start_epoch = 0
    best_map = 0.0
    if args.resume:
        print(f"\n[4] Resuming from {args.resume}")
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if ema is not None and state.get("ema") is not None:
            ema.load_state_dict(state["ema"])
        if state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        start_epoch = state.get("epoch", 0) + 1
        best_map = state.get("best_metric", 0.0)

    # ---- 训练 ----
    print("\n[5] Training starts!")
    print("=" * 70)

    save_dir = config["checkpoint"]["save_dir"]
    save_freq = config["checkpoint"]["save_freq"]
    log_interval = config.get("log", {}).get("log_interval", 20)
    val_interval = config["train"].get("val_interval", 5)
    epochs = config["train"]["epochs"]
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        print(f"\n  Epoch {epoch + 1}/{epochs}")
        avg_loss, items = train_one_epoch(
            model, train_loader, optimizer, device, scaler, ema,
            max_norm=config["train"].get("grad_clip", 10.0),
            use_amp=use_amp,
            log_interval=log_interval,
        )
        scheduler.step()
        lr = optimizer.param_groups[-1]["lr"]
        print(
            f"  Loss: {avg_loss:.4f} | "
            f"box={items['box']:.4f} cls={items['cls']:.4f} dfl={items['dfl']:.4f} | "
            f"LR: {lr:.6f}"
        )

        # 验证（best 选择用原始模型；EMA 仅作参考输出。
        # 注意: ema_decay 若过大（如 0.9999）而每 epoch 步数少，
        # EMA 权重会严重滞后，用 EMA 选 best 会把最好的 checkpoint 错过）
        if val_loader and (epoch + 1) % val_interval == 0:
            print("\n  --- Validation ---")
            results = evaluate(
                model, val_loader, device, num_classes=nc,
                conf_thres=cfg_dict["conf_thres"],
                iou_thres=cfg_dict["iou_thres"],
                max_det=cfg_dict["max_det"],
                use_amp=use_amp,
            )
            if ema is not None:
                print("  --- Validation (EMA) ---")
                results_ema = evaluate(
                    ema.model, val_loader, device, num_classes=nc,
                    conf_thres=cfg_dict["conf_thres"],
                    iou_thres=cfg_dict["iou_thres"],
                    max_det=cfg_dict["max_det"],
                    use_amp=use_amp,
                )

            map_5095 = results.get("mAP50-95", 0.0)
            if map_5095 > best_map:
                best_map = map_5095
                save_yolo_checkpoint(
                    os.path.join(save_dir, "best.pth"),
                    model, ema, optimizer, epoch, best_map, cfg_dict,
                )
                print(f"  🏆 New Best mAP@50-95: {best_map:.4f}")

        # 定期保存
        if (epoch + 1) % save_freq == 0:
            save_yolo_checkpoint(
                os.path.join(save_dir, "latest.pth"),
                model, ema, optimizer, epoch, best_map, cfg_dict,
            )

    # 最终保存
    save_yolo_checkpoint(
        os.path.join(save_dir, "final.pth"),
        model, ema, optimizer, epochs - 1, best_map, cfg_dict,
    )
    print("\n" + "=" * 70)
    print(f"  Training complete! Best mAP@50-95: {best_map:.4f}")
    print(f"  Checkpoints: {save_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
