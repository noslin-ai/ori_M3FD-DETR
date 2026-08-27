"""M3F-DINO 主训练脚本 — 比赛级训练框架。

支持:
    1. 单GPU / 多GPU (DDP via torchrun)
    2. AMP 混合精度
    3. EMA 模型平均
    4. 5-Fold 交叉验证
    5. 分阶段训练（冻结→解冻）
    6. 断点恢复
    7. mAP@50-95 验证

单卡运行:
    python train.py

多卡运行 (4×4090):
    torchrun --nproc_per_node=4 train.py

Debug 模式:
    python train.py --config configs/debug.yaml

指定 fold:
    python train.py --fold 1

断点恢复:
    python train.py --resume checkpoints/latest.pth
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.rgb_ir_depth_dataset import RGBIRDepthDataset
from models.m3f_detr import M3F_DETR
from models.losses import DINOLoss
from engine.ema import EMA
from engine.trainer import train_one_epoch, collate_fn
from engine.evaluator import validate
from utils import (
    set_seed, get_scaler, build_scheduler,
    save_checkpoint, load_checkpoint,
    init_distributed_mode, is_main_process,
    get_rank, get_world_size,
)


def get_fold_indices(dataset, fold, split_dir="splits"):
    """获取 fold 训练/验证索引。

    Args:
        dataset: 数据集
        fold: fold 编号 (1-based)
        split_dir: split 文件目录

    Returns:
        train_indices, val_indices
    """
    train_file = os.path.join(split_dir, f"fold{fold}_train.txt")
    val_file = os.path.join(split_dir, f"fold{fold}_val.txt")

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        if is_main_process():
            print(f"  ⚠ Fold split 未找到: {train_file}")
            print(f"  运行: python tools/split_5fold.py")
        return list(range(len(dataset))), []

    with open(train_file) as f:
        train_stems = set(line.strip() for line in f)
    with open(val_file) as f:
        val_stems = set(line.strip() for line in f)

    train_indices, val_indices = [], []
    for idx in range(len(dataset)):
        name = dataset.rgb_names[idx]
        stem = os.path.splitext(name)[0]
        if stem in val_stems:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    return train_indices, val_indices


def build_optimizer(model, config):
    """构建 AdamW 优化器（backbone 使用更小学习率）。"""
    backbone_params, head_params = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "rgb" in name or "backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    lr = float(config["optimizer"]["lr"])
    backbone_lr = float(config["optimizer"].get("backbone_lr", lr * 0.1))
    weight_decay = float(config["optimizer"].get("weight_decay", 0.05))

    return torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": lr},
    ], weight_decay=weight_decay)


def get_input_size(config):
    input_cfg = config.get("input", {})
    height = int(input_cfg.get("height", 384))
    width = int(input_cfg.get("width", 640))
    return height, width


def get_normalize_rgb(config):
    input_cfg = config.get("input", {})
    return bool(input_cfg.get("normalize_rgb", False))


def get_detector_cfg(config):
    model_cfg = config.get("model", {})
    anchor_box_size = model_cfg.get("anchor_box_size", (0.06, 0.12))
    return {
        "decoder_feature_level": model_cfg.get("decoder_feature_level", -1),
        "decoder_feature_levels": model_cfg.get("decoder_feature_levels"),
        "use_anchor_boxes": bool(model_cfg.get("use_anchor_boxes", False)),
        "anchor_box_size": tuple(anchor_box_size),
    }


def main():
    parser = argparse.ArgumentParser(description="M3F-DINO Training")
    parser.add_argument("--config", default="configs/m3f_dino.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # ---- 分布式初始化 ----
    dist_info = init_distributed_mode()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    local_rank = dist_info["local_rank"]
    is_main = is_main_process()

    # ---- 配置 ----
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.epochs:
        config["train"]["epochs"] = args.epochs

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(config.get("seed", 42))

    if is_main:
        print("=" * 70)
        print("  M3F-DINO Training System (Competition Ready)")
        print("=" * 70)
        print(f"  GPUs: {world_size}")
        print(f"  Config: {args.config}")
        print(f"  Batch size: {config['train']['batch_size']} (per GPU)")
        print(f"  Total epochs: {config['train']['epochs']}")
        print(f"  AMP: {config['train'].get('amp', True)}")
        print(f"  EMA: {config['train'].get('ema', False)}")

    # ---- 数据集 ----
    if is_main:
        print("\n[1] Loading dataset...")

    input_size = get_input_size(config)
    normalize_rgb = get_normalize_rgb(config)
    detector_cfg = get_detector_cfg(config)
    full_dataset = RGBIRDepthDataset(
        config["dataset"]["root"], train=True, size=input_size,
        normalize_rgb=normalize_rgb,
    )

    if is_main:
        print(f"  Total samples: {len(full_dataset)}")

    # Fold 划分
    if args.fold > 0:
        train_indices, val_indices = get_fold_indices(
            full_dataset, args.fold, config["cv"]["split_dir"]
        )
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices) if val_indices else None
        if is_main:
            print(f"  Fold {args.fold}: train={len(train_indices)}, val={len(val_indices)}")
    else:
        train_dataset = full_dataset
        val_dataset = None

    num_workers = config.get("num_workers", 8)

    # DDP: 使用 DistributedSampler 确保各 GPU 处理不同数据
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=3,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # ---- 模型 ----
    if is_main:
        print("\n[2] Building model...")

    model = M3F_DETR(
        num_classes=config["dataset"]["num_classes"],
        hidden_dim=config["model"]["hidden_dim"],
        num_queries=config["model"]["queries"],
        backbone_name=config["model"].get("backbone", "swin_large"),
        use_dn=config["model"].get("use_dn", False),
        pretrained=config["model"].get("pretrained", False),
        input_size=input_size,
        **detector_cfg,
    ).to(device)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )

    # torch.compile JIT 编译（PyTorch>=2.0, RTX 5090 Blackwell 优化）
    # 默认关闭：编译后 state_dict 键名可能带 `_orig_mod.` 前缀导致推理加载失败，
    # 确认 checkpoint 兼容（utils/checkpoint.py 已做前缀剥离）后再按需开启 compile: true
    if hasattr(torch, 'compile') and config.get("compile", False):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            if is_main:
                print("  torch.compile: enabled (mode=reduce-overhead)")
        except Exception:
            if is_main:
                print("  torch.compile: skipped (not supported)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"  Total params: {total_params / 1e6:.1f}M")
        print(f"  Trainable: {trainable_params / 1e6:.1f}M")

    # 模型配置字典（用于存入 checkpoint，推理时自动恢复）
    train_cfg = {
        "backbone": config["model"].get("backbone", "swin_large"),
        "hidden_dim": config["model"]["hidden_dim"],
        "num_queries": config["model"]["queries"],
        "num_classes": config["dataset"]["num_classes"],
        "use_dn": config["model"].get("use_dn", False),
        "image_size": input_size,
        "normalize_rgb": normalize_rgb,
        **detector_cfg,
    }

    # ---- 损失 ----
    loss_cfg = config["loss"]
    criterion = DINOLoss(
        num_classes=config["dataset"]["num_classes"],
        cost_class=loss_cfg.get("cost_class", 2.0),
        cost_bbox=loss_cfg.get("cost_bbox", 5.0),
        cost_giou=loss_cfg.get("cost_giou", 2.0),
        focal_alpha=loss_cfg.get("focal_alpha", 0.25),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
        class_weights=loss_cfg.get("class_weights"),
        cost_ce=loss_cfg.get("cost_ce", 0.0),
        aux_loss_weight=loss_cfg.get("aux_loss_weight", 0.5),
        quality_class_targets=loss_cfg.get("quality_class_targets", False),
        quality_floor=loss_cfg.get("quality_floor", 0.05),
    ).to(device)

    # ---- 优化器 & 调度器 ----
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(
        optimizer,
        config["train"]["epochs"],
        scheduler_type=config["scheduler"]["type"],
        warmup_epochs=config["scheduler"].get("warmup_epochs", 5),
    )

    # ---- AMP ----
    scaler = get_scaler(enabled=config["train"]["amp"])

    # ---- EMA ----
    ema = None
    if config["train"].get("ema"):
        ema_decay = config["train"].get("ema_decay", 0.9999)
        if is_main:
            print("\n[3] Initializing EMA...")
        ema = EMA(model, decay=ema_decay)
        ema.model = ema.model.to(device)

    # ---- 恢复训练 ----
    start_epoch = 0
    if args.resume:
        if is_main:
            print(f"\n[4] Resuming from {args.resume}")
        result = load_checkpoint(args.resume, model, optimizer, scaler, ema, device)
        start_epoch = result["epoch"]

    # ---- 训练阶段 ----
    stages = config["train"].get("stages", [
        {"name": "full", "epochs": config["train"]["epochs"], "freeze": [], "lr": config["optimizer"]["lr"]}
    ])

    if is_main:
        print("\n[5] Training starts!")
        print("=" * 70)

    best_map = 0.0
    save_dir = config["checkpoint"]["save_dir"]
    save_freq = config["checkpoint"]["save_freq"]
    log_interval = config["log"]["log_interval"]
    use_amp = config["train"]["amp"]
    total_epochs = config["train"]["epochs"]
    num_classes = config["dataset"]["num_classes"]

    current_epoch = start_epoch

    for stage in stages:
        stage_name = stage["name"]
        stage_epochs = stage["epochs"]
        freeze_list = stage.get("freeze", [])

        if is_main:
            print(f"\n{'─' * 70}")
            print(f"  Stage: {stage_name} | Epochs: {current_epoch}→{current_epoch + stage_epochs}")
            print(f"{'─' * 70}")

        # 冻结/解冻
        for p in model.parameters():
            p.requires_grad_(True)
        for mod_name in freeze_list:
            if hasattr(model, mod_name):
                for p in getattr(model, mod_name).parameters():
                    p.requires_grad_(False)
                if is_main:
                    print(f"  Frozen: {mod_name}")

        # 重建优化器（显式 float 转换防止 YAML 1e-4 被解析为字符串）
        if stage.get("lr") is not None:
            config["optimizer"]["lr"] = float(stage["lr"])
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(
            optimizer, stage_epochs,
            scheduler_type=config["scheduler"]["type"],
            warmup_epochs=config["scheduler"].get("warmup_epochs", 5),
        )

        for epoch in range(stage_epochs):
            if current_epoch >= total_epochs:
                break

            if is_main:
                print(f"\n  Epoch {current_epoch + 1}/{total_epochs} [{stage_name}]")

            if train_sampler is not None:
                train_sampler.set_epoch(current_epoch)

            # 训练
            avg_loss = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, scaler, ema,
                max_norm=config["train"]["grad_clip"],
                use_amp=use_amp,
                log_interval=log_interval,
            )

            scheduler.step()

            if is_main:
                print(f"  Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            # 验证
            if val_loader and is_main and (current_epoch + 1) % 10 == 0:
                print("\n  --- Validation ---")
                eval_model = ema.model if ema else model
                results = validate(eval_model, val_loader, device, num_classes, use_amp)
                map_5095 = results.get("mAP50-95", 0.0)

                if map_5095 > best_map:
                    best_map = map_5095
                    save_checkpoint(
                        model, optimizer, current_epoch,
                        os.path.join(save_dir, "best.pth"),
                        scaler=scaler, ema=ema, loss=avg_loss, best_metric=map_5095,
                        cfg=train_cfg,
                    )
                    print(f"  🏆 New Best mAP@50-95: {best_map:.4f}")

            # 定期保存
            if is_main and (current_epoch + 1) % save_freq == 0:
                save_checkpoint(
                    model, optimizer, current_epoch,
                    os.path.join(save_dir, "latest.pth"),
                    scaler=scaler, ema=ema, loss=avg_loss,
                    cfg=train_cfg,
                )

            current_epoch += 1

    # 最终保存
    if is_main:
        save_checkpoint(
            model, optimizer, current_epoch - 1,
            os.path.join(save_dir, "final.pth"),
            scaler=scaler, ema=ema, loss=avg_loss, best_metric=best_map,
            cfg=train_cfg,
        )
        print("\n" + "=" * 70)
        print(f"  Training complete! Best mAP@50-95: {best_map:.4f}")
        print(f"  Checkpoints: {save_dir}/")
        print("=" * 70)


if __name__ == "__main__":
    main()
