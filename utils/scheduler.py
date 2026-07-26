"""Scheduler — 学习率调度器。

提供:
    1. Cosine Annealing LR（默认）
    2. Warmup + Cosine（推荐，前几个 epoch 用小 lr 预热）
"""

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import math


def build_scheduler(optimizer, epochs, scheduler_type="cosine", warmup_epochs=5):
    """构建学习率调度器。

    Args:
        optimizer: 优化器
        epochs: 总训练轮数
        scheduler_type: "cosine" 或 "warmup_cosine"
        warmup_epochs: warmup 轮数（仅 warmup_cosine）

    Returns:
        scheduler: LR scheduler
    """
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)

    elif scheduler_type == "warmup_cosine":
        def warmup_fn(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, warmup_fn)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
