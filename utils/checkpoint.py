"""Checkpoint 管理 — 保存与恢复。

支持:
    1. 保存: model + optimizer + scaler + ema + epoch
    2. 加载: 安全恢复训练状态
    3. Best checkpoint 跟踪
"""

import os
import torch
from packaging import version


def _safe_torch_load(path, map_location="cpu"):
    """安全 torch.load，兼容 PyTorch < 2.4 版本。"""
    if version.parse(torch.__version__) >= version.parse("2.4"):
        return torch.load(path, map_location=map_location, weights_only=False)
    else:
        return torch.load(path, map_location=map_location)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    path,
    scaler=None,
    ema=None,
    loss=None,
    best_metric=None,
    extra=None,
    cfg=None,
):
    """保存训练 checkpoint。

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        path: 保存路径
        scaler: AMP GradScaler (可选)
        ema: EMA 实例 (可选)
        loss: 当前 loss (可选)
        best_metric: 最佳指标值 (可选)
        extra: 额外信息 dict (可选)
        cfg: 模型配置 dict (可选，推荐保存以便推理时自动恢复)
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if ema is not None:
        state["ema"] = ema.state_dict() if hasattr(ema, "state_dict") else None
    if loss is not None:
        state["loss"] = loss
    if best_metric is not None:
        state["best_metric"] = best_metric
    if extra is not None:
        state.update(extra)
    if cfg is not None:
        state["cfg"] = cfg

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scaler=None,
    ema=None,
    device="cuda",
):
    """加载训练 checkpoint。

    Args:
        path: checkpoint 路径
        model: 模型
        optimizer: 优化器 (可选)
        scaler: AMP GradScaler (可选)
        ema: EMA 实例 (可选)
        device: 设备

    Returns:
        dict: {
            "epoch": int,
            "loss": float | None,
            "best_metric": float | None,
        }
    """
    if not os.path.exists(path):
        print(f"  ⚠ Checkpoint 不存在: {path}")
        return {"epoch": 0, "loss": None, "best_metric": None}

    state = _safe_torch_load(path, map_location=device)

    # 加载模型
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    # 加载优化器
    if optimizer is not None and "optimizer" in state:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except Exception as e:
            print(f"  ⚠ 优化器状态恢复失败: {e}")

    # 加载 scaler
    if scaler is not None and "scaler" in state and state.get("scaler"):
        try:
            scaler.load_state_dict(state["scaler"])
        except Exception as e:
            print(f"  ⚠ Scaler 状态恢复失败: {e}")

    # 加载 EMA
    if ema is not None and "ema" in state and state.get("ema"):
        try:
            if hasattr(ema, "load_state_dict"):
                ema.load_state_dict(state["ema"])
        except Exception as e:
            print(f"  ⚠ EMA 状态恢复失败: {e}")

    epoch = state.get("epoch", 0)
    loss = state.get("loss", None)
    best_metric = state.get("best_metric", None)

    print(f"  Checkpoint loaded: {path} (epoch={epoch + 1})")
    return {"epoch": epoch + 1, "loss": loss, "best_metric": best_metric}
