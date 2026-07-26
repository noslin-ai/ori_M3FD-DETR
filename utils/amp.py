"""AMP — 混合精度训练工具。

大模型训练必备：FP16 前向 + FP32 梯度更新，显存减半、速度翻倍。
"""

import torch
from torch.cuda.amp import GradScaler

# 全局 scaler（单例）
_scaler = None


def get_scaler(enabled=True):
    """获取 AMP GradScaler 单例。

    Args:
        enabled: 是否启用 AMP

    Returns:
        GradScaler 实例
    """
    global _scaler
    if _scaler is None:
        _scaler = GradScaler(enabled=enabled)
    return _scaler


def reset_scaler():
    """重置 scaler（用于重新开始训练）。"""
    global _scaler
    _scaler = None
