"""EMA — Exponential Moving Average 模型参数滑动平均。

比赛小数据场景下非常有效：维护一份权重的滑动平均副本，
推理时使用 EMA 权重通常比直接用最终权重更稳定、泛化更好。
"""

import copy
import torch
import torch.nn as nn


class EMA:
    """指数移动平均模型。

    维护模型参数的滑动平均:
        ema_param = decay * ema_param + (1 - decay) * model_param

    Args:
        model: 训练中的模型
        decay: EMA 衰减率（默认 0.9999）
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()

        # EMA 模型不需要梯度
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """更新 EMA 参数。

        Args:
            model: 当前训练模型（正在训练的那个）
        """
        model_params = dict(model.named_parameters())
        ema_params = dict(self.model.named_parameters())

        for name, ema_p in ema_params.items():
            if name in model_params:
                ema_p.data.mul_(self.decay).add_(
                    model_params[name].data, alpha=1 - self.decay
                )

    @torch.no_grad()
    def update_buffers(self, model):
        """同步 buffer（如 BatchNorm 的 running_mean/var）。"""
        model_buffers = dict(model.named_buffers())
        ema_buffers = dict(self.model.named_buffers())
        for name, ema_b in ema_buffers.items():
            if name in model_buffers:
                ema_b.data.copy_(model_buffers[name].data)

    def state_dict(self):
        """返回 EMA 模型的 state_dict。"""
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """加载 EMA 权重。"""
        self.model.load_state_dict(state_dict)
