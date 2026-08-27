"""Box Head — 边界框回归预测头。

将 Transformer decoder 输出的每个 query embedding 映射到 4 个归一化坐标:
    cx, cy, w, h（均归一化到 [0, 1]）

输出: (B, num_queries, 4)
"""

import torch.nn as nn
import torch


class BoxHead(nn.Module):
    """边界框回归头。

    Args:
        hidden_dim: 输入特征维度
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),  # 归一化到 [0, 1]
        )

    def forward_logits(self, x):
        return self.net[:-1](x)

    def reset_delta_init(self):
        final = self.net[2]
        nn.init.constant_(final.weight, 0)
        nn.init.constant_(final.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, num_queries, hidden_dim)

        Returns:
            boxes: (B, num_queries, 4) — (cx, cy, w, h) 归一化坐标
        """
        return torch.sigmoid(self.forward_logits(x))
