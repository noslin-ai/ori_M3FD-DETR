"""Class Head — 分类预测头。

将 Transformer decoder 输出的每个 query embedding 映射到类别 logits。
+1 是因为需要背景类别（"无目标"）。

输出: (B, num_queries, num_classes + 1)
"""

import torch.nn as nn


class ClassHead(nn.Module):
    """分类预测头。

    Args:
        hidden_dim: 输入特征维度
        num_classes: 检测类别数（不含背景）
    """

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        # +1: 背景类别
        self.fc = nn.Linear(hidden_dim, num_classes + 1)

    def forward(self, x):
        """
        Args:
            x: (B, num_queries, hidden_dim)

        Returns:
            logits: (B, num_queries, num_classes + 1)
        """
        return self.fc(x)
