"""DINO Detector — 完整检测器主体。

整合:
    1. Input Projection: backbone 通道 → hidden_dim
    2. Position Encoding: 2D Sine 位置编码
    3. Transformer Decoder: Object Query 从特征中聚合信息
    4. Class Head: 分类预测 (num_classes + 1)
    5. Box Head: 边界框回归 (cx, cy, w, h)

输入: (B, C, H, W) 融合特征图
输出: dict(pred_logits, pred_boxes)
    - pred_logits: (B, num_queries, num_classes+1)
    - pred_boxes:  (B, num_queries, 4)
"""

import torch
import torch.nn as nn

from .transformer import DINOTransformer
from .position_encoding import PositionEmbeddingSine
from .class_head import ClassHead
from .box_head import BoxHead


class DINO(nn.Module):
    """DINO 检测器。

    Args:
        num_classes: 检测类别数（不含背景）
        in_channels: 输入特征通道数
        hidden_dim: Transformer 隐藏维度
        num_queries: Object Query 数量
    """

    def __init__(
        self,
        num_classes,
        in_channels=256,
        hidden_dim=256,
        num_queries=300,
    ):
        super().__init__()

        self.num_queries = num_queries

        # 输入投影: backbone 通道 → hidden_dim
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        # 位置编码
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2)

        # Transformer Decoder + Object Queries
        self.transformer = DINOTransformer(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
        )

        # 分类头: +1 背景
        self.class_head = ClassHead(hidden_dim, num_classes)

        # 回归头: cx, cy, w, h
        self.box_head = BoxHead(hidden_dim)

    def forward(self, feature):
        """
        Args:
            feature: (B, C, H, W) 融合特征图

        Returns:
            dict:
                pred_logits: (B, num_queries, num_classes+1)
                pred_boxes:  (B, num_queries, 4)
        """
        B = feature.shape[0]

        # 投影到 hidden_dim
        src = self.input_proj(feature)  # (B, hidden_dim, H, W)

        # 位置编码
        pos = self.pos_embed(src)  # (B, hidden_dim, H, W)

        # Flatten 并转置为 (seq, batch, feat) 格式
        src_flat = src.flatten(2).permute(2, 0, 1)    # (HW, B, C)
        pos_flat = pos.flatten(2).permute(2, 0, 1)    # (HW, B, C)

        # 初始 tgt (zeros)
        tgt = torch.zeros(self.num_queries, B, src.shape[1], device=src.device)

        # Transformer Decoder
        hs = self.transformer(tgt, src_flat, pos_flat)  # list of [(Nq, B, C)]
        hs = hs[-1].transpose(0, 1)  # (B, Nq, C)

        # 分类 + 回归
        logits = self.class_head(hs)  # (B, num_queries, num_classes+1)
        boxes = self.box_head(hs)      # (B, num_queries, 4)

        return {
            "pred_logits": logits,
            "pred_boxes": boxes,
        }
