"""Position Embedding — Sine 正弦位置编码。

Transformer 本身不知道空间位置信息，
需要为特征图的每个像素生成 2D 正弦位置编码。

参考: DETR (End-to-End Object Detection with Transformers)
"""

import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """2D Sine Position Embedding。

    为 (B, C, H, W) 特征图生成位置编码 (B, num_pos_feats*2, H, W)。

    Args:
        num_pos_feats: 每个维度的位置编码特征数（总输出 = 2*num_pos_feats）。
            必须等于 hidden_dim//2，使总输出 == hidden_dim。
    """

    def __init__(self, num_pos_feats=128):
        super().__init__()
        assert num_pos_feats * 2 in (128, 256, 512), \
            f"num_pos_feats={num_pos_feats}，总输出={num_pos_feats*2}。通常应为 hidden_dim//2"
        self.num_pos_feats = num_pos_feats

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 特征图

        Returns:
            pos: (B, num_pos_feats*2, H, W) 位置编码
        """
        B, C, H, W = x.shape

        # 坐标范围 [0, 2*pi]
        mask = torch.zeros(B, H, W, device=x.device)
        y_embed = mask.cumsum(1)
        x_embed = mask.cumsum(2)

        eps = 1e-6
        x_embed = x_embed / (x_embed[:, -1:, :] + eps) * 2 * 3.14159265
        y_embed = y_embed / (y_embed[:, :, -1:] + eps) * 2 * 3.14159265

        # 频率分母
        dim = torch.arange(self.num_pos_feats, device=x.device)
        dim = 10000 ** (2 * (dim // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim
        pos_y = y_embed[:, :, :, None] / dim

        # sin/cos 交错
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=4).flatten(3)

        pos = torch.cat([pos_y, pos_x], dim=3)  # (B, H, W, num_pos_feats*2)
        return pos.permute(0, 3, 1, 2)  # (B, num_pos_feats*2, H, W)
