"""Position Embedding — Sine 正弦位置编码。

Transformer 本身不知道空间位置信息，
需要为特征图的每个像素生成 2D 正弦位置编码。

参考: DETR (End-to-End Object Detection with Transformers)
  官方实现: 对偶数维做 sin、奇数维做 cos，输出维度 = 2 * num_pos_feats = hidden_dim。
"""

import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """2D Sine Position Embedding。

    为 (B, C, H, W) 特征图生成位置编码，输出 (B, 2*num_pos_feats, H, W)。

    Args:
        num_pos_feats: 每个空间维度的位置编码特征数。
            输出总通道数 = 2 * num_pos_feats，应对齐 hidden_dim。
            例如 hidden_dim=256 时应传 128。
    """

    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 特征图

        Returns:
            pos: (B, 2*num_pos_feats, H, W) 位置编码
        """
        B, _, H, W = x.shape

        # 归一化坐标到 [0, 2*pi]
        not_mask = torch.ones(B, H, W, device=x.device, dtype=torch.float32)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * torch.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * torch.pi

        # 频率: (num_pos_feats,)
        dim_t = torch.arange(self.num_pos_feats, device=x.device, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t   # (B, H, W, num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_t

        # 官方 DETR 写法: 偶数维 sin, 奇数维 cos
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(),
             pos_x[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)   # (B, H, W, num_pos_feats)

        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(),
             pos_y[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)   # (B, H, W, num_pos_feats)

        pos = torch.cat((pos_y, pos_x), dim=3)   # (B, H, W, 2*num_pos_feats)
        return pos.permute(0, 3, 1, 2)           # (B, 2*num_pos_feats, H, W)
