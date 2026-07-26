"""Deformable Attention — 可变形注意力。

用于 Deformable-DETR 风格的多尺度交叉注意力。
参考: Deformable DETR (https://arxiv.org/abs/2010.04159)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableAttention(nn.Module):
    """可变形多头注意力。

    从参考点周围采样 K 个偏移点，做多头注意力。
    """

    def __init__(
        self,
        d_model=256,
        n_heads=8,
        n_levels=4,
        n_points=4,
    ):
        """
        Args:
            d_model: 特征维度
            n_heads: 注意力头数
            n_levels: 多尺度特征层数
            n_points: 每层采样点数
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        # 采样偏移预测: 2 * n_levels * n_points * n_heads (x, y)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)

        # 注意力权重: n_heads * n_levels * n_points
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        # 输出投影
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0)
        nn.init.constant_(self.sampling_offsets.bias, 0)

        # 初始化 attention_weights bias，让初始权重均匀
        prior = torch.zeros(self.n_heads * self.n_levels * self.n_points)
        nn.init.constant_(self.attention_weights.bias, prior)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    @staticmethod
    def _is_power_of_2(n):
        return (n & (n - 1) == 0) and n != 0

    def forward(self, query, reference_points, src_flatten, spatial_shapes):
        """
        Args:
            query: (B, Nq, d_model) 查询向量
            reference_points: (B, Nq, n_levels, 2) 归一化参考点
            src_flatten: (B, sum(H_l*W_l), d_model) 多尺度特征（展平拼接）
            spatial_shapes: (n_levels, 2) 每层的 (H, W)

        Returns:
            output: (B, Nq, d_model)
        """
        B, Nq, _ = query.shape

        # 预测采样偏移
        offsets = self.sampling_offsets(query)  # (B, Nq, n_heads * n_levels * n_points * 2)
        offsets = offsets.reshape(B, Nq, self.n_heads, self.n_levels, self.n_points, 2)

        # 预测注意力权重
        attn_weights = self.attention_weights(query)  # (B, Nq, n_heads * n_levels * n_points)
        attn_weights = attn_weights.reshape(B, Nq, self.n_heads, self.n_levels, self.n_points)
        attn_weights = attn_weights.softmax(-1)

        # 对每一层采样
        level_start_idx = [0]
        for h, w in spatial_shapes:
            level_start_idx.append(level_start_idx[-1] + h * w)

        output = torch.zeros(B, Nq, self.d_model, device=query.device, dtype=query.dtype)

        for level in range(self.n_levels):
            h, w = spatial_shapes[level]
            level_src = src_flatten[:, level_start_idx[level]:level_start_idx[level + 1]]
            level_src = level_src.reshape(B, h, w, self.d_model)

            # 采样位置: reference_point + offset
            ref = reference_points[:, :, level, :]  # (B, Nq, 2)
            sample_locs = ref.unsqueeze(2).unsqueeze(2) + offsets[:, :, :, level]  # (B, Nq, n_heads, n_points, 2)

            # 归一化坐标 → 像素坐标
            sample_y = sample_locs[..., 1] * (h - 1)  # (B, Nq, n_heads, n_points)
            sample_x = sample_locs[..., 0] * (w - 1)

            # 双线性采样
            sample_x_norm = sample_x / (w - 1) * 2 - 1  # → [-1, 1]
            sample_y_norm = sample_y / (h - 1) * 2 - 1
            grid = torch.stack([sample_x_norm, sample_y_norm], dim=-1)  # (B, Nq, n_heads, n_points, 2)

            # 对每个 head 单独 grid_sample
            for head in range(self.n_heads):
                head_grid = grid[:, :, head]  # (B, Nq, n_points, 2)

                # Reshape: (B*Nq, n_points, 1, 2)
                head_grid = head_grid.reshape(B * Nq, self.n_points, 1, 2)

                # level_src: (B, h, w, d_model) → (B, d_model, h, w)
                level_src_permuted = level_src.permute(0, 3, 1, 2)

                # Expand d_model into n_heads * head_dim
                sampled = F.grid_sample(
                    level_src_permuted,
                    head_grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )  # (B, d_model, Nq, n_points)

                sampled = sampled.permute(0, 2, 3, 1)  # (B, Nq, n_points, d_model)
                sampled = sampled.reshape(B, Nq, self.n_points, self.n_heads, self.head_dim)

                # 加权求和
                attn = attn_weights[:, :, head, level]  # (B, Nq, n_points)
                attn = attn.unsqueeze(-1).unsqueeze(-1)  # (B, Nq, n_points, 1, 1)

                weighted = (sampled * attn).sum(dim=2)  # (B, Nq, n_heads, head_dim)

                head_start = head * self.head_dim
                output[:, :, head_start:head_start + self.head_dim] += weighted

        output = self.output_proj(output)
        return output
