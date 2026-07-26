"""DINO Detector — 完整 DINO 检测器。

将现有 DINO Transformer + 分类头 + 回归头封装为一个完整模块，
支持多尺度特征输入。

特性:
    1. 多尺度特征输入（FPN 输出)
    2. Position Encoding
    3. Transformer Decoder
    4. Denoising Query (可选)
    5. Class Head + Box Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .position_encoding import PositionEmbeddingSine
from .transformer import DINOTransformer
from .class_head import ClassHead
from .box_head import BoxHead
from .dn_query import prepare_for_dn, dn_post_process


class DINODetector(nn.Module):
    """DINO 检测器主体。

    Args:
        num_classes: 检测类别数（不含背景）
        hidden_dim: 隐层维度
        num_queries: Object Query 数量
        nhead: 注意力头数
        num_decoder_layers: Decoder 层数
        use_dn: 是否使用 Denoising Query
        dn_number: DN 的噪声查询数
        dn_label_noise: 标签噪声比例
        dn_box_noise: 框噪声比例
    """

    def __init__(
        self,
        num_classes=12,
        hidden_dim=256,
        num_queries=900,
        nhead=8,
        num_decoder_layers=6,
        use_dn=True,
        dn_number=100,
        dn_label_noise=0.2,
        dn_box_noise=0.4,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.use_dn = use_dn

        # Position Encoding
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)

        # Input Projection (多尺度特征 → hidden_dim)
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(8, hidden_dim),
            )
        ])

        # Transformer Decoder
        self.transformer = DINOTransformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
        )

        # Object Queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 分类头
        self.class_head = ClassHead(
            hidden_dim=hidden_dim,
            num_classes=num_classes,  # ClassHead 内部已 +1 处理背景
        )

        # 回归头
        self.box_head = BoxHead(
            hidden_dim=hidden_dim,
        )

        # DN 相关
        if use_dn:
            self.dn_query_embed = nn.Embedding(num_queries, hidden_dim)
            nn.init.normal_(self.dn_query_embed.weight)
            self.dn_number = dn_number
            self.dn_label_noise = dn_label_noise
            self.dn_box_noise = dn_box_noise

        self._init_weights()

    def _init_weights(self):
        # 所有参数已在各子模块中初始化
        pass

    def forward(self, features, targets=None):
        """
        Args:
            features: list of (B, C, H_i, W_i) 多尺度特征
            targets: list of dict，每个元素 {"boxes": (N,4), "labels": (N,)}（训练时需要）

        Returns:
            dict:
                pred_logits: (B, num_queries, num_classes+1) 或 list of (B, num_queries, num_classes+1) 按 decoder 层
                pred_boxes: (B, num_queries, 4) 或 list of
                dn_results: (可选) DN 相关的输出
        """
        # 取最后一层（最高层）做 position encoding
        feat_flat = features[-1]  # (B, C, H, W)
        pos_embed = self.position_embedding(feat_flat)  # (B, C, H, W)

        # Flatten 特征
        B, C, H, W = feat_flat.shape
        src = feat_flat.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        pos = pos_embed.flatten(2).permute(2, 0, 1)  # (HW, B, C)

        # Object queries
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (Nq, B, C)

        # Denoising (训练时)
        dn_meta = None
        if self.use_dn and self.training and targets is not None:
            query_embed, targets, attn_mask, dn_meta = prepare_for_dn(
                query_embed.transpose(0, 1),  # (B, Nq, C)
                targets,
                self.dn_query_embed.weight,
                self.dn_number,
                self.num_classes,
                self.dn_label_noise,
                self.dn_box_noise,
            )
            # query_embed: (B, Nq+dn_Nq, C)
            query_embed = query_embed.transpose(0, 1)  # (Nq+dn_Nq, B, C)
            if attn_mask is not None:
                attn_mask = attn_mask.to(feat_flat.device)

        # Transformer Decoder
        # 使用简化版: 只有单尺度特征
        tgt = torch.zeros_like(query_embed)
        hs = self.transformer(
            tgt,
            src,
            pos,
            query_embed,
            mask=attn_mask,  # DN attention mask: 限制原始 query 不可见 DN query
        )  # list of (Nq, B, C) per decoder layer

        # 取最后一层结果
        hs_final = hs[-1]  # (Nq, B, C)
        hs_final = hs_final.transpose(0, 1)  # (B, Nq, C)

        # DN 后处理 (训练时去除 DN 部分)
        if dn_meta is not None:
            hs_final, targets_out = dn_post_process(hs_final, targets, dn_meta)
        else:
            targets_out = targets

        # 分类 + 回归
        pred_logits = self.class_head(hs_final)  # (B, Nq, num_classes+1)
        pred_boxes = self.box_head(hs_final)     # (B, Nq, 4)

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "targets": targets_out,
            "aux_outputs": None,  # 后续可添加中间层输出
            "dn_meta": dn_meta,
        }
