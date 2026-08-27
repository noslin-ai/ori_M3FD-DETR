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


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


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
        decoder_feature_level=-1,
        decoder_feature_levels=None,
        use_anchor_boxes=False,
        anchor_box_size=(0.06, 0.12),
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.use_dn = use_dn
        self.decoder_feature_level = decoder_feature_level
        self.decoder_feature_levels = decoder_feature_levels
        self.use_anchor_boxes = use_anchor_boxes

        # Position Encoding
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)
        self.level_embed = nn.Parameter(torch.zeros(4, hidden_dim))

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
        if use_anchor_boxes:
            self.box_head.reset_delta_init()

        anchors = self._make_anchor_boxes(num_queries, anchor_box_size)
        self.register_buffer("anchor_boxes", anchors, persistent=False)

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
        nn.init.normal_(self.level_embed, std=0.02)
        pass

    @staticmethod
    def _make_anchor_boxes(num_queries, box_size):
        sizes = torch.as_tensor(box_size, dtype=torch.float32)
        if sizes.ndim == 1:
            sizes = sizes.view(1, 2)
        if sizes.ndim != 2 or sizes.shape[1] != 2:
            raise ValueError(
                "anchor_box_size must be [w, h] or a list of [w, h] pairs"
            )

        num_sizes = sizes.shape[0]
        num_centers = max(1, (num_queries + num_sizes - 1) // num_sizes)
        if num_centers == 300:
            rows, cols = 15, 20
        elif num_centers == 900:
            rows, cols = 30, 30
        elif num_centers == 100:
            rows, cols = 10, 10
        else:
            cols = int(num_centers ** 0.5)
            rows = max(1, num_centers // max(cols, 1))
            while rows * cols < num_centers:
                cols += 1

        ys = (torch.arange(rows, dtype=torch.float32) + 0.5) / rows
        xs = (torch.arange(cols, dtype=torch.float32) + 0.5) / cols
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        centers = centers[:num_centers]
        if centers.shape[0] < num_centers:
            pad = centers[-1:].repeat(num_centers - centers.shape[0], 1)
            centers = torch.cat([centers, pad], dim=0)

        centers = centers[:, None, :].expand(-1, num_sizes, -1).reshape(-1, 2)
        wh = sizes[None, :, :].expand(num_centers, -1, -1).reshape(-1, 2)
        return torch.cat([centers, wh], dim=-1)[:num_queries].clamp(1e-4, 1 - 1e-4)

    def _select_levels(self, features):
        if self.decoder_feature_levels is None:
            levels = [self.decoder_feature_level]
        else:
            levels = list(self.decoder_feature_levels)

        selected = []
        for level in levels:
            idx = level if level >= 0 else len(features) + level
            if idx < 0 or idx >= len(features):
                raise IndexError(
                    f"decoder feature level {level} resolves to {idx}, "
                    f"but only {len(features)} levels are available"
                )
            selected.append(idx)
        return selected

    def _flatten_features(self, features):
        srcs, poss = [], []
        for idx in self._select_levels(features):
            feat = self.input_proj[0](features[idx])
            pos_embed = self.position_embedding(feat)
            assert pos_embed.shape[1] == self.hidden_dim, (
                f"位置编码输出 {pos_embed.shape[1]} 通道 ≠ hidden_dim {self.hidden_dim}。"
                f"请将 PositionEmbeddingSine 的 num_pos_feats 设为 hidden_dim//2={self.hidden_dim//2}")

            level_pos = self.level_embed[idx].view(1, self.hidden_dim, 1, 1)
            pos_embed = pos_embed + level_pos

            srcs.append(feat.flatten(2).permute(2, 0, 1))
            poss.append(pos_embed.flatten(2).permute(2, 0, 1))

        return torch.cat(srcs, dim=0), torch.cat(poss, dim=0)

    def _predict_boxes(self, hs):
        if not self.use_anchor_boxes:
            return self.box_head(hs)

        raw = self.box_head.forward_logits(hs)
        anchors = self.anchor_boxes[:hs.shape[1]].to(device=hs.device, dtype=hs.dtype)
        anchors = anchors.unsqueeze(0).expand(hs.shape[0], -1, -1)
        return torch.sigmoid(inverse_sigmoid(anchors) + raw)

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
        # input_proj: Conv + GroupNorm 归一化特征，避免大数值淹没位置编码。
        # v8 可把 P3/P4/P5 等多尺度特征拼成 decoder memory，减轻单尺度定位瓶颈。
        src, pos = self._flatten_features(features)
        B = src.shape[1]

        # Object queries
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (Nq, B, C)

        # Denoising (训练时)
        dn_meta = None
        attn_mask = None
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

        hs_layers = [layer.transpose(0, 1) for layer in hs]  # each: (B, Nq, C)

        # DN 后处理 (训练时去除 DN 部分)
        if dn_meta is not None:
            dn_num = dn_meta["dn_num"]
            hs_layers = [layer[:, :-dn_num] if dn_num > 0 else layer for layer in hs_layers]
            _, targets_out = dn_post_process(hs[-1].transpose(0, 1), targets, dn_meta)
        else:
            targets_out = targets

        # 分类 + 回归；最后一层作为主输出，中间层作为辅助监督。
        hs_final = hs_layers[-1]
        pred_logits = self.class_head(hs_final)  # (B, Nq, num_classes+1)
        pred_boxes = self._predict_boxes(hs_final)     # (B, Nq, 4)
        aux_outputs = [
            {
                "pred_logits": self.class_head(layer),
                "pred_boxes": self._predict_boxes(layer),
            }
            for layer in hs_layers[:-1]
        ]

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "targets": targets_out,
            "aux_outputs": aux_outputs,
            "dn_meta": dn_meta,
        }
