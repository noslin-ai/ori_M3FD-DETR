"""DINO Transformer — Transformer Decoder + Object Queries。

核心组件:
    1. TransformerDecoder: 6 层 decoder，每个 query 从 encoder 特征中聚合信息
    2. Object Queries: 300 个可学习 embedding，每个对应一个检测槽位

输入: (B, C, H, W) 特征图 + 位置编码
输出: (B, num_queries, hidden_dim) decoder hidden states
"""

import torch.nn as nn
import torch.nn.functional as F


class DINOTransformerDecoderLayer(nn.Module):
    """带 query_pos / memory_pos 的 DETR 风格 decoder layer。

    属性名保持和 nn.TransformerDecoderLayer 一致，方便加载旧 checkpoint。
    """

    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None, query_pos=None, tgt_mask=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DINOTransformerDecoder(nn.Module):
    """轻量 decoder 容器，保留 decoder.layers.* 的 state_dict 键名。"""

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer])
        for _ in range(1, num_layers):
            self.layers.append(
                DINOTransformerDecoderLayer(
                    d_model=decoder_layer.linear2.out_features,
                    nhead=decoder_layer.self_attn.num_heads,
                    dim_feedforward=decoder_layer.linear1.out_features,
                    dropout=decoder_layer.dropout.p,
                )
            )

    def forward(self, tgt, memory, pos=None, query_pos=None, tgt_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                pos=pos,
                query_pos=query_pos,
                tgt_mask=tgt_mask,
            )
        return output


class DINOTransformer(nn.Module):
    """DINO Transformer Decoder。

    Args:
        hidden_dim: 隐藏维度（默认 256）
        num_queries: Object Query 数量（默认 300）
        num_heads: 多头注意力头数
        num_layers: Decoder 层数
        dim_feedforward: FFN 中间层维度
    """

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_decoder_layers=6,
        num_queries=300,
        dim_feedforward=2048,
    ):
        super().__init__()

        decoder_layer = DINOTransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
        )

        self.decoder = DINOTransformerDecoder(
            decoder_layer,
            num_decoder_layers,
        )

    def forward(self, tgt, memory, pos, query_embed=None, mask=None):
        """
        Args:
            tgt: (Nq, B, C) 初始 decoder 输入（zeros）
            memory: (HW, B, C) 已 flatten 的 encoder 特征
            pos: (HW, B, C) 已 flatten 的位置编码
            query_embed: (Nq, B, C) Object Queries（来自 dino_detector，可选）
            mask: 可选的 attention mask（DN 训练时使用）

        Returns:
            hs: list of [(Nq, B, C)] per decoder layer
        """
        # 使用传入的 query_embed；当前由 dino_detector.py 传入，不支持内部 fallback
        if query_embed is None:
            raise ValueError(
                "DINOTransformer.forward 必须传入 query_embed 参数。"
                "请通过 dino_detector.py 调用，不要直接使用 DINOTransformer。"
            )

        # 处理 attention mask（DN 训练时限制原始 query 看到 DN query）
        tgt_mask = None
        if mask is not None:
            # mask 来自 dn_query: (B, Nq+dn_num, Nq+dn_num) dtype=bool
            # 转换为 float mask，True→-inf, False→0
            # 取第一个 batch 的 mask（所有 batch 的 DN 注意力模式相同）
            if mask.ndim == 3:
                mask = mask[0]
            tgt_mask = mask.float()
            tgt_mask.masked_fill_(tgt_mask.bool(), float('-inf'))

        # Decoder: tgt 是内容向量，query_embed 是 object query 的位置向量。
        # 旧实现把 query_embed 直接当 tgt，训练后容易被 cross-attention/LayerNorm 拉成同质输出。
        hs = self.decoder(
            tgt,
            memory,
            pos=pos,
            query_pos=query_embed,
            tgt_mask=tgt_mask,
        )  # (Nq, B, C)

        return [hs]  # 返回 list 保持与多尺度输出的一致性
