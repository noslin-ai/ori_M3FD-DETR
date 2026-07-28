"""DINO Transformer — Transformer Decoder + Object Queries。

核心组件:
    1. TransformerDecoder: 6 层 decoder，每个 query 从 encoder 特征中聚合信息
    2. Object Queries: 300 个可学习 embedding，每个对应一个检测槽位

输入: (B, C, H, W) 特征图 + 位置编码
输出: (B, num_queries, hidden_dim) decoder hidden states
"""

import torch
import torch.nn as nn


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

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=False,   # (seq, batch, feat) 格式，与 dino_detector 传入一致
            dropout=0.1,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
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
        # 将位置编码加到 memory 上
        memory = memory + pos

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

        # Decoder: query 从 memory 中聚合信息
        hs = self.decoder(query_embed, memory, tgt_mask=tgt_mask)  # (Nq, B, C)

        return [hs]  # 返回 list 保持与多尺度输出的一致性
