"""RGB Backbone — Swin Transformer。

使用 timm 库的 Swin 系列，输出多尺度特征。
后续可替换为 InternImage 等更强 backbone，接口保持一致即可。

注意: timm Swin 的 features_only=True 返回 NHWC 格式 [B,H,W,C]，
forward() 中已统一转换为 NCHW [B,C,H,W] 与下游模块对齐。
"""

import torch
import torch.nn as nn
import timm


class RGBBackbone(nn.Module):
    """Swin Transformer 作为 RGB 模态主干网络。

    输出 4 层多尺度特征，通道数由 backbone 变体决定。
    """

    _MODEL_MAP = {
        "swin_tiny":   "swin_tiny_patch4_window7_224",
        "swin_small":  "swin_small_patch4_window7_224",
        "swin_base":   "swin_base_patch4_window7_224",
        "swin_large":  "swin_large_patch4_window7_224",
    }

    def __init__(self, backbone_name="swin_small", pretrained=False):
        super().__init__()

        model_name = self._MODEL_MAP.get(backbone_name, backbone_name)

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True,
            img_size=(384, 640),
        )

        self.backbone.feature_info.out_indices = (0, 1, 2, 3)
        self.channels = list(self.backbone.feature_info.channels())

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) RGB 图像  [NCHW]

        Returns:
            feats: list of (B, C_i, H_i, W_i)  多尺度特征 [NCHW]
        """
        feats = self.backbone(x)          # timm Swin: [B, H, W, C]
        out = []
        for f in feats:
            if f.dim() == 4:
                f = f.permute(0, 3, 1, 2).contiguous()  # [B,H,W,C] → [B,C,H,W]
            out.append(f)
        return out
