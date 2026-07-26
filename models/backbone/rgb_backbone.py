"""RGB Backbone — Swin Transformer Large.

使用 timm 库的 swin_large_patch4_window7_224，输出多尺度特征。
后续可替换为 InternImage 等更强 backbone，接口保持一致即可。
"""

import torch
import torch.nn as nn
import timm


class RGBBackbone(nn.Module):
    """Swin Transformer 作为 RGB 模态主干网络。

    输出 4 层多尺度特征:
        - P2: 192/96 通道, stride 4
        - P3: 384/192 通道, stride 8
        - P4: 768/384 通道, stride 16
        - P5: 1536/768 通道, stride 32
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

        self.channels = self.backbone.feature_info.channels()

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) RGB 图像

        Returns:
            feats: list of (B, C_i, H_i, W_i) 多尺度特征
        """
        feats = self.backbone(x)
        return feats
