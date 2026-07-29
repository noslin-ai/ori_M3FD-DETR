"""RGB Backbone — Swin Transformer。

使用 timm 库的 Swin 系列，输出多尺度特征。
后续可替换为 InternImage 等更强 backbone，接口保持一致即可。
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

        # 显式设置 out_indices，防止 timm 版本差异导致索引错位
        self.backbone.feature_info.out_indices = (0, 1, 2, 3)

        # 用微型前向验证实际输出通道，避免 feature_info.channels() 与 forward 不一致
        self.channels = self._verify_channels()

    def _verify_channels(self):
        """用微型输入验证 backbone 实际输出通道数，确保与 feature_info 声明一致。"""
        declared = list(self.backbone.feature_info.channels())
        try:
            with torch.no_grad():
                dummy = torch.randn(1, 3, 64, 64)
                feats = self.backbone(dummy)
                actual = [f.shape[1] for f in feats]
        except Exception:
            # 微型 forward 失败（如某些 backbone 需要更大输入），回退到声明值
            return declared

        if len(actual) != len(declared):
            print(f"[RGBBackbone] 实际输出 {len(actual)} 层 ≠ 声明 {len(declared)} 层, "
                  f"已按实际值修正: channels={actual}")
            return actual

        if actual != declared:
            print(f"[RGBBackbone] 实际通道 {actual} ≠ 声明 {declared}, "
                  f"已按实际值修正")
            return actual

        return declared

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) RGB 图像

        Returns:
            feats: list of (B, C_i, H_i, W_i) 多尺度特征
        """
        feats = self.backbone(x)
        return feats
