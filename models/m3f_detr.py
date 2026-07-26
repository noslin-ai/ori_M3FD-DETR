"""M3F-DINO — 三模态 Deformable-DINO 目标检测模型。

网络架构:
    RGB ────► Swin-L Backbone ──► 多尺度特征 [P2, P3, P4, P5]
                                       │
    IR ─────► Thermal Adapter ──► IR Feature (192ch)
                                       │
    Depth ──► Depth Encoder ────► Depth Feature (192ch)
                                       │
                              ┌────────┼────────┐
                              │    CMFA Fusion   │
                              │ Cross Modal Attn │
                              └────────┬────────┘
                                       │
                              MultiScaleFPN
                                       │
                              DINO Detector
                                       │
                           ┌───────────┴───────────┐
                           │                       │
                       Class Head              Box Head
                     (B, 900, 13)           (B, 900, 4)

特性:
    - 三模态融合: RGB + Infrared + Depth
    - CMFA 交叉注意力融合
    - 多尺度 FPN
    - DINO Detection Head (支持 Denoising Query)
    - 支持训练和推理
"""

import torch
import torch.nn as nn

from .backbone.rgb_backbone import RGBBackbone
from .backbone.thermal_adapter import ThermalAdapter
from .backbone.depth_encoder import DepthEncoder
from .fusion.cmfa import CMFA
from .neck.multimodal_fpn import MultiScaleFusion
from .detector.dino_detector import DINODetector


class M3F_DETR(nn.Module):
    """M3F-DINO 三模态目标检测模型。

    Args:
        num_classes: 检测类别数（不含背景）
        hidden_dim: 隐层维度
        num_queries: Object Query 数量
        use_dn: 是否使用 Denoising Query
    """

    def __init__(
        self,
        num_classes=12,
        hidden_dim=256,
        num_queries=900,
        use_dn=True,
        backbone_name="swin_small",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # ---- 模态编码器 ----
        self.rgb_backbone = RGBBackbone(backbone_name=backbone_name, pretrained=False)

        # 从 backbone 自动获取各层通道数
        backbone_channels = self.rgb_backbone.channels

        self.ir_encoder = ThermalAdapter(backbone_channels[0])
        self.depth_encoder = DepthEncoder(backbone_channels[0])

        # ---- 三模态融合 ----
        self.fusion = CMFA(dim=backbone_channels[0])


        # ---- 多尺度 FPN ----
        self.fpn = MultiScaleFusion(
            in_channels_list=backbone_channels,
            out_channels=hidden_dim,
        )

        # ---- DINO 检测器 ----
        self.detector = DINODetector(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            use_dn=use_dn,
        )

    def forward(self, rgb, ir, depth, targets=None):
        """
        Args:
            rgb:   (B, 3, H, W) RGB 图像
            ir:    (B, 3, H, W) 红外图像（三通道灰度堆叠）
            depth: (B, 3, H, W) 深度图像（3通道: depth, gx, gy）
            targets: list of dict，每个元素 {"boxes": (N, 4), "labels": (N,)}
                     训练时需要，推理时为 None

        Returns:
            dict:
                pred_logits: (B, num_queries, num_classes+1) 分类 logits
                pred_boxes:  (B, num_queries, 4) 边界框 (cx, cy, w, h) 归一化
                targets: list of dict (DN 处理后的 targets)
        """
        # (1) RGB Backbone → 多尺度特征
        rgb_features = self.rgb_backbone(rgb)
        # [P2(ch0, H/4, W/4), P3(ch1, H/8, W/8),
        #  P4(ch2, H/16, W/16), P5(ch3, H/32, W/32)]

        # (2) IR Encoder → IR 特征
        ir_feature = self.ir_encoder(ir)  # (B, ch0, H/4, W/4)

        # (3) Depth Encoder → Depth 特征
        depth_feature = self.depth_encoder(depth)  # (B, ch0, H/4, W/4)

        # (4) CMFA 三模态融合
        # 使用 P2 (B, 192, H/4, W/4) 作为 RGB 输入，与 IR/Depth 通道数和分辨率匹配
        fused = self.fusion(
            rgb_features[0],      # P2 (B, 192, H/4, W/4)
            ir_feature,           # (B, 192, H/4, W/4)
            depth_feature,        # (B, 192, H/4, W/4)
        )  # 输出 (B, 192, H/4, W/4)

        # (5) 多尺度 FPN — 将 P2 替换为融合后特征
        fused_features = [fused] + list(rgb_features[1:])
        features = self.fpn(fused_features)  # [(B,256,H/4,W/4), ..., (B,256,H/32,W/32)]

        # (6) DINO 检测器
        outputs = self.detector(features, targets)

        return outputs
