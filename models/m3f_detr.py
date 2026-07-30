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

        # 从 backbone 获取各层通道数（由 timm feature_info 提供，已通过 NHWC→NCHW 转换验证）
        backbone_channels = self.rgb_backbone.channels
        assert len(backbone_channels) == 4, \
            f"期望 backbone 输出 4 层多尺度特征，实际 {len(backbone_channels)} 层: {backbone_channels}"
        print(f"[M3F-DETR] backbone={backbone_name}, channels={list(backbone_channels)}, "
              f"hidden_dim={hidden_dim}, num_queries={num_queries}")

        self.ir_encoder = ThermalAdapter(backbone_channels[0])
        self.depth_encoder = DepthEncoder(backbone_channels[0])

        # ---- 三模态融合 ----
        self.fusion = CMFA(dim=backbone_channels[0], use_downsample=True)


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

        # ---- 运行时通道校验: 确保 Backbone/编码器/CMFA/FPN 通道一致 ----
        ch = self.rgb_backbone.channels  # [C0, C1, C2, C3]
        assert len(rgb_features) == len(ch), (
            f"Backbone 返回 {len(rgb_features)} 层特征, 期望 {len(ch)} 层")
        for i in range(len(ch)):
            assert rgb_features[i].shape[1] == ch[i], (
                f"RGB 特征层 {i} 通道 {rgb_features[i].shape[1]} ≠ 期望 {ch[i]}")

        # (2) IR Encoder → IR 特征
        ir_feature = self.ir_encoder(ir)  # (B, ch0, H/4, W/4)
        assert ir_feature.shape[1] == ch[0], (
            f"IR 编码器输出 {ir_feature.shape[1]} 通道 ≠ 期望 {ch[0]}")
        assert ir_feature.shape[2:] == rgb_features[0].shape[2:], (
            f"IR 空间尺寸 {ir_feature.shape[2:]} ≠ RGB P2 {rgb_features[0].shape[2:]}")

        # (3) Depth Encoder → Depth 特征
        depth_feature = self.depth_encoder(depth)  # (B, ch0, H/4, W/4)
        assert depth_feature.shape[1] == ch[0], (
            f"Depth 编码器输出 {depth_feature.shape[1]} 通道 ≠ 期望 {ch[0]}")
        assert depth_feature.shape[2:] == rgb_features[0].shape[2:], (
            f"Depth 空间尺寸 {depth_feature.shape[2:]} ≠ RGB P2 {rgb_features[0].shape[2:]}")

        # (4) CMFA 三模态融合
        # 使用 P2 作为 RGB 输入，与 IR/Depth 通道数和分辨率匹配
        fused = self.fusion(
            rgb_features[0],      # P2 (B, ch[0], H/4, W/4)
            ir_feature,           # (B, ch[0], H/4, W/4)
            depth_feature,        # (B, ch[0], H/4, W/4)
        )  # 输出 (B, ch[0], H/4, W/4)
        assert fused.shape[1] == ch[0], (
            f"CMFA 融合输出 {fused.shape[1]} 通道 ≠ 期望 {ch[0]}")

        # (5) 多尺度 FPN — 将 P2 替换为融合后特征
        fused_features = [fused] + list(rgb_features[1:])
        features = self.fpn(fused_features)  # [(B,256,H/4,W/4), ..., (B,256,H/32,W/32)]

        # (6) DINO 检测器
        outputs = self.detector(features, targets)

        return outputs

    def forward_debug(self, rgb, ir, depth, targets=None):
        """带调试输出的 forward，用于排查通道不匹配问题。
        确认无误后可切换回 forward()。
        """
        rgb_features = self.rgb_backbone(rgb)
        print(f"  [DEBUG] RGB feats: {[f.shape for f in rgb_features]}")

        ir_feature = self.ir_encoder(ir)
        print(f"  [DEBUG] IR feat:   {ir_feature.shape}")

        depth_feature = self.depth_encoder(depth)
        print(f"  [DEBUG] Depth feat: {depth_feature.shape}")

        fused = self.fusion(rgb_features[0], ir_feature, depth_feature)
        print(f"  [DEBUG] Fused:     {fused.shape}")

        fused_features = [fused] + list(rgb_features[1:])
        print(f"  [DEBUG] FPN input channels: {[f.shape[1] for f in fused_features]}")

        features = self.fpn(fused_features)
        outputs = self.detector(features, targets)
        return outputs
