"""YOLO 模型构建与预训练权重迁移（v0.9.0 支持双分支架构）。

支持的架构:
    yolo11 (单分支):
        3ch=RGB 对照 / 5ch=RGB+IR+Depth 早期融合；
    dual (双分支 + P3-P5 注意力融合, v0.9.0):
        RGB(3ch) 一个预训练 backbone，IR+Depth(2ch) 第二个 backbone，
        在 P3/P4/P5 三个尺度用 CrossModalFusion 做通道+空间注意力融合，
        融合特征送入 yolo11 Detect 头。

权重迁移要点:
    1. 从官方 yolo11n.pt（COCO 预训练）迁移形状匹配的权重；
    2. 首层卷积通道数变化时，前 3 通道继承预训练 RGB 权重，
       新增通道（或 2ch 辅助分支的全部通道）用 RGB 权重均值初始化，
       避免融合/辅助分支一开始就是随机噪声（v0.9.0 修复早期 fusion 的 stem bug）。
"""

import os

import torch
import torch.nn as nn

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel

from .fusion import CrossModalFusion, ZeroConv2d


def _load_pretrained_sd(pretrained_path):
    """加载 ultralytics 官方 checkpoint 的 state_dict。

    官方 checkpoint 序列化了 DetectionModel 全局对象，必须用
    weights_only=False（来源为官方 assets，可信）。
    """
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    src = ckpt.get("model", ckpt)
    return src.state_dict() if hasattr(src, "state_dict") else src


def _init_stem_from_pretrained(model_sd, sd):
    """用预训练 RGB 首层权重初始化目标 stem（通道数不一致时）。

    - 目标通道 > 3（如 5ch 早期融合）: 前 3 通道继承预训练权重，
      新增通道用 RGB 均值初始化；
    - 目标通道 < 3（如 2ch 辅助分支）: 全部通道用 RGB 均值初始化。
    """
    stem_key = "model.0.conv.weight"
    if stem_key not in sd or stem_key not in model_sd:
        return
    src_w = sd[stem_key]        # 预训练 (out, 3, k, k)
    w = model_sd[stem_key]      # 目标 (out, ch, k, k)
    if w.shape[1] == src_w.shape[1]:
        return
    mean_w = src_w.mean(dim=1, keepdim=True)
    if w.shape[1] > 3:
        extra = w.shape[1] - 3
        model_sd[stem_key] = torch.cat(
            [src_w, mean_w.repeat(1, extra, 1, 1)], dim=1
        )
    else:
        model_sd[stem_key] = mean_w.repeat(1, w.shape[1], 1, 1)


def transfer_pretrained_weights(model, pretrained_path):
    """从官方 checkpoint 迁移形状匹配的权重，并初始化 stem 新通道。

    Returns:
        (transferred, skipped): 迁移/跳过的参数量
    """
    sd = _load_pretrained_sd(pretrained_path)
    model_sd = model.state_dict()
    transferred, skipped = 0, 0
    for k, v in model_sd.items():
        if k in sd and sd[k].shape == v.shape:
            model_sd[k] = sd[k]
            transferred += 1
        else:
            skipped += 1

    _init_stem_from_pretrained(model_sd, sd)
    model.load_state_dict(model_sd)
    return transferred, skipped


def build_yolo_model(ch=3, nc=12, pretrained="yolo11n.pt", arch="yolo11", verbose=False):
    """构建 YOLO 检测模型（按 arch 分派单分支/双分支）。

    Args:
        ch: 输入通道数（3=RGB，5=RGB+IR+Depth）
        nc: 类别数（赛题 12 类）
        pretrained: 官方预训练权重路径；不存在时使用随机初始化
        arch: "yolo11" 单分支 或 "dual" 双分支+注意力融合
        verbose: 是否打印模型结构
    """
    if arch == "dual_mcf":
        return build_mcf_yolo_model(nc=nc, pretrained=pretrained, verbose=verbose)

    if arch == "dual":
        return build_dual_yolo_model(nc=nc, pretrained=pretrained, verbose=verbose)

    # yolo11n.yaml 默认 nc=80，这里覆盖为赛题类别数
    model = DetectionModel("yolo11n.yaml", ch=ch, nc=nc, verbose=verbose)

    # 官方训练器在调用模型前会设置 model.args（超参命名空间），
    # 自定义训练循环里需要手动补上，否则 v8DetectionLoss 初始化会报
    # `'DetectionModel' object has no attribute 'args'`。
    model.args = get_cfg(overrides={})

    if pretrained and os.path.exists(pretrained):
        transferred, skipped = transfer_pretrained_weights(model, pretrained)
        print(f"  Pretrained: {pretrained}")
        print(f"  Transferred: {transferred} keys, skipped: {skipped} keys")
    else:
        print(f"  ⚠ Pretrained 权重不存在（{pretrained}），使用随机初始化")

    return model


def build_dual_yolo_model(nc=12, pretrained="yolo11n.pt", verbose=False):
    """构建双分支 + P3-P5 注意力融合检测器（v0.9.0）。

    结构:
        RGB 分支: yolo11n backbone（3ch，预训练）
        Aux 分支: yolo11n backbone（IR+Depth 2ch，stem 用 RGB 均值初始化）
        融合:    CrossModalFusion 在 P3/P4/P5（层 16/19/22）做注意力融合
        头:      yolo11n Detect head（nc=12）
    """
    model_rgb = DetectionModel("yolo11n.yaml", ch=3, nc=nc, verbose=verbose)
    model_aux = DetectionModel("yolo11n.yaml", ch=2, nc=nc, verbose=verbose)
    model_rgb.args = get_cfg(overrides={})
    model_aux.args = get_cfg(overrides={})

    if pretrained and os.path.exists(pretrained):
        sd = _load_pretrained_sd(pretrained)
        for tag, model in (("RGB", model_rgb), ("Aux", model_aux)):
            model_sd = model.state_dict()
            transferred, skipped = 0, 0
            for k, v in model_sd.items():
                if k in sd and sd[k].shape == v.shape:
                    model_sd[k] = sd[k]
                    transferred += 1
                else:
                    skipped += 1
            _init_stem_from_pretrained(model_sd, sd)
            model.load_state_dict(model_sd)
            print(f"  [{tag} branch] transferred: {transferred} keys, skipped: {skipped} keys")
    else:
        print(f"  ⚠ Pretrained 权重不存在（{pretrained}），使用随机初始化")

    # backbone = 除 Detect 头外的全部层（yolo11n 为 0..22）
    backbone_rgb = nn.Sequential(*list(model_rgb.model)[:-1])
    backbone_aux = nn.Sequential(*list(model_aux.model)[:-1])
    head = model_rgb.model[-1]

    model = DualBranchYOLO(backbone_rgb, backbone_aux, head)
    model.args = model_rgb.args
    return model


class DualBranchYOLO(nn.Module):
    """双分支 + P3-P5 注意力融合检测器。

    接口与单分支 DetectionModel 对齐:
        train: model(batch_dict) -> (loss, loss_items)
        eval:  model(img) -> (y, preds)   # Detect head 原生输出
    """

    def __init__(self, backbone_rgb, backbone_aux, head, fuse_levels=(16, 19, 22)):
        super().__init__()
        self.backbone_rgb = backbone_rgb   # RGB 分支 (3ch)
        self.backbone_aux = backbone_aux   # IR+Depth 分支 (2ch)
        self.head = head                   # yolo11 Detect 头 (nc=12)
        self.fuse_levels = fuse_levels     # P3/P4/P5 对应 backbone 层号

        # P3/P4/P5 通道数（yolo11: 64/128/256）
        self.fusion = nn.ModuleList([
            CrossModalFusion(c) for c in (64, 128, 256)
        ])

        # 与 v8DetectionLoss 的约定对齐：它通过 model.model[-1] 取 Detect 头。
        # 用纯列表而非 ModuleList，避免 head 在 named_parameters 里重复出现。
        self.model = [head]
        self.args = get_cfg(overrides={})  # 构建函数里会覆盖为单分支模型的 args
        self.stride = head.stride
        self.names = getattr(head, "names", {i: str(i) for i in range(head.nc)})
        self.loss_fn = None

    def _extract(self, backbone, x):
        """按 ultralytics _predict_once 语义提取 P3/P4/P5 特征。

        backbone 中的层保留了 .f（输入来源）与 .i（层号）属性，
        Concat/Upsample 等层需要按 .f 取前面的输出。
        """
        y = []
        feats = []
        for m in backbone:
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
            if m.i in self.fuse_levels:
                feats.append(x)
        return feats

    def _forward_fused(self, img):
        """双分支前向并做 P3-P5 注意力融合，返回 [P3, P4, P5]。"""
        rgb = img[:, :3]
        aux = img[:, 3:]
        feats_rgb = self._extract(self.backbone_rgb, rgb)
        feats_aux = self._extract(self.backbone_aux, aux)
        return [
            fuse(f_rgb, f_aux)
            for fuse, f_rgb, f_aux in zip(self.fusion, feats_rgb, feats_aux)
        ]

    def forward(self, x):
        if isinstance(x, dict):
            # 训练态: 返回 (loss, loss_items)，与单分支 DetectionModel 一致
            fused = self._forward_fused(x["img"])
            preds = self.head(fused)
            if self.loss_fn is None:
                from ultralytics.utils.loss import v8DetectionLoss
                self.loss_fn = v8DetectionLoss(self)
            return self.loss_fn(preds, x)
        # 推理态: 返回 Detect head 原生输出 (y, preds)
        return self.head(self._forward_fused(x))


def apply_freeze(model, freeze_layers):
    """按参数名前缀冻结（兼容单分支层索引、双分支模块前缀、按模块按层）。

    Args:
        model: DetectionModel 或 DualBranchYOLO
        freeze_layers: 冻结项列表，支持三种写法：
            - int（单分支层索引），如 [0, ..., 9] 冻结 backbone；
            - str（参数名前缀），如 ["backbone_rgb", "backbone_aux"]；
            - dict（按模块按层），如 {"backbone_rgb": [0..9], "backbone_aux": [0..9]}
              （双分支下只冻结各分支 backbone，保留 neck 可训练）。
    """
    # YAML 里 freeze 直接写 dict 时（{"模块": [层,...]}），包成单元素列表统一处理
    if isinstance(freeze_layers, dict):
        freeze_layers = [freeze_layers]
    freeze_layers = freeze_layers or []
    frozen = 0
    for name, p in model.named_parameters():
        frozen_flag = False
        for f in freeze_layers:
            if isinstance(f, dict):
                # {"模块前缀": [层索引,...]}：按模块按层冻结
                frozen_flag = any(
                    name.startswith(f"{prefix}.{idx}.")
                    for prefix, indices in f.items()
                    for idx in indices
                )
            elif isinstance(f, int):
                frozen_flag = name.startswith(f"model.{f}.")
            else:
                frozen_flag = name.startswith(f"{f}.")
            if frozen_flag:
                break
        p.requires_grad_(not frozen_flag)
        frozen += int(frozen_flag)

    total = sum(1 for _ in model.parameters())
    print(f"  Frozen prefixes: {freeze_layers} ({frozen}/{total} params frozen)")


def count_parameters(model):
    """统计模型参数量（对齐旧 train.py 的打印格式）。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class MCFDualYOLO(nn.Module):
    """MCF 双分支（YOLOv11-RGBT midfusion-MCF 结构）。

    与 DualBranchYOLO 的区别:
        - 融合单元: P3/P4/P5 每层一个 ZeroConv2d，fused = rgb_feat + ZeroConv(aux_feat)。
          初始 ZeroConv 权重为 0，融合特征 = RGB 主分支特征（可控微调）；
        - 主分支（RGB）建议冻结预训练权重，只训练辅助分支 + ZeroConv + head。
    接口与 DualBranchYOLO 一致（train/eval 兼容现有训练循环与评估链路）。
    """

    def __init__(self, backbone_rgb, backbone_aux, head, fuse_levels=(16, 19, 22)):
        super().__init__()
        self.backbone_rgb = backbone_rgb
        self.backbone_aux = backbone_aux
        self.head = head
        self.fuse_levels = fuse_levels
        # P3/P4/P5 通道数（yolo11: 64/128/256），ZeroConv 将 aux 映射到同通道后 ADD
        self.fusion = nn.ModuleList([
            ZeroConv2d(c, c, kernel_size=1) for c in (64, 128, 256)
        ])
        self.model = [head]
        self.args = get_cfg(overrides={})
        self.stride = head.stride
        self.names = getattr(head, "names", {i: str(i) for i in range(head.nc)})
        self.loss_fn = None

    def _extract(self, backbone, x):
        y = []
        feats = []
        for m in backbone:
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
            if m.i in self.fuse_levels:
                feats.append(x)
        return feats

    def _forward_fused(self, img):
        rgb = img[:, :3]
        aux = img[:, 3:]
        feats_rgb = self._extract(self.backbone_rgb, rgb)
        feats_aux = self._extract(self.backbone_aux, aux)
        return [
            f_rgb + zc(f_aux)
            for zc, f_rgb, f_aux in zip(self.fusion, feats_rgb, feats_aux)
        ]

    def forward(self, x):
        if isinstance(x, dict):
            fused = self._forward_fused(x["img"])
            preds = self.head(fused)
            if self.loss_fn is None:
                from ultralytics.utils.loss import v8DetectionLoss
                self.loss_fn = v8DetectionLoss(self)
            return self.loss_fn(preds, x)
        return self.head(self._forward_fused(x))


def build_mcf_yolo_model(nc=12, pretrained="yolo11n.pt", verbose=False):
    """构建 MCF 双分支（RGB 主分支 + IR/Depth 辅助分支 + ZeroConv ADD 融合）。

    权重迁移:
        - RGB 分支: 3ch 预训练权重直接复制；
        - Aux 分支: 2ch，stem 用 RGB 权重均值初始化（v0.9.0 同款策略）；
        - ZeroConv2d: 零初始化（MCF 核心）。
    """
    model_rgb = DetectionModel("yolo11n.yaml", ch=3, nc=nc, verbose=verbose)
    model_aux = DetectionModel("yolo11n.yaml", ch=2, nc=nc, verbose=verbose)
    model_rgb.args = get_cfg(overrides={})
    model_aux.args = get_cfg(overrides={})

    if pretrained and os.path.exists(pretrained):
        sd = _load_pretrained_sd(pretrained)
        for tag, model in (("RGB", model_rgb), ("Aux", model_aux)):
            model_sd = model.state_dict()
            transferred, skipped = 0, 0
            for k, v in model_sd.items():
                if k in sd and sd[k].shape == v.shape:
                    model_sd[k] = sd[k]
                    transferred += 1
                else:
                    skipped += 1
            _init_stem_from_pretrained(model_sd, sd)
            model.load_state_dict(model_sd)
            print(f"  [{tag} branch] transferred: {transferred} keys, skipped: {skipped} keys")
    else:
        print(f"  Warning: pretrained {pretrained} not found, random init")

    backbone_rgb = nn.Sequential(*list(model_rgb.model)[:-1])
    backbone_aux = nn.Sequential(*list(model_aux.model)[:-1])
    head = model_rgb.model[-1]

    model = MCFDualYOLO(backbone_rgb, backbone_aux, head)
    model.args = model_rgb.args
    return model

