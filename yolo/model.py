"""YOLO 模型构建与预训练权重迁移。

核心思路（v0.8.0 早期融合方案）:
    1. 用 `yolo11n.yaml` 构建 `ch=5/3`、`nc=12` 的模型，保证首层卷积输入
       通道与类别数与赛题一致；
    2. 从官方 `yolo11n.pt`（COCO 预训练）迁移**形状匹配**的权重：backbone/neck
       全部继承，首层卷积与分类头因通道/类别数变化而从零训练；
    3. 首层卷积新增的 IR/Depth 通道用 RGB 三通道权重均值初始化，
       避免融合分支一开始就是随机噪声、梯度消失。
"""

import os

import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel


def build_yolo_model(ch=3, nc=12, pretrained="yolo11n.pt", verbose=False):
    """构建 YOLO11 检测模型。

    Args:
        ch: 输入通道数（3=RGB 对照，5=RGB+IR+Depth 早期融合）
        nc: 类别数（赛题 12 类）
        pretrained: 官方预训练权重路径；不存在时使用随机初始化
        verbose: 是否打印模型结构

    Returns:
        model: ultralytics.nn.tasks.DetectionModel
    """
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


def transfer_pretrained_weights(model, pretrained_path):
    """从官方 checkpoint 迁移形状匹配的权重，并初始化 stem 新通道。

    Args:
        model: 构建好的目标模型
        pretrained_path: yolo11n.pt 路径

    Returns:
        (transferred, skipped): 迁移/跳过的参数量
    """
    # ultralytics 官方 checkpoint 里序列化了 DetectionModel 全局对象，
    # 必须用 weights_only=False（来源为官方 assets，可信）。
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    src = ckpt.get("model", ckpt)
    sd = src.state_dict() if hasattr(src, "state_dict") else src

    model_sd = model.state_dict()
    transferred, skipped = 0, 0
    for k, v in model_sd.items():
        if k in sd and sd[k].shape == v.shape:
            model_sd[k] = sd[k]
            transferred += 1
        else:
            skipped += 1

    # 首层卷积新通道初始化：RGB 三通道权重均值复制到 IR/Depth 通道
    stem_key = "model.0.conv.weight"
    if stem_key in model_sd:
        w = model_sd[stem_key]
        if w.shape[1] > 3:
            rgb_w = w[:, :3]
            mean_w = rgb_w.mean(dim=1, keepdim=True)
            extra = w.shape[1] - 3
            model_sd[stem_key] = torch.cat(
                [rgb_w, mean_w.repeat(1, extra, 1, 1)], dim=1
            )

    model.load_state_dict(model_sd)
    return transferred, skipped


def apply_freeze(model, freeze_layers):
    """冻结指定层索引的参数（ultralytics freeze 语义，0-based）。

    Args:
        model: DetectionModel（model.model 为 nn.Sequential）
        freeze_layers: 要冻结的层索引列表，如 [0,...,9] 冻结 backbone；
                       空列表表示全部可训练
    """
    freeze_layers = freeze_layers or []
    for name, p in model.named_parameters():
        # 解析 "model.<idx>...." 形式的参数名
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
            layer_idx = int(parts[1])
            if layer_idx in freeze_layers:
                p.requires_grad_(False)
        else:
            p.requires_grad_(True)

    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total = sum(1 for p in model.parameters())
    print(f"  Frozen layers: {sorted(freeze_layers)} ({frozen}/{total} params frozen)")


def count_parameters(model):
    """统计模型参数量（对齐旧 train.py 的打印格式）。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
