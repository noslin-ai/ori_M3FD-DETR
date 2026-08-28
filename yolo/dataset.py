"""YOLO 三模态数据集 — RGB-only 对照与 5ch 早期融合共用。

目录结构（与 datasets/rgb_ir_depth_dataset.py 保持一致）:
    root/
    ├── visible/     RGB 3 通道 8bit（jpg/png）
    ├── infrared/    红外 3 通道 8bit 灰度堆叠（三通道内容一致，只取单通道）
    ├── depth/       16bit 深度图（单位毫米，300~19999 有效，0 为无效）
    └── labels/      YOLO 归一化标签 [cls cx cy w h]（测试集无此目录）

mode:
    rgb:    输出 (3, H, W)，仅 RGB，用于对照验证环境/数据/评估链路
    fusion: 输出 (5, H, W)，[RGB(3), IR 灰度(1), Depth 归一化(1)] 早期融合

说明:
    1. 与旧 M3F-DETR 链路不同，这里直接输出 YOLO 归一化标签 (cx, cy, w, h)，
       不再转换为像素坐标，避免训练/评估之间的坐标约定分叉。
    2. 训练增强与旧 MultiModalTransform 保持一致思路（同步 resize + 随机水平翻转），
       另加 RGB 光度扰动；几何增强对所有通道同步生效，光度增强只作用于 RGB。
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# 深度传感器有效范围（毫米）— 赛题规定 30cm~20m
DEPTH_MIN = 300
DEPTH_MAX = 19999


def load_yolo_label(label_path):
    """加载 YOLO 归一化标签 [cls cx cy w h]，越界值裁剪到合法范围。

    Args:
        label_path: txt 路径，每行一个目标

    Returns:
        cls:    (N,) int64
        boxes:  (N, 4) float32，归一化 [cx, cy, w, h]
    """
    cls_list = []
    box_list = []
    with open(label_path, "r") as f:
        for line in f:
            data = line.strip().split()
            if len(data) < 5:
                continue
            cls_id = int(float(data[0]))
            cx = min(max(float(data[1]), 0.0), 1.0)
            cy = min(max(float(data[2]), 0.0), 1.0)
            w = min(max(float(data[3]), 1e-6), 1.0)
            h = min(max(float(data[4]), 1e-6), 1.0)
            cls_list.append(cls_id)
            box_list.append([cx, cy, w, h])

    if not box_list:
        return (
            torch.zeros(0, dtype=torch.int64),
            torch.zeros((0, 4), dtype=torch.float32),
        )
    return (
        torch.tensor(cls_list, dtype=torch.int64),
        torch.tensor(box_list, dtype=torch.float32),
    )


class YOLOFusionDataset(Dataset):
    """三模态 YOLO 数据集（rgb / fusion 两种模式）。"""

    def __init__(self, root, mode="rgb", size=(384, 640), train=True, nc=12):
        # 竞赛 zip 解压后的实际目录名
        self.rgb_dir = os.path.join(root, "visible")
        self.ir_dir = os.path.join(root, "infrared")
        self.depth_dir = os.path.join(root, "depth")
        self.label_dir = os.path.join(root, "labels")

        self.mode = mode
        self.size = tuple(size)          # (H, W)
        self.train = train
        self.nc = nc
        self.has_labels = os.path.isdir(self.label_dir)

        # 以 visible 目录为基准构建 stem -> 完整路径映射，兼容 .jpg/.png 混用
        self.names = sorted(self._list_stems(self.rgb_dir))
        self.rgb_map = self._build_file_map(self.rgb_dir)
        self.ir_map = self._build_file_map(self.ir_dir)
        self.depth_map = self._build_file_map(self.depth_dir)

    def _list_stems(self, dir_path):
        """返回目录中所有图片的 stem（不含扩展名）。"""
        stems = []
        for f in os.listdir(dir_path):
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXTENSIONS:
                stems.append(stem)
        return stems

    def _build_file_map(self, dir_path):
        """构建 {stem: full_path} 映射，兼容不同图片格式。"""
        file_map = {}
        for f in os.listdir(dir_path):
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXTENSIONS:
                file_map[stem] = os.path.join(dir_path, f)
        return file_map

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        stem = self.names[idx]
        rgb_path = self.rgb_map[stem]
        ir_path = self.ir_map[stem]
        depth_path = self.depth_map[stem]

        # ---- RGB: BGR -> RGB uint8 ----
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # ---- IR: 三通道灰度堆叠，取单通道即可 ----
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

        # ---- Depth: 16bit 毫米值 -> [0,1] 归一化单通道（无效值掩码为 0）----
        depth = self._load_depth(depth_path)

        # ---- 标签（测试集无标签或标签文件缺失则返回空目标）----
        label_path = os.path.join(self.label_dir, stem + ".txt")
        if self.has_labels and os.path.exists(label_path):
            cls, boxes = load_yolo_label(label_path)
        else:
            cls = torch.zeros(0, dtype=torch.int64)
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        # ---- 同步几何增强（resize + 随机水平翻转），与旧链路一致 ----
        rgb, ir, depth = self._resize(rgb, ir, depth)
        if self.train:
            rgb = self._rgb_photometric_jitter(rgb)
            if random.random() < 0.5:
                rgb = np.fliplr(rgb)
                ir = np.fliplr(ir)
                depth = np.fliplr(depth)
                if boxes.shape[0] > 0:
                    boxes[:, 0] = 1.0 - boxes[:, 0]

        # ---- 通道拼接: rgb(3) + ir(1) + depth(1) ----
        h, w = self.size
        if self.mode == "fusion":
            img = np.concatenate(
                [rgb.astype(np.float32) / 255.0,
                 ir.astype(np.float32)[..., None] / 255.0,
                 depth.astype(np.float32)[..., None]],
                axis=-1,
            )
        else:
            img = rgb.astype(np.float32) / 255.0

        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

        return {
            "img": img,          # (C, H, W) float32，值域 [0,1]
            "cls": cls,          # (N,) int64
            "bboxes": boxes,     # (N, 4) float32，归一化 [cx, cy, w, h]
            "name": stem,
        }

    def _load_depth(self, depth_path):
        """加载深度图并归一化到 [0,1]（单通道 float32 HxW）。

        兼容两种格式（与 datasets/depth_process.py 逻辑一致）:
            uint16 PNG: 原始毫米值，裁剪到 300~19999 后归一化
            uint8 JPG:  有损可视化深度，1~255 线性映射到 0~1
        """
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"无法读取深度图: {depth_path}")

        # JPG 会被读取为 3 通道，取第一个通道
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        depth = depth.astype(np.float32)
        invalid = depth <= 0
        if depth.max() > 255:
            # uint16 PNG：原始深度值（毫米）
            depth = np.clip(depth, DEPTH_MIN, DEPTH_MAX)
            depth = (depth - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
        else:
            # uint8 JPG：有损可视化深度（赛题样例实际为此格式）
            valid = depth > 0
            if valid.any():
                depth[valid] = (depth[valid] - 1.0) / 254.0
        depth[invalid] = 0.0
        return depth

    def _resize(self, rgb, ir, depth):
        """同步 resize 三模态到 (H, W)。"""
        # self.size 为 (H, W)，cv2.resize 的参数是 (W, H)
        h, w = self.size
        rgb = cv2.resize(rgb, (w, h))
        ir = cv2.resize(ir, (w, h))
        depth = cv2.resize(depth, (w, h))
        return rgb, ir, depth

    @staticmethod
    def _rgb_photometric_jitter(rgb):
        """RGB 光度扰动：轻微亮度/对比度变化，增强可见光分支鲁棒性。"""
        alpha = 1.0 + random.uniform(-0.2, 0.2)
        beta = random.uniform(-25, 25)
        return cv2.convertScaleAbs(rgb, alpha=alpha, beta=beta)


def collate_fn(batch):
    """将 Dataset 返回的 list of dict 整理为模型训练/推理所需的 batch 格式。

    与 ultralytics v8DetectionLoss 的约定对齐:
        img:       (B, C, H, W) float32
        cls:       (N,) int64，全 batch 拼接
        bboxes:    (N, 4) float32，归一化 [cx, cy, w, h]
        batch_idx: (N,) int64，每行目标所属的 batch 下标
    """
    imgs = torch.stack([b["img"] for b in batch])
    cls_list, box_list, bidx_list = [], [], []
    for i, b in enumerate(batch):
        n = b["cls"].shape[0]
        cls_list.append(b["cls"])
        box_list.append(b["bboxes"])
        bidx_list.append(torch.full((n,), i, dtype=torch.int64))

    cls = torch.cat(cls_list) if cls_list else torch.zeros(0, dtype=torch.int64)
    bboxes = torch.cat(box_list) if box_list else torch.zeros((0, 4), dtype=torch.float32)
    batch_idx = torch.cat(bidx_list) if bidx_list else torch.zeros(0, dtype=torch.int64)

    return {
        "img": imgs,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx,
        "names": [b["name"] for b in batch],
    }


def get_fold_indices(dataset, fold, split_dir="splits"):
    """获取 fold 训练/验证索引（与 train.py 的 get_fold_indices 等价）。

    Args:
        dataset: YOLOFusionDataset
        fold: fold 编号 (1-based)
        split_dir: split 文件目录

    Returns:
        train_indices, val_indices
    """
    train_file = os.path.join(split_dir, f"fold{fold}_train.txt")
    val_file = os.path.join(split_dir, f"fold{fold}_val.txt")

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"  ⚠ Fold split 未找到: {train_file}")
        print(f"  运行: python tools/split_5fold.py")
        return list(range(len(dataset))), []

    with open(train_file) as f:
        train_stems = set(line.strip() for line in f)
    with open(val_file) as f:
        val_stems = set(line.strip() for line in f)

    train_indices, val_indices = [], []
    for idx, stem in enumerate(dataset.names):
        if stem in val_stems:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
    return train_indices, val_indices
