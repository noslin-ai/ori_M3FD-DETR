import os

import cv2
import torch
from torch.utils.data import Dataset

from .depth_process import process_depth
from .label_parser import load_yolo_label
from .transforms import MultiModalTransform

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class RGBIRDepthDataset(Dataset):
    """RGB + Infrared + Depth 三模态目标检测数据集。

    目录结构（匹配竞赛 zip 解压后的实际目录名）:
        root/
        ├── visible/     *.png / *.jpg (3通道8bit)
        ├── infrared/    *.png / *.jpg (3通道8bit灰度堆叠)
        ├── depth/       *.png (16bit uint16) / *.jpg (8bit)
        └── labels/      *.txt (YOLO格式)

    输出适配 DETR / DINO 的 target 格式。
    """

    def __init__(self, root, train=True, size=(384, 640), normalize_rgb=False):
        # 竞赛 zip 解压后的实际目录名
        self.rgb_dir = os.path.join(root, "visible")
        self.ir_dir = os.path.join(root, "infrared")
        self.depth_dir = os.path.join(root, "depth")
        self.label_dir = os.path.join(root, "labels")

        self.train = train
        self.normalize_rgb = normalize_rgb
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        # 以 visible 目录的文件列表为基准
        self.rgb_names = sorted(os.listdir(self.rgb_dir))

        # 构建 stem -> 完整路径映射，兼容 .jpg / .png 混用
        self.rgb_map = self._build_file_map(self.rgb_dir)
        self.ir_map = self._build_file_map(self.ir_dir)
        self.depth_map = self._build_file_map(self.depth_dir)

        # 标签是否存在（测试集没有 labels）
        self.has_labels = os.path.isdir(self.label_dir)

        self.transform = MultiModalTransform(size=tuple(size), train=train)

    def _build_file_map(self, dir_path):
        """构建 {stem: full_path} 映射，兼容不同图片格式。"""
        file_map = {}
        for f in os.listdir(dir_path):
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXTENSIONS:
                file_map[stem] = os.path.join(dir_path, f)
        return file_map

    def __len__(self):
        return len(self.rgb_names)

    def __getitem__(self, idx):
        name = self.rgb_names[idx]
        stem, _ = os.path.splitext(name)

        # 查找各模态对应的文件（扩展名可能不同）
        rgb_path = self.rgb_map[stem]
        ir_path = self.ir_map[stem]
        depth_path = self.depth_map[stem]

        # ---- RGB ----
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # ---- IR ----
        ir = cv2.imread(ir_path)
        ir = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)

        # ---- Depth ----
        depth = process_depth(depth_path)

        # ---- 标签（测试集无标签则创建空 target）----
        if self.has_labels:
            label_path = os.path.join(self.label_dir, stem + ".txt")
            boxes, labels = load_yolo_label(label_path, w, h)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # ---- 数据增强 ----
        rgb, ir, depth, boxes = self.transform(rgb, ir, depth, boxes)

        # ---- BBox 格式转换: 像素(xmin,ymin,xmax,ymax) → DETR 归一化(cx,cy,w,h) ----
        if boxes.shape[0] > 0:
            new_h, new_w = self.transform.size  # 增强后的图像尺寸
            xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            cx = (xmin + xmax) / (2.0 * new_w)
            cy = (ymin + ymax) / (2.0 * new_h)
            bw = (xmax - xmin) / new_w
            bh = (ymax - ymin) / new_h
            boxes = torch.stack([cx, cy, bw, bh], dim=1)

        # ---- 转 tensor ----
        rgb = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
        ir = torch.from_numpy(ir.copy()).permute(2, 0, 1).float() / 255.0
        depth = torch.from_numpy(depth.copy()).float()

        if self.normalize_rgb:
            rgb = (rgb - self.rgb_mean) / self.rgb_std

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return {"rgb": rgb, "ir": ir, "depth": depth, "target": target, "name": stem}
