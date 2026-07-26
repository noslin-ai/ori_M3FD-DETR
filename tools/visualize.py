"""可视化工具 — 绘制检测框和三模态图像。

用于:
    1. 查看训练样本中的标注
    2. 对比 RGB/IR/Depth 三模态
    3. 验证数据增强效果
    4. 查看预测结果

运行方式:
    cd M3F-DETR
    # 可视化训练样本
    python tools/visualize.py --data-root data/train --num 10 --output vis_samples

    # 可视化特定索引
    python tools/visualize.py --data-root data/train --indices 0 10 20 30 --output vis_samples
"""

import os
import sys
import argparse
import random
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from datasets.rgb_ir_depth_dataset import RGBIRDepthDataset

# AIC 2026 竞赛类别颜色映射（12类）
CLASS_COLORS = [
    (0, 255, 0),     # 0: person - green
    (255, 0, 0),     # 1: boat - blue
    (0, 0, 255),     # 2: animal - red
    (255, 255, 0),   # 3: seat - cyan
    (0, 255, 255),   # 4: sign - yellow
    (255, 0, 255),   # 5: bicycle - magenta
    (128, 0, 128),   # 6: car - purple
    (128, 128, 0),   # 7: ball - olive
    (0, 128, 128),   # 8: light - teal
    (128, 128, 255), # 9: garbagecan - pink
    (255, 128, 0),   # 10: uav - orange
    (0, 128, 255),   # 11: tricycle - light orange
]

CLASS_NAMES = {
    0: "person", 1: "boat", 2: "animal", 3: "seat",
    4: "sign", 5: "bicycle", 6: "car", 7: "ball",
    8: "light", 9: "garbagecan", 10: "uav", 11: "tricycle",
}


def draw_boxes(img, boxes, labels, confs=None):
    """在图像上绘制检测框。

    Args:
        img: BGR 图像 (H, W, 3)
        boxes: (N, 4) [xmin, ymin, xmax, ymax] pixel coords
        labels: (N,) class labels
        confs: (N,) 置信度（可选）

    Returns:
        img: 绘制后的图像
    """
    img = img.copy()
    for i, (box, label) in enumerate(zip(boxes, labels)):
        label = int(label)
        color = CLASS_COLORS[label % len(CLASS_COLORS)]
        name = CLASS_NAMES.get(label, f"cls_{label}")

        xmin, ymin, xmax, ymax = map(int, box)

        # 画框
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

        # 标签文字
        if confs is not None:
            text = f"{name} {confs[i]:.2f}"
        else:
            text = name

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (xmin, ymin - th - 4), (xmin + tw, ymin), color, -1)
        cv2.putText(img, text, (xmin, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def normalize_ir(ir_img):
    """归一化 IR 图像用于显示。"""
    ir_img = ir_img.copy()
    if ir_img.max() > 0:
        ir_img = ir_img / ir_img.max() * 255
    return ir_img.astype(np.uint8)


def normalize_depth(depth_img):
    """归一化 Depth 图像用于显示。"""
    depth_img = depth_img.copy()
    if depth_img.max() > 0:
        depth_img = depth_img / depth_img.max() * 255
    return depth_img.astype(np.uint8)


def visualize_samples(dataset, indices, output_dir, prefix="sample"):
    """可视化指定索引的样本。

    Args:
        dataset: RGBIRDepthDataset
        indices: 要可视化的索引列表
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx in indices:
        data = dataset[idx]

        # RGB 图像
        rgb = data["rgb"].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # IR 图像（取第一通道）
        ir = data["ir"].permute(1, 2, 0).cpu().numpy()
        ir_disp = normalize_ir(ir[..., 0])  # 单通道灰度
        ir_disp = cv2.cvtColor(ir_disp, cv2.COLOR_GRAY2BGR)

        # Depth 图像（取第一通道）
        depth = data["depth"].permute(1, 2, 0).cpu().numpy()
        depth_disp = normalize_depth(depth[..., 0])
        depth_disp = cv2.cvtColor(depth_disp, cv2.COLOR_GRAY2BGR)
        depth_disp = cv2.applyColorMap(depth_disp, cv2.COLORMAP_JET)

        # 绘制标注
        target = data["target"]
        boxes = target["boxes"].cpu().numpy()
        labels = target["labels"].cpu().numpy()

        if len(boxes) > 0:
            # cxcywh → xyxy
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            img_h, img_w = rgb.shape[:2]
            xyxy_boxes = np.stack([
                (cx - w / 2) * img_w,
                (cy - h / 2) * img_h,
                (cx + w / 2) * img_w,
                (cy + h / 2) * img_h,
            ], axis=1)
            rgb_vis = draw_boxes(rgb, xyxy_boxes, labels)
        else:
            rgb_vis = rgb

        # 三模态对比图
        # Layout: [ RGB w/ boxes | IR | Depth ]
        target_h = max(rgb.shape[0], ir_disp.shape[0], depth_disp.shape[0])
        gap = 10

        h = target_h + 30  # for title
        w = rgb.shape[1] * 3 + gap * 2

        canvas = np.ones((h, w, 3), dtype=np.uint8) * 240

        # RGB + boxes
        y_offset = 30
        canvas[y_offset:y_offset + rgb.shape[0], 0:rgb.shape[1]] = rgb_vis
        cv2.putText(canvas, "RGB + GT Boxes", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # IR
        x_ir = rgb.shape[1] + gap
        canvas[y_offset:y_offset + ir_disp.shape[0], x_ir:x_ir + ir_disp.shape[1]] = ir_disp
        cv2.putText(canvas, "Infrared", (x_ir + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Depth
        x_depth = x_ir + ir_disp.shape[1] + gap
        canvas[y_offset:y_offset + depth_disp.shape[0], x_depth:x_depth + depth_disp.shape[1]] = depth_disp
        cv2.putText(canvas, "Depth", (x_depth + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # 保存
        out_path = os.path.join(output_dir, f"{prefix}_{idx:04d}.jpg")
        cv2.imwrite(out_path, canvas)

    print(f"\n  Saved {len(indices)} visualizations → {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="M3F-DINO Visualization Tool")
    parser.add_argument("--data-root", default="data/train", help="数据根目录")
    parser.add_argument("--num", type=int, default=10, help="可视化数量")
    parser.add_argument("--indices", type=int, nargs="+", default=None, help="指定索引")
    parser.add_argument("--output", default="vis_samples", help="输出目录")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("  M3F-DINO Visualization Tool")
    print("=" * 70)

    # 数据集
    dataset = RGBIRDepthDataset(args.data_root, train=True)
    print(f"  Dataset: {len(dataset)} samples")

    if args.indices is not None:
        indices = args.indices
    else:
        random.seed(args.seed)
        indices = random.sample(range(len(dataset)), min(args.num, len(dataset)))

    print(f"  Visualizing {len(indices)} samples")
    visualize_samples(dataset, indices, args.output)

    print("\n" + "=" * 70)
    print(f"  Visualization complete! → {args.output}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
