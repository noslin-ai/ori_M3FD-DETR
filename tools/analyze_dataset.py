"""数据集分析工具 — 统计 12 类目标分布、小目标占比、类别不均衡等信息。

帮助:
    1. 了解类别分布
    2. 发现小目标比例
    3. 指导数据增强策略
    4. 辅助超参数调整

运行方式:
    cd M3F-DETR
    python tools/analyze_dataset.py --data-root data/train
"""

import os
import sys
import argparse
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from datasets.label_parser import load_yolo_label


# AIC 2026 竞赛类别名称映射（12类）
CLASS_NAMES = {
    0:  "person",
    1:  "boat",
    2:  "animal",
    3:  "seat",
    4:  "sign",
    5:  "bicycle",
    6:  "car",
    7:  "ball",
    8:  "light",
    9:  "garbagecan",
    10: "uav",
    11: "tricycle",
}


def analyze_dataset(data_root):
    """分析数据集。

    Args:
        data_root: 数据根目录（含 labels/ 子目录）

    Returns:
        dict: 分析结果
    """
    labels_dir = os.path.join(data_root, "labels")
    if not os.path.isdir(labels_dir):
        print(f"Error: labels directory not found: {labels_dir}")
        return None

    label_files = sorted([
        f for f in os.listdir(labels_dir) if f.endswith(".txt")
    ])

    if len(label_files) == 0:
        print(f"Error: no label files found in {labels_dir}")
        return None

    print(f"Total label files: {len(label_files)}")

    # 统计
    total_boxes = 0
    class_counts = Counter()
    img_box_counts = []  # 每张图的框数
    areas = []
    boxes_per_class = defaultdict(list)  # class_id → [w, h]

    for fname in label_files:
        path = os.path.join(labels_dir, fname)
        boxes, labels = load_yolo_label(path, 1920, 1080)

        if len(boxes) == 0:
            img_box_counts.append(0)
            continue

        total_boxes += len(boxes)
        img_box_counts.append(len(boxes))

        for box, label in zip(boxes, labels):
            # box: [xmin, ymin, xmax, ymax] in pixel coords
            xmin, ymin, xmax, ymax = box
            w = xmax - xmin
            h = ymax - ymin
            area = w * h

            class_counts[int(label)] += 1
            areas.append(area)
            boxes_per_class[int(label)].append((w, h))

    # ---- 打印统计 ----
    print("\n" + "=" * 70)
    print("  Dataset Analysis Results")
    print("=" * 70)

    print(f"\n  Total images:     {len(label_files)}")
    print(f"  Total boxes:      {total_boxes}")
    print(f"  Avg boxes/image:  {total_boxes / len(label_files):.1f}")
    print(f"  Max boxes/image:  {max(img_box_counts)}")
    print(f"  Min boxes/image:  {min(img_box_counts)}")
    print(f"  Images w/ 0 box:  {img_box_counts.count(0)} ({img_box_counts.count(0) / len(label_files) * 100:.1f}%)")

    # 面积统计
    areas_arr = np.array(areas)
    # 相对于 768×1280 的归一化面积
    norm_areas = areas_arr / (768 * 1280)

    print(f"\n  Area statistics (pixels):")
    print(f"    Mean:   {areas_arr.mean():.0f}")
    print(f"    Median: {np.median(areas_arr):.0f}")
    print(f"    Min:    {areas_arr.min():.0f}")
    print(f"    Max:    {areas_arr.max():.0f}")

    # 小目标比例 (面积 < 32^2 = 1024)
    small_thresh = 32 * 32
    small_count = (areas_arr < small_thresh).sum()
    print(f"\n  Small objects (<32×32 px): {small_count} ({small_count / len(areas) * 100:.1f}%)")
    medium_thresh = 96 * 96
    medium_count = ((areas_arr >= small_thresh) & (areas_arr < medium_thresh)).sum()
    print(f"  Medium objects (32×32~96×96): {medium_count} ({medium_count / len(areas) * 100:.1f}%)")
    large_count = (areas_arr >= medium_thresh).sum()
    print(f"  Large objects (>96×96): {large_count} ({large_count / len(areas) * 100:.1f}%)")

    # 各类别统计
    print(f"\n  Class distribution:")
    print(f"  {'ID':<4} {'Name':<16} {'Count':<8} {'%':<8} {'Avg W':<10} {'Avg H':<10}")
    print(f"  {'─' * 60}")
    for cls_id in sorted(class_counts.keys()):
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        count = class_counts[cls_id]
        pct = count / total_boxes * 100
        bb = np.array(boxes_per_class[cls_id])
        avg_w = bb[:, 0].mean()
        avg_h = bb[:, 1].mean()
        print(f"  {cls_id:<4} {name:<16} {count:<8} {pct:<8.1f} {avg_w:<10.1f} {avg_h:<10.1f}")

    # 类别不均衡警告
    print(f"\n  Imbalance analysis:")
    img_size = 768 * 1280
    counts_arr = np.array([class_counts.get(i, 0) for i in range(12)])
    max_cls = counts_arr.max()
    min_cls = counts_arr[counts_arr > 0].min() if (counts_arr > 0).any() else 0
    if min_cls > 0:
        ratio = max_cls / min_cls
        print(f"    Max/Min ratio: {ratio:.1f}x")
        if ratio > 10:
            print(f"    ⚠ Severe class imbalance! Consider:")
            print(f"       - Class-aware sampling")
            print(f"       - Focal Loss (already used)")
            print(f"       - Data augmentation for minority classes")

    print(f"\n  Suggestions:")
    if small_count / len(areas) > 0.1:
        print(f"    - Use multi-scale training (default augment already handles)")
        print(f"    - Consider Mosaic augmentation for small objects")
    if img_box_counts.count(0) > 0:
        print(f"    - {img_box_counts.count(0)} images have no annotations")

    print("\n" + "=" * 70)

    return {
        "total_images": len(label_files),
        "total_boxes": total_boxes,
        "class_counts": dict(class_counts),
        "areas": areas_arr,
        "small_pct": small_count / len(areas),
        "max_min_ratio": max_cls / min_cls if min_cls > 0 else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser(description="AIC 2026 Dataset Analysis")
    parser.add_argument("--data-root", default="data/train", help="数据根目录")
    args = parser.parse_args()
    analyze_dataset(args.data_root)


if __name__ == "__main__":
    main()
