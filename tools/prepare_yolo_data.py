"""把赛题三模态数据整理成 ultralytics 原生 YOLO 格式（RGB-only 用）。

输出目录结构（图片用符号链接，不复制，节省磁盘）:
    out/
    ├── train/
    │   ├── images/*.jpg        # 指向 data/train/visible
    │   └── labels/*.txt        # 指向 data/train/labels
    ├── val/
    │   ├── images/*.jpg
    │   └── labels/*.txt
    └── data.yaml               # ultralytics 数据集配置

用法:
    python tools/prepare_yolo_data.py --out data/yolo_x --fold 1
"""

import os
import argparse
import shutil

# 赛题 12 类名称（与官方编号一致）
CLASS_NAMES = [
    "person", "boat", "animal", "seat", "sign", "bicycle",
    "car", "ball", "light", "garbage can", "uav", "tricycle",
]


def load_stems(split_file):
    """读取 split 文件中的 stem 列表。"""
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


def symlink_by_stem(src_dir, dst_dir, stems):
    """按 stem 把 src_dir 中的文件符号链接到 dst_dir。"""
    os.makedirs(dst_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        stem = os.path.splitext(f)[0]
        if stem in stems:
            src = os.path.abspath(os.path.join(src_dir, f))
            dst = os.path.join(dst_dir, f)
            if not os.path.exists(dst):
                os.symlink(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO-format RGB data")
    parser.add_argument("--root", default="data/train", help="三模态训练数据根目录")
    parser.add_argument("--splits", default="splits", help="fold split 目录")
    parser.add_argument("--fold", type=int, default=1, help="fold 编号")
    parser.add_argument("--out", default="data/yolo_rgb", help="输出目录")
    args = parser.parse_args()

    visible_dir = os.path.join(args.root, "visible")
    label_dir = os.path.join(args.root, "labels")
    train_stems = load_stems(os.path.join(args.splits, f"fold{args.fold}_train.txt"))
    val_stems = load_stems(os.path.join(args.splits, f"fold{args.fold}_val.txt"))
    print(f"train={len(train_stems)} val={len(val_stems)}")

    for split, stems in (("train", train_stems), ("val", val_stems)):
        symlink_by_stem(visible_dir, os.path.join(args.out, split, "images"), set(stems))
        symlink_by_stem(label_dir, os.path.join(args.out, split, "labels"), set(stems))

    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_content = (
        f"path: {os.path.abspath(args.out)}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_yaml}\n"
    )
    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    n_train = len(os.listdir(os.path.join(args.out, "train", "images")))
    n_val = len(os.listdir(os.path.join(args.out, "val", "images")))
    print(f"done: {args.out} train_images={n_train} val_images={n_val} yaml={yaml_path}")


if __name__ == "__main__":
    main()
