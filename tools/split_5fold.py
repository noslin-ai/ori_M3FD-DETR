"""5-Fold 数据划分脚本。

将 2000 个训练样本划分为 5 折，每折 1600 train / 400 val。
输出: splits/fold{N}_train.txt, splits/fold{N}_val.txt

运行方式:
    cd M3F-DETR
    python tools/split_5fold.py
"""

import os
import random
import argparse


def split_5fold(data_root, output_dir="splits", seed=42, num_folds=5):
    """5-Fold 划分。

    Args:
        data_root: 数据根目录（含 visible/ 子目录）
        output_dir: 输出目录
        seed: 随机种子
        num_folds: fold 数量
    """
    # 获取所有样本名（以 visible 目录为准）
    visible_dir = os.path.join(data_root, "visible")
    if not os.path.isdir(visible_dir):
        print(f"错误: 找不到 {visible_dir}")
        print("请先解压训练数据到 data/train/")
        return

    all_names = sorted(os.listdir(visible_dir))
    # 去掉非图片文件
    img_exts = {".png", ".jpg", ".jpeg"}
    all_names = [
        n for n in all_names
        if os.path.splitext(n)[1].lower() in img_exts
    ]

    print(f"总样本数: {len(all_names)}")

    # 打乱
    random.seed(seed)
    random.shuffle(all_names)

    # 划分 fold
    fold_size = len(all_names) // num_folds
    os.makedirs(output_dir, exist_ok=True)

    for fold in range(num_folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < num_folds - 1 else len(all_names)

        val_names = all_names[start:end]
        train_names = all_names[:start] + all_names[end:]

        # 保存
        train_file = os.path.join(output_dir, f"fold{fold+1}_train.txt")
        val_file = os.path.join(output_dir, f"fold{fold+1}_val.txt")

        with open(train_file, "w") as f:
            for name in train_names:
                stem = os.path.splitext(name)[0]
                f.write(stem + "\n")

        with open(val_file, "w") as f:
            for name in val_names:
                stem = os.path.splitext(name)[0]
                f.write(stem + "\n")

        print(
            f"  Fold {fold+1}: "
            f"train={len(train_names)}, val={len(val_names)} "
            f"→ {train_file}, {val_file}"
        )

    print(f"\n✅ {num_folds}-Fold 划分完成，输出目录: {output_dir}")
    print(f"训练命令: python train.py --fold 1  (训练第 1 折)")


def main():
    parser = argparse.ArgumentParser(description="5-Fold 数据划分")
    parser.add_argument(
        "--data-root", default="data/train",
        help="数据根目录（含 visible/ 子目录）"
    )
    parser.add_argument(
        "--output-dir", default="splits",
        help="输出目录"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-folds", type=int, default=5)

    args = parser.parse_args()
    split_5fold(args.data_root, args.output_dir, args.seed, args.num_folds)


if __name__ == "__main__":
    main()
