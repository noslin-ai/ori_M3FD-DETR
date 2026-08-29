#!/usr/bin/env python
"""对含少数类的训练图做整图复制，平衡类别分布（YOLO 图级过采样）。

复制倍数（按稀有度）:
    boat(1)=2x, ball(7)=3x, uav(10)=2x, tricycle(11)=10x, garbage can(9)=1x
每张图的复制倍数 = 该图所含稀有类对应倍数的最大值。
输出: dst/train/{images,labels}（原始 + 副本），val 原样复制。
"""
import os, argparse, shutil
from collections import Counter

MULT = {1: 2, 7: 3, 10: 2, 11: 10, 9: 1}
NAMES = {0:"person",1:"boat",2:"animal",3:"seat",4:"sign",5:"bicycle",
         6:"car",7:"ball",8:"light",9:"garbage can",10:"uav",11:"tricycle"}

def read_cls(label_path):
    with open(label_path) as fh:
        return [int(l.split()[0]) for l in fh if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/yolo_sar_m")
    ap.add_argument("--dst", default="data/yolo_sar_v3")
    args = ap.parse_args()
    src, dst = args.src, args.dst

    for split in ("train", "val"):
        os.makedirs(os.path.join(dst, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dst, split, "labels"), exist_ok=True)

    # ---- val 原样复制 ----
    val_lbl = os.path.join(src, "val", "labels")
    for f in sorted(os.listdir(val_lbl)):
        stem = os.path.splitext(f)[0]
        shutil.copy2(os.path.join(src, "val", "images", stem + ".jpg"),
                     os.path.join(dst, "val", "images", stem + ".jpg"))
        shutil.copy2(os.path.join(val_lbl, f), os.path.join(dst, "val", "labels", f))

    # ---- 分析 train 每张图的稀有类，确定复制倍数 ----
    train_lbl = os.path.join(src, "train", "labels")
    train_img = os.path.join(src, "train", "images")
    files = sorted(os.listdir(train_lbl))
    before = Counter()
    plan = {}
    for f in files:
        stem = os.path.splitext(f)[0]
        cls = read_cls(os.path.join(train_lbl, f))
        before.update(cls)
        m = 1
        for c in set(cls):
            if c in MULT:
                m = max(m, MULT[c])
        plan[stem] = m

    # ---- 复制原始 + 副本 ----
    after = Counter()
    n_dup = 0
    for f in files:
        stem = os.path.splitext(f)[0]
        cls = read_cls(os.path.join(train_lbl, f))
        m = plan[stem]
        img_src = os.path.join(train_img, stem + ".jpg")
        for k in range(m):
            ns = stem if k == 0 else f"{stem}_dup{k}"
            shutil.copy2(img_src, os.path.join(dst, "train", "images", ns + ".jpg"))
            shutil.copy2(os.path.join(train_lbl, f), os.path.join(dst, "train", "labels", ns + ".txt"))
            after.update(cls)
            if k > 0:
                n_dup += 1

    print("=== 过采样结果 ===")
    print(f"{'类别':<12}{'前':>8}{'后':>8}{'倍数':>8}")
    for c in sorted(before):
        mult = after[c] / before[c] if before[c] else 0
        print(f"{NAMES[c]:<12}{before[c]:>8}{after[c]:>8}{mult:>8.2f}")
    print(f"\n原始图 {len(files)} 张 -> 新增副本 {n_dup} 张, 训练图共 {len(files)+n_dup} 张")

    # ---- 写 data.yaml ----
    names_yaml = "\n".join(f"  {i}: {n}" for i, n in NAMES.items())
    yaml_content = (
        f"path: {os.path.abspath(dst)}\n"
        "train: train/images\nval: val/images\n"
        f"nc: {len(NAMES)}\nnames:\n{names_yaml}\n"
    )
    with open(os.path.join(dst, "data.yaml"), "w") as fh:
        fh.write(yaml_content)
    print(f"\n已生成 {os.path.abspath(dst)}/data.yaml")

if __name__ == "__main__":
    main()
