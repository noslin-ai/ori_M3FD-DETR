"""Prepare true tri-modal 3-channel YOLO data.

This keeps the proven Ultralytics YOLO path while changing the image signal to
use the three competition modalities:

    channel 0: visible luminance with local contrast and detail sharpening
    channel 1: infrared local-contrast intensity
    channel 2: depth structure, blended from normalized depth and depth edges

The design is intentionally conservative: labels and geometry are unchanged,
test images are only transformed for inference, and no pseudo labels are
generated. It borrows the practical lesson from RGB-depth-thermal fusion papers
that modality-specific complementary cues should be preserved instead of mixed
away before detection.
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = [
    "person", "boat", "animal", "seat", "sign", "bicycle",
    "car", "ball", "light", "garbage can", "uav", "tricycle",
]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
DEPTH_MIN = 300.0
DEPTH_MAX = 19999.0


def read_stems(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_map(directory):
    mapping = {}
    if not os.path.isdir(directory):
        return mapping
    for name in os.listdir(directory):
        stem, ext = os.path.splitext(name)
        if ext.lower() in IMG_EXTS:
            mapping[stem] = os.path.join(directory, name)
    return mapping


def load_depth(path):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = depth.astype(np.float32)
    invalid = depth <= 0
    if depth.max() > 255:
        depth = np.clip(depth, DEPTH_MIN, DEPTH_MAX)
        depth = (depth - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
    else:
        valid = depth > 0
        if valid.any():
            depth[valid] = (depth[valid] - 1.0) / 254.0
    depth[invalid] = 0.0
    return np.clip(depth, 0.0, 1.0)


def normalize_u8(x):
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    lo, hi = float(x.min()), float(x.max())
    if hi > lo:
        x = (x - lo) / (hi - lo)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def clahe_gray(x, clip_limit=2.0):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(x)


def sobel_edges(x):
    x8 = normalize_u8(x)
    gx = cv2.Sobel(x8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x8, cv2.CV_32F, 0, 1, ksize=3)
    return normalize_u8(cv2.magnitude(gx, gy))


def fuse_trimodal(rgb_path, ir_path, depth_path, sharpen=0.25, depth_edge_weight=0.45):
    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(rgb_path)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        raise FileNotFoundError(ir_path)
    depth = load_depth(depth_path)

    h, w = bgr.shape[:2]
    if ir.shape[:2] != (h, w):
        ir = cv2.resize(ir, (w, h), interpolation=cv2.INTER_LINEAR)
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    bgr_dn = cv2.bilateralFilter(bgr, d=5, sigmaColor=24, sigmaSpace=3)
    lab = cv2.cvtColor(bgr_dn, cv2.COLOR_BGR2LAB)
    visible_l = clahe_gray(lab[:, :, 0], clip_limit=2.0)
    blur = cv2.GaussianBlur(visible_l, (0, 0), sigmaX=1.0)
    visible_detail = cv2.addWeighted(visible_l, 1.0 + sharpen, blur, -sharpen, 0)

    ir_detail = clahe_gray(cv2.medianBlur(ir, 3), clip_limit=2.0)
    depth_u8 = clahe_gray(normalize_u8(depth), clip_limit=2.0)
    depth_edge = sobel_edges(depth)
    depth_mix = cv2.addWeighted(
        depth_u8,
        1.0 - depth_edge_weight,
        depth_edge,
        depth_edge_weight,
        0,
    )

    return cv2.merge([visible_detail, ir_detail, depth_mix])


def ensure_clean_dir(path, overwrite=False):
    if os.path.isdir(path) and overwrite:
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def link_or_copy(src, dst, copy=False):
    if os.path.exists(dst):
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def process_stems(split, stems, maps, label_root, out_root, args, write_labels=True):
    img_dir = os.path.join(out_root, split, "images")
    label_out = os.path.join(out_root, split, "labels")
    ensure_clean_dir(img_dir, overwrite=False)
    if write_labels:
        ensure_clean_dir(label_out, overwrite=False)

    n_img, n_label, n_skip = 0, 0, 0
    for stem in stems:
        if stem not in maps["rgb"] or stem not in maps["ir"] or stem not in maps["depth"]:
            n_skip += 1
            print(f"skip missing modality: {stem}")
            continue
        image = fuse_trimodal(
            maps["rgb"][stem],
            maps["ir"][stem],
            maps["depth"][stem],
            sharpen=args.sharpen,
            depth_edge_weight=args.depth_edge_weight,
        )
        cv2.imwrite(
            os.path.join(img_dir, f"{stem}.jpg"),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), args.quality],
        )
        n_img += 1

        if write_labels:
            label_src = os.path.join(label_root, "labels", f"{stem}.txt")
            if os.path.exists(label_src):
                link_or_copy(label_src, os.path.join(label_out, f"{stem}.txt"), copy=args.copy_labels)
                n_label += 1
    return n_img, n_label, n_skip


def write_data_yaml(out):
    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    content = (
        f"path: {os.path.abspath(out)}\n"
        "train: train/images\n"
        "val: val/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_yaml}\n"
    )
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(content)


def prepare_train_val(args):
    train_stems = read_stems(os.path.join(args.splits, f"fold{args.fold}_train.txt"))
    val_stems = read_stems(os.path.join(args.splits, f"fold{args.fold}_val.txt"))
    maps = {
        "rgb": build_map(os.path.join(args.root, "visible")),
        "ir": build_map(os.path.join(args.root, "infrared")),
        "depth": build_map(os.path.join(args.root, "depth")),
    }
    if args.overwrite and os.path.isdir(args.out):
        shutil.rmtree(args.out)
    ensure_clean_dir(args.out)

    print(f"fold={args.fold} train={len(train_stems)} val={len(val_stems)} out={args.out}")
    tr = process_stems("train", train_stems, maps, args.root, args.out, args, write_labels=True)
    va = process_stems("val", val_stems, maps, args.root, args.out, args, write_labels=True)
    write_data_yaml(args.out)
    print(f"done train_images={tr[0]} train_labels={tr[1]} skipped={tr[2]}")
    print(f"done val_images={va[0]} val_labels={va[1]} skipped={va[2]}")
    print(f"yaml={os.path.join(args.out, 'data.yaml')}")


def prepare_test(args):
    maps = {
        "rgb": build_map(os.path.join(args.test_root, "visible")),
        "ir": build_map(os.path.join(args.test_root, "infrared")),
        "depth": build_map(os.path.join(args.test_root, "depth")),
    }
    stems = sorted(set(maps["rgb"]) & set(maps["ir"]) & set(maps["depth"]))
    if args.overwrite and os.path.isdir(args.test_out):
        shutil.rmtree(args.test_out)
    visible_dir = os.path.join(args.test_out, "visible")
    ensure_clean_dir(visible_dir)
    print(f"test={len(stems)} out={visible_dir}")
    n_img, n_skip = 0, 0
    for stem in stems:
        try:
            image = fuse_trimodal(
                maps["rgb"][stem],
                maps["ir"][stem],
                maps["depth"][stem],
                sharpen=args.sharpen,
                depth_edge_weight=args.depth_edge_weight,
            )
        except FileNotFoundError:
            n_skip += 1
            continue
        cv2.imwrite(
            os.path.join(visible_dir, f"{stem}.jpg"),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), args.quality],
        )
        n_img += 1
    print(f"done test_images={n_img} skipped={n_skip}")


def main():
    parser = argparse.ArgumentParser(description="Prepare tri-modal fused 3ch YOLO data")
    parser.add_argument("--root", default="data/train")
    parser.add_argument("--test-root", default="data/test")
    parser.add_argument("--splits", default="splits")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--out", default="data/yolo_trimodal_fusion_m")
    parser.add_argument("--test-out", default="data/test_trimodal_fusion")
    parser.add_argument("--quality", type=int, default=96)
    parser.add_argument("--sharpen", type=float, default=0.25)
    parser.add_argument("--depth-edge-weight", type=float, default=0.45)
    parser.add_argument("--copy-labels", action="store_true", help="copy labels instead of symlinking")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-train-val", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    if not args.skip_train_val:
        prepare_train_val(args)
    if not args.skip_test:
        prepare_test(args)


if __name__ == "__main__":
    main()
