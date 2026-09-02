"""Prepare SAR-paper-inspired 3-channel enhanced YOLO data.

The current best baseline is native Ultralytics YOLO11m on RGB images. Earlier
5-channel and dual-branch fusion did not clearly beat RGB, so this script keeps
3-channel ImageNet-pretrained compatibility and only changes the image signal.

Enhancement ideas borrowed from recent SAR detection papers:
- DenoDet-style robust front end: small median/bilateral denoising before detail
  amplification, useful when IR/depth/RGB contain sensor noise.
- SARLite/QGPG-style small-target detail emphasis: CLAHE and unsharp masking on
  luminance, with weak IR/depth saliency injected into luminance only.
- Domain-adaptation caution: keep color channels and label geometry unchanged so
  the native YOLO pipeline remains comparable with the RGB baseline.

Output is a regular Ultralytics dataset:
    out/{train,val}/images/*.jpg
    out/{train,val}/labels/*.txt
    out/data.yaml
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


def to_uint8_norm(x):
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if x.max() > x.min():
        x = (x - x.min()) / (x.max() - x.min())
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def sobel_edges(x):
    x8 = to_uint8_norm(x)
    gx = cv2.Sobel(x8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return to_uint8_norm(mag)


def enhance_image(rgb_path, ir_path, depth_path, ir_weight=0.12, depth_weight=0.08, sharpen=0.35, fusion_mode="soft"):
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

    # Light denoising before local contrast/detail amplification.
    bgr_dn = cv2.bilateralFilter(bgr, d=5, sigmaColor=28, sigmaSpace=3)
    lab = cv2.cvtColor(bgr_dn, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    ir_eq = clahe.apply(cv2.medianBlur(ir, 3))
    depth_edge = sobel_edges(depth)

    if fusion_mode == "soft":
        base_weight = max(0.0, 1.0 - ir_weight - depth_weight)
        l_mix = (
            base_weight * l_eq.astype(np.float32)
            + ir_weight * ir_eq.astype(np.float32)
            + depth_weight * depth_edge.astype(np.float32)
        )
    elif fusion_mode == "gated":
        base = l_eq.astype(np.float32)
        ir_local = ir_eq.astype(np.float32) - cv2.GaussianBlur(ir_eq, (0, 0), sigmaX=5.0).astype(np.float32)
        ir_saliency = to_uint8_norm(np.maximum(ir_local, 0.0)).astype(np.float32)
        edge = depth_edge.astype(np.float32)
        gate = np.maximum(ir_saliency, edge) / 255.0
        gate = cv2.GaussianBlur(gate, (0, 0), sigmaX=1.0)
        ir_residual = np.clip(ir_local, -48.0, 48.0)
        l_mix = base + gate * (ir_weight * ir_residual + depth_weight * edge)
    else:
        raise ValueError(f"unsupported fusion_mode: {fusion_mode}")
    l_mix = np.clip(l_mix, 0, 255).astype(np.uint8)

    blur = cv2.GaussianBlur(l_mix, (0, 0), sigmaX=1.0)
    l_sharp = cv2.addWeighted(l_mix, 1.0 + sharpen, blur, -sharpen, 0)
    out = cv2.cvtColor(cv2.merge([l_sharp, a, b]), cv2.COLOR_LAB2BGR)
    return out


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


def process_split(split, stems, maps, root, out, args):
    img_dir = os.path.join(out, split, "images")
    label_out = os.path.join(out, split, "labels")
    ensure_clean_dir(img_dir, overwrite=False)
    ensure_clean_dir(label_out, overwrite=False)

    n_img, n_label = 0, 0
    for stem in stems:
        if stem not in maps["rgb"] or stem not in maps["ir"] or stem not in maps["depth"]:
            print(f"skip missing modality: {stem}")
            continue
        image = enhance_image(
            maps["rgb"][stem], maps["ir"][stem], maps["depth"][stem],
            ir_weight=args.ir_weight, depth_weight=args.depth_weight, sharpen=args.sharpen, fusion_mode=args.fusion_mode,
        )
        cv2.imwrite(os.path.join(img_dir, f"{stem}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        n_img += 1

        label_src = os.path.join(root, "labels", f"{stem}.txt")
        if os.path.exists(label_src):
            link_or_copy(label_src, os.path.join(label_out, f"{stem}.txt"), copy=args.copy_labels)
            n_label += 1
    return n_img, n_label


def process_test(maps, out, args):
    visible_out = os.path.join(out, "visible")
    ensure_clean_dir(visible_out, overwrite=False)
    stems = sorted(set(maps["rgb"]) & set(maps["ir"]) & set(maps["depth"]))
    n_img = 0
    for stem in stems:
        image = enhance_image(
            maps["rgb"][stem], maps["ir"][stem], maps["depth"][stem],
            ir_weight=args.ir_weight, depth_weight=args.depth_weight, sharpen=args.sharpen, fusion_mode=args.fusion_mode,
        )
        cv2.imwrite(os.path.join(visible_out, f"{stem}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        n_img += 1
    return n_img


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


def main():
    parser = argparse.ArgumentParser(description="Prepare SAR-style enhanced 3ch YOLO data")
    parser.add_argument("--root", default="data/train")
    parser.add_argument("--test-root", default="data/test")
    parser.add_argument("--splits", default="splits")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--out", default="data/yolo_sar_m")
    parser.add_argument("--test-out", default=None, help="optional test root with fused images under visible/")
    parser.add_argument("--ir-weight", type=float, default=0.12)
    parser.add_argument("--depth-weight", type=float, default=0.08)
    parser.add_argument("--sharpen", type=float, default=0.35)
    parser.add_argument("--fusion-mode", choices=("soft", "gated"), default="soft")
    parser.add_argument("--quality", type=int, default=96)
    parser.add_argument("--copy-labels", action="store_true", help="copy labels instead of symlinking")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-train-val", action="store_true")
    args = parser.parse_args()

    root = args.root
    out = args.out
    if not args.skip_train_val and args.overwrite and os.path.isdir(out):
        shutil.rmtree(out)
    if not args.skip_train_val:
        ensure_clean_dir(out)
        train_stems = read_stems(os.path.join(args.splits, f"fold{args.fold}_train.txt"))
        val_stems = read_stems(os.path.join(args.splits, f"fold{args.fold}_val.txt"))
        maps = {
            "rgb": build_map(os.path.join(root, "visible")),
            "ir": build_map(os.path.join(root, "infrared")),
            "depth": build_map(os.path.join(root, "depth")),
        }

        print(f"fold={args.fold} train={len(train_stems)} val={len(val_stems)} out={out}")
        tr = process_split("train", train_stems, maps, root, out, args)
        va = process_split("val", val_stems, maps, root, out, args)
        write_data_yaml(out)
        print(f"done train_images={tr[0]} train_labels={tr[1]} val_images={va[0]} val_labels={va[1]}")
        print(f"yaml={os.path.join(out, 'data.yaml')}")

    if args.test_out:
        if args.overwrite and os.path.isdir(args.test_out):
            shutil.rmtree(args.test_out)
        test_maps = {
            "rgb": build_map(os.path.join(args.test_root, "visible")),
            "ir": build_map(os.path.join(args.test_root, "infrared")),
            "depth": build_map(os.path.join(args.test_root, "depth")),
        }
        n_test = process_test(test_maps, args.test_out, args)
        print(f"done test_images={n_test} out={os.path.join(args.test_out, 'visible')}")


if __name__ == "__main__":
    main()
