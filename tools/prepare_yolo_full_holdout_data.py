"""Prepare a full-data YOLO train/val split with optional tri-modal enhancement.

This script is for platform-oriented runs where fold1 validation was shown to be
misleading. It uses all 2000 labeled training images, keeps a small deterministic
validation holdout, and writes an Ultralytics-compatible dataset. Images are
generated through the existing SAR/tri-modal enhancement pipeline so soft and
gated variants stay comparable.
"""

import argparse
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from prepare_yolo_sar_enhanced_data import CLASS_NAMES, build_map, enhance_image, link_or_copy, write_data_yaml


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_stems(label_dir):
    stems = []
    for path in sorted(Path(label_dir).glob("*.txt")):
        stems.append(path.stem)
    return stems


def label_classes(label_path):
    classes = set()
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                classes.add(int(parts[0]))
    return classes


def stratified_holdout(stems, label_dir, val_count):
    classes_by_stem = {stem: label_classes(Path(label_dir) / f"{stem}.txt") for stem in stems}
    class_counts = Counter(cls for classes in classes_by_stem.values() for cls in classes)
    target = {cls: max(1, round(count * val_count / max(1, len(stems)))) for cls, count in class_counts.items()}
    selected = set()
    val_class_counts = Counter()

    # Greedy rare-first multilabel holdout. It is deterministic and avoids making
    # the tiny validation set accidentally miss rare classes such as tricycle.
    ordered = sorted(
        stems,
        key=lambda stem: (
            min((class_counts[c] for c in classes_by_stem[stem]), default=10**9),
            stem,
        ),
    )
    for stem in ordered:
        if len(selected) >= val_count:
            break
        classes = classes_by_stem[stem]
        if any(val_class_counts[c] < target[c] for c in classes):
            selected.add(stem)
            val_class_counts.update(classes)

    if len(selected) < val_count:
        for stem in stems:
            if stem not in selected:
                selected.add(stem)
                if len(selected) >= val_count:
                    break

    val = sorted(selected)
    train = [stem for stem in stems if stem not in selected]
    return train, val, class_counts, val_class_counts


def ensure_clean_dir(path, overwrite=False):
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def process_split(split, stems, maps, root, out, args):
    img_dir = out / split / "images"
    label_dir = out / split / "labels"
    ensure_clean_dir(img_dir)
    ensure_clean_dir(label_dir)
    n_img, n_label = 0, 0
    for stem in stems:
        if stem not in maps["rgb"] or stem not in maps["ir"] or stem not in maps["depth"]:
            print(f"skip missing modality: {stem}")
            continue
        image = enhance_image(
            maps["rgb"][stem],
            maps["ir"][stem],
            maps["depth"][stem],
            ir_weight=args.ir_weight,
            depth_weight=args.depth_weight,
            sharpen=args.sharpen,
            fusion_mode=args.fusion_mode,
        )
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        n_img += 1
        label_src = Path(root) / "labels" / f"{stem}.txt"
        if label_src.exists():
            link_or_copy(str(label_src), str(label_dir / f"{stem}.txt"), copy=args.copy_labels)
            n_label += 1
    return n_img, n_label


def process_test(maps, out, args):
    visible_out = out / "visible"
    ensure_clean_dir(visible_out)
    stems = sorted(set(maps["rgb"]) & set(maps["ir"]) & set(maps["depth"]))
    for stem in stems:
        image = enhance_image(
            maps["rgb"][stem],
            maps["ir"][stem],
            maps["depth"][stem],
            ir_weight=args.ir_weight,
            depth_weight=args.depth_weight,
            sharpen=args.sharpen,
            fusion_mode=args.fusion_mode,
        )
        cv2.imwrite(str(visible_out / f"{stem}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
    return len(stems)


def main():
    parser = argparse.ArgumentParser(description="Prepare full-data YOLO holdout dataset")
    parser.add_argument("--root", default="data/train")
    parser.add_argument("--test-root", default="data/test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--test-out", default=None)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--ir-weight", type=float, default=0.18)
    parser.add_argument("--depth-weight", type=float, default=0.10)
    parser.add_argument("--sharpen", type=float, default=0.24)
    parser.add_argument("--fusion-mode", choices=("soft", "gated"), default="gated")
    parser.add_argument("--quality", type=int, default=96)
    parser.add_argument("--copy-labels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    ensure_clean_dir(out, overwrite=args.overwrite)
    stems = list_stems(root / "labels")
    train_stems, val_stems, class_counts, val_class_counts = stratified_holdout(stems, root / "labels", args.val_count)

    maps = {
        "rgb": build_map(root / "visible"),
        "ir": build_map(root / "infrared"),
        "depth": build_map(root / "depth"),
    }
    print(f"full holdout total={len(stems)} train={len(train_stems)} val={len(val_stems)} out={out}")
    print("val class coverage:", dict(sorted(val_class_counts.items())))
    print("all class coverage:", dict(sorted(class_counts.items())))
    tr = process_split("train", train_stems, maps, root, out, args)
    va = process_split("val", val_stems, maps, root, out, args)
    write_data_yaml(str(out))
    print(f"done train_images={tr[0]} train_labels={tr[1]} val_images={va[0]} val_labels={va[1]}")

    if args.test_out:
        test_root = Path(args.test_root)
        test_out = Path(args.test_out)
        ensure_clean_dir(test_out, overwrite=args.overwrite)
        test_maps = {
            "rgb": build_map(test_root / "visible"),
            "ir": build_map(test_root / "infrared"),
            "depth": build_map(test_root / "depth"),
        }
        n_test = process_test(test_maps, test_out, args)
        print(f"done test_images={n_test} out={test_out / 'visible'}")


if __name__ == "__main__":
    main()
