"""Fuse multiple YOLO submission folders/zips with weighted box fusion.

The script is inference-only: it reads existing prediction txt files and writes
a new official-format submission. It is meant for conservative multi-modal late
fusion, where a strong primary model keeps a high weight and weaker auxiliary
modal predictions can only adjust or add boxes when confidence is sufficient.
"""

import argparse
import math
import os
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path


def unpack_if_zip(path, tmp_dir):
    path = Path(path)
    if path.is_dir():
        return path
    if path.suffix.lower() != ".zip":
        raise ValueError(f"input must be a directory or zip: {path}")
    out = Path(tmp_dir) / path.stem
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(out)
    return out


def read_boxes(path):
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        cls, cx, cy, w, h, conf = parts[:6]
        cls = int(float(cls))
        cx, cy, w, h, conf = map(float, (cx, cy, w, h, conf))
        if conf <= 0 or w <= 0 or h <= 0:
            continue
        x1 = max(0.0, cx - w / 2.0)
        y1 = max(0.0, cy - h / 2.0)
        x2 = min(1.0, cx + w / 2.0)
        y2 = min(1.0, cy + h / 2.0)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([cls, x1, y1, x2, y2, conf])
    return boxes


def iou(a, b):
    ix1 = max(a[1], b[1])
    iy1 = max(a[2], b[2])
    ix2 = min(a[3], b[3])
    iy2 = min(a[4], b[4])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[3] - a[1]) * (a[4] - a[2])
    area_b = (b[3] - b[1]) * (b[4] - b[2])
    return inter / max(area_a + area_b - inter, 1e-12)


def weighted_fuse(cluster):
    score_sum = sum(b[5] * b[6] for b in cluster)
    if score_sum <= 0:
        score_sum = 1e-12
    cls = cluster[0][0]
    x1 = sum(b[1] * b[5] * b[6] for b in cluster) / score_sum
    y1 = sum(b[2] * b[5] * b[6] for b in cluster) / score_sum
    x2 = sum(b[3] * b[5] * b[6] for b in cluster) / score_sum
    y2 = sum(b[4] * b[5] * b[6] for b in cluster) / score_sum
    weighted_conf = sum(b[5] * b[6] for b in cluster) / sum(b[6] for b in cluster)
    bonus = min(1.0, 0.04 * (len({b[7] for b in cluster}) - 1))
    conf = min(1.0, weighted_conf + bonus)
    return [cls, x1, y1, x2, y2, conf]


def fuse_class(boxes, iou_thr):
    boxes = sorted(boxes, key=lambda b: b[5] * b[6], reverse=True)
    clusters = []
    for box in boxes:
        matched = None
        best_iou = 0.0
        for cluster in clusters:
            rep = weighted_fuse(cluster)
            ov = iou(box, rep)
            if ov > best_iou:
                best_iou = ov
                matched = cluster
        if matched is not None and best_iou >= iou_thr:
            matched.append(box)
        else:
            clusters.append([box])
    return [weighted_fuse(cluster) for cluster in clusters]


def write_boxes(path, boxes):
    lines = []
    for cls, x1, y1, x2, y2, conf in boxes:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        values = (cx, cy, w, h, conf)
        if not all(math.isfinite(v) for v in values):
            continue
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
    path.write_text("".join(lines), encoding="utf-8")


def make_zip(output_dir, zip_path):
    base_name = zip_path[:-4] if zip_path.endswith(".zip") else zip_path
    archive_path = shutil.make_archive(base_name, "zip", root_dir=output_dir)
    print(f"Packed submission zip -> {archive_path}")


def main():
    parser = argparse.ArgumentParser(description="Fuse YOLO submission dirs/zips")
    parser.add_argument("--inputs", nargs="+", required=True, help="submission dirs/zips")
    parser.add_argument("--weights", nargs="+", type=float, required=True, help="per-input fusion weights")
    parser.add_argument("--output", required=True, help="output submission directory")
    parser.add_argument("--zip", default=None, help="optional zip output")
    parser.add_argument("--iou", type=float, default=0.55, help="class-wise fusion IoU")
    parser.add_argument("--min-conf", type=float, default=0.001, help="drop fused boxes below this confidence")
    parser.add_argument("--aux-min-conf", type=float, default=0.015, help="drop boxes from non-primary inputs below this confidence")
    parser.add_argument("--max-det", type=int, default=100, help="max boxes per image")
    args = parser.parse_args()

    if len(args.inputs) != len(args.weights):
        raise ValueError("--inputs and --weights must have the same length")

    with tempfile.TemporaryDirectory() as tmp:
        roots = [unpack_if_zip(path, tmp) for path in args.inputs]
        stems = sorted({p.stem for root in roots for p in root.glob("*.txt")})
        out = Path(args.output)
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        total_boxes = 0
        for stem in stems:
            by_cls = defaultdict(list)
            for src_idx, (root, weight) in enumerate(zip(roots, args.weights)):
                for box in read_boxes(root / f"{stem}.txt"):
                    if src_idx > 0 and box[5] < args.aux_min_conf:
                        continue
                    by_cls[box[0]].append(box + [weight, src_idx])

            fused = []
            for boxes in by_cls.values():
                fused.extend(fuse_class(boxes, args.iou))
            fused = [b for b in fused if b[5] >= args.min_conf]
            fused.sort(key=lambda b: b[5], reverse=True)
            fused = fused[: args.max_det]
            total_boxes += len(fused)
            write_boxes(out / f"{stem}.txt", fused)

        print(f"Generated {len(stems)} files -> {out}")
        print(f"Total boxes: {total_boxes}")
        if args.zip:
            make_zip(str(out), args.zip)


if __name__ == "__main__":
    main()
