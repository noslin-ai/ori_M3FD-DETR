"""Single-model multi-scale TTA merge (compliant).

Merges two inference passes of the SAME model (same weights) at different input
sizes (e.g. 1280 and 1024) into one submission. This is test-time augmentation
applied to one model - NOT multi-model ensembling - so it stays within the rule
that forbids fusing multiple different-structure/training-stage MODELS.

Per class: boxes from the higher-confidence scale are kept; a box from the
other scale that overlaps (IoU >= thr) boosts confidence slightly (consensus
between two views of the same model, like flip-TTA views). No cross-model logic.
"""
import argparse
import glob
import math
import os
import shutil


def read_boxes(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        p = line.strip().split()
        if len(p) < 6:
            continue
        cls = int(float(p[0]))
        cx, cy, w, h, conf = map(float, p[1:6])
        if conf <= 0 or w <= 0 or h <= 0:
            continue
        out.append([cls, cx, cy, w, h, conf])
    return out


def iou(a, b):
    ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2
    ax2, ay2 = a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2
    bx2, by2 = b[1] + b[3] / 2, b[2] + b[4] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def merge_image(primary, secondary, iou_thr=0.5, boost=0.02):
    """primary: list of boxes (main scale). secondary: other scale.
    Keep all primary boxes; if a secondary box overlaps a primary box of same
    class, raise primary conf by boost (consensus of same model, two views).
    Add secondary box only if it overlaps nothing (recall gain) - capped conf.
    """
    by_cls = {}
    for b in primary:
        by_cls.setdefault(b[0], []).append(b)
    merged = []
    used_sec = [False] * len(secondary)
    for b in primary:
        merged.append(list(b))  # copy
    # boost primary boxes with secondary agreement
    for si, s in enumerate(secondary):
        for p in merged:
            if p[0] == s[0] and iou(p, s) >= iou_thr:
                p[5] = min(1.0, p[5] + boost)
                used_sec[si] = True
                break
    # add unmatched secondary (same-model second view catches boxes main missed)
    for si, s in enumerate(secondary):
        if used_sec[si]:
            continue
        if s[5] >= 0.65:  # only high-conf adds to avoid FP
            merged.append(list(s))
    merged.sort(key=lambda x: -x[5])
    return merged[:100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, help="main scale dir (usually 1280)")
    ap.add_argument("--secondary", required=True, help="second scale dir (usually 1024)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--boost", type=float, default=0.02)
    ap.add_argument("--zip", default=None)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    total = 0
    for f in sorted(glob.glob(os.path.join(args.primary, "*.txt"))):
        stem = os.path.basename(f)
        sec_f = os.path.join(args.secondary, stem)
        prim = read_boxes(f)
        sec = read_boxes(sec_f) if os.path.exists(sec_f) else []
        merged = merge_image(prim, sec, args.iou, args.boost)
        with open(os.path.join(args.output, stem), "w", encoding="utf-8") as w:
            for b in merged:
                w.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f} {b[5]:.6f}\n")
        total += len(merged)
    print(f"Merged {len(glob.glob(os.path.join(args.primary,'*.txt')))} images, {total} boxes")
    if args.zip:
        shutil.make_archive(args.zip[:-4] if args.zip.endswith('.zip') else args.zip, 'zip', root_dir=args.output)
        print(f"Packed -> {args.zip}")


if __name__ == "__main__":
    main()
