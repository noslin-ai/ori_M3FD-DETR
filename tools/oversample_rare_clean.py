#!/usr/bin/env python
"""Clean image-level oversampling for genuinely-rare classes (boat/ball/tricycle).

Why NOT uav/garbage/person: v0.13 over-sampled uav (2x) and tricycle (10x) on the
same images that also carry many person/car boxes, which shifted class priors and
*reduced* person/uav/sign recall on the official test set (platform 48.876->47.928).

Design rules learned from that failure:
  - Only duplicate images that contain a *genuinely* rare class with very few boxes:
      boat(1) 107 boxes / 48 images,  ball(7) 72 boxes / 59 images,
      tricycle(11) 25 boxes / 22 images   (fold1 train, 1600 imgs).
  - Keep multipliers mild (2x/2x/4x) so rare classes get more examples WITHOUT
    changing the person/car prior much.
  - Never touch data/test and never write pseudo-labels (rule + tested: hurts score).
  - Work directly on the enhanced (SAR/soft-fusion) images the model actually trains
    on, so train/val input distribution stays identical to the baseline soft run.

Space-safety: duplications are HARD LINKS on the same filesystem (zero extra disk),
labels are hard links too. The source is untouched. Output is a normal Ultralytics
dataset dir + data.yaml.

Usage:
    python tools/oversample_rare_clean.py \
      --src data/yolo_trimodal_soft_m \
      --dst data/yolo_trimodal_soft_m_rareos \
      --mult 1:2,7:2,11:4
"""
import argparse
import os
import shutil
from collections import Counter

NAMES = {0:"person",1:"boat",2:"animal",3:"seat",4:"sign",5:"bicycle",
         6:"car",7:"ball",8:"light",9:"garbage can",10:"uav",11:"tricycle"}

def read_cls(label_path):
    with open(label_path) as fh:
        return [int(l.split()[0]) for l in fh if l.strip()]

def hardlink_or_copy(src, dst):
    """Same-fs hard link (0 extra space); fallback to copy if cross-device."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def parse_mult(s):
    out = {}
    for tok in s.split(","):
        k, v = tok.split(":")
        out[int(k)] = int(v)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/yolo_trimodal_soft_m")
    ap.add_argument("--dst", default="data/yolo_trimodal_soft_m_rareos")
    ap.add_argument("--mult", default="1:2,7:2,11:4",
                    help="class:multiplier, e.g. 1:2,7:2,11:4")
    ap.add_argument("--img-ext", default=".jpg")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    mult = parse_mult(args.mult)
    rare = set(mult)
    names_extra = {**NAMES, **{i: f"class{i}" for i in range(12)}}

    # ---- gather train/val file lists from src ----
    for split in ("train", "val"):
        si = os.path.join(src, split, "images")
        sl = os.path.join(src, split, "labels")
        di = os.path.join(dst, split, "images")
        dl = os.path.join(dst, split, "labels")
        os.makedirs(di, exist_ok=True)
        os.makedirs(dl, exist_ok=True)

    # val untouched (hard links keep zero extra disk)
    for f in sorted(os.listdir(os.path.join(src, "val", "labels"))):
        stem = os.path.splitext(f)[0]
        hardlink_or_copy(os.path.join(src, "val", "images", stem + args.img_ext),
                         os.path.join(dst, "val", "images", stem + args.img_ext))
        hardlink_or_copy(os.path.join(src, "val", "labels", f),
                         os.path.join(dst, "val", "labels", f))

    # ---- plan duplication on train ----
    tl = os.path.join(src, "train", "labels")
    ti = os.path.join(src, "train", "images")
    files = sorted(os.listdir(tl))
    before = Counter()
    dup_plan = {}   # stem -> how many EXTRA copies
    rare_imgs = Counter()
    for f in files:
        stem = os.path.splitext(f)[0]
        cls = read_cls(os.path.join(tl, f))
        before.update(cls)
        m = 1
        for c in set(cls):
            if c in mult:
                m = max(m, mult[c])
                rare_imgs[c] += 1
        if m > 1:
            dup_plan[stem] = m - 1  # extra copies

    # ---- write originals + duplicates ----
    after = Counter()
    total_dup = 0
    for f in files:
        stem = os.path.splitext(f)[0]
        cls = read_cls(os.path.join(tl, f))
        n_extra = dup_plan.get(stem, 0)
        # original
        hardlink_or_copy(os.path.join(ti, stem + args.img_ext),
                         os.path.join(dst, "train", "images", stem + args.img_ext))
        hardlink_or_copy(os.path.join(tl, f),
                         os.path.join(dst, "train", "labels", f))
        after.update(cls)
        # duplicates
        for k in range(1, n_extra + 1):
            ns = f"{stem}_os{k}"
            hardlink_or_copy(os.path.join(ti, stem + args.img_ext),
                             os.path.join(dst, "train", "images", ns + args.img_ext))
            hardlink_or_copy(os.path.join(tl, f),
                             os.path.join(dst, "train", "labels", ns + ".txt"))
            after.update(cls)
            total_dup += 1

    print("=== clean rare-class oversampling ===")
    print(f"source train imgs: {len(files)}  ->  dst train imgs: {len(files) + total_dup}  (+{total_dup} dups)")
    print(f"rare multiplier map: {mult}")
    print(f"{'class':<12}{'before':>8}{'after':>8}{'mult':>8}   (dup imgs)")
    for c in sorted(before):
        b = before[c]
        a = after[c]
        print(f"{names_extra[c]:<12}{b:>8}{a:>8}{(a/b if b else 0):>8.2f}   ({rare_imgs.get(c,0)} img)")
    # confirm rare dup images don't carry disproportionate person/car
    print("\n[guard] person/car multiplier must stay ~1.0 (prior unchanged):")
    for c in (0, 6):
        print(f"  {names_extra[c]}: {before[c]} -> {after[c]}  ({after[c]/before[c]:.2f}x)")

    # ---- data.yaml (keep same names/nc) ----
    yaml_lines = [f"path: {os.path.abspath(dst)}", "train: train/images", "val: val/images",
                  f"nc: {len(NAMES)}", "names:"]
    for i in range(len(NAMES)):
        yaml_lines.append(f"  {i}: {NAMES[i]}")
    with open(os.path.join(dst, "data.yaml"), "w") as fh:
        fh.write("\n".join(yaml_lines) + "\n")
    print(f"\nwrote data.yaml -> {os.path.abspath(dst)}/data.yaml")

if __name__ == "__main__":
    main()
