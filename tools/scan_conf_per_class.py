#!/usr/bin/env python
"""Per-class confidence-threshold scan on the fold1 val split (GPU-once, CPU-sweep).

Motivation (report line 2339 + our analysis):
  mAP@50-95 averages each of the 12 classes EQUALLY. Rare/under-recalled classes
  (boat/ball/tricycle) and tiny-object classes (sign/uav) benefit from a LOW conf
  threshold (keep recall); well-recalled classes (person/car/animal/light) can take
  a slightly HIGHER threshold to cut false positives that cost precision. Because a
  class's AP depends only on its OWN ranked predictions matched to its own GT, each
  class can be thresholded independently to maximize the global mean.

How it works:
  1. [GPU, once]  predict the fold1 val split (the split the best model was validated
                  on) at conf=0.001 and cache every (box, cls, score) to a .npz.
  2. [CPU, fast]  for each class, sweep its confidence cutoff on a fine grid and keep
                  the cutoff that maximizes that class's mAP50-95 contribution.
  3. Reports a global single-threshold baseline (conf=0.001) vs. the per-class map,
                  plus the predicted delta on the SAME val (in-sample estimate; the
                  real gain on the hidden test is smaller but same-direction).

The AP computation mirrors the competition metric exactly:
  - 10 IoU thresholds 0.50..0.95 step 0.05, greedy TP/FP matching per class per IoU,
    predictions ranked by confidence descending,
  - per-class AP via 101-point interpolation over recall, averaged over the 10 IoUs,
  - final mAP = mean over the 12 classes.

Usage:
    /root/miniconda3/envs/race/bin/python tools/scan_conf_per_class.py \
      --weights runs/detect/runs/native_m_trimodal/soft768_labelrefresh_from_sar_best/weights/best.pt \
      --val-images data/yolo_trimodal_soft_m/val/images \
      --val-labels data/yolo_trimodal_soft_m/val/labels \
      --cache /root/autodl-tmp/aic_race/M3F-DETR/.scan_cache.npz \
      --imgsz 768 --tta --batch 8

Use the per-class conf thresholds in the inference script at submission time
(after confirming they don't over-cut: cap max boxes/img at 100).
"""
import argparse
import os
import glob
import numpy as np

CLASS_NAMES = ["person","boat","animal","seat","sign","bicycle",
               "car","ball","light","garbage can","uav","tricycle"]
NC = 12
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


def parse_label(txt):
    """Return list of [cx,cy,w,h] in pixel coords for a label file (YOLO norm)."""
    boxes = []
    with open(txt) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 5:
                continue
            cls = int(p[0])
            cx, cy, w, h = map(float, p[1:5])
            boxes.append((cls, cx, cy, w, h))
    return boxes


def norm_to_xyxy(b, imw, imh):
    cls, cx, cy, w, h = b
    x1 = (cx - w / 2) * imw
    y1 = (cy - h / 2) * imh
    x2 = (cx + w / 2) * imw
    y2 = (cy + h / 2) * imh
    return cls, [x1, y1, x2, y2]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def ap_at_iou(gts, preds, iou_t):
    """One class, one IoU threshold -> AP via 101-point interpolation."""
    # preds already sorted by score desc; each = (score, xyxy)
    preds = sorted(preds, key=lambda x: -x[0])
    matched = [False] * len(gts)
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, (score, pb) in enumerate(preds):
        best_j, best_iou = -1, iou_t
        for j, gt in enumerate(gts):
            if matched[j]:
                continue
            ov = iou(pb, gt)
            if ov >= best_iou:
                best_iou, best_j = ov, j
        if best_j >= 0:
            matched[best_j] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0
    tpc = np.cumsum(tp)
    fpc = np.cumsum(fp)
    n_gt = max(1, len(gts))
    recall = tpc / n_gt
    precision = tpc / np.maximum(tpc + fpc, 1e-9)
    # 101-point interpolation
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    grid = np.arange(0, 1.01, 0.01)
    ap = 0.0
    for r in grid:
        idx = np.where(mrec >= r)[0]
        ap += mpre[idx[0]] if len(idx) else 0.0
    return ap / 101.0


def map_for_conf(per_class_data, conf_map, conf_global=0.001):
    """Given cached preds, compute mAP50-95 when each class uses its own cutoff.
    conf_map: dict cls -> cutoff (or the global default)."""
    ap_sum = 0.0
    ap_by_class = {}
    for c in range(NC):
        cutoff = conf_map.get(c, conf_global)
        # preds for class c across all images: (score, xyxy)
        preds = [p for p in per_class_data[c]["preds"] if p[0] >= cutoff]
        gts = per_class_data[c]["gts"]
        aps = [ap_at_iou(gts, preds, t) for t in IOU_THRESHOLDS]
        ap_c = float(np.mean(aps))
        ap_by_class[c] = ap_c
        ap_sum += ap_c
    return ap_sum / NC, ap_by_class


def run_predict(weights, val_images, val_labels, imgsz, tta, batch, device, cache):
    """Ultralytics predict once at conf=0.001, cache per-class preds+GT."""
    from ultralytics import YOLO
    print("=" * 70)
    print("  [GPU pass] collecting detections at conf=0.001 ...")
    print(f"  weights={weights}  imgsz={imgsz}  tta={tta}")
    model = YOLO(weights)
    results = model.predict(
        source=val_images,
        imgsz=imgsz,
        conf=0.001,
        iou=0.6,
        max_det=300,
        augment=tta,
        batch=batch,
        device=device,
        verbose=False,
    )
    # get image size (first result)
    imw, imh = results[0].orig_shape[1], results[0].orig_shape[0]

    # map stems: label file stem == image stem
    gt_by_stem = {}
    for lp in sorted(glob.glob(os.path.join(val_labels, "*.txt"))):
        stem = os.path.splitext(os.path.basename(lp))[0]
        gt_by_stem[stem] = parse_label(lp)

    per_class = {c: {"gts": [], "preds": []} for c in range(NC)}
    matched_img = 0
    for r in results:
        stem = os.path.splitext(os.path.basename(r.path))[0]
        # GT
        for (cls, cx, cy, w, h) in gt_by_stem.get(stem, []):
            _, box = norm_to_xyxy((cls, cx, cy, w, h), imw, imh)
            per_class[cls]["gts"].append(box)
        # preds
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), sc, cl in zip(xyxy, scores, clss):
                per_class[int(cl)]["preds"].append((float(sc), [float(x1), float(y1), float(x2), float(y2)]))
        matched_img += 1

    print(f"  processed {matched_img} images")
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, preds=per_class)
        print(f"  cached -> {cache}")
    return per_class


def ap_class_only(pcd, c, cutoff):
    """mAP50-95 for a single class c only (fast for sweeping one class)."""
    preds = [p for p in pcd[c]["preds"] if p[0] >= cutoff]
    gts = pcd[c]["gts"]
    if not gts:
        return 0.0
    aps = [ap_at_iou(gts, preds, t) for t in IOU_THRESHOLDS]
    return float(np.mean(aps))


def sweep(per_class):
    print("=" * 70)
    print("  [CPU sweep] per-class confidence thresholds")
    grid = np.concatenate([np.arange(0.001, 0.01, 0.002),
                           np.arange(0.01, 0.11, 0.01),
                           np.arange(0.11, 0.31, 0.02),
                           np.arange(0.31, 0.51, 0.05)])
    # baseline: global conf=0.001
    base_map, base_ap = map_for_conf(per_class, {})
    print(f"\nbaseline (all classes conf=0.001): mAP50-95 = {base_map:.4f}")
    print(f"{'class':<12}{'baseAP':>8}{'bestConf':>9}{'bestAP':>8}{'delta':>7}")
    best_conf = {}
    for c in range(NC):
        name = CLASS_NAMES[c]
        best_ap, best_t = base_ap[c], 0.001
        for t in grid:
            ap_c = ap_class_only(per_class, c, float(t))
            if ap_c > best_ap:
                best_ap, best_t = ap_c, float(t)
        best_conf[c] = best_t
        delta = best_ap - base_ap[c]
        print(f"{name:<12}{base_ap[c]:>8.4f}{best_t:>9.4f}{best_ap:>8.4f}{delta:>+7.4f}")

    # combined map with per-class best
    comb_map, comb_ap = map_for_conf(per_class, best_conf)
    print("=" * 70)
    print(f"combined per-class thresholds: mAP50-95 = {comb_map:.4f}  "
          f"(delta {comb_map - base_map:+.4f})")
    print("\n[per-class thresholds to use at submission]")
    for c in range(NC):
        print(f"  {CLASS_NAMES[c]:<12} -> conf = {best_conf[c]:.4f}")
    print("\nNOTE: in-sample (val) estimate; hidden-test gain is smaller but same sign.")
    return best_conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--val-labels", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweep-only", action="store_true",
                    help="load cache and only sweep (no GPU)")
    ap.add_argument("--out-json", default=None,
                    help="write per-class best conf to a JSON file for inference (--perclass-conf)")
    args = ap.parse_args()

    if args.sweep_only and args.cache and os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        per_class = z["preds"].item()
        print(f"loaded cache {args.cache}")
    else:
        per_class = run_predict(args.weights, args.val_images, args.val_labels,
                                args.imgsz, args.tta, args.batch, args.device, args.cache)
    best_conf = sweep(per_class)

    if args.out_json:
        import json
        with open(args.out_json, "w") as fh:
            json.dump({str(c): best_conf[c] for c in range(NC)}, fh, indent=2)
        print(f"\nwrote per-class thresholds -> {args.out_json}")


if __name__ == "__main__":
    main()
