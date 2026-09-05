#!/usr/bin/env python
"""Per-class confidence-threshold scan on the fold1 val split (GPU-once, CPU-sweep).

Motivation: mAP@50-95 averages each of the 12 classes EQUALLY. Rare/under-recalled
classes benefit from a LOW conf; well-recalled classes can take a slightly HIGHER
threshold. Because a class's AP depends only on its OWN ranked predictions matched
to its own GT, each class can be thresholded independently to maximize the global
mean.

KEY OPTIMIZATION (exact, not approximate):
  Greedy matching runs by score DESC. Box k only takes a GT that boxes 1..k-1
  rejected, so dropping low-score TAIL boxes never changes the TP/FP of the kept
  high-score PREFIX. Hence per (class, IoU) we greedy-match ONCE over the full
  sorted list -> cumulative TP/FP arrays; AP at any conf cutoff = evaluate the
  prefix above that cutoff (O(1) per grid point). Sweep over ~30 cutoffs becomes
  near-instant even for person (thousands of preds).

Steps:
  1. [GPU, once]  predict fold1 val at conf=0.001, cache per-image dets to .npz.
  2. [CPU, fast]  precompute per-class cumulative TP/FP per IoU; sweep cutoffs.
  3. Optionally write per-class best conf to JSON for inference (--perclass-conf).

AP math matches the competition: 10 IoU 0.50..0.95, greedy TP/FP, 101-pt interp,
mean over IoUs = class AP; final = mean over the 12 classes.

Usage:
  python tools/scan_conf_per_class.py \\
    --weights <best.pt> --val-images <dir> --val-labels <dir> \\
    --cache /root/.../.scan.npz --imgsz 768 --tta --out-json .scan.json
"""
import argparse
import os
import glob
import numpy as np

CLASS_NAMES = ["person","boat","animal","seat","sign","bicycle",
               "car","ball","light","garbage can","uav","tricycle"]
NC = 12
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
RECALL_GRID = np.arange(0, 1.01, 0.01)  # 101 points


def parse_label(txt):
    boxes = []
    with open(txt) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 5:
                continue
            boxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return boxes


def norm_to_xyxy(b, imw, imh):
    cls, cx, cy, w, h = b
    return cls, [ (cx-w/2)*imw, (cy-h/2)*imh, (cx+w/2)*imw, (cy+h/2)*imh ]


def _xyxy_to_arr(boxes):
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def _iou_matrix(preds, gts):
    """Vectorized IoU, preds:(P,4) gts:(G,4) -> (P,G)."""
    if len(preds) == 0 or len(gts) == 0:
        return np.zeros((len(preds), len(gts)))
    p = preds
    g = gts
    ix1 = np.maximum(p[:, None, 0], g[None, :, 0])
    iy1 = np.maximum(p[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(p[:, None, 2], g[None, :, 2])
    iy2 = np.minimum(p[:, None, 3], g[None, :, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    pa = np.maximum(0.0, (p[:, 2]-p[:, 0]) * (p[:, 3]-p[:, 1]))[:, None]
    ga = np.maximum(0.0, (g[:, 2]-g[:, 0]) * (g[:, 3]-g[:, 1]))[None, :]
    union = pa + ga - inter
    return np.divide(inter, np.maximum(union, 1e-12))


def _greedy_tp_fp(ious, n_gt):
    """Greedy match rows (sorted by score desc already). Returns tp/fp cumsum arrays.
    ious: (P, G) IoU of the box row against each gt; caller already sorted rows desc."""
    p = ious.shape[0]
    matched = np.zeros(n_gt, dtype=bool)
    tp = np.zeros(p, dtype=np.float64)
    fp = np.zeros(p, dtype=np.float64)
    if p == 0:
        return tp, fp
    # Process sequentially (greedy is inherently order-dependent but vectorized per row).
    for i in range(p):
        row = ious[i]
        cand = np.where(~matched & (row >= 0.0))[0]  # candidate gts not yet matched
        if cand.size == 0:
            fp[i] = 1.0
            continue
        j = int(cand[np.argmax(row[cand])])
        # accept only if IoU >= the (class-level handled below) — here caller filters
        matched[j] = True
        tp[i] = 1.0
    return tp, fp


def _ap_from_cum(tp_cum, fp_cum, n_gt):
    """101-point interpolated AP from cumulative tp/fp arrays over a (kept) prefix."""
    if n_gt == 0 or len(tp_cum) == 0:
        return 0.0
    recall = tp_cum / float(n_gt)
    denom = np.maximum(tp_cum + fp_cum, 1e-9)
    precision = tp_cum / denom
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    ap = 0.0
    for r in RECALL_GRID:
        idx = np.where(mrec >= r)[0]
        ap += mpre[idx[0]] if idx.size else 0.0
    return ap / 101.0


def run_predict(weights, val_images, val_labels, imgsz, tta, batch, device, cache):
    from ultralytics import YOLO
    print("=" * 70)
    print("  [GPU pass] collecting detections at conf=0.001 ...")
    model = YOLO(weights)
    results = model.predict(source=val_images, imgsz=imgsz, conf=0.001, iou=0.6,
                            max_det=300, augment=tta, batch=batch, device=device,
                            verbose=False)
    gt_by_stem = {}
    for lp in sorted(glob.glob(os.path.join(val_labels, "*.txt"))):
        stem = os.path.splitext(os.path.basename(lp))[0]
        gt_by_stem[stem] = parse_label(lp)
    per_class = {c: {"gts": [], "preds": []} for c in range(NC)}
    for r in results:
        # each image may have its OWN native size; use it for that image's GT.
        imh, imw = int(r.orig_shape[0]), int(r.orig_shape[1])
        stem = os.path.splitext(os.path.basename(r.path))[0]
        for (cls, cx, cy, w, h) in gt_by_stem.get(stem, []):
            _, box = norm_to_xyxy((cls, cx, cy, w, h), imw, imh)
            per_class[cls]["gts"].append(box)
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            sc = r.boxes.conf.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), s, c in zip(xyxy, sc, cl):
                per_class[int(c)]["preds"].append((float(s), [float(x1), float(y1), float(x2), float(y2)]))
    if cache:
        d = os.path.dirname(cache)
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez_compressed(cache, preds=per_class)
        print(f"  cached -> {cache}")
    return per_class


def precompute(per_class):
    """For each class & IoU, sort preds desc, greedy match once -> (scores, tp_cum, fp_cum)."""
    tables = {}
    for c in range(NC):
        gts = _xyxy_to_arr(per_class[c]["gts"])
        preds = per_class[c]["preds"]
        preds_sorted = sorted(preds, key=lambda x: -x[0])
        scores = np.array([p[0] for p in preds_sorted], dtype=np.float64)
        pb = _xyxy_to_arr([p[1] for p in preds_sorted])
        g = len(gts)
        entry = {"n_gt": g, "scores": scores}
        for t in IOU_THRESHOLDS:
            if len(pb) == 0 or g == 0:
                entry[float(t)] = (np.array([], dtype=np.float64),
                                   np.array([], dtype=np.float64))
                continue
            ious = _iou_matrix(pb, gts)
            # mask rows below threshold -> treat as unmatched (fp)
            valid = ious >= float(t)
            matched = np.zeros(g, dtype=bool)
            p = len(pb)
            tp = np.zeros(p, dtype=np.float64)
            fp = np.zeros(p, dtype=np.float64)
            for i in range(p):
                cand = np.where(valid[i] & ~matched)[0]
                if cand.size == 0:
                    fp[i] = 1.0
                else:
                    j = int(cand[np.argmax(ious[i, cand])])
                    matched[j] = True
                    tp[i] = 1.0
            entry[float(t)] = (np.cumsum(tp), np.cumsum(fp))
        tables[c] = entry
    return tables


def class_ap_at_conf(tables, c, cutoff):
    """mAP50-95 for ONE class c with the given conf cutoff (prefix exact)."""
    e = tables[c]
    if e["n_gt"] == 0:
        return 0.0
    scores = e["scores"]
    k = int(np.searchsorted(-scores, -cutoff, side="right"))  # # preds with score>=cutoff
    if k == 0:
        return 0.0
    aps = []
    for t in IOU_THRESHOLDS:
        tc, fc = e[float(t)]
        if len(tc) < k:
            aps.append(0.0)
            continue
        aps.append(_ap_from_cum(tc[:k], fc[:k], e["n_gt"]))
    return float(np.mean(aps))


def sweep(tables):
    print("=" * 70)
    print("  [CPU sweep] per-class confidence thresholds (prefix method)")
    grid = np.concatenate([np.arange(0.001, 0.01, 0.002),
                           np.arange(0.01, 0.11, 0.01),
                           np.arange(0.11, 0.31, 0.02),
                           np.arange(0.31, 0.51, 0.05)])
    base_ap = {c: class_ap_at_conf(tables, c, 0.001) for c in range(NC)}
    base_map = float(np.mean(list(base_ap.values())))
    print(f"\nbaseline (all classes conf=0.001): mAP50-95 = {base_map:.4f}")
    print(f"{'class':<12}{'baseAP':>8}{'bestConf':>9}{'bestAP':>8}{'delta':>7}")
    best_conf = {}
    for c in range(NC):
        best_ap, best_t = base_ap[c], 0.001
        for t in grid:
            a = class_ap_at_conf(tables, c, float(t))
            if a > best_ap:
                best_ap, best_t = a, float(t)
        best_conf[c] = best_t
        print(f"{CLASS_NAMES[c]:<12}{base_ap[c]:>8.4f}{best_t:>9.4f}{best_ap:>8.4f}{best_ap-base_ap[c]:>+7.4f}")
    comb_map = float(np.mean([class_ap_at_conf(tables, c, best_conf[c]) for c in range(NC)]))
    print("=" * 70)
    print(f"combined per-class thresholds: mAP50-95 = {comb_map:.4f}  (delta {comb_map-base_map:+.4f})")
    return best_conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--val-images")
    ap.add_argument("--val-labels")
    ap.add_argument("--cache")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sweep-only", action="store_true")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.sweep_only and args.cache and os.path.exists(args.cache):
        per_class = np.load(args.cache, allow_pickle=True)["preds"].item()
        print(f"loaded cache {args.cache}")
    else:
        per_class = run_predict(args.weights, args.val_images, args.val_labels,
                                args.imgsz, args.tta, args.batch, args.device, args.cache)
    print("precomputing greedy-match tables ...")
    tables = precompute(per_class)
    best_conf = sweep(tables)
    if args.out_json:
        import json
        with open(args.out_json, "w") as fh:
            json.dump({str(c): best_conf[c] for c in range(NC)}, fh, indent=2)
        print(f"\nwrote per-class thresholds -> {args.out_json}")


if __name__ == "__main__":
    main()
