"""Official competition metric — implemented exactly as the problem statement specifies.

Rules (problem statement, sec. 6):
  * per class, pool predictions from ALL images, sort by confidence descending
  * greedily walk that order; a prediction is TP iff it matches a still-unmatched GT of the
    same class with IoU >= t, else FP; unmatched GT are FN
  * AP = 101-point interpolated precision-recall area (precision envelope)
  * mAP@t = mean over classes; final = mean over t in {0.50,0.55,...,0.95}
  * score = mAP@50-95 * 100

Predictions: dict stem -> list of (cls, cx, cy, w, h, conf)   (normalised xywh)
GT:           dict stem -> list of (cls, cx, cy, w, h)
"""
import numpy as np

IOU_TS = [round(0.50 + 0.05 * i, 2) for i in range(10)]
REC_THRS = np.linspace(0.0, 1.0, 101)


def _xyxy(b):
    cx, cy, w, h = b
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _iou_matrix(P, G):
    """P: (N,4) xyxy, G: (M,4) xyxy -> (N,M) IoU."""
    if len(P) == 0 or len(G) == 0:
        return np.zeros((len(P), len(G)), dtype=np.float64)
    x1 = np.maximum(P[:, None, 0], G[None, :, 0])
    y1 = np.maximum(P[:, None, 1], G[None, :, 1])
    x2 = np.minimum(P[:, None, 2], G[None, :, 2])
    y2 = np.minimum(P[:, None, 3], G[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ap = (P[:, 2] - P[:, 0]) * (P[:, 3] - P[:, 1])
    ag = (G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1])
    return inter / (ap[:, None] + ag[None, :] - inter + 1e-12)


def _ap_101(rec, prec):
    """101-point interpolated AP with the precision envelope (COCO style)."""
    if len(rec) == 0:
        return 0.0
    # precision envelope: running max from the right
    env = np.maximum.accumulate(prec[::-1])[::-1]
    out = np.zeros(len(REC_THRS))
    idx = np.searchsorted(rec, REC_THRS, side="left")
    for i, k in enumerate(idx):
        out[i] = env[k] if k < len(env) else 0.0
    return float(out.mean())


def evaluate(preds, gts, num_classes=12, verbose=False, names=None):
    """preds: {stem: [(cls, cx, cy, w, h, conf)]}, gts: {stem: [(cls, cx, cy, w, h)]}
    returns (score_x100, {cls: ap}, {t: mAP_t})"""
    dets = {}
    gtb = {}
    # Collect detections and GT independently.  Iterating GT only through
    # preds.items() silently drops every GT on an image with zero detections.
    for stem, arr in preds.items():
        for d in arr:
            dets.setdefault(int(d[0]), []).append((float(d[5]), stem, d[1:5]))
    for stem, arr in gts.items():
        for g in arr:
            gtb.setdefault(int(g[0]), []).append((stem, g[1:5]))
    per_cls_t = {c: [] for c in range(num_classes)}
    per_t = {t: [] for t in IOU_TS}
    for c in range(num_classes):
        plist = dets.get(c, [])
        glist = gtb.get(c, [])
        n_gt = len(glist)
        if n_gt == 0:
            continue                      # classes absent from GT are not scored
        per_image_gt = {}
        for stem, box in glist:
            per_image_gt.setdefault(stem, []).append(box)
        plist.sort(key=lambda x: -x[0])
        for t in IOU_TS:
            matched = {s: np.zeros(len(v), bool) for s, v in per_image_gt.items()}
            tp = np.zeros(len(plist), np.float64)
            for i, (conf, stem, box) in enumerate(plist):
                G = per_image_gt.get(stem)
                if G is None:
                    continue
                ious = _iou_matrix(np.array([_xyxy(box)]), np.array([_xyxy(g) for g in G]))[0]
                order = np.argsort(-ious)
                for j in order:
                    if ious[j] < t:
                        break
                    if not matched[stem][j]:
                        matched[stem][j] = True
                        tp[i] = 1.0
                        break
            ctp = np.cumsum(tp)
            cfp = np.cumsum(1.0 - tp)
            rec = ctp / n_gt
            prec = ctp / np.maximum(ctp + cfp, 1e-12)
            ap = _ap_101(rec, prec)
            per_t[t].append(ap)
            per_cls_t[c].append(ap)
    per_cls = {c: float(np.mean(v)) for c, v in per_cls_t.items() if v}   # AP50-95 per class
    mAPs = [float(np.mean(v)) for v in per_t.values() if v]
    score = float(np.mean(mAPs)) * 100 if mAPs else 0.0
    if verbose and names:
        for c in range(num_classes):
            if c in per_cls:
                print(f"    {names[c]:<12} AP50-95={per_cls[c]:.4f}")
    return score, per_cls, {t: float(np.mean(v)) if v else 0.0 for t, v in per_t.items()}


def load_yolo_labels(path, num_classes=12):
    rows = []
    import os
    if os.path.exists(path):
        for ln in open(path):
            q = ln.split()
            if len(q) >= 5:
                rows.append((int(float(q[0])), *[float(v) for v in q[1:5]]))
    return rows
