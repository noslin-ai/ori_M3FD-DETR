"""Tile-aware Ultralytics YOLO inference for small-object submissions.

This script keeps the existing full-image inference path intact and adds optional
overlapping tile inference. It is inference-only: it never reads labels, never
writes training data, and never creates pseudo labels.
"""

import argparse
import math
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(visible_dir: Path, limit: int = 0) -> List[Path]:
    images = [p for p in visible_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    images.sort()
    return images[:limit] if limit > 0 else images


def make_zip(output_dir: Path, zip_path: Path) -> None:
    base_name = str(zip_path)[:-4] if str(zip_path).endswith(".zip") else str(zip_path)
    archive_path = shutil.make_archive(base_name, "zip", root_dir=output_dir)
    print(f"  Packed submission zip -> {archive_path}")


def tile_windows(width: int, height: int, tile: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    if tile <= 0:
        return []
    tile_w = min(tile, width)
    tile_h = min(tile, height)
    stride = max(1, int(tile * (1.0 - overlap)))

    xs = list(range(0, max(width - tile_w + 1, 1), stride))
    ys = list(range(0, max(height - tile_h + 1, 1), stride))
    if not xs or xs[-1] != width - tile_w:
        xs.append(width - tile_w)
    if not ys or ys[-1] != height - tile_h:
        ys.append(height - tile_h)

    windows = []
    seen = set()
    for y in ys:
        for x in xs:
            win = (int(x), int(y), int(x + tile_w), int(y + tile_h))
            if win not in seen:
                windows.append(win)
                seen.add(win)
    return windows


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def classwise_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float) -> np.ndarray:
    keep: List[int] = []
    for cls_id in sorted(set(classes.astype(int).tolist())):
        inds = np.where(classes == cls_id)[0]
        order = inds[np.argsort(-scores[inds])]
        while order.size > 0:
            current = order[0]
            keep.append(int(current))
            if order.size == 1:
                break
            ious = box_iou_one_to_many(boxes[current], boxes[order[1:]])
            order = order[1:][ious <= iou_thr]
    return np.array(keep, dtype=np.int64)


def append_result(
    preds: List[Tuple[float, float, float, float, float, int]],
    result,
    dx: int,
    dy: int,
    width: int,
    height: int,
) -> None:
    if result.boxes is None or len(result.boxes) == 0:
        return
    xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    confs = result.boxes.conf.cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int64)
    xyxy[:, [0, 2]] += dx
    xyxy[:, [1, 3]] += dy
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)
    for box, conf, cls_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = map(float, box)
        if conf <= 0.0 or cls_id < 0 or x2 <= x1 or y2 <= y1:
            continue
        preds.append((x1, y1, x2, y2, float(conf), int(cls_id)))


def predict_batch(model, sources: Sequence[np.ndarray], imgsz: int, conf: float, iou: float, max_det: int, augment: bool, device: str):
    if not sources:
        return []
    return model.predict(
        source=list(sources),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        augment=augment,
        device=device,
        verbose=False,
        stream=False,
    )


def write_submission(
    out_path: Path,
    preds: List[Tuple[float, float, float, float, float, int]],
    width: int,
    height: int,
    fuse_iou: float,
    max_det: int,
    perclass_conf: dict = None,
) -> int:
    if not preds:
        out_path.write_text("")
        return 0

    arr = np.array(preds, dtype=np.float32)
    boxes = arr[:, :4]
    scores = arr[:, 4]
    classes = arr[:, 5].astype(np.int64)
    keep = classwise_nms(boxes, scores, classes, fuse_iou)
    if keep.size > 0:
        # optional per-class confidence threshold (applied AFTER fusion NMS)
        if perclass_conf:
            cls_of = classes[keep]
            sc_of = scores[keep]
            mask = np.ones(len(keep), dtype=bool)
            for c, thr in perclass_conf.items():
                c = int(c)
                if thr and thr > 0:
                    mask &= ~((cls_of == c) & (sc_of < thr))
            keep = keep[mask]
        keep = keep[np.argsort(-scores[keep])[:max_det]]

    lines = []
    for idx in keep:
        x1, y1, x2, y2 = boxes[idx]
        conf = float(scores[idx])
        cls_id = int(classes[idx])
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        values = (cx, cy, bw, bh, conf)
        if not all(math.isfinite(v) for v in values):
            continue
        cx = min(max(float(cx), 0.0), 1.0)
        cy = min(max(float(cy), 0.0), 1.0)
        bw = min(max(float(bw), 1e-6), 1.0)
        bh = min(max(float(bh), 1e-6), 1.0)
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.6f}\n")

    out_path.write_text("".join(lines))
    return len(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-image + tiled Ultralytics inference for official submission files")
    parser.add_argument("--weights", required=True, help="Path to Ultralytics best.pt")
    parser.add_argument("--data-root", required=True, help="Test/validation data root containing visible/")
    parser.add_argument("--output", default="submission_ultra_tiled", help="Output directory for txt files")
    parser.add_argument("--zip", default=None, help="Optional zip path")
    parser.add_argument("--imgsz", type=int, default=768, help="YOLO input size for both full image and tiles")
    parser.add_argument("--tile", type=int, default=512, help="Tile side length in original pixels; 0 disables tiles")
    parser.add_argument("--overlap", type=float, default=0.25, help="Tile overlap ratio in [0, 0.9)")
    parser.add_argument("--conf", type=float, default=0.001, help="Prediction confidence threshold")
    parser.add_argument("--perclass-conf", default=None,
                        help="Optional JSON file mapping class_id -> conf threshold, e.g. {\"7\":0.003,\"11\":0.001}. Applied AFTER fusion NMS: a box is dropped if its class's threshold > 0 and score < threshold. Class ids absent from the map keep --conf.")
    parser.add_argument("--iou", type=float, default=0.6, help="Ultralytics per-view NMS IoU")
    parser.add_argument("--fuse-iou", type=float, default=0.55, help="Final class-wise NMS IoU across full image and tiles")
    parser.add_argument("--max-det", type=int, default=100, help="Max boxes per image after fusion")
    parser.add_argument("--batch", type=int, default=8, help="Tile prediction batch size")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tta", action="store_true", help="Enable Ultralytics augment=True for each view")
    parser.add_argument("--no-full-image", action="store_true", help="Use tiles only; default also includes full-image prediction")
    parser.add_argument("--limit", type=int, default=0, help="Optional image limit for smoke tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.overlap < 0.9:
        raise ValueError("--overlap must be in [0, 0.9)")

    # optional per-class conf json
    perclass_conf = None
    if args.perclass_conf:
        import json as _json
        with open(args.perclass_conf) as fh:
            perclass_conf = {int(k): float(v) for k, v in _json.load(fh).items()}
        print(f"  perclass_conf loaded: {perclass_conf}")

    visible_dir = Path(args.data_root) / "visible"
    if not visible_dir.is_dir():
        raise FileNotFoundError(f"visible directory not found: {visible_dir}")

    images = list_images(visible_dir, args.limit)
    if not images:
        raise FileNotFoundError(f"no images found in {visible_dir}")

    from ultralytics import YOLO

    output_dir = Path(args.output)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Tile-aware Ultralytics Inference (inference-only, no pseudo labels)")
    print("=" * 72)
    print(f"  Weights: {args.weights}")
    print(f"  Images: {len(images)} from {visible_dir}")
    print(f"  imgsz={args.imgsz} tile={args.tile} overlap={args.overlap} full={not args.no_full_image}")
    print(f"  conf={args.conf} iou={args.iou} fuse_iou={args.fuse_iou} max_det={args.max_det} tta={args.tta}")

    model = YOLO(args.weights)
    total_boxes = 0
    for image_idx, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        height, width = image.shape[:2]
        preds: List[Tuple[float, float, float, float, float, int]] = []

        if not args.no_full_image:
            full_result = predict_batch(model, [image], args.imgsz, args.conf, args.iou, args.max_det, args.tta, args.device)[0]
            append_result(preds, full_result, 0, 0, width, height)

        crops: List[np.ndarray] = []
        offsets: List[Tuple[int, int]] = []
        for x1, y1, x2, y2 in tile_windows(width, height, args.tile, args.overlap):
            if not args.no_full_image and x1 == 0 and y1 == 0 and x2 == width and y2 == height:
                continue
            crops.append(image[y1:y2, x1:x2])
            offsets.append((x1, y1))
            if len(crops) == args.batch:
                results = predict_batch(model, crops, args.imgsz, args.conf, args.iou, args.max_det, args.tta, args.device)
                for result, (dx, dy) in zip(results, offsets):
                    append_result(preds, result, dx, dy, width, height)
                crops.clear()
                offsets.clear()

        if crops:
            results = predict_batch(model, crops, args.imgsz, args.conf, args.iou, args.max_det, args.tta, args.device)
            for result, (dx, dy) in zip(results, offsets):
                append_result(preds, result, dx, dy, width, height)

        n = write_submission(output_dir / f"{image_path.stem}.txt", preds, width, height, args.fuse_iou, args.max_det, perclass_conf)
        total_boxes += n
        if image_idx % 100 == 0 or image_idx == len(images):
            print(f"  [{image_idx:4d}/{len(images)}] total_boxes={total_boxes}")

    print(f"\n  Generated {len(images)} submission files -> {output_dir}/ ({total_boxes} boxes)")
    if args.zip:
        make_zip(output_dir, Path(args.zip))
    print("  Done.")


if __name__ == "__main__":
    main()
