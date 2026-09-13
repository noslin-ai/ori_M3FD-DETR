"""Exact champion-style FP32 TTA inference for six-channel adapter models."""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/v2"))

import adapter_model  # noqa: E402,F401 - checkpoint pickle dependency
import distribution_aligned_adapter  # noqa: E402,F401 - checkpoint pickle dependency
import mage_exchange_adapter  # noqa: E402,F401 - checkpoint pickle dependency


def path_map(directory: Path):
    output = {}
    for extension in ("*.png", "*.jpg", "*.jpeg"):
        for path in glob.glob(str(directory / extension)):
            stem = Path(path).stem
            if stem in output:
                raise RuntimeError(f"duplicate stem {stem} in {directory}")
            output[stem] = path
    return output


def read_auxiliary(ir_path: str, depth_path: str):
    infrared = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if infrared.ndim == 3:
        infrared = cv2.cvtColor(infrared, cv2.COLOR_BGR2GRAY)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.dtype == np.uint16:
        valid = depth > 0
        value = np.zeros(depth.shape, np.float32)
        value[valid] = np.log1p(np.clip(depth[valid].astype(np.float32), 0, 20000)) / np.log1p(20000.0)
        value *= 255.0
    else:
        valid = depth > 0
        value = depth.astype(np.float32)
    return np.stack((infrared.astype(np.float32), value, valid.astype(np.float32) * 255.0), axis=-1)


def letterbox_pair(rgb, auxiliary, letterbox):
    params = letterbox.get_params({"img": rgb})
    rgb = letterbox.apply_image({"img": rgb}, params)["img"]
    new_unpad = params["new_unpad"]
    if auxiliary.shape[:2][::-1] != new_unpad:
        auxiliary = np.stack(
            (
                cv2.resize(auxiliary[..., 0], new_unpad, interpolation=cv2.INTER_LINEAR),
                cv2.resize(auxiliary[..., 1], new_unpad, interpolation=cv2.INTER_NEAREST),
                cv2.resize(auxiliary[..., 2], new_unpad, interpolation=cv2.INTER_NEAREST),
            ),
            axis=-1,
        )
    auxiliary = cv2.copyMakeBorder(
        auxiliary,
        params["top"],
        params["bottom"],
        params["left"],
        params["right"],
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return rgb, np.ascontiguousarray(auxiliary)


def box_iou_one_to_many(box, boxes):
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod(1)
    area = np.maximum(box[2:] - box[:2], 0).prod()
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1)
    return intersection / np.maximum(area + areas - intersection, 1e-9)


def second_classwise_nms(detections, iou_threshold=0.55, max_det=100):
    if len(detections) == 0:
        return detections
    boxes = detections[:, :4]
    scores = detections[:, 4]
    classes = detections[:, 5].astype(np.int64)
    keep = []
    for class_id in sorted(set(classes.tolist())):
        order = np.where(classes == class_id)[0]
        order = order[np.argsort(-scores[order])]
        while len(order):
            current = int(order[0])
            keep.append(current)
            if len(order) == 1:
                break
            overlap = box_iou_one_to_many(boxes[current], boxes[order[1:]])
            order = order[1:][overlap <= iou_threshold]
    keep = np.asarray(keep, dtype=np.int64)
    keep = keep[np.argsort(-scores[keep])[:max_det]]
    return detections[keep]


def infer(weights: str, cache_path: Path, limit: int = 0):
    visible = path_map(ROOT / "data/test/visible")
    infrared = path_map(ROOT / "data/test/infrared")
    depth = path_map(ROOT / "data/test/depth")
    stems = sorted(visible)
    if len(stems) != 1000 or set(stems) != set(infrared) or set(stems) != set(depth):
        raise RuntimeError("test modalities are not aligned 1000-way")
    if limit:
        stems = stems[:limit]

    network = YOLO(weights).model.cuda().float().eval()
    letterbox = LetterBox(new_shape=(1280, 1280), auto=True, stride=int(network.stride.max()))
    output = {}
    with torch.inference_mode():
        for index, stem in enumerate(stems, 1):
            original = cv2.imread(visible[stem], cv2.IMREAD_COLOR)
            auxiliary = read_auxiliary(infrared[stem], depth[stem])
            image, auxiliary = letterbox_pair(original, auxiliary, letterbox)
            rgb = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)).astype(np.float32)
            tensor = torch.from_numpy(np.concatenate((rgb, auxiliary.transpose(2, 0, 1)), 0))
            tensor = tensor.cuda().float().div_(255.0).unsqueeze(0)
            raw = network(tensor, augment=True)
            prediction = raw[0] if isinstance(raw, (tuple, list)) else raw
            detection = non_max_suppression(prediction, 0.001, 0.6, max_det=100)[0]
            if len(detection):
                detection[:, :4] = ops.scale_boxes(tensor.shape[2:], detection[:, :4], original.shape)
                detection = second_classwise_nms(detection.float().cpu().numpy(), 0.55, 100)
                xywh = ops.xyxy2xywh(torch.from_numpy(detection[:, :4])).numpy()
                xywh[:, [0, 2]] /= original.shape[1]
                xywh[:, [1, 3]] /= original.shape[0]
                output[stem] = [
                    (int(detection[row, 5]), *[float(v) for v in xywh[row]], float(detection[row, 4]))
                    for row in range(len(detection))
                ]
            else:
                output[stem] = []
            if index % 50 == 0 or index == len(stems):
                print(f"TTAEXACT {index}/{len(stems)}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as handle:
        json.dump(output, handle)
    return output


def emit(raw, threshold: float, output_dir: Path, zip_path: Path):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    boxes = 0
    for stem in sorted(raw):
        rows = []
        for row in raw[stem]:
            class_id, x, y, width, height, score = row
            values = (x, y, width, height, score)
            if score >= threshold and 0 <= class_id < 12 and all(math.isfinite(v) for v in values):
                if 0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1:
                    rows.append(row)
            if len(rows) == 100:
                break
        boxes += len(rows)
        with (output_dir / f"{stem}.txt").open("w") as handle:
            for class_id, x, y, width, height, score in rows:
                handle.write(f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f} {score:.6f}\n")
    files = sorted(output_dir.glob("*.txt"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)
    print(f"PACKAGE {zip_path} files={len(files)} boxes={boxes} bytes={zip_path.stat().st_size}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.45])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    raw = json.load(args.cache.open()) if args.cache.exists() else infer(args.weights, args.cache, args.limit)
    for threshold in args.thresholds:
        suffix = f"conf{threshold:.2f}"
        emit(raw, threshold, ROOT / "submissions" / f"{args.prefix}_{suffix}", ROOT / "submissions" / f"{args.prefix}_{suffix}.zip")


if __name__ == "__main__":
    main()
