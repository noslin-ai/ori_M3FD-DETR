#!/usr/bin/env python
"""对测试集做 SAR 增强 + best 模型推理，生成伪标签并入训练集。

输出: out/images/pseudo_*.jpg + out/labels/pseudo_*.txt (YOLO 格式, 第6列置信度)
"""
import os, sys, argparse
import cv2
from ultralytics import YOLO

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from prepare_yolo_sar_enhanced_data import enhance_image, build_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/test")
    ap.add_argument("--out", default="data/yolo_sar_v3/train")
    ap.add_argument("--weights", default="runs/detect/runs/native_m_sar/rgb_sar768-2/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=768)
    args = ap.parse_args()

    rgb_map = build_map(os.path.join(args.root, "visible"))
    ir_map = build_map(os.path.join(args.root, "infrared"))
    depth_map = build_map(os.path.join(args.root, "depth"))
    img_dir = os.path.join(args.out, "images")
    lbl_dir = os.path.join(args.out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    model = YOLO(args.weights)
    n_kept, total_boxes = 0, 0
    for stem in sorted(rgb_map):
        img = enhance_image(rgb_map[stem], ir_map[stem], depth_map[stem])
        out_stem = "pseudo_" + stem
        cv2.imwrite(os.path.join(img_dir, out_stem + ".jpg"), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        boxes = res.boxes
        lines = []
        if boxes is not None and len(boxes) > 0:
            xy = boxes.xyxyn.cpu().numpy()
            cl = boxes.cls.cpu().numpy().astype(int)
            cf = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, s in zip(xy, cl, cf):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {s:.6f}\n")
            total_boxes += len(lines)
        with open(os.path.join(lbl_dir, out_stem + ".txt"), "w") as f:
            f.writelines(lines)
        if lines:
            n_kept += 1
    print(f"伪标签完成: 处理 {len(rgb_map)} 张, {n_kept} 张有检测, 共 {total_boxes} 个框")

if __name__ == "__main__":
    main()
