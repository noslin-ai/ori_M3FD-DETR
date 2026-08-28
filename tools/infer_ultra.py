"""ultralytics 原生 checkpoint 的提交文件生成器（v0.10.0）。

用于加载原生训练的 best.pt（如 runs/native_x/rgb640/weights/best.pt），
在测试集上推理并生成官方格式提交文件:
    [class_id, norm_center_x, norm_center_y, norm_w, norm_h, confidence]

与 tools/infer_yolo.py 的区别:
    - 原生模型按 ultralytics 约定用 letterbox 预处理（imgsz=640 正方形），
      这里直接复用 YOLO.predict，避免手工预处理与训练不一致；
    - 支持 augment=True（ultralytics 内置测试时增强）。

用法:
    python tools/infer_ultra.py --weights runs/native_x/rgb640/weights/best.pt \
        --data-root data/test --output submission_ultra --tta --zip submission_ultra.zip
"""

import os
import argparse
import math
import shutil

from ultralytics import YOLO


def make_zip(output_dir, zip_path):
    """将提交目录打包为 zip，zip 根目录直接包含所有 txt 文件。"""
    base_name = zip_path[:-4] if zip_path.endswith(".zip") else zip_path
    archive_path = shutil.make_archive(base_name, "zip", root_dir=output_dir)
    print(f"  Packed submission zip → {archive_path}")


def main():
    parser = argparse.ArgumentParser(description="Ultralytics native inference & submission")
    parser.add_argument("--weights", required=True, help="best.pt 路径")
    parser.add_argument("--data-root", required=True, help="测试数据根目录（含 visible/）")
    parser.add_argument("--output", default="submission_ultra", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU 阈值")
    parser.add_argument("--max-det", type=int, default=100, help="每图最多保留框数")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸（与训练一致）")
    parser.add_argument("--tta", action="store_true", help="启用 augment 测试时增强")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--zip", default=None, help="可选：生成提交 zip 路径")
    args = parser.parse_args()

    visible_dir = os.path.join(args.data_root, "visible")
    if not os.path.isdir(visible_dir):
        raise FileNotFoundError(f"visible 目录不存在: {visible_dir}")

    print("=" * 70)
    print("  Ultralytics Native Inference & Submission (v0.10.0)")
    print("=" * 70)
    print(f"  Weights: {args.weights}")
    print(f"  Data: {visible_dir} | imgsz={args.imgsz} | TTA={args.tta}")

    model = YOLO(args.weights)
    results = model.predict(
        source=visible_dir,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        augment=args.tta,
        batch=args.batch,
        device=args.device,
        verbose=False,
    )

    if os.path.isdir(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)

    n_files = 0
    for r in results:
        stem = os.path.splitext(os.path.basename(r.path))[0]
        img_w, img_h = r.orig_shape[1], r.orig_shape[0]
        out_path = os.path.join(args.output, f"{stem}.txt")
        lines = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for j in range(len(confs)):
                x1, y1, x2, y2 = xyxy[j]
                cx = (x1 + x2) / 2.0 / img_w
                cy = (y1 + y2) / 2.0 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                conf = float(confs[j])
                if cls[j] < 0 or conf <= 0.0 or w <= 0.0 or h <= 0.0:
                    continue
                if not all(math.isfinite(v) for v in (cx, cy, w, h, conf)):
                    continue
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                w = min(max(w, 1e-6), 1.0)
                h = min(max(h, 1e-6), 1.0)
                lines.append(
                    f"{int(cls[j])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}\n"
                )
        with open(out_path, "w") as f:
            f.writelines(lines)
        n_files += 1

    print(f"\n  Generated {n_files} submission files → {args.output}/")
    if args.zip:
        make_zip(args.output, args.zip)
    print("\n  Submission generation complete!")


if __name__ == "__main__":
    main()
