"""YOLO 推理脚本 — v0.8.0 官方提交文件生成器。

在测试集上推理，输出符合官方要求的 txt 文件:
    [class_id, norm_center_x, norm_center_y, norm_w, norm_h, confidence]
    每行一个检测框，空格分隔；每图最多 100 框；空图提交空 txt。

运行方式:
    cd M3F-DETR
    python tools/infer_yolo.py --checkpoint checkpoints/yolo_fusion/best.pth \
        --data-root data/test --output submission_yolo --zip submission_yolo.zip

    # 使用 EMA 权重 / 调整阈值
    python tools/infer_yolo.py --checkpoint checkpoints/yolo_fusion/best.pth \
        --data-root data/test --output submission_yolo --use-ema \
        --conf-threshold 0.25 --nms-iou 0.6
"""

import os
import sys
import argparse
import math
import shutil

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from yolo.dataset import YOLOFusionDataset, collate_fn
from yolo.model import build_yolo_model
from engine.evaluator import class_aware_nms
from ultralytics.utils.nms import non_max_suppression


def load_checkpoint(checkpoint_path, device):
    """加载 checkpoint 并返回 (state_dict, cfg)。"""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = state.get("cfg", {}) if isinstance(state, dict) else {}
    return state, cfg


def det_to_submission_lines(det, img_w, img_h):
    """把 NMS 输出的 (N,6) xyxy 像素框转成提交行 [cls cx cy w h conf]。"""
    lines = []
    if det is None or det.shape[0] == 0:
        return lines

    for j in range(det.shape[0]):
        cls_id = int(det[j, 5])
        conf = float(det[j, 4])
        x1, y1, x2, y2 = det[j, :4].tolist()

        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        # 比赛会判非法类别、非法坐标、置信度缺失为无效预测
        if cls_id < 0 or conf <= 0.0 or w <= 0.0 or h <= 0.0:
            continue
        if not all(math.isfinite(v) for v in (cx, cy, w, h, conf)):
            continue

        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w = min(max(w, 1e-6), 1.0)
        h = min(max(h, 1e-6), 1.0)
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
    return lines


def make_zip(output_dir, zip_path):
    """将提交目录打包为 zip，zip 根目录直接包含所有 txt 文件。"""
    base_name = zip_path[:-4] if zip_path.endswith(".zip") else zip_path
    archive_path = shutil.make_archive(base_name, "zip", root_dir=output_dir)
    print(f"  Packed submission zip → {archive_path}")


@torch.no_grad()
def generate_submission(model, loader, output_dir, device, cfg,
                        conf_threshold=0.25, nms_iou=0.6, max_det=100,
                        use_ema=False, clean_output=False, tta=False):
    """在测试集上推理并生成提交文件。

    Args:
        tta: 是否启用水平翻转 TTA（原图 + 翻转图各推理一次，
             合并后用类别内 NMS 去重；实测 RGB 模型可提升约 +0.02 mAP）。
    """
    model.eval()

    if clean_output and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    num_files = 0
    for batch in loader:
        img = batch["img"].to(device, non_blocking=True)
        names = batch["names"]
        B, _, H, W = img.shape

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(img)
            if tta:
                out_flip = model(torch.flip(img, dims=[3]))

        y = out[0] if isinstance(out, tuple) else out
        if tta:
            y_flip = out_flip[0] if isinstance(out_flip, tuple) else out_flip
        dets = non_max_suppression(
            y, conf_thres=conf_threshold, iou_thres=nms_iou, max_det=max_det
        )
        if tta:
            dets_flip = non_max_suppression(
                y_flip, conf_thres=conf_threshold, iou_thres=nms_iou, max_det=max_det
            )

        for i in range(B):
            out_path = os.path.join(output_dir, f"{names[i]}.txt")
            det = dets[i]
            if tta:
                merged = []
                if det is not None and det.shape[0]:
                    merged.append(det)
                df = dets_flip[i]
                if df is not None and df.shape[0]:
                    df = df.clone()
                    df[:, [0, 2]] = W - df[:, [2, 0]]  # 翻转回原坐标
                    merged.append(df)
                if merged:
                    merged = torch.cat(merged, dim=0)
                    # 合并后是 (N,6) xyxy+conf+cls，用类别内 NMS 去重
                    cxcywh = torch.stack([
                        (merged[:, 0] + merged[:, 2]) / 2 / W,
                        (merged[:, 1] + merged[:, 3]) / 2 / H,
                        (merged[:, 2] - merged[:, 0]) / W,
                        (merged[:, 3] - merged[:, 1]) / H,
                    ], dim=1).clamp(0.0, 1.0)
                    keep = class_aware_nms(
                        cxcywh, merged[:, 4], merged[:, 5].long(),
                        iou_threshold=nms_iou, max_dets=max_det,
                    )
                    det = merged[keep]
                else:
                    det = None
            lines = det_to_submission_lines(det, W, H)
            with open(out_path, "w") as f:
                f.writelines(lines)
            num_files += 1

    print(f"\n  Generated {num_files} submission files → {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="YOLO Inference & Submission")
    parser.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径")
    parser.add_argument("--data-root", required=True, help="测试数据根目录")
    parser.add_argument("--output", default="submission_yolo", help="输出目录")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--max-det", type=int, default=100, help="每张图最多保留预测框数量")
    parser.add_argument("--nms-iou", type=float, default=0.6, help="同类别 NMS IoU 阈值")
    parser.add_argument("--clean-output", action="store_true", help="生成前清空输出目录")
    parser.add_argument("--zip", default=None, help="可选：生成提交 zip 路径")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-ema", action="store_true", help="使用 EMA 权重")
    parser.add_argument("--tta", action="store_true", help="启用水平翻转 TTA（+约0.02 mAP）")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, switching to CPU")
        device = "cpu"
    device = torch.device(device)

    print("=" * 70)
    print("  YOLO Inference & Submission Generator (v0.8.0)")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data root: {args.data_root}")
    print(f"  Conf threshold: {args.conf_threshold} | NMS IoU: {args.nms_iou}")
    print(f"  Device: {device} | EMA: {args.use_ema}")
    print(f"  TTA: {args.tta}")

    state, cfg = load_checkpoint(args.checkpoint, device)
    mode = cfg.get("mode", "fusion")
    ch = cfg.get("ch", 5)
    nc = cfg.get("nc", 12)
    image_size = tuple(cfg.get("image_size", (384, 640)))

    # ---- 数据集 ----
    print("\n[1] Loading test dataset...")
    dataset = YOLOFusionDataset(
        args.data_root, mode=mode, size=image_size, train=False, nc=nc
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    print(f"  Total samples: {len(dataset)}")

    # ---- 模型 ----
    print("\n[2] Loading model...")
    model = build_yolo_model(ch=ch, nc=nc, pretrained=None).to(device)
    if args.use_ema and state.get("ema") is not None:
        print("  Using EMA weights")
        model.load_state_dict(state["ema"])
    else:
        model.load_state_dict(state["model"])
    model.eval()

    # ---- 推理 ----
    print("\n[3] Generating submissions...")
    generate_submission(
        model, loader, args.output, device, cfg,
        conf_threshold=args.conf_threshold,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        use_ema=args.use_ema,
        clean_output=args.clean_output,
        tta=args.tta,
    )

    if args.zip:
        make_zip(args.output, args.zip)

    print("\n" + "=" * 70)
    print("  Submission generation complete!")
    print(f"  Results saved to: {args.output}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
