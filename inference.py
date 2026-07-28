"""M3F-DINO 推理脚本 — 官方提交文件生成器。

在测试集上推理，输出符合官方要求的 YOLO 格式 txt 文件。

官方格式:
    [class_id, norm_center_x, norm_center_y, norm_w, norm_h, confidence]
    每行一个检测框，空格分隔

运行方式:
    cd M3F-DETR
    # 单模型推理
    python inference.py --checkpoint checkpoints/best.pth --data-root data/test --output submission

    # 使用 EMA 权重
    python inference.py --checkpoint checkpoints/best.pth --data-root data/test --output submission --use-ema

    # 指定设备和阈值
    python inference.py --checkpoint checkpoints/best.pth --data-root data/test --output submission --conf-threshold 0.3 --device cuda:0
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.rgb_ir_depth_dataset import RGBIRDepthDataset
from models.m3f_detr import M3F_DETR
from utils.checkpoint import _safe_torch_load


@torch.no_grad()
def generate_submission(
    model,
    loader,
    output_dir,
    device,
    conf_threshold=0.3,
    use_amp=True,
):
    """在测试集上推理并生成提交文件。

    Args:
        model: M3F-DINO 模型
        loader: 测试集 DataLoader
        output_dir: 输出目录
        device: 设备
        conf_threshold: 置信度阈值
        use_amp: 是否使用混合精度
    """
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    sample_idx = 0

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        ir = batch["ir"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        names = batch["names"] if "names" in batch else None

        with autocast(enabled=use_amp):
            output = model(rgb, ir, depth)

        B = rgb.shape[0]
        pred_logits = output["pred_logits"]   # (B, Q, C+1)
        pred_boxes = output["pred_boxes"]      # (B, Q, 4)

        for i in range(B):
            # Softmax + 过滤背景
            scores = pred_logits[i].softmax(-1)       # (Q, C+1)
            max_scores, labels = scores.max(dim=-1)   # (Q,), (Q,)

            # 过滤低置信度和背景类
            num_classes = scores.shape[1] - 1
            valid = (labels < num_classes) & (max_scores > conf_threshold)
            valid_labels = labels[valid]               # (K,)
            valid_scores = max_scores[valid]           # (K,)
            valid_boxes = pred_boxes[i][valid]         # (K, 4)

            # 写入 txt 文件
            # 按置信度排序，截断到最多 100 个框（竞赛要求）
            if len(valid_labels) > 100:
                sorted_idx = valid_scores.argsort(descending=True)[:100]
                valid_labels = valid_labels[sorted_idx]
                valid_scores = valid_scores[sorted_idx]
                valid_boxes = valid_boxes[sorted_idx]

            # 文件名: 使用数据集中的原始 stem
            file_stem = names[i] if names and i < len(names) else f"{sample_idx:06d}"
            sample_idx += 1

            out_path = os.path.join(output_dir, f"{file_stem}.txt")
            with open(out_path, "w") as f:
                for j in range(len(valid_labels)):
                    cls_id = valid_labels[j].item()
                    conf = valid_scores[j].item()
                    box = valid_boxes[j]  # (cx, cy, w, h) normalized

                    # 格式: class_id cx cy w h confidence
                    f.write(
                        f"{cls_id} "
                        f"{box[0].item():.6f} {box[1].item():.6f} "
                        f"{box[2].item():.6f} {box[3].item():.6f} "
                        f"{conf:.6f}\n"
                    )

    print(f"\n  Generated {sample_idx} submission files → {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="M3F-DINO Inference & Submission")
    parser.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径")
    parser.add_argument("--data-root", required=True, help="测试数据根目录")
    parser.add_argument("--output", default="submission", help="输出目录")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--use-ema", action="store_true", help="使用 EMA 权重")
    parser.add_argument("--backbone", default=None,
                        choices=["swin_tiny","swin_small","swin_base","swin_large"],
                        help="backbone 类型 (默认从 checkpoint 自动读取，或 fallback 为 swin_large)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default="configs/m3f_dino.yaml")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, switching to CPU")
        device = "cpu"

    print("=" * 70)
    print("  M3F-DINO Inference & Submission Generator")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data root: {args.data_root}")
    print(f"  Output: {args.output}/")
    print(f"  Conf threshold: {args.conf_threshold}")
    print(f"  Device: {device}")
    print(f"  EMA: {args.use_ema}")

    # ---- 数据集 ----
    print("\n[1] Loading test dataset...")
    # test mode: 不需要 labels
    dataset = RGBIRDepthDataset(args.data_root, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: {
            "rgb": torch.stack([b["rgb"] for b in batch]),
            "ir": torch.stack([b["ir"] for b in batch]),
            "depth": torch.stack([b["depth"] for b in batch]),
            "names": [b.get("name", "") for b in batch],
        },
        pin_memory=True,
    )
    print(f"  Total samples: {len(dataset)}")

    # ---- 模型 ----
    print("\n[2] Loading model...")
    # 先加载 checkpoint 获取模型配置
    state = _safe_torch_load(args.checkpoint, map_location="cpu")
    cfg = state.get("cfg", {}) if isinstance(state, dict) else {}

    # 确定 backbone: 优先使用命令行参数 > checkpoint cfg > fallback
    backbone_name = args.backbone
    if backbone_name is None:
        backbone_name = cfg.get("backbone", "swin_large")
    num_classes = cfg.get("num_classes", args.num_classes)
    hidden_dim  = cfg.get("hidden_dim", getattr(args, 'hidden_dim', 256))
    num_queries = cfg.get("num_queries", getattr(args, 'num_queries', 900))

    print(f"  Backbone: {backbone_name} (from {'checkpoint cfg' if cfg.get('backbone') else 'fallback'})")
    model = M3F_DETR(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        backbone_name=backbone_name,
        use_dn=False,  # 推理时不需要 DN
    ).to(device)

    # 加载权重（如果是 CPU 加载的 state 需确保到 device）
    if device != "cpu":
        state = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in state.items() if k != "cfg"}

    if args.use_ema and "ema" in state and state.get("ema"):
        print("  Using EMA weights")
        model.load_state_dict(state["ema"])
    elif "model" in state:
        model.load_state_dict(state["model"])
    else:
        # 移除 cfg 后再加载
        state_to_load = {k: v for k, v in state.items() if k != "cfg"}
        model.load_state_dict(state_to_load)

    model.eval()

    # ---- 推理 ----
    print("\n[3] Generating submissions...")
    generate_submission(
        model, loader, args.output, device,
        conf_threshold=args.conf_threshold,
        use_amp=True,
    )

    print("\n" + "=" * 70)
    print("  Submission generation complete!")
    print(f"  Results saved to: {args.output}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
