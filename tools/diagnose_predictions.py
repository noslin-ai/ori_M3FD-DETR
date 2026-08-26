"""诊断预测置信度分布，定位“验证 0 个框”问题。

用法（在服务器上，需存在数据集目录）:
    python tools/diagnose_predictions.py --checkpoint checkpoints/rush/latest.pth

输出:
    - pred_logits 数值分布（按类别均值，背景类应明显偏高）
    - softmax argmax 落在背景类的比例
    - 去掉背景类后 sigmoid 最大置信度分布（> 0.1/0.2/0.3/0.5 的 query 数）
    - 原始权重 vs EMA 权重对比（验证 EMA 滞后假设）
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.rgb_ir_depth_dataset import RGBIRDepthDataset
from engine.trainer import collate_fn
from models.m3f_detr import M3F_DETR
from utils.checkpoint import _safe_torch_load, strip_state_dict_prefixes


@torch.no_grad()
def diagnose(checkpoint, data_root, max_batches, device, batch_size):
    state = _safe_torch_load(checkpoint, map_location="cpu")
    cfg = state.get("cfg", {}) or {}
    image_size = tuple(cfg.get("image_size", (384, 640)))
    print("Checkpoint cfg:", cfg)

    dataset = RGBIRDepthDataset(data_root, train=True, size=image_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    print(f"Dataset: {len(dataset)} samples, scanning first {max_batches} batches\n")

    for tag, key in [("raw (model)", "model"), ("EMA", "ema")]:
        weights = state.get(key)
        if not isinstance(weights, dict) or len(weights) == 0:
            print(f"=== {tag}: checkpoint 中无该权重，跳过 ===")
            continue

        model = M3F_DETR(
            num_classes=cfg.get("num_classes", 12),
            hidden_dim=cfg.get("hidden_dim", 256),
            num_queries=cfg.get("num_queries", 900),
            backbone_name=cfg.get("backbone", "swin_tiny"),
            use_dn=cfg.get("use_dn", False),
            input_size=image_size,
        ).to(device).eval()
        model.load_state_dict(strip_state_dict_prefixes(weights))

        logits_list = []
        softmax_bg_ratio = []
        sigmoid_obj = []
        for n, batch in enumerate(loader):
            if n >= max_batches:
                break
            rgb = batch["rgb"].to(device, non_blocking=True)
            ir = batch["ir"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            out = model(rgb, ir, depth)
            logits = out["pred_logits"].float()          # (B, Q, C+1)
            logits_list.append(logits)

            sm = logits.softmax(-1)
            softmax_bg_ratio.append(
                (sm.argmax(-1) == logits.shape[-1] - 1).float().mean().item()
            )
            obj_conf = logits[:, :, :-1].sigmoid().max(-1).values  # (B, Q)
            sigmoid_obj.append(obj_conf)

        logits = torch.cat(logits_list)
        obj_conf = torch.cat(sigmoid_obj)
        n_queries = obj_conf.numel()
        bg_ratio = sum(softmax_bg_ratio) / len(softmax_bg_ratio)

        print(f"=== {tag} ===")
        print(f"  pred_logits: mean={logits.mean():.4f} std={logits.std():.4f} "
              f"min={logits.min():.4f} max={logits.max():.4f}")
        class_means = ", ".join(
            f"{i}:{logits[:, :, i].mean():.3f}" for i in range(logits.shape[-1])
        )
        print(f"  logits mean by class: {class_means}")
        print(f"  softmax argmax == 背景 的比例: {bg_ratio:.4f}")
        print(f"  去背景后 sigmoid 最大置信度: mean={obj_conf.mean():.4f} "
              f"max={obj_conf.max():.4f}")
        for thr in (0.1, 0.2, 0.3, 0.5):
            cnt = (obj_conf > thr).sum().item()
            print(f"    conf > {thr}: {cnt} / {n_queries} ({cnt / n_queries * 100:.2f}%)")
        print()


def main():
    parser = argparse.ArgumentParser(description="诊断预测置信度分布")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 路径")
    parser.add_argument("--data-root", default="data/train", help="数据集根目录")
    parser.add_argument("--max-batches", type=int, default=8, help="扫描多少 batch")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    diagnose(
        args.checkpoint, args.data_root,
        args.max_batches, args.device, args.batch_size,
    )


if __name__ == "__main__":
    main()
