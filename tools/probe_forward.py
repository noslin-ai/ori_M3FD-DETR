"""前向探针：定位“所有 query 输出相同 logits”的塌缩环节。

用法（在服务器上运行，无需数据集）:
    # 随机权重（架构本身是否塌缩）
    python tools/probe_forward.py

    # 加载已训练 checkpoint（训练后是否塌缩）
    python tools/probe_forward.py --checkpoint checkpoints/rush/latest.pth

输出各环节的空间/query 方差，方差 ≈ 0 的环节即塌缩点。
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector.position_encoding import PositionEmbeddingSine
from models.m3f_detr import M3F_DETR
from utils.checkpoint import _safe_torch_load, strip_state_dict_prefixes


def spatial_std(t):
    """跨空间位置的标准差，平均到每个 (B, C) 通道。"""
    return t.std(dim=tuple(range(2, t.dim()))).mean().item()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="前向探针")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    B = 2

    cfg = {}
    if args.checkpoint:
        state = _safe_torch_load(args.checkpoint, map_location="cpu")
        cfg = state.get("cfg", {}) or {}
        print("Load checkpoint:", args.checkpoint)
    image_size = tuple(cfg.get("image_size", (384, 640)))

    model = M3F_DETR(
        num_classes=cfg.get("num_classes", 12),
        hidden_dim=cfg.get("hidden_dim", 256),
        num_queries=cfg.get("num_queries", 900),
        backbone_name=cfg.get("backbone", "swin_tiny"),
        pretrained=False,
        use_dn=cfg.get("use_dn", False),
        input_size=image_size,
    ).to(device).eval()

    if args.checkpoint:
        model.load_state_dict(strip_state_dict_prefixes(state["model"]))

    # 输入尺寸必须与 backbone 的 img_size 一致（timm 会断言 H/W）
    img_h, img_w = getattr(
        model.rgb_backbone.backbone, "img_size", (384, 640)
    )
    print(f"Input size: {img_h} x {img_w}")

    img = torch.randn(B, 3, img_h, img_w, device=device)
    ir = torch.randn(B, 3, img_h, img_w, device=device)
    depth = torch.randn(B, 3, img_h, img_w, device=device)

    # 2) backbone 各层特征的空间方差
    feats = model.rgb_backbone(img)
    for i, f in enumerate(feats):
        print(f"[BACKBONE] P{i + 2} shape={tuple(f.shape)} "
              f"mean={f.mean():.4f} spatial_std={spatial_std(f):.6f}")

    # 3) FPN 输出空间方差（features[-1] 是 decoder 的 memory 来源）
    ir_f = model.ir_encoder(ir)
    depth_f = model.depth_encoder(depth)
    fused = model.fusion(feats[0], ir_f, depth_f)
    fpn_out = model.fpn([fused] + list(feats[1:]))
    for i, f in enumerate(fpn_out):
        print(f"[FPN]       out{i} shape={tuple(f.shape)} "
              f"spatial_std={spatial_std(f):.6f}")

    # 1) 位置编码是否退化（每个位置是否相同），用 decoder 实际输入的特征尺寸
    pe = PositionEmbeddingSine(model.hidden_dim // 2).to(device)
    pos = pe(fpn_out[-1])
    print(f"[PE]        shape={tuple(pos.shape)} spatial_std={spatial_std(pos):.6f}"
          "  <- 若≈0，位置编码塌缩")

    # 4) query embedding 行间方差
    det = model.detector
    q = det.query_embed.weight
    print(f"[QUERY]     row_std={q.std(dim=0).mean().item():.6f}"
          "  <- 若≈0，query 已互相塌缩")

    # 5) decoder 输出跨 query 方差
    src = det.input_proj[0](fpn_out[-1])   # 与模型实际 forward 一致
    pos_emb = det.position_embedding(src)
    src_flat = src.flatten(2).permute(2, 0, 1)      # (HW, B, C)
    pos_flat = pos_emb.flatten(2).permute(2, 0, 1)  # (HW, B, C)
    print(f"[MEMORY]    src spatial_std={src_flat.std(dim=0).mean().item():.6f} "
          f"pos spatial_std={pos_flat.std(dim=0).mean().item():.6f}")

    q_emb = q.unsqueeze(1).repeat(1, B, 1)          # (Nq, B, C)
    tgt = torch.zeros_like(q_emb)
    hs = det.transformer(tgt, src_flat, pos_flat, q_emb)[-1].transpose(0, 1)
    print(f"[DECODER]   hs shape={tuple(hs.shape)} "
          f"query_std={hs.std(dim=1).mean().item():.6f}"
          "  <- 若≈0，decoder 输出对 query 塌缩")

    # 5b) 依赖测试：输出是否真的依赖 query / memory
    hs_mem2 = det.transformer(tgt, src_flat + 1.0, pos_flat, q_emb)[-1].transpose(0, 1)
    hs_q2 = det.transformer(tgt, src_flat, pos_flat, torch.zeros_like(q_emb))[-1].transpose(0, 1)
    print(f"[DEP]       hs 随 memory 变化: {(hs - hs_mem2).abs().mean().item():.6f} "
          f"| hs 随 query 变化: {(hs - hs_q2).abs().mean().item():.6f}")

    # 6) 最终 logits：跨 query 方差 + 不同输入是否产生不同输出
    out1 = model(img, ir, depth)
    out2 = model(img + 1.0, ir + 1.0, depth + 1.0)
    lg1, lg2 = out1["pred_logits"], out2["pred_logits"]
    print(f"[LOGITS]    query_std={lg1.std(dim=1).mean().item():.6f} "
          f"img1_vs_img2_diff={(lg1 - lg2).abs().mean().item():.6f}"
          "  <- 后者≈0 则输出与输入无关")

    # 7) 框：跨 query 是否塌缩 + 是否依赖图像内容 + 均值位置/尺寸
    bx1 = out1["pred_boxes"]
    bx2 = out2["pred_boxes"]
    print(f"[BOX]       query_std={bx1.std(dim=1).mean().item():.6f} "
          f"img1_vs_img2_diff={(bx1 - bx2).abs().mean().item():.6f} "
          f"cx={bx1[:, :, 0].mean().item():.3f} cy={bx1[:, :, 1].mean().item():.3f} "
          f"w={bx1[:, :, 2].mean().item():.3f} h={bx1[:, :, 3].mean().item():.3f}"
          "  <- query_std≈0 则所有框相同; img_diff≈0 则框不随图像变化(模板)")


if __name__ == "__main__":
    main()
