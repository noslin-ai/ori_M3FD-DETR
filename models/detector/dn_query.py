"""Denoising Query — DINO 对比去噪训练。

在 Object Query 中混入噪声 Ground Truth 框，让模型学习
区分正样本（去噪）和负样本（原始查询），加速收敛。

参考: DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection
"""

import torch
import torch.nn.functional as F


def prepare_for_dn(
    query_embed,        # (B, Nq, C)
    targets,            # list of dict {"boxes": (N_i,4), "labels": (N_i,)}
    dn_query_embed,     # (Nq, C) denoising query embed
    dn_number=100,       # 每个 batch 添加多少 DN 查询
    num_classes=12,
    label_noise_ratio=0.2,   # 标签翻转概率
    box_noise_ratio=0.4,      # 框抖动标准差
):
    """为 query_embed 添加 Denoising Query。

    Args:
        query_embed: 原始 Object Query (B, Nq, C)
        targets: list of dict, GT 标注
        dn_query_embed: DN 专用的 query embed
        dn_number: DN 查询数
        num_classes: 类别数
        label_noise_ratio: 标签噪声比例
        box_noise_ratio: 框噪声标准差

    Returns:
        padded_query_embed: (B, Nq+P, C) 拼接后的 queries
        dn_targets: 对应的 targets（DN 部分用 GT，正常部分不变）
        attn_mask: (Nq+P, Nq+P) 的自注意力 mask，让 DN queries 互相可见但原始部分不能看到 DN
        dn_meta: dict，DN 元信息用于后处理
    """
    if not targets or dn_number == 0:
        return query_embed, targets, None, None

    B, Nq, C = query_embed.shape
    device = query_embed.device

    # --- 收集所有 GT boxes ---
    all_boxes = []
    all_labels = []
    gt_indices = []

    for i, t in enumerate(targets):
        boxes = t["boxes"]  # (Ni, 4) cxcywh normalized
        labels = t["labels"]  # (Ni,)
        n = boxes.shape[0]
        if n == 0:
            continue
        all_boxes.append(boxes)
        all_labels.append(labels)
        for _ in range(n):
            gt_indices.append(i)

    if len(all_boxes) == 0:
        return query_embed, targets, None, None

    all_boxes = torch.cat(all_boxes, dim=0)  # (total_N, 4)
    all_labels = torch.cat(all_labels, dim=0)  # (total_N,)

    total_gt = all_boxes.shape[0]

    # 每组 DN queries = min(dn_number, total_gt) 个
    # 如果需要更多，可以重复采样
    dn_num = min(dn_number, total_gt)
    if dn_num == 0:
        return query_embed, targets, None, None

    # 随机选取 dn_num 个 GT
    perm = torch.randperm(total_gt, device=device)[:dn_num]
    dn_boxes = all_boxes[perm]   # (dn_num, 4)
    dn_labels = all_labels[perm] # (dn_num,)

    # --- 添加噪声 ---
    # 标签: 以 label_noise_ratio 的概率随机翻转
    label_noise = torch.rand(dn_labels.shape, device=device) < label_noise_ratio
    random_labels = torch.randint(0, num_classes, dn_labels.shape, device=device)
    noisy_labels = torch.where(label_noise, random_labels, dn_labels.long())

    # 框: 添加高斯噪声，然后 clamp 到 [0, 1]
    box_noise = torch.randn(dn_boxes.shape, device=device) * box_noise_ratio
    noisy_boxes = dn_boxes + box_noise
    noisy_boxes = noisy_boxes.clamp(0, 1)

    # --- 构造 DN queries ---
    # 每个样本分配 dn_num 个 DN queries
    # 但并非所有样本都有 GT，这里简化：把 dn queries 堆在 batch 维度

    # 实际上，DINO 的 DN 做法更复杂：对每个样本独立分配
    # 这里简化为：取前 B 个 batch 中的样本，分配 dn_num 个

    # 简化实现：把 DN queries 作为额外的 batch 维度
    # padded_query_embed: (B, Nq, C) → (B + 1, Nq + dn_num, C) 行不通

    # 正确做法：构造扩展后的 query embed
    # 对每个样本，如果它有 GT，就把 GT (加噪) 编码进 DN queries
    # 如果它没有 GT，就补零
    dn_mask = torch.zeros(B, dn_num, dtype=torch.bool, device=device)
    dn_query_list = []
    dn_label_list = []
    dn_box_list = []

    for i in range(B):
        t = targets[i]
        n_gt = t["boxes"].shape[0]
        if n_gt == 0:
            # 无 GT，用零填充
            dn_query_list.append(torch.zeros(dn_num, C, device=device))
            dn_label_list.append(torch.zeros(dn_num, dtype=torch.long, device=device))
            dn_box_list.append(torch.zeros(dn_num, 4, device=device))
        else:
            # 有 GT，取 min(n_gt, dn_num) 个
            take = min(n_gt, dn_num)
            p = torch.randperm(n_gt, device=device)[:take]

            gt_boxes = t["boxes"][p]  # (take, 4)
            gt_labels = t["labels"][p]  # (take,)

            # 加噪
            box_n = torch.randn(gt_boxes.shape, device=device) * box_noise_ratio
            noisy = (gt_boxes + box_n).clamp(0, 1)

            label_n = torch.rand(gt_labels.shape, device=device) < label_noise_ratio
            random_l = torch.randint(0, num_classes, gt_labels.shape, device=device)
            nl = torch.where(label_n, random_l, gt_labels.long())

            # 用 GT 框坐标编码 DN query (简化: 直接拼坐标和标签)
            # 实际 DINO: 用 MLP 将 (bbox, label) 映射到 query 空间
            dn_q = torch.cat([
                gt_boxes.repeat(1, C // 16)[:, :C - 4],  # 用 GT 位置信息编码
                gt_boxes.repeat(1, 1),  # 4 dims for box
            ], dim=1)[:, :C]
            # 简化: 直接用 GT box 扩展填充
            dn_q = torch.zeros(take, C, device=device)
            dn_q[:, :8] = gt_boxes.repeat(1, 2)  # 前 8 维放 box info

            if take < dn_num:
                # 补齐
                pad_q = torch.zeros(dn_num - take, C, device=device)
                dn_q = torch.cat([dn_q, pad_q], dim=0)
                nl = torch.cat([nl, torch.zeros(dn_num - take, dtype=torch.long, device=device)])
                noisy = torch.cat([noisy, torch.zeros(dn_num - take, 4, device=device)])

            dn_query_list.append(dn_q)
            dn_label_list.append(nl)
            dn_box_list.append(noisy)
            dn_mask[i, :take] = True

    dn_queries = torch.stack(dn_query_list, dim=0)  # (B, dn_num, C)
    dn_labels_t = torch.stack(dn_label_list, dim=0)  # (B, dn_num)
    dn_boxes_t = torch.stack(dn_box_list, dim=0)     # (B, dn_num, 4)

    # 拼接 DN query embed
    dn_embed = dn_query_embed[:dn_num].unsqueeze(0).expand(B, -1, -1)  # (B, dn_num, C)
    dn_queries = dn_queries + dn_embed

    # 拼接: (B, Nq+dn_num, C)
    padded_query_embed = torch.cat([query_embed, dn_queries], dim=1)

    # 构造新的 targets (DN 部分用 GT)
    new_targets = []
    for i in range(B):
        new_targets.append({
            "boxes": torch.cat([targets[i]["boxes"], dn_boxes_t[i][dn_mask[i]]], dim=0),
            "labels": torch.cat([targets[i]["labels"], dn_labels_t[i][dn_mask[i]]], dim=0),
            "is_dn": torch.cat([
                torch.zeros(targets[i]["boxes"].shape[0], dtype=torch.bool, device=device),
                dn_mask[i][dn_mask[i]],
            ]),
        })

    # 注意力 mask: DN queries 互相可见但原始不能看 DN
    total_q = Nq + dn_num
    attn_mask = torch.zeros(B, total_q, total_q, dtype=torch.bool, device=device)
    for i in range(B):
        # 原始 → DN: 不可见
        attn_mask[i, :Nq, Nq:] = True
        # 原始 → 原始: 可见（不需要 mask）
        # DN → DN: 可见（不需要 mask）
        # DN → 原始: 可见（不需要 mask）

    dn_meta = {
        "dn_num": dn_num,
        "dn_mask": dn_mask,
    }

    return padded_query_embed, new_targets, attn_mask, dn_meta


def dn_post_process(hs, targets, dn_meta):
    """DN 后处理: 去除 DN 部分的结果。

    Args:
        hs: (B, Nq+dn_num, C) 模型输出
        targets: list of dict (包含 DN ones)
        dn_meta: DN 元信息

    Returns:
        hs_clean: (B, Nq, C)
        targets_clean: list of dict (去除 DN ones)
    """
    dn_num = dn_meta["dn_num"]
    B = hs.shape[0]

    # 分离
    hs_clean = hs[:, :-dn_num] if dn_num > 0 else hs

    targets_clean = []
    for i, t in enumerate(targets):
        is_dn = t.get("is_dn", None)
        if is_dn is None:
            targets_clean.append({"boxes": t["boxes"], "labels": t["labels"]})
            continue
        nondn = ~is_dn
        targets_clean.append({
            "boxes": t["boxes"][nondn],
            "labels": t["labels"][nondn],
            "image_id": t.get("image_id", torch.tensor([i])),
        })

    return hs_clean, targets_clean
