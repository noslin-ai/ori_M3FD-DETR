# 代码修改记录 (CHANGELOG)

> 项目: M3F-DETR | 基于: `项目问题分析报告.md`

---

---

## v0.7.0 (experiment) — rush_v2 完整 60 轮训练 + 首次非零 mAP + 阈值/NMS 扫描 + 新提交

**日期:** 2026-08-26  
**交接:** 本次为纯实验/评估操作，未改代码；记录续训、诊断、扫描、提交全流程，便于其他人接力。

### 背景

- 8/23 的 rush_v2 训练在 epoch 17 被中断（见 `.ipynb_checkpoints/rush_v2_train-checkpoint.log`）；8/24 notebook 重试因 CUDA 不可见失败（`torch.cuda.is_available()=False`，`rush_v2_train.log` 止于 CUDA 禁用提示）。
- 8/26 容器重启后 GPU 恢复（RTX 5090；base python torch 2.12.1+cu130），当时无任何训练进程在跑。

### 操作流程（复现命令）

1. 续训（从 `checkpoints/rush_v2/latest.pth` 的 epoch 15 继续，至 60 轮跑完）：
   ```bash
   python -u train.py --config configs/rush_v2.yaml --fold 1 --resume checkpoints/rush_v2/latest.pth
   ```
   日志：`rush_v2_resume_train.log`（约 30 秒/epoch，约 35 分钟跑完剩余 46 轮）。
2. 诊断（最终模型）：
   ```bash
   python tools/diagnose_predictions.py --checkpoint checkpoints/rush_v2/latest.pth
   python tools/diagnose_iou.py --checkpoint checkpoints/rush_v2/latest.pth --conf-threshold 0.01
   ```
3. 阈值/NMS 扫描（全量训练集 2000 张，40 组）：
   ```bash
   python tools/sweep_eval.py --checkpoint checkpoints/rush_v2/latest.pth --data-root data/train --batch-size 4
   ```
   日志：`sweep_rush_v2.log`。
4. 生成测试提交（用扫描最佳配置）：
   ```bash
   python inference.py --checkpoint checkpoints/rush_v2/latest.pth --data-root data/test --output submission_rush_v2_ep60 --conf-threshold 0.001 --nms-iou 0.5 --max-det 100 --zip submission_rush_v2_ep60.zip
   ```

### 训练结果

| 指标 | epoch 15（续训起点） | epoch 60（结束） |
|------|------|------|
| loss | 2.45 | 1.84 |
| cls | 0.093 | 0.074 |
| ce | 1.03 | 0.59 |
| bbox | 0.042 | 0.025 |
| giou | 0.81 | 0.67 |

- 验证（EMA，每 10 轮一次）：epoch 20 = mAP50-95 0.0001 / mAP50 0.0004（历史首次非零）；epoch 30 归零；epoch 40 = mAP50 0.0003；epoch 60 = mAP50-95 0.0001 / mAP50 0.0007（New Best）。
- checkpoint：`checkpoints/rush_v2/{best,latest,final}.pth`（8/26 07:30，605M，cfg: swin_tiny / hidden 256 / 300 queries / use_dn=False）。

### 诊断结论（最终模型）

- 类别分布：pred 仅 class 10/8/0/6；GT 中 class 4（33 个）和 class 3（2 个）完全没出 → 少数类召回缺失。
- 定位：GT recall@IoU0.5（同类别）= 22.5%；每图最佳框 IoU ≈ 0.53，但最高分框平均 IoU 仅 0.14 → 置信度排序与定位质量脱节。
- 框形态：预测框偏高偏大（h 均值 0.157 vs GT 0.114）、中心偏右（cx 0.55 vs 0.48）。
- 置信度：去背景 sigmoid 均值约 0.21（EMA 约 0.24）；conf 大于 0.5 的 query 仅 0~3.6%。

### 阈值/NMS 扫描结果（40 组，全量训练集，数值偏乐观）

| conf | nms | preds | mAP50-95 | mAP50 |
|------|-----|-------|----------|-------|
| 0.001 | 0.50 | 198802 | **0.0022** | **0.0096** |
| 0.001 | 0.60 | 199820 | 0.0021 | 0.0092 |
| 0.080 | 0.60 | 196324 | 0.0020 | 0.0077 |
| 0.005 | 0.50 | 198795 | 0.0020 | 0.0085 |

- 结论：低阈值 + nms 0.5~0.6 最优；提高阈值不改善 mAP（因置信度排序差）。
- 注意：sweep 在含训练图的 2000 张上评估，仅用于配置间相对比较，不代表测试集真实分数。

### 提交

- `submission_rush_v2_ep60/`：1000 个 txt（格式 `class cx cy w h conf`），平均约 99.7 框/图。
- `submission_rush_v2_ep60.zip`（1.9MB），可直接上传。
- 本次用 raw model（非 EMA）与 sweep 一致；如需对比可加 `--use-ema` 再生成一版。

### 结论与接力建议

- 管线全通、训练收敛正常，但模型精度仍为 mAP 约 0.1~0.2%（验证/训练集）量级；主要瓶颈是**定位精度、少数类召回、置信度标定**。
- 下一步建议（按优先级）：
  1. 定位：检查 box head / 提高 GIoU 权重或延长训练；换更大预训练 backbone（swin_large）做 A/B。
  2. 少数类：class 4/3 样本极少，考虑类别重采样、复制增强或 focal 调参。
  3. 置信度排序：考虑 score 校准或排序损失。
  4. 用 EMA checkpoint 与 raw 各出一版提交对比。
- 服务器 GPU 当前空闲，可直接续跑；注意容器重启会杀掉 nohup 进程，但 checkpoint 每 5 轮保存。

---

## v0.6.12 (patch) — 提交后处理：同类别 NMS + 阈值扫描（补充记录）

**日期:** 2026-08-23  
**说明:** 该版本改动此前未写入 CHANGELOG，现补充记录，方便交接。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `engine/evaluator.py` | 增强 | 新增 `class_aware_nms`；`evaluate_model` 支持 `nms_iou` 参数 |
| `evaluate.py` | 增强 | 新增 `--nms-iou` 参数 |
| `inference.py` | 修复/增强 | 提交生成加入同类别 NMS（默认 IoU 0.6）；完善测试集提交 txt 生成 |
| `tools/sweep_eval.py` | 新增 | 阈值 × NMS IoU 扫描脚本，输出每组 mAP |

### 验证

- 提交：`submission_rush_v2_nms/` + `submission_rush_v2_nms.zip`（8/23 23:30）。
- 后续 v0.7.0 的 sweep 与提交即基于该后处理链路。

---
## v0.6.11 (patch) — 用正样本 CE 替代手写类别权重，修复单类塌缩

**日期:** 2026-08-22  
**目标:** v0.6.10 后模型从 class 0/2 塌缩转为几乎全预测 class 6，说明手写类别权重仍会把模型推向单一类别。改为对 Hungarian 匹配到的正样本 query 增加前景 CE 分类损失。

### 现场证据

```text
pred: [(6, 6400)]
gt:   [(10, 102), (6, 68), (8, 67), ...]
GT recall@IoU0.5: any=0.0000 same_cls=0.0000
```

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/losses/dino_loss.py` | 增强 | 对匹配到的正样本 query 增加 `loss_ce = cross_entropy(matched_logits, matched_labels)` |
| `train.py` | 增强 | 从配置读取 `loss.cost_ce` |
| `engine/trainer.py` | 增强 | 训练日志新增 `ce=` 分项 |
| `configs/debug.yaml` / `configs/rush_v2.yaml` | 调参 | 移除手写 `class_weights`，改用 `cost_ce: 0.5`，`cost_class: 1.0`，`focal_alpha: 0.5` |

### 原因

Sigmoid focal 主要负责前景/背景和多标签式分类信号，但在当前短 debug 训练中容易被负样本和类别频次带偏。正样本 CE 只作用于 Hungarian 匹配到的 query，不影响 unmatched query，可更直接地教模型“这个框对应哪个类别”。

---

## v0.6.10 (patch) — 类别权重仅作用于正样本，避免少数类负梯度被放大

**日期:** 2026-08-22  
**目标:** 修复 v0.6.9 中 class-weighted focal 的副作用：提高 class 10/6/8 权重时，同时放大了这些类在大量 unmatched query 上的负样本损失。

### 现场证据

```text
pred: [(0, 6336), (2, 64)]
gt:   [(10, 102), (6, 68), (8, 67), ...]
mAP@50: 0.0001
```

v0.6.9 后 mAP@50 已出现极小非 0，但类别分布仍几乎只落在 class 0/2。原因之一是类别权重被乘到了正负样本全部 loss 上。对于 sigmoid focal，未匹配 query 对每个前景类都是负样本；若 class 10 权重更高，它在海量负样本上受到更强抑制，抵消了正样本增益。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/losses/focal_loss.py` | 修复 | `class_weights` 只作用于 `targets==1` 的正样本项，负样本权重保持 1 |

### 核心逻辑

```python
pos_weights = 1.0 + (class_weights - 1.0) * targets
loss = loss * pos_weights
```

### 下一步

重新训练 debug，观察 `pred` 类别分布是否开始出现 class 10/6/8；若 mAP@50 继续非 0，再进入正式 `rush_v2` 长训。

---

## v0.6.9 (patch) — 针对 class 0/2 塌缩加入类别加权 focal

**日期:** 2026-08-22  
**目标:** 在定位 recall 已有改善但 mAP 仍为 0 的情况下，处理预测类别几乎集中到 class 0/2 的分类塌缩。

### 现场证据

```text
GT recall@IoU0.5: any=0.0730 same_cls=0.0730
pred: [(0, 6226), (2, 174)]
gt:   [(10, 102), (6, 68), (8, 67), (0, 43), ...]
conf > 0.1: 923 / 9600
```

结论：定位比上一轮明显改善，前景置信度也抬升；但预测类别严重偏向 class 0/2，和 GT 主要类别不一致，导致 AP 仍难以非 0。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/losses/focal_loss.py` | 增强 | 新增 `class_weights`，支持对不同前景类的 focal loss 加权 |
| `models/losses/dino_loss.py` | 增强 | `DINOLoss` 透传 `class_weights` 到 `FocalLoss` |
| `train.py` | 增强 | 从配置读取 `loss.class_weights` |
| `configs/debug.yaml` | 调参 | `cost_class=2.0`、`focal_alpha=0.75`，并降低 class 0/2 权重、提高 class 10/6/8 权重 |
| `configs/rush_v2.yaml` | 调参 | 同步分类塌缩修复参数 |

### 新分类权重

```yaml
class_weights: [0.5, 1.0, 0.7, 1.2, 1.2, 1.0, 2.0, 1.0, 2.0, 1.2, 2.5, 1.0]
```

含义：降低当前过度预测的 class 0/2 权重，提高 GT 中更常见但模型没学到的 class 10/6/8 权重。

### 下一步

```bash
python train.py --config configs/debug.yaml --fold 1
python tools/diagnose_iou.py --checkpoint checkpoints/debug/latest.pth --conf-threshold 0.01
python tools/diagnose_predictions.py --checkpoint checkpoints/debug/latest.pth
python evaluate.py --checkpoint checkpoints/debug/latest.pth --data-root data/train --conf-threshold 0.01
```

重点观察 `pred` 类别分布是否从 class 0/2 扩散到 class 10/6/8。

---

## v0.6.8 (patch) — 根据 IoU 诊断重新加强定位损失权重

**日期:** 2026-08-22  
**目标:** 在总 loss 已进入 1 左右但 mAP 仍为 0 的情况下，把优化重点重新转向定位质量。

### 现场证据

```text
每图最佳 IoU(不看类别): 0.1735
每图最佳 IoU(同类别):   0.0483
GT recall@IoU0.3: any=0.0159 same_cls=0.0000
GT recall@IoU0.5: any=0.0000 same_cls=0.0000
pred: [(0, 6336), (2, 64)]
gt:   [(10, 102), (6, 68), (8, 67), ...]
```

结论：框定位还远不到 AP50 要求，类别也明显塌到 class 0。先加强定位项，让候选框至少能接近 GT；否则分类即使改善，mAP 仍然无法非 0。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `configs/debug.yaml` | 调参 | loss 权重改为 `cost_class=1, cost_bbox=5, cost_giou=2` |
| `configs/rush_v2.yaml` | 调参 | 同步恢复更强定位权重，同时按当前 loss 分项估算总 loss 仍约 1.9 |
| `tools/diagnose_iou.py` | 增强 | 框分布新增逐坐标 `cx/cy/w/h` 的 mean/min/max，便于判断中心偏移或宽高偏置 |

### 下一步

重新训练 debug 后观察：

```bash
python train.py --config configs/debug.yaml --fold 1
python tools/diagnose_iou.py --checkpoint checkpoints/debug/latest.pth --conf-threshold 0.01
python tools/diagnose_predictions.py --checkpoint checkpoints/debug/latest.pth
python evaluate.py --checkpoint checkpoints/debug/latest.pth --data-root data/train --conf-threshold 0.01
```

预期优先看到 `GT recall@IoU0.1/0.3` 上升；若 `any IoU` 上升但 `same_cls` 仍为 0，再处理类别塌缩。

---

## v0.6.7 (tool) — 新增 IoU 诊断脚本定位 mAP=0 的框匹配问题

**日期:** 2026-08-21  
**目标:** 在 loss 已降到约 1、前景置信度开始抬升但 mAP 仍为 0 的情况下，区分“类别预测错误”和“框 IoU 过低 / 坐标尺度异常”。

### 新增工具

| 文件 | 类型 | 摘要 |
|------|------|------|
| `tools/diagnose_iou.py` | 新增 | 统计 top-k 预测框与 GT 的 class-agnostic / same-class 最大 IoU、GT recall@IoU、预测/GT 框分布和类别分布 |

### 使用命令

```bash
python tools/diagnose_iou.py --checkpoint checkpoints/debug/latest.pth --conf-threshold 0.01
```

重点看：

- `每图最佳 IoU(不看类别)`：若仍接近 0，优先查框坐标/resize/box head；
- `每图最佳 IoU(同类别)`：若远低于不看类别 IoU，说明分类错得更多；
- `GT recall@IoU0.5 same_cls`：接近 0 时 mAP@50 必然为 0；
- `pred` vs `gt` 框分布：快速发现预测框过小、过大或中心偏移。

---

## v0.6.6 (patch) — 调整 loss 量纲、训练日志与评估候选数

**日期:** 2026-08-21  
**目标:** 处理“mAP 仍为 0、总 loss 长期偏高”的训练可观测性和损失量纲问题，避免只看总 loss 误判模型状态。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/losses/focal_loss.py` | 修复 | Focal loss 先对类别维度取平均，再按 GT 数量归一化，避免分类项随类别数线性放大 |
| `engine/trainer.py` | 增强 | 训练日志新增 `cls/bbox/giou` 三项平均值，便于判断 loss 高来自分类还是定位 |
| `engine/evaluator.py` | 优化 | 每张图按置信度最多保留 100 个预测框，与 COCOeval `maxDets=100` 对齐，减少低分候选干扰和评估开销 |
| `configs/debug.yaml` | 调整 | debug 改为 `swin_tiny + 300 queries + num_workers=2`，并降低损失权重量纲 |
| `configs/rush_v2.yaml` | 调整 | loss 权重改为 `class=1, bbox=2, giou=1`，让总 loss 更接近 1~2 的可观察区间 |

### 注意

总 loss 不是越低越好。若只是把权重整体缩小，loss 可以马上变成 1 以下，但 mAP 不会因此变好。本轮修改重点是：

1. 修正分类项的类别维度量纲；
2. 降低 900 query / 大 backbone 带来的 debug 噪声；
3. 打印分项 loss，后续按根因调参。

### 建议验证命令

```bash
python -m py_compile models/losses/focal_loss.py engine/trainer.py engine/evaluator.py
python train.py --config configs/debug.yaml --fold 1
python tools/diagnose_predictions.py --checkpoint checkpoints/debug/latest.pth
python tools/probe_forward.py --checkpoint checkpoints/debug/latest.pth
python evaluate.py --checkpoint checkpoints/debug/latest.pth --data-root data/train --conf-threshold 0.01
```

训练日志中优先观察：

- `avg` 是否进入约 1~2；
- `cls` 是否持续下降；
- `bbox/giou` 是否仍明显偏高；
- `mAP@50` 是否先于 `mAP@50-95` 出现非 0。

---

## v0.6.5 (patch) — 移除 sigmoid focal 中的显式背景类监督

**日期:** 2026-08-20  
**目标:** 继续处理“背景 logit 长期最高、前景置信度难以抬升、mAP 仍为 0”的分类训练目标冲突。

### 问题定位

v0.6.3 后，评估和推理已经改为只在前景类中取最大 sigmoid 分数；v0.6.4 后，decoder query 也开始分化。但训练损失仍把未匹配 query 的目标设为 `background=1`：

```python
target_classes = torch.full((B, Q), self.num_classes)
target_onehot.scatter_(1, target_classes.reshape(-1, 1), 1)
```

这会让 900 个 query 中绝大多数 unmatched query 都强烈监督背景类为正样本，继续把模型推向背景 logit 最高的吸引子。对于 sigmoid focal 目标，更合理的做法是只监督前景 `C` 类：匹配 query 对应类别为 1，未匹配 query 所有前景类为 0，不再需要显式背景类。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/losses/dino_loss.py` | 修复 | 分类目标由 `num_classes+1` 改为仅前景 `num_classes`；未匹配 query 全 0 |
| `models/losses/dino_loss.py` | 兼容 | 模型仍输出 `num_classes+1`，训练时只取 `pred_logits[:, :, :num_classes]`，旧 checkpoint 结构可继续加载 |

### 验证命令

```bash
python -m py_compile models/losses/dino_loss.py
python train.py --config configs/debug.yaml --fold 1
python tools/diagnose_predictions.py --checkpoint checkpoints/debug/latest.pth
python tools/probe_forward.py --checkpoint checkpoints/debug/latest.pth
python evaluate.py --checkpoint checkpoints/debug/latest.pth --data-root data/train --conf-threshold 0.01
```

观察重点：

- 去背景后前景 sigmoid 最大置信度是否高于之前的 `max≈0.0537`；
- `conf > 0.1` 的 query 是否开始出现；
- `[LOGITS] query_std`、`[BOX] query_std` 是否继续提升；
- 若 mAP 仍为 0，再优先检查候选框 IoU / 坐标尺度。

---

## v0.6.4 (patch) — 修复 decoder 未按 query_pos 使用 object query 导致跨 query 塌缩

**日期:** 2026-08-20  
**目标:** 根据 `probe_forward.py` 的输出，修复 decoder 输出几乎不随 object query 变化的问题，让不同 query 能学习不同检测槽位。

### 现场证据

```text
[DECODER] query_std=0.000121
[DEP]     hs 随 memory 变化: 0.209253 | hs 随 query 变化: 0.000135
[LOGITS]  query_std=0.000062
[BOX]     query_std=0.000021 img1_vs_img2_diff=0.154687
```

该结果说明 backbone、FPN、位置编码和输入图像依赖都不是完全失效；真正的问题是 decoder 输出对不同 query 几乎一致，导致 900 个 query 生成高度重复的类别和框。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/detector/transformer.py` | 修复 | 自定义 DETR 风格 decoder layer：`tgt` 作为内容向量，`query_embed` 作为 `query_pos` 注入 self-attention / cross-attention |
| `models/detector/transformer.py` | 兼容 | 保留 `decoder.layers.*`、`self_attn`、`multihead_attn`、`linear/norm/dropout` 等 state_dict 键名，便于加载旧 checkpoint |

### 根因说明

旧实现虽然从 `dino_detector.py` 传入了 `tgt = zeros_like(query_embed)`，但在 transformer 内部实际执行的是：

```python
hs = self.decoder(query_embed, memory, tgt_mask=tgt_mask)
```

这等于把 object query 直接当 decoder 内容输入，而不是作为 query position。标准 DETR/DINO 的接法应是：

```python
self_attn(q=tgt + query_pos, k=tgt + query_pos, v=tgt)
cross_attn(q=tgt + query_pos, k=memory + pos, v=memory)
```

本轮已按该逻辑重写 decoder forward，避免 query 信息在多层 cross-attention 和 LayerNorm 后被拉成同质输出。

### 验证命令

```bash
python -m py_compile models/detector/transformer.py
python tools/probe_forward.py --checkpoint checkpoints/debug/latest.pth
```

观察重点：

- `[DECODER] query_std` 应明显高于 `0.0001`；
- `[DEP] hs 随 query 变化` 应明显增大；
- `[LOGITS] query_std`、`[BOX] query_std` 应不再接近 0；
- 旧 checkpoint 可用于结构诊断，但正式效果仍建议重新训练，因为 decoder 前向语义已经改变。

---

## v0.6.3 (patch) — 修复 sigmoid 后处理仍被背景类吞掉导致评估 0 框

**日期:** 2026-08-20  
**目标:** 根据服务器 debug checkpoint 的诊断输出，修复“前景 sigmoid 已有低置信候选，但 evaluate 仍显示预测框 0”的后处理问题。

### 现场证据

```text
预测框: 0
真实框: 14417
softmax argmax == 背景 的比例: 1.0000
去背景后 sigmoid 最大置信度: mean=0.0506 max=0.0537
```

该结果说明模型并非完全没有前景响应；问题出在推理阶段把背景类也放进 `sigmoid().max()`，再用 `labels < num_classes` 过滤。由于背景 logit 始终最大，所有 query 都被提前丢弃，导致 `conf_threshold=0.001` 下仍然 0 框。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `engine/evaluator.py` | 修复 | 后处理由“含背景全类 sigmoid max + 过滤背景”改为“仅前景类 sigmoid max”，避免背景类吞掉全部候选 |
| `inference.py` | 修复 | 提交文件生成逻辑同步改为仅前景类取最大分数，保持评估和推理一致 |

### 验证命令

```bash
python evaluate.py --checkpoint checkpoints/debug/latest.pth --data-root data/train --conf-threshold 0.001
python tools/diagnose_predictions.py --checkpoint checkpoints/debug/latest.pth
```

预期 `预测框` 应从 0 变为非 0；若 mAP 仍为 0，则下一步继续检查候选框 IoU、坐标尺度和类别分布，而不是再停留在“无预测框”问题。

---

## v0.6.2 (patch) — 针对 mAP 持续为 0 的匹配、分类归一化与评估阈值修复

**日期:** 2026-08-14  
**目标:** 在 v0.6.0 已修复 0 框 / query 塌缩主链路的基础上，继续处理“验证有框但 mAP 仍为 0”或“epoch 后期又回到全背景”的剩余风险点。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/detector/matcher.py` | 修复 | Hungarian matcher 的分类代价由 `softmax` 改为 `sigmoid` 目标类概率，和当前 sigmoid Focal Loss 训练目标对齐；同时对 GIoU 和总代价矩阵加入 NaN/Inf 防护 |
| `models/losses/focal_loss.py` | 修复 | FocalLoss 新增 `normalizer` 参数，支持按 batch 内 GT 数量归一化 |
| `models/losses/dino_loss.py` | 修复 | 分类损失传入 `num_boxes` 作为归一化因子，避免正样本信号继续被 query 数量稀释 |
| `engine/evaluator.py` | 修复 | mAP 评估默认阈值由 `0.3` 降为 `0.001`；评估前对预测框做 clamp 和最小宽高保护 |
| `evaluate.py` | 增强 | 新增 `--conf-threshold` 参数，便于评估时显式调整候选框过滤阈值 |

### 根因补充

v0.6.0 已解决位置编码、pretrained、input_proj、FocalLoss 元素级平均、后处理不匹配等问题；但 mAP 仍可能为 0 的剩余原因包括：

1. **matcher 与 loss 不一致**：训练用 sigmoid Focal，但 Hungarian matcher 仍用 softmax 分类概率，背景类会和目标类强制竞争，可能导致正样本匹配质量差。
2. **分类 loss 仍按 query 平均**：`loss.sum(-1).mean()` 虽然不再除以 `num_classes`，但仍会按 `B × Q` 平均。单图约 7 个 GT、300 queries 时，正样本梯度仍偏弱。
3. **评估前阈值过高**：COCO AP 应尽量使用完整候选排序，默认 `conf_threshold=0.3` 会提前过滤低置信候选，可能让“有学习迹象但置信度未抬升”的模型评估为 0 框 / 0 mAP。
4. **坏框数值影响评估**：预测框越界或极小宽高会干扰 COCO 格式评估，已在评估转换阶段做 clamp 和最小宽高保护。

### 建议验证命令

```bash
python train.py --config configs/rush_v2.yaml --fold 1
python evaluate.py --checkpoint checkpoints/rush_v2/latest.pth --data-root data/train --conf-threshold 0.001
python tools/probe_forward.py --checkpoint checkpoints/rush_v2/latest.pth
python tools/diagnose_predictions.py --checkpoint checkpoints/rush_v2/latest.pth
```

### 观察重点

- `预测框数量` 是否稳定非 0；
- `[BOX] img1_vs_img2_diff` 是否明显非 0；
- `loss_class` 是否不再长期过小；
- 目标类 sigmoid 置信度是否逐步高于背景吸引子；
- `mAP@50` 是否先于 `mAP@50-95` 出现非 0。

---

## v0.6.1 (audit) — 按队长交接报告对齐代码：确认 v0.6.0 五处根因修复未被覆盖

**日期:** 2026-08-09  
**目标:** 以队长交接报告为准，先恢复并核验 v0.6.0 “0 框 / mAP=0”根因闭环代码，避免未验证的结构显存优化实验覆盖已闭环的关键修复。

### 本轮处理原则

1. **先修 bug 闭环，再做结构实验**：本轮不混入 P2/P3-P5 融合结构调整，优先保证队长已验证的训练链路修复保持完整。
2. **不新增 Word 报告**：修改说明直接写入 `CHANGELOG.md` 和 `项目问题分析报告.md`。
3. **保留 v0.6.0 baseline**：后续训练应先跑 `configs/rush_v2.yaml`，确认类别不平衡实验结果，再单独开分支做显存优化结构 A/B。

### 已核验的五处根因修复

| 根因 | 文件 | 当前状态 |
|------|------|:--:|
| 位置编码失效：`cumsum(zeros)` 导致所有位置编码相同 | `models/detector/position_encoding.py` | 已确认使用 `torch.ones(...)` 后再 `cumsum` |
| `pretrained` 被忽略，rush 配置预训练从未生效 | `models/backbone/rgb_backbone.py` | 已确认 `pretrained` 透传 `timm.create_model`，并支持本地 `.safetensors/.pth` 权重路径 |
| decoder 输入特征未归一化，FPN 大方差淹没位置编码 | `models/detector/dino_detector.py` | 已确认 `features[-1]` 先经过 `input_proj[0]`（Conv + GroupNorm）再进入 position encoding / decoder |
| 分类梯度被稀释，模型只学框不学分类 | `models/losses/focal_loss.py` | 已确认 `return loss.sum(-1).mean()`，按 query 汇总类别维度 |
| 后处理与 sigmoid Focal 训练目标不匹配 | `engine/evaluator.py` / `inference.py` | 已确认使用 `sigmoid + 含背景全类 max/argmax + 过滤背景类 + 置信度阈值` |

### 同步确认的工具与配置

| 文件 | 状态 |
|------|------|
| `tools/probe_forward.py` | 保持队长版本：用于定位 decoder/query/box 是否塌缩，包含 `[DEP]` 和 `[BOX]` 检测 |
| `tools/diagnose_predictions.py` | 保持队长版本：用于诊断 logits、背景占比和 sigmoid 置信度分布 |
| `configs/rush_v2.yaml` | 保持队长版本：`queries: 300` + `focal_alpha: 0.5`，用于类别不平衡实验 |

### 后续执行建议

```bash
python train.py --config configs/rush_v2.yaml --fold 1
python tools/probe_forward.py --checkpoint checkpoints/rush_v2/latest.pth
python tools/diagnose_predictions.py --checkpoint checkpoints/rush_v2/latest.pth
```

重点观察：

- `[BOX] img1_vs_img2_diff`：若约等于 0，优先改 decoder query_pos 接法；
- 验证框数量：epoch 10/20 是否保持非 0；
- 背景类 logit 与目标类 logit 分布：判断是否仍被类别不平衡吸回全背景；
- `latest.pth`：mAP 仍为 0 时不要等 `best.pth`，评估/诊断用 latest。

### 注意

此前提出的“P2 保留 RGB，P3-P5 三模态 CMFA 融合、Swin-Small、hidden_dim 192、decoder 4、queries 300”的显存优化方案，适合作为后续独立实验分支；本轮按队长报告对齐代码时不将其混入主线，以免影响 v0.6.0 根因闭环的可解释性。

---

## v0.6.0 (patch) — 训练"0 框 / mAP=0"根因闭环：五处修复 + 诊断工具 + 不平衡实验配置

**日期:** 2026-08-08
**目标:** 解决 rush 正式训练后验证 "预测 0 个框 / mAP=0"（错误 #8 复发），恢复可训练、可出框、可评估链路

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/detector/position_encoding.py` | 修复 | 位置编码 `mask = torch.zeros(...)` → `torch.ones(...)`：原实现坐标全 0，每个空间位置拿到相同编码，decoder 完全失去空间信息（首因） |
| `models/backbone/rgb_backbone.py` | 修复 | `pretrained` 真正透传 `timm.create_model`（原来硬编码 False，rush 的 `pretrained: true` 从未生效）；新增本地权重路径支持（`pretrained=<path>`，兼容 safetensors/pth，适配无外网服务器） |
| `models/detector/dino_detector.py` | 修复 | decoder 特征先过 `input_proj`（Conv+GroupNorm）归一化：FPN 输出 std≈8~11 淹没位置编码（≈0.1）且注意力 logits 饱和，训练中 query 互相塌缩、所有框相同 |
| `models/losses/focal_loss.py` | 修复 | 分类损失 `loss.mean()` → `loss.sum(-1).mean()`：原实现平均到 900×13 个元素，分类梯度被稀释约 11700 倍，模型只学框不学分类，最终全背景 |
| `engine/evaluator.py` | 修复 | ① 后处理 softmax→sigmoid，与 sigmoid Focal 训练目标对齐；② 最终版"含背景类全类 argmax + 阈值"，背景胜出的 query 直接丢弃；③ per-class COCOeval 空 stats 崩溃保护 |
| `inference.py` | 修复 | 同 evaluator 后处理（sigmoid + 含背景 argmax + 阈值） |
| `configs/rush.yaml` | 注释 | 补充 pretrained 本地路径用法 |
| `configs/rush_v2.yaml` | 新增 | 类别不平衡实验配置：queries 900→300、focal_alpha 0.25→0.5（待验证） |
| `tools/diagnose_predictions.py` | 新增 | 诊断：逐类 logits 均值、softmax 背景占比、sigmoid 置信度分布（raw vs EMA） |
| `tools/probe_forward.py` | 新增 | 前向探针：逐环节方差定位塌缩（backbone/FPN/PE/query/decoder/logits/box），含 [DEP] 依赖测试与 [BOX] 图像依赖检测 |

### 根因链（错误 #8 闭环）

1. **位置编码失效**：`cumsum(zeros)` 导致所有空间位置编码相同 → decoder 无空间信息
2. **pretrained 未生效**：`timm.create_model` 硬编码 `pretrained=False` → Swin-Tiny 全程随机初始化
3. **特征未归一化**：`input_proj` 已定义但从未接线；FPN 输出 std≈8~11 淹没位置信号、注意力饱和 → 训练 5 轮内 query 塌缩（所有 query 输出相同 logits/框）
4. **分类损失被稀释**：`FocalLoss.mean()` 平均到 900×13 元素，分类梯度远小于 box 梯度 → 只学框不学分类 → 全背景（bg logit +1.5、目标类 -3）
5. **后处理不匹配**：softmax+argmax 与 sigmoid Focal 不兼容，且早期版本忽略背景类竞争

### 验证记录（服务器实测）

- 随机权重探针：decoder query_std=0.2375（架构健康）
- 修复前旧 checkpoint：decoder query_std=0.000006、[DEP] 随 query 变化 0.000006、[BOX] 全 query 相同
- input_proj 修复后重训（epoch 20）：decoder query_std=0.50、[BOX] query_std=0.184（塌缩解决）
- focal 修复后重训：epoch 10 验证 2143 框（数量级正确，mAP 仍 0）；epoch 20 又塌回 0 框（类别不平衡，见遗留）

### 验证清单

- [x] 本地 `py_compile` 通过（evaluator/inference/focal_loss/dino_detector/rgb_backbone/position_encoding/probe/diagnose）
- [x] 服务器探针确认 decoder 塌缩解决（query_std 0.50）
- [x] v0.5.0 遗留项"前 20 epoch val mAP > 0"已推进：从 0 框 → 2143 框，但 mAP 仍为 0
- [ ] `rush_v2` 重训 10~20 轮：验证框数正常且 mAP > 0（当前遗留）

### 遗留问题（接力必读）

1. **类别极端不平衡**：900 queries vs 约 7 GT/图（约 126:1），分类在"全背景"吸引子与"乱出框"之间摆动。实验配置 `rush_v2`（300 queries + alpha 0.5）待验证。
2. **框的图像依赖未确认**：探针 `[BOX] img1_vs_img2_diff` 待跑；若 ≈0 说明框仍是模板，需将 decoder 改为标准 DETR 接法（tgt=zeros + query_embed 作为 query_pos 逐层加入），并考虑多尺度特征（当前 decoder 只用 P5 单尺度）。
3. **训练分辨率**：backbone 硬编码 `img_size=(384, 640)`，数据集实际输出 384×640；配置 `input.width/height: 1024×640` 未生效。提分辨率需同步改 backbone 与数据集。
4. **best.pth 仅在 mAP>0 时保存**：当前一直为 0 所以只有 latest.pth；评估/推理请用 latest.pth。
5. **AMP FutureWarning**：`torch.cuda.amp.autocast` 弃用警告，低优先级（可换 `torch.amp.autocast("cuda", ...)`）。

---

## v0.5.0 (minor) — 快速出分改造：pretrained 开关 + 训练可靠性修复 + rush 配置

**日期:** 2026-08-08  
**目标:** 尽快产出一版可提交、非零 mAP 的结果（先出分，再冲精度）

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/m3f_detr.py` | 增强 | `M3F_DETR` 新增 `pretrained` 参数并透传 `RGBBackbone`，支持加载 ImageNet 预训练权重 |
| `train.py` | 修复 | ① `use_dn` 默认关闭并写入 checkpoint cfg；② `compile` 默认关闭（避免 `_orig_mod.` 键名导致推理加载失败）；③ 模型按 `config.model.use_dn / pretrained` 创建 |
| `inference.py` / `evaluate.py` | 修复 | `use_dn` 从 checkpoint cfg 读取（`cfg.get("use_dn", True)` 兼容旧 checkpoint）；加载时剥离 `_orig_mod.` / `module.` 前缀 |
| `utils/checkpoint.py` | 加固 | 新增 `strip_state_dict_prefixes()`，保存/加载统一剥离 compile/DDP 前缀，保证 checkpoint 与裸模型互操作 |
| `configs/m3f_dino.yaml` / `configs/train.yaml` | 修复 | `freeze: ["rgb"]` → `["rgb_backbone"]`（原冻结阶段因模块名不匹配从未生效）；`train.yaml` 的 `compiler` 键名统一为 `compile` |
| `configs/rush.yaml` | 新增 | 快速出分配置：swin_tiny + pretrained + 1024×640 + 60 epochs + EMA + compile off |

### 背景与说明

- **冻结 backbone 阶段此前是静默 no-op**：配置文件写 `freeze: ["rgb"]`，但模型属性名为 `rgb_backbone`，`hasattr` 永远不命中，第一阶段"冻结 RGB"从未执行。
- **use_dn 分支无有效监督**：DN query 在进分类头前被 `dn_post_process` 切除，训练损失用的是原始 targets，DN 拿不到任何梯度，只是白占参数和计算。统一关闭，训练/推理结构由 checkpoint cfg 驱动保持一致。
- **torch.compile 默认关闭**：编译后 state_dict 键名可能带 `_orig_mod.` 前缀，裸模型 strict 加载会失败。`utils/checkpoint.py` 已加前缀剥离做兜底，确认"compile 训练 → inference 加载"链路 OK 后可按需开启 `compile: true`。
- **pretrained 只影响初始化**：推理时权重由 checkpoint 覆盖，因此不需要在推理侧配置。

### 验证清单

- [x] 本地 `py_compile` 通过（train/inference/evaluate/m3f_detr/checkpoint）
- [ ] 服务器 `git pull` 后 `python -m py_compile` 通过
- [ ] `python tools/split_5fold.py --data-root data/train` 生成 splits
- [ ] `python train.py --config configs/rush.yaml --fold 1` 前向/反向正常、loss 下降
- [ ] 前 20 epoch val mAP > 0
- [ ] `python inference.py --checkpoint checkpoints/rush/best.pth --data-root data/test --output submission_rush` 生成 1000 个文件且非空

---

## v0.3.3 (patch) — 修复 Transformer 接口对齐

**日期:** 2026-07-30  
**触发:** v0.3.2 backbone 修复后，错误下移到 Transformer
- Test 2: `pos.shape[-1]=512 ≠ hidden_dim=256`（位置编码维度不匹配）
- Test 3-4: `attn_mask` 未初始化（`use_dn=False` 时）

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/detector/dino_detector.py` | 修复 | `attn_mask` 显式初始化为 `None`；新增位置编码维度断言 |

### 详细修改

#### `models/detector/dino_detector.py`

**变更 1 — attn_mask 初始化（UnboundLocalError）**

第 132 行: DN 训练分支前显式 `attn_mask = None`，确保 `use_dn=False` / 推理时变量始终定义。

**变更 2 — 位置编码维度断言**

第 121-124 行: 新增 `assert pos_embed.shape[1] == self.hidden_dim`，在不匹配时打印含修复建议的明确错误（`num_pos_feats` 应为 `hidden_dim//2`）。

---

## v0.3.2 (patch) — 修复 timm Swin NHWC → NCHW 格式转换

**日期:** 2026-07-30  
**根因确认:** timm Swin `features_only=True` 返回 NHWC `[B,H,W,C]`，M3F-DETR 全线按 NCHW `[B,C,H,W]` 设计，`rgb_backbone.py` 的 `forward()` 缺少 `permute(0,3,1,2)` 导致所有下游模块维度解读错位。此前观察到的"48/80/160 通道"均为被误读的空间维度。`feature_info.channels()` 和 backbone 实际输出通道数自始至终正确。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/backbone/rgb_backbone.py` | 修复 | `forward()` 添加 NHWC→NCHW 转换；移除 v0.3.1 错误诊断引入的 `_verify_channels()` |
| `models/m3f_detr.py` | 加固 | 恢复 `len(backbone_channels)==4` 断言；更新注释 |

### 详细修改

#### `models/backbone/rgb_backbone.py`

**变更前:**
```python
def forward(self, x):
    feats = self.backbone(x)   # timm Swin 返回 [B,H,W,C]
    return feats               # 直接透传 NHWC → 下游全部解读错位
```

**变更后:**
```python
def forward(self, x):
    feats = self.backbone(x)          # timm Swin: [B,H,W,C]
    out = []
    for f in feats:
        if f.dim() == 4:
            f = f.permute(0, 3, 1, 2).contiguous()  # NHWC → NCHW
        out.append(f)
    return out
```

同时移除了 v0.3.1 的 `_verify_channels()` 方法。该方法基于错误的诊断假设（"channels 声明值错误"），它在 NHWC 格式下用 `f.shape[1]` 读到了 H 而非 C，从而产生了"通道数不一致"的假阳性。实际 `feature_info.channels()` 完全正确，移除该额外机制，`self.channels` 恢复为直接读取 `feature_info.channels()`。

#### `models/m3f_detr.py`

恢复 `assert len(backbone_channels) == 4` 断言（v0.3.1 曾移除）。因根因（NHWC）已修正，通道层数不再需要自适应。

### 验证清单

- [x] `RGBBackbone(backbone_name='swin_small')` 正确输出 NCHW 格式
- [x] 服务器 timm 直接测试验证：forward 原始输出为 NHWC，修复后为 NCHW
- [ ] 服务器 `python test.py` 全部 4 项测试通过
- [ ] 下游模块（IR/Depth/CMFA/FPN）通道链路全部对齐
- [ ] CMFA OOM 在 use_downsample 模式下不触发

---

## v0.3.1 (patch) — 修复 Backbone 通道声明与实际输出不一致

**日期:** 2026-07-29  
**触发:** v0.3.0 运行时断言捕获 `RGB 特征层 1 通道 48 ≠ 期望 192`  
**根因:** `timm.feature_info.channels()` 返回值与 backbone 实际 forward 输出不一致（timm 版本/配置差异导致）

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/backbone/rgb_backbone.py` | 修复 | `self.channels` 不再信任 `feature_info.channels()` 声明，改为微型 forward 验证后按实际值修正 |

### 详细修改

#### `models/backbone/rgb_backbone.py` — 通道自检机制

**变更前:**
```python
self.channels = self.backbone.feature_info.channels()
```
仅读取 timm 声明的理论值，不验证实际 forward 输出。

**变更后:**
1. 显式设置 `self.backbone.feature_info.out_indices = (0, 1, 2, 3)` — 防止 timm 版本不同导致默认 out_indices 不一致
2. 新增 `_verify_channels()` — 用 64×64 微型 tensor 做一次 forward，读取实际输出通道
3. 若实际输出层数/通道数与声明不一致，打印警告并按实际值修正
4. 若微型 forward 失败（极小概率），回退到声明值

**效果:**
- 服务器上 swin_small 实际输出 `[48, 96, 192, 384]` 而声明为 `[96, 192, 384, 768]` 时，`self.channels` 会按实际 `[48, 96, 192, 384]` 存储
- 所有下游模块（ThermalAdapter、DepthEncoder、CMFA、FPN）通过 `backbone_channels` 自动适配正确通道
- 切 backbone 或换 timm 版本后无需手动修改通道值

**副作用:**
- `__init__` 增加一次无梯度的微型 forward（64×64），耗时 < 10ms，不累积到训练
- 与 DDP 无冲突（DDP 包装发生在 `__init__` 之后，已完成 self.channels 的确定）

### 验证清单

- [x] `M3F_DETR(backbone_name='swin_small')` 正常初始化，channels 通过微型 forward 验证
- [x] 声明值与实际值一致时不打印警告（零误报）
- [ ] 服务器 `python test.py` 4 项测试全部通过
- [ ] IR/Depth/CMFA/FPN 通道链路全部自动对齐
- [ ] 训练不 OOM（CMFA use_downsample 生效）

---

## v0.3.0 (minor) — 运行时通道校验 + CMFA 降采样内存优化

**日期:** 2026-07-29  
**基于:** 服务器 2026-07-29 运行日志诊断 (48 通道断点定位 + CMFA OOM 确认)  
**目标:** 根治多模态管线通道不一致问题，大分辨率训练可运行

---

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/m3f_detr.py` | 加固 | forward 中增加 Backbone / IR / Depth / CMFA / FPN 逐层通道 & 空间校验断言 |
| `models/fusion/cmfa.py` | 优化 | 新增 `use_downsample` 模式：2×2 pool → Attention → upsample，token 减少 4×，attention 显存降低 ~16× |

### 详细修改

#### 1. `models/m3f_detr.py` — 运行时通道+空间校验

**修改原因:** 服务器日志确认 Backbone 初始化正常 (`channels=[96,192,384,768]`)，但 forward 中 CMFA 输出 48 通道。根源在 IR/Depth 编码器或 CMFA 内部存在未同步的降维代码。新增运行时断言确保即使代码版本不一致，也能在通道不匹配的第一时间抛出含具体通道值的明确错误。

**变更内容:**

- 第 116-130 行: Backbone 输出后逐层验证 RGB 特征通道数与 `backbone_channels` 一致
- 第 132-141 行: IR/Depth 编码器输出验证通道数 = `ch[0]`，验证空间尺寸与 RGB P2 对齐
- 第 148-150 行: CMFA 输出验证通道数 = `ch[0]`

诊断效果: 服务器报错从模糊的 `Conv2d expected 96 but got 48` 变为清晰的 `CMFA 融合输出 48 通道 ≠ 期望 96`，直接定位故障模块。

#### 2. `models/fusion/cmfa.py` — 降采样注意力模式

**修改原因:** 服务器日志确认 CMFA 在 P2 层做全局 MultiheadAttention 导致 OOM（1920×1080 输入下 P2 约 270×480=129600 token，attention score 矩阵 ≈ 67 GB float32）。报告问题 #2 建议降采样方案而非直接删除 P2 融合。

**变更内容:**

- `CMFA.__init__` 新增 `use_downsample` 参数 (默认 `True`)
- `forward` 新增降采样分支: 输入 H>32 且 W>32 时 2×2 avg_pool，token 减少 4×，attention 显存降低 ~16×
- 注意力后 bilinear upsample 回原始分辨率
- `use_downsample=False` 时行为与旧版完全一致，向后兼容

**副作用评估:**

| 场景 | 影响 |
|------|------|
| 小分辨率输入 (H≤32 或 W≤32) | 自动跳过降采样，行为不变 |
| 大分辨率训练 | 显存大幅降低，P2 细节通过上采样保留 |
| 精度 (mAP) | 轻微下降可预期（降采样丢失部分高频细节），建议 A/B 实验确认 |

#### 3. `models/m3f_detr.py` — CMFA 启用降采样

第 80 行: `CMFA(dim=backbone_channels[0])` → `CMFA(dim=backbone_channels[0], use_downsample=True)`

---

### 验证清单

- [x] `CMFA(96, use_downsample=True)` 小图输入自动跳过降采样 (H≤32 分支)
- [x] `CMFA(96, use_downsample=True)` 大图输入正确 pool→attention→upsample
- [ ] 服务器 `python test.py` 全部 4 个测试通过
- [ ] `forward_debug` 打印各层 shape 确认 IR/Depth/CMFA 输出均为 96 通道
- [ ] 训练 `batch_size=4` 在 24GB GPU 上不 OOM

---

## v0.2.0 — 接口统一与防御性加固

**日期:** 2026-07-28  
**版本:** 初始版本库之后的第一轮修订
**提交:** `e777116`

---

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/m3f_detr.py` | 加固 | 添加启动期通道数断言 + 前向调试方法 + 修正注释 |
| `models/neck/multimodal_fpn.py` | 加固 | 默认值改为必填 + forward 断言 + 修正注释 |
| `models/neck/simple_fpn.py` | 清理 | 修正 docstring 中的硬编码通道示例 |
| `inference.py` | 修复 | 添加 `--backbone` 参数 + checkpoint cfg 自动恢复 |
| `evaluate.py` | 修复 | 同上 + 修正重复加载 bug |
| `utils/checkpoint.py` | 增强 | `save_checkpoint` 新增 `cfg` 参数 |
| `train.py` | 增强 | 保存 checkpoint 时附带 `train_cfg` |
| `models/detector/transformer.py` | 加固 | query_embed 缺失时抛出明确错误而非静默 AttributeError |
| `engine/__init__.py` | 清理 | 删除残留注释行 |

---

### 详细修改

#### 1. `models/m3f_detr.py` — 通道配置注释 + 启动断言 + 调试方法

**修改原因:** 报告中问题 #1（通道配置脱节）和 #4（注释硬编码）。

**变更内容:**

- 第 69-74 行: 添加启动期 `backbone_channels` 数量断言和启动信息打印
  ```python
  assert len(backbone_channels) == 4, \
      f"期望 backbone 输出 4 层多尺度特征，实际 {len(backbone_channels)} 层: {backbone_channels}"
  print(f"[M3F-DETR] backbone={backbone_name}, channels={backbone_channels}, ...")
  ```
- 第 123-130 行: 修正 CMFA 融合处注释，将 `(B, 192, ...)` 改为 `(B, ch[0], ...)`
- 第 141-163 行: 新增 `forward_debug()` 方法，在前向传播中逐层打印所有模块的 tensor shape

#### 2. `models/neck/multimodal_fpn.py` — 消除默认值 + forward 断言

**修改原因:** 报告中问题 #3-#4。旧默认值 `[96, 192, 384, 768]` 只匹配 swin_small，切换 backbone 时可能被误触发。

**变更内容:**

- 第 16-22 行: 修正 docstring，`[P2(192), P3(384), ...]` → `[P2, P3, P4, P5]`
- 第 31-33 行: 修正参数说明，移除 `"默认 [192, 384, 768, 1536] (SwinV2-L)"`
- 第 38-42 行: 默认值从 `[96, 192, 384, 768]` 改为显式抛出 `ValueError`，要求调用方必须传入
- 第 77-78 行: `forward()` 开头增加特征层数断言

#### 3. `models/neck/simple_fpn.py` — 注释修正

**修改原因:** 报告中问题 #4。

**变更内容:** 第 23 行 docstring 参数示例 `如 [192, 384, 768, 1536]` → `（由 backbone 决定，需显式传入）`

#### 4. `inference.py` — backbone 配置统一管理

**修改原因:** 报告中问题 #4。原代码默认 `swin_small`，与训练 `swin_large` 不一致。

**变更内容:**

- 第 125-127 行: 新增 `--backbone` 命令行参数（choices: tiny/small/base/large）
- 第 167-178 行: 重写模型加载逻辑 — 先从 checkpoint 读取 `cfg`，自动恢复 backbone/hidden_dim/num_queries/num_classes
- 第 180-198 行: 修复 checkpoint 双次加载问题，单次 CPU 加载后移 tensors 到 device

#### 5. `evaluate.py` — 同上 + 重复加载修复

**修改原因:** 同上。

**变更内容:**

- 第 42-45 行: 新增 `--backbone` 参数
- 第 71-93 行: 重写模型加载逻辑，从 checkpoint cfg 自动恢复配置
- 第 97-112 行: 修复 checkpoint 双次加载问题

#### 6. `utils/checkpoint.py` — 支持保存模型配置

**修改原因:** 报告中问题 #3 方案 A。checkpoint 需自带模型配置信息以便推理时自动恢复。

**变更内容:**

- `save_checkpoint` 新增 `cfg` 参数
- 第 65-66 行: `cfg` 不为 None 时写入 state dict

#### 7. `train.py` — 保存 checkpoint 时附带配置

**修改原因:** 配合 `checkpoint.py` 改动。

**变更内容:**

- 第 226-231 行: 构建 `train_cfg` 字典（backbone/hidden_dim/num_queries/num_classes）
- 第 350/358/371 行: 三处 `save_checkpoint` 调用均传入 `cfg=train_cfg`

#### 8. `models/detector/transformer.py` — query_embed 防御性代码

**修改原因:** 报告中问题 #6。原 fallback 引用 `self.query_embed.weight` 但该属性不存在。

**变更内容:** 第 65-69 行: `query_embed=None` 时抛出明确 `ValueError`，而非静默 `AttributeError`

#### 9. `engine/__init__.py` — 代码清理

**修改原因:** 报告中问题 #7a。

**变更内容:** 第 4 行删除残留注释 `#from ..utils.checkpoint import save_checkpoint, load_checkpoint`

---

### 未做修改（按报告建议，需实验验证后决策）

- ❌ CMFA 融合层从 P2 下移到 P3（需 A/B 实验验证精度影响）
- ❌ 添加 ModelConfig 集中式配置类（下一阶段重构）
- ❌ Depth 编码器改进（比赛后期精度优化）
- ❌ `datasets/` 模块补充（模块在服务器上已存在，暂不处理）
- ❌ `losses/` 空目录合并/删除（低优先级）

---

### 验证清单

- [x] `python -c "from models.m3f_detr import M3F_DETR; m = M3F_DETR(backbone_name='swin_small')"` — 启动断言和打印正常
- [x] `MultiscaleFusion` 默认值 `in_channels_list=None` 时正确抛出 `ValueError`
- [x] `DINOTransformer.forward(query_embed=None)` 正确抛出 `ValueError`
- [x] `engine/__init__.py` 无残留注释，import 正常
- [ ] 在比赛数据集上运行 `python train.py` 验证完整的训练前向通过
- [ ] 使用新 checkpoint 运行 `python inference.py --checkpoint latest.pth --data-root data/test` 验证 cfg 自动恢复

---

## v0.3.5 (patch) — 修复 YAML 科学计数法解析 + optimizer float() 守卫

**日期:** 2026-08-01
**根因:** 项目环境 PyYAML 将 `lr: 1e-4` 解析为字符串，`train.py:315` 无类型转换赋值给 `config["optimizer"]["lr"]`，导致 `"1e-4" * 0.1` TypeError

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `train.py` | 加固 | `build_optimizer` 入口 `float()` 守卫 + stage lr 赋值 `float()` 转换 |
| `configs/debug.yaml` | 修复 | `lr: 1e-4` → `lr: 0.0001` (兼容 PyYAML 解析差异) |

### 详细修改

#### `train.py` — build_optimizer float() 守卫

第 101-103 行: 统一 `float()` 转换，兼容 YAML 字符串或浮点

```python
lr = float(config["optimizer"]["lr"])
backbone_lr = float(config["optimizer"].get("backbone_lr", lr * 0.1))
weight_decay = float(config["optimizer"].get("weight_decay", 0.05))
```

#### `train.py` — stage lr 赋值 float() 转换

第 315-316 行: `stage.get("lr") is not None` + `float(stage["lr"])`

#### `configs/debug.yaml` — 科学计数法改为纯浮点

第 28 行: `lr: 1e-4` → `lr: 0.0001` (`1e-4` 在项目环境 PyYAML 中被解析为 str)

### 验证清单

- [x] `yaml.safe_load` 解析 `0.0001` 为 float ✓
- [x] `lr: 1e-4` 被 PyYAML 解析为 str — 根因确认
- [ ] 服务器 `python train.py --config configs/debug.yaml` 正常启动训练

---

## v0.4.0 (minor) — RTX 5090 训练加速 + 全局注释审计修复

**日期:** 2026-08-01

### 训练加速

| 文件 | 改动 | 效果 |
|------|------|------|
| `utils/seed.py` | cudnn.benchmark=True, deterministic=False, TF32 启用 | 卷积 10-25% + matmul 15-30% 加速 |
| `train.py` | find_unused_parameters=False | DDP 通信 20-40% 加速 |
| `train.py` | torch.compile(model, mode="reduce-overhead") | 15-30% 加速 (PyTorch>=2.0) |
| `engine/trainer.py` | autocast dtype=torch.bfloat16 | 数值更稳 + 5-10% |
| `train.py` | DataLoader persistent_workers=True, prefetch_factor=3 | epoch 间切换 0 开销 |
| `configs/m3f_dino.yaml` | batch_size 4→8, num_workers 8→12 | GPU 利用率提升 |

### 注释修复（14 处）

| 文件 | 问题 |
|------|------|
| `models/m3f_detr.py` | 架构图 192ch→ch[0], Args 增加 backbone_name |
| `tools/test_dataloader.py` | 300→num_queries |
| `engine/evaluator.py` | 删除残留注释 import 行 |
| `models/backbone/depth_encoder.py` | normal→gradient 术语统一 |
| `models/detector/matcher.py` | 300→Nq |
| `models/neck/simple_fpn.py` | "第三阶段"→"备选实现" |
| `models/detector/dino_head.py` | "占位"→"已废弃" |

### 验证清单

- [x] cudnn.benchmark=True 不影响精度
- [x] BF16 autocast RTX 5090 兼容
- [x] torch.compile 有 try/except 兜底
- [ ] 服务器训练 batch_size=8 不 OOM

---

## v0.4.1 (patch) — 修复 inference.py use_dn 与训练不一致

**日期:** 2026-08-01 17:21
**根因:** inference.py:185 `use_dn=False` 训练时 `use_dn=True`（默认），导致 checkpoint 含 `dn_query_embed.weight` 而推理模型无此参数 → `RuntimeError`

### 修改

| 文件 | 行号 | 改动 |
|------|------|------|
| `inference.py` | 185 | `use_dn=False` → `use_dn=True` |

### 验证清单

- [x] `grep -R use_dn` 确认训练 + 推理 now 统一为 True
- [ ] 服务器 `python inference.py --checkpoint checkpoints/debug/final.pth` 正常加载
