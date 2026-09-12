# V2 新路线实验日志

> 面向城市场景视觉多模态目标检测的新项目记录。**最新记录置顶**，供协作者快速接力；旧路线见 `CHANGELOG.md`。

## 项目基线与硬约束

- 当前已知合规平台最佳：**57.0240**；目标：**60+**。
- 平台指标：12 类等权 COCO mAP@50–95（101 点插值）。
- 最终方案必须是单模型、单权重、单次可复现推理；禁止模型框级融合。
- 测试集仅用于最终推理，严禁伪标签训练或任何形式的测试集训练。
- 原始训练集：2000 组三模态数据；测试集：1000 组。
- 大图 depth 为真实 uint16 毫米 PNG（0 表示无效）；少量小图 depth 为退化 uint8 JPEG；IR 三通道完全相同。

## v2.0.7 — 三模态 Adapter 低学习率续训（运行中，2026-09-12）

- 原 run 的最佳点仍在最后一个 epoch 20，未显示明确收敛。
- 从 `trimodal_adapter_p23_frozen/weights/best.pt` 继续训练 10 epoch；RGB backbone/neck/head 仍冻结，仅更新 IR/depth Adapter。
- LR 从 `5e-4` 降至 `2e-4`，避免已获得的稳定增益被大步更新破坏。
- Run：`runs/detect/runs/s3/trimodal_adapter_p23_frozen_cont`。

## v2.0.6 — RGB+IR+depth 三模态 Adapter（完成，2026-09-12）

### 方案与验收

- 从 `ir_adapter_p23_frozen/weights/best.pt` 暖启动，保留已学习的 IR P2/P3 Adapter。
- 新增 depth 两通道分支：真实 uint16 数据使用 `log1p(depth)/log1p(20000) + valid mask`；退化 uint8 JPEG 使用 `value/255 + valid mask`，确保全部 2000 张训练图都实际使用 RGB、IR、depth。
- Depth 在 P2/P3 通过独立零初始化 1×1 residual injection 接入；加入 depth 前后模型输出 `max_abs_diff = 0.0`。
- 单模型、单 checkpoint、单次六通道在线推理，不做模型集成。

### 正式运行

- Run：`runs/detect/runs/s3/trimodal_adapter_p23_frozen`
- 配置：folds5_v2/fold0、imgsz 1280、batch 8、20 epoch、AdamW lr0=5e-4；RGB backbone/neck/head 继续冻结，IR 与 depth Adapter 联合训练。
- 日志：`runs/detect/runs/s3/logs/trimodal_adapter_p23_frozen.log`

### 结果

- 最佳 epoch：20；官方原图级 fold0：**44.1243**。
- 相对 RGB-1280 `43.5798`：**+0.5445**；相对 IR-only `43.8725`：**+0.2518**。
- 大图相对 RGB：`42.3936 → 43.0337`（**+0.6401**）；小图：`64.8201 → 64.6097`（−0.2104）。
- 逐类 9/12 类高于 RGB；boat +3.118、light +1.105、tricycle +1.084，主要下降为 uav −0.970、person −0.293。
- 200 次配对 bootstrap（三模态−RGB）：均值 **+0.5250**，95% CI **[+0.1589, +1.0992]**，`P(delta>0)=0.99`。这是 V2 首个 fold0 置信区间完全高于 0 的结构增益，保留三模态路线。

## v2.0.5 — IR P2/P3 零残差 Adapter（完成，2026-09-12）

### 结构

- 起点：`runs/detect/runs/s2/full1280_v2/weights/best.pt`。
- RGB YOLO11m 主干、neck、Detect 全部冻结并保持原 checkpoint 对象结构。
- IR 取单通道，经 32/64/128 通道轻量分支，在 layer 2（P2，stride 4）和 layer 4（P3，stride 8）注入零初始化 1×1 残差。
- 新增 175,584 参数，占 RGB 基线 **0.875%**；单模型、单 checkpoint。
- 在线读取 RGB/IR/depth，不生成六通道中间数据；RGB/IR/depth 共用 affine、flip、letterbox 几何参数。第一臂仅启用 IR，depth 通道暂不进入模型。

### 训练前验收

- 零初始化模型与 RGB checkpoint 前向 bit-exact：`max_abs_diff = 0.0`。
- P2/P3 injection 首次 backward 梯度和均非零。
- 1% 数据端到端 smoke 完成：六通道 Dataset → 同步增强 → 原生检测 loss → validator → checkpoint。
- 正式训练 epoch 1 checkpoint：P2/P3 injection 权重已非零；所有 `model.*` RGB backbone/neck/head tensor 相对起点 `max_diff = 0`，确认 `freeze=24` 生效。

### 正式运行

- Run：`runs/detect/runs/s3/ir_adapter_p23_frozen`
- 配置：folds5_v2/fold0、imgsz 1280、batch 8、20 epoch、AdamW lr0=1e-3、warmup 2、仅 Adapter 可训练。
- 日志：`runs/detect/runs/s3/logs/ir_adapter_p23_frozen.log`
- 停止判据：与固定 RGB-1280 基线做相同原图级官方指标；若无明确正收益，则不进入 depth/双 Adapter 和其余折。

### 结果

- 最佳 epoch：12；官方原图级 fold0：**43.8725**，RGB-1280 基线 **43.5798**，提升 **+0.2927**。
- 大图：`42.3936 → 42.8070`（**+0.4134**）；小图：`64.8201 → 65.6052`（**+0.7851**）。
- 主要逐类变化：boat +2.566、light +1.068、seat +0.716；uav -1.386、tricycle -0.673、person -0.264。
- 200 次原图级配对 bootstrap：均值 +0.2192，95% CI `[-0.2109, +0.5009]`，`P(delta>0)=0.90`。方向为正但单 fold 尚未显著，作为三模态暖启动继续验证。

## v2.0.4 — 全图 1280/1920 统一测评（2026-09-12）

两个训练均完成 40 epoch：1280 的训练内 best mAP50–95 为 0.43421（epoch 40），1920 为 0.42145（epoch 38）。统一测评使用 `folds5_v2/fold0` 的全部 395 张原图、自研 101 点指标、每图最多 100 框；预测缓存于 `runs/detect/runs/s2/eval/`。

### 训练分辨率 × 推理分辨率

| 模型 | 推理 1280 | 推理 1920 |
|---|---:|---:|
| 训练 1280 | **43.5798** | 34.6948 |
| 训练 1920 | 34.1145 | **42.7762** |

训练和推理分辨率必须匹配；两个交叉格均大幅下降。匹配格中，1920 相对 1280 为 **−0.8036**。

### 子集与逐类差值（1920 − 1280，匹配格）

- 大图（365 张）：−0.3834
- 小图（30 张）：−1.6630
- 正向：ball +1.95、sign +1.09、garbage can +0.42、person +0.35、uav +0.16
- 负向：light −3.49、seat −2.82、tricycle −2.70、boat −2.44、bicycle −1.12、car −0.77
- 原始低阈值预测框：17883 → 20303（+2420），但 AP 下降。

### 判定

全图 1920 原生尺度路线在正确 fold0 上明确低于 1280；不进入其余 4 折。保留 1280 作为 V2 RGB 基线，下一步转向 IR/depth 的轻量 Adapter。该结论仅关闭“全图 1920”实现，不重新开启已失败的静态 crop。

## v2.0.3 — Step 2 全图纯尺度 A/B（运行中，2026-09-11）

脚本：`/tmp/chain_s2_fullscale.sh`；数据 manifest：`data/s2_fullscale/`；正式划分：`data/folds5_v2/fold0.txt`。

| Run | 输入 | Batch | Epoch | 状态 |
|---|---:|---:|---:|---|
| `runs/detect/runs/s2/full1280_v2` | 1280（大图约 0.667×） | 8 | 40 | 已完成 |
| `runs/detect/runs/s2/full1920_v2` | 1920（大图基本 1:1） | 4 | 40 | 已完成 |

控制变量：同一 RGB 数据、fold、YOLO11m 初始权重、优化器、增强、seed；无 crop、无漏 GT、无接缝、保留全局上下文。

训练日志：

- `runs/detect/runs/s2/logs/full1280_v2.log`
- `runs/detect/runs/s2/logs/full1920_v2.log`
- 完成摘要将归档到 `runs/detect/runs/s2/logs/s2_fullscale_summary.txt`

关机策略：`full1920_v2` 成功完成并出现 `TRAINING_DONE` 后，守护脚本先将 `/tmp` 日志复制为上述持久文件，执行 `sync`，随后 `shutdown -h now`。若训练链异常退出，则不关机。统一测评延期到下次开机后执行。

采用规则：fold0 只筛选；若 1920 有明确收益，再进入更多折。最终候选必须使用 `folds5_v2` OOF，并以平台单模型 A/B 判定能否超过 57.024。

### 下一步（尚未运行）

主线不是直接 6 通道替换首层，而是：

- RGB：保留原 YOLO11m 预训练主干。
- IR：独立轻量 Adapter。
- Depth：`log-depth + valid mask` 独立 Adapter；退化 JPEG depth 第一版关闭。
- 在 P2/P3 做零初始化门控残差注入；初始输出严格等于 RGB 基线。
- 消融顺序：RGB → RGB+IR → RGB+depth/mask → 双 Adapter → P2 与 P2+P3。

## v2.0.2 — Step 1 静态 crop pilot（2026-09-11）

### 设置

- 模型：YOLO11m，COCO 预训练。
- 两臂均 40 epoch、batch 8、imgsz 1280、AdamW、seed 42、mosaic=0。
- A：整图 RGB 训练。
- B：大图每张永久固定一个随机 1280×1080 crop；小图填充保持 1:1。
- 评测：旧 fold0 的 370 张大图，使用修正后的原图级指标。

### 数据审计

B 并非“双窗口训练”。A/B 都有 1601 个训练样本，但 B 只保留 **9121 / 12203 = 74.74%** 的训练 GT，各类别保留率不一致。因此本轮只能作为完整 crop pipeline pilot：B 失败不能否定原生尺度假设。

### 完整性

- Arm A：40/40；best epoch 40；训练内 best mAP 0.41974。
- Arm B：40/40；best epoch 36；训练内 best mAP 0.40664。

### 四格结果（fold0-large-local mAP50–95 ×100）

| 训练几何 | 推理几何 | 分数 |
|---|---|---:|
| A 整图 | 整图 | **41.5859** |
| A 整图 | tile | 30.1525 |
| B 静态 crop | 整图 | 34.3781 |
| B 静态 crop | tile | **39.6258** |

效应：

- A 模型 tile 推理效应：−11.4334
- B 模型 tile 推理效应：+5.2477
- 相同 tile 推理下，B 相对 A：+9.4733
- 训练×推理交互效应：+16.6811
- 最终 B-tile 相对 A-full：**−1.9601**

### 判定

- 训练—推理尺度匹配显著修复 tile 崩溃，机制信号成立。
- 当前静态 crop 端到端方案仍低于整图对照，不进入完整 5 折。
- 因 B 永久漏掉 25.26% GT，本结果不能否定原生尺度；下一步使用不裁图的 1920 全图对照。

## v2.0.1 — 评测基础设施审计与修复（2026-09-11）

### 官方指标

文件：`/tmp/official_metric.py`（本地 staging：`v2/official_metric.py`）。

修复：GT 必须独立遍历 `gts.items()`；旧实现通过 `preds.items()` 收集 GT，会漏掉零预测图片的全部 GT 并虚高 AP。两图合成单测：修复后得分 **50.4950**。

### 5 折划分

旧 `data/folds5` 的统计有 bookkeeping bug：图片被分配后只更新负责分配的类别，没有更新同图其他类别。

正式划分改用：`data/folds5_v2/`。

真实实例分布：

- fold 图片数：395 / 399 / 401 / 405 / 400
- person：1110 / 1110 / 1110 / 1109 / 1110
- boat：27 / 27 / 27 / 27 / 27
- ball：19 / 19 / 19 / 19 / 19
- uav：41 / 42 / 42 / 42 / 41
- tricycle：6 / 6 / 5 / 5 / 5

### 评测后处理

- 每张原图最终最多 100 框。
- tile 重叠区采用独占责任边界，再做跨 tile 类内 NMS。
- 修复小图文件误入 `_0/_1` tile 名称解析的问题。
- 该仓库的 `YOLO.predict()` warmup 在 1280 上异常占用超过 24 GB；V2 评测改用未融合的原始 `m.model` FP16 前向，自行执行 LetterBox、NMS 和坐标还原。

## v2.0.0 — 新工作区与路线重置（2026-09-11）

### 目标

摆脱旧 soft-fusion 输入和单一 200 张验证集，从原始 RGB/IR/depth 重建可复现的单模型方案。

### 工作区

- 主目录：`/root/autodl-tmp/aic_race/M3F-DETR`
- 新服务器作为 V2 唯一实验区；旧服务器只保留备份。
- 删除 `CityViMD-Net`（约 1.4 GB）及历史派生产物，保留原始 train/test 数据。

### 路线

1. 正确的 5 折 OOF 与官方口径指标。
2. 原生像素尺度训练/推理。
3. 保留 RGB 主干的 IR/depth Adapter，而不是直接替换 RGB 输入。
4. 只对单折明确正收益的候选运行完整 OOF。
