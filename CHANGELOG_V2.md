# V2 新路线实验日志

> 面向城市场景视觉多模态目标检测的新项目记录。**最新记录置顶**，供协作者快速接力；旧路线见 `CHANGELOG.md`。

## 项目基线与硬约束

- 当前已知合规平台最佳：**57.0240**；目标：**60+**。
- 平台指标：12 类等权 COCO mAP@50–95（101 点插值）。
- 最终方案必须是单模型、单权重、单次可复现推理；禁止模型框级融合。
- 测试集仅用于最终推理，严禁伪标签训练或任何形式的测试集训练。
- 原始训练集：2000 组三模态数据；测试集：1000 组。
- 大图 depth 为真实 uint16 毫米 PNG（0 表示无效）；少量小图 depth 为退化 uint8 JPEG；IR 三通道完全相同。

## v2.0.14 — 57.024 champion 的 IR-only 分布锚定 Adapter（进行中，2026-09-12）

- 证据：V2 的 IR Adapter 在 fold0、fold1 均相对匹配 RGB 基线为正，而额外 depth 在 fold1 下降且 v2.0.13 的 valid gate 无法稳定重载复现；因此本轮只移除不稳定的额外 depth residual。
- 起点仍为平台 **57.0240** champion；soft 主输入已由 RGB/IR/depth 像素级融合生成，所以整体输入信息仍是三模态，新增 P2/P3 Adapter 仅使用更稳定的 IR 纠偏。
- 分布锚定、数据划分、最新标签、增强、优化器和 20 epoch 预算均与 v2.0.13 相同；新增 `--adapter-modalities ir` 开关，训练前必须保持 `max_abs_diff=0.0`。
- Run 计划：`runs/detect/runs/s5/champion_da_ir_p23`；以 ungated 双辅助分支重载约 0.650 为保留门槛。

## v2.0.13 — 57.024 champion 的 valid-aware 分布锚定 Adapter（完成，淘汰，2026-09-12）

- 按用户要求只从平台最佳 **57.0240** 的 `full2000cont_1280_refine/weights/best.pt` 起步，不继承平台 54.7800 的 V2 fold0 三模态权重。
- 首次 ungated 探索已完成 20 epoch：最佳 epoch 12、当前 200-val 内置 mAP50–95 **0.65043**；原 champion 历史末轮为约 0.64660。由于标签已更新，该差值仅作训练趋势参考，不宣称平台增益；原 YOLO 参数逐项 `max_diff=0.0`。
- 下一组从 champion 重新零初始化 IR/depth P2/P3 Adapter，保留 CVPR 2026 启发的逐通道均值/标准差分布锚定，并加入 v2.0.12 已在 fold0 验证为正的 depth valid-mask 空间门控。
- 除 depth gate 外保持首组配置不变：soft 1800/200、最新 `data/train/labels`、1280、batch 8、20 epoch、AdamW `lr0=5e-4`、alignment weight 0.05；训练前仍要求输出 `max_abs_diff=0.0`。
- Run 计划：`runs/detect/runs/s5/champion_da_trimodal_validgate`；以超过 ungated 0.65043 为本地保留门槛。
- 20/20 epoch 完整训练结束；逐轮 CSV 峰值为 epoch 7 的 **0.65104**，仅比 ungated 0.65043 高 0.00061。
- 训练结束重新加载同一 `best.pt` 后验证仅约 **0.647**，无法复现逐轮峰值且低于 ungated 权重重载约 0.650；checkpoint 已确认保留 `depth_valid_gate=True`、`align_weight=0.05`，并非属性丢失。
- 按“重载后可复现”标准淘汰 valid-gate 权重，不生成测试提交包；保留 ungated `champion_da_trimodal_p23/weights/best.pt` 为实验候选，平台最佳仍为 57.0240 原权重。

## v2.0.12 — Valid-aware depth Adapter（fold1 复验中，2026-09-12）

- 问题：普通 depth Adapter 在 fold0 提升、fold1 相对 IR-only 下降 0.1626，疑似无效深度区及稀疏 JPEG depth 污染残差。
- 改动：在 depth 分支 P2/P3 特征注入前，用 valid mask 的对应尺度平均有效率做逐位置门控；RGB 与 IR 路径不变。
- 从 fold0 IR-only best 重新加入零初始化 depth 分支；门控模型与 IR 起点 `max_abs_diff=0.0`。
- Run：`runs/detect/runs/s6/fold0_trimodal_validgate`，20 epoch、Adapter-only；门槛为超过普通三模态 fold0 的 44.1243。
- fold0 最佳 epoch 15，官方 **44.2721**，较普通三模态 **+0.1478**；大图 43.1125，小图 64.6669。
- 已启动 fold1 同配置复验：`runs/detect/runs/s6/fold1_trimodal_validgate`；保留门槛为超过 fold1 普通三模态 43.4263。

## v2.0.11 — 平台首测与全量三模态候选（2026-09-12）

### 平台反馈

- fold0 三模态 `conf=0.001`：**48.5130**。
- 同一权重 `conf=0.47`：**54.7800**，高 **+6.2670**；平台结论覆盖两折 OOF 阈值选择，后续锁定高阈值窄区间。
- 该模型仍低于合规 RGB 冠军 57.0240，差 **2.2440**。

### 全量 2000 张快速候选

- 从平台实测 54.7800 对应的 fold0 三模态权重出发，将训练集由 1605 张补全到全部 2000 张。
- RGB backbone/neck/head 冻结，仅联合续训 IR/depth Adapter 10 epoch，AdamW `lr0=1e-4`；固定预算使用 `last.pt`，不以训练重叠的 fold0 val 选择 epoch。
- Run：`runs/detect/runs/s5/trimodal_full2000_cont10`；10/10 完成。
- 测试集 `conf=0.45/0.47/0.49` 分别输出 4524/4461/4394 框；`conf=0.47` 与旧包 4395 框仅差 66，作为首选平台 A/B。

## v2.0.10 — 首个三模态测试集提交包（2026-09-12）

- 权重：fold0 稳健主权重 `runs/detect/runs/s3/trimodal_adapter_p23_frozen/weights/best.pt`；训练使用 1605 张、未使用测试集训练。
- 对测试集 1000 组 RGB+IR+depth 做单模型、单次 1280 推理，缓存每图 top-100。
- 两折 OOF conf sweep 推荐 `conf=0.001`；其 AP 43.2385，高于 0.30 的 39.0040 和 0.47 的 36.5144。
- 推荐包：`submissions/trimodal_fold0_conf0.00.zip`（对外交付名 `submission_trimodal_fold0_conf0.001.zip`），1000 TXT、31,632 框、每图最多 100。
- 备用包：`submissions/trimodal_fold0_conf0.47.zip`，1000 TXT、4,395 框、每图最多 30。
- 推荐包 SHA256：`15a7f851478901ab8fe3f74e0117a74641a54638f97942956bcbc7d6718c9a66`。
- 备用包 SHA256：`91da57ac960cbfaeaa5d076507ccbe099d773f785642a14b7821f43563cf8429`。

## v2.0.9 — fold1 独立复验（完成，2026-09-12）

- 不再继续利用 fold0 调参，改用 `folds5_v2/fold1` 独立验证三模态增益。
- 串行训练匹配的 RGB-1280 基线 40 epoch → IR Adapter 20 epoch → RGB+IR+depth Adapter 20 epoch；三者训练集、验证集和几何配置一致。
- 使用 zero-copy manifest/symlink，不复制原始数据；Run 统一位于 `runs/detect/runs/s4/`。
- fold1 官方分数：RGB **43.2538**；IR-only **43.5889**（+0.3351）；三模态 **43.4263**（相对 RGB +0.1725、相对 IR −0.1626）。IR 增益跨折复现，depth 在 fold1 未提供额外增益。
- fold0+fold1 共 794 张 OOF bootstrap：IR−RGB 均值 +0.2364、`P>0=0.81`；三模态−RGB 均值 +0.2204、95% CI `[-0.2429,+0.6989]`、`P>0=0.805`。两折点估计为正但尚未显著。

## v2.0.8 — 三模态 neck/head 适配（候选，不替代主权重，2026-09-12）

- 从保留的三模态最佳 `44.1243` 起步；冻结 RGB backbone layer 0–10，解冻 neck、Detect 和两个 Adapter。
- 以 AdamW `lr0=2e-5` 微调 10 epoch，使后半网络适配 P2/P3 三模态残差，同时尽量避免破坏 RGB 表示。
- Run：`runs/detect/runs/s3/trimodal_neckhead_ft`。
- 官方 fold0 **44.2401**，相对原三模态 44.1243 为 +0.1158；但 200 次 bootstrap CI `[-0.6633,+0.8759]`、`P(delta>0)=0.71`，且 8 个常见类下降。因此仅保留候选，不替代更稳健的原三模态权重。
- 使用标准 `pycocotools COCOeval` 复核 RGB/IR/三模态/neck-head 四组缓存，结果与 `official_metric.py` 逐项完全一致。

## v2.0.7 — 三模态 Adapter 低学习率续训（淘汰，2026-09-12）

- 原 run 的最佳点仍在最后一个 epoch 20，未显示明确收敛。
- 从 `trimodal_adapter_p23_frozen/weights/best.pt` 继续训练 10 epoch；RGB backbone/neck/head 仍冻结，仅更新 IR/depth Adapter。
- LR 从 `5e-4` 降至 `2e-4`，避免已获得的稳定增益被大步更新破坏。
- Run：`runs/detect/runs/s3/trimodal_adapter_p23_frozen_cont`。
- 最佳内置 val 出现在续训 epoch 2；官方原图级结果 **43.9046**，比起点 **44.1243** 下降 **0.2197**，不保留该权重。

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
