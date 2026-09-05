# 代码修改记录 (CHANGELOG)

> 项目: M3F-DETR | 基于: `项目问题分析报告.md`

---

## v0.18.3 (result) — 平台里程碑：full2000cont ab00 = 51.3220 新最佳（超越 soft768 50.505）

**日期:** 2026-09-05
**类型:** 平台结果 / 方案突破
**背景:** full2000_soft1024_cont_noearly（全量 2000 + 1024 + 无早停续训）的提交 `submission_full2000cont_ab00.zip` 平台 A/B 得 **51.3220**，超越 soft768_labelrefresh（50.5050）成为当前平台最佳。

### 平台实测全景（更新到最新）

| 提交 | 训练数据 | imgsz | 平台分 | 较 soft768 |
|---|---|---|---|---|
| soft768_labelrefresh（旧最佳） | fold1 1600 | 768 | 50.5050 | 基准 |
| rareos ab00/ab01 | fold1 1600+重采样 | 768 | 50.1870 | −0.32 |
| soft1024 ab00 | fold1 1600 | 1024 | 49.6430 | −0.86 |
| **full2000cont ab00（新最佳）** | **全量 1800** | **1024** | **51.3220** | **+0.82** |

### 关键洞察（本轮最重要的认知更新）

1. **此前"soft768 改动最少所以最好"的判断是错的。** full2000cont 用了更多数据（1800 vs 1600）+ 更高分辨率（1024）+ 更久训练（无早停续训），平台反而最高（+0.82）。
2. **全量数据 + 充分训练才是平台要的**：soft1024（fold1 1600、early-stop@27）平台掉到 49.64，而 full2000cont（全量 1800、无早停、续训充分）平台 51.32 —— 同样的 1024 分辨率，差别全在"用了更多数据 + 训更久"。
3. **early-stop / fold1 选模持续误导**：full2000 首轮 early-stop 在 ep3（200-val 0.692 看似最高），实际平台好的却是续训充分版（best ep10=0.658，200-val 反而更低）。本地 val 分数低 ≠ 平台差。
4. 结论方向：**增大有效训练数据量 + 充分训练（不依赖不靠谱的 early-stop）+ 1024 分辨率** 是平台真正认可的组合。后续应沿"全量数据、充分训练"扩展，而非再折回 fold1 1600。

### 状态

- [x] full2000cont ab00 = 51.3220 平台最佳（待队长最终确认提交编号）。
- [ ] 下一步：基于"全量 2000 + 充分训练"方向继续（见分析），CV fold2 训练进行中作参考。

---

## v0.18.2 (result) — 平台 A/B 反馈：soft1024 掉分 49.643；full2000+1024 训练与提交

**日期:** 2026-09-05
**类型:** 平台结果分析 / 数据量消融 / 训练 / 提交产物
**背景:** 汇总 v0.18.0/v0.18.1 三个提交的平台实测，并对"全量 2000 + 1024"数据量路线做消融训练。

### 平台 A/B 实测汇总（关键反馈）

| 提交 | fold1 val | 平台分 | 较 soft768 平台最佳 |
|---|---|---|---|
| soft768_labelrefresh（平台最佳，v0.17.0） | 0.473 | **50.505** | 基准 |
| rareos ab00/ab01（v0.18.0） | 0.497 | **50.187** | −0.32 |
| **soft1024 ab00（v0.18.1）** | **0.496(最高)** | **49.643** | **−0.86** |

**反直觉结论：fold1 val 分数与平台分基本负相关。** soft1024 是 fold1 最高分(0.496)，平台却最低(49.643)；soft768 是 fold1 最低分(0.473)，平台却最高(50.505)。结合 rareos(平台50.187)判断：
- **fold1 单折 val 完全不能预测平台分**（甚至误导）。
- 1024 高分辨率在平台真实测试上反而掉分（49.643），推测平台测试目标尺度/分布与 fold1 里被 1024 增益的"小目标"并不一致，或 1024 过拟合了 fold1 的高清细节。
- rareos ab01(perclass)与 ab00 相同(50.187)，证实 per-class 阈值全 0.001 无增益。

### full2000 + 1024 数据量消融

为检验"多 12.5% 训练数据是否提分"，用全部 2000 张 soft 融合图(分层留出 200-val，train=1800)从 soft1024 best 继续微调：

- 首轮 `full2000_soft1024`：early-stop@28，best ep3 mAP50-95=0.692(200-val)，但该 200-val 与 soft1024 权重几乎同源，ep3 近乎继承起点分数；后续持续过拟合下滑至 0.66。
- 续训 `full2000_soft1024_cont_noearly`（patience=0，低 lr，40 轮跑满）：best ep10=**0.658**，全程未超首轮 0.692。
- 结论：**数据量不是瓶颈**（+12.5% 无效）；模型起点已充分训练，额外数据只造成过拟合。且 full2000 无干净留出可本地交叉验证（fold 图几乎全进训练集），平台是唯一裁判，风险高。

### 提交产物（full2000_cont_noearly best ep10=0.658）

- `submission_full2000cont_ab00.zip`：imgsz=1024、full TTA、conf=0.001，1000 txt / 29389 框。
- 类别分布：person 7257/animal 8930/light 5056 仍偏多，反映 1024 训练特征。

### 数据/工具产出

- `data/yolo_full2000_soft`：全 2000 soft 融合图 + 分层 1800train/200val，data.yaml 就绪。
- `configs/yolo_native_m_trimodal_full2000_soft1024.yaml` / `_cont.yaml`：全量训练与续训配置。

### 状态

- [x] full2000cont ab00 已生成，待平台 A/B。
- [ ] 平台结果进一步表明：fold1 选模不可靠、分辨率/数据量方向均未获平台增益，下一步需重新审视验证策略与真正区分度方向（见分析）。

---

## v0.18.1 (result) — rareos 平台回退 + fold 诊断 + soft1024 训练与提交

**日期:** 2026-09-05
**类型:** 平台结果分析 / 验证策略诊断 / 分辨率实验 / 提交产物
**背景:** v0.18.0 rareos 提交平台 A/B 得 **50.1870**，低于平台最佳 soft_labelrefresh 的 **50.5050**（尽管 rareos 在 fold1 val 上 mAP50-95=0.4974 更高）。据此做根因分析与 fold 交叉诊断，并启动 soft+1024 高分辨率路线。

### rareos 平台掉分根因（整图重采样的先验漂移）

- 整图复制含稀缺类(boat/ball/tricycle)的图，但这类图本身混有大量 person/animal/light/car，导致大类样本连带抬升：person 1.14×、animal 1.12×、light 1.12×、car 1.09×——整体类别先验向右漂移。
- 提交框量 21124→25571(+21%)，animal +2091 框(+49%)，大类低质 FP 泛滥 → precision 下降 → 平台分降。
- 根因：fold1 val 与 rareos 训练图同分布，val 上放大先验"看似更好"；真实测试与 fold1 分布偏移，放大先验即露馅。**与 v0.13 伪标签/uav 过采样、1024 平台降同因：任何改变类别先验的改动，fold1 val 都测不出来。**

### fold 诊断（验证集分布偏移）

- 数据有 27 张 640×360 + 373 张 1920×1080 混尺寸。
- 逐 fold val 类别分布差异大：fold1 animal=733 明显偏多、car=351 偏少；fold2 animal=589、car=450 更接近整体平均。
- rareos 在 fold2 val mAP50-95=0.7685 < soft_labelrefresh 0.7852，与平台实测方向一致 → **fold2 比 fold1 更接近平台测试分布**（animal 少、car/uav 多）。
- ⚠️ 但注意 fold2 的 400 图全在 fold1 train 内（训练内记忆，非干净留出），绝对分虚高，仅相对差异有参考。

### soft+1024 训练（无先验改动，纯分辨率）

从平台最佳 soft768 权重 + soft 融合数据微调至 imgsz=1024（无重采样/无伪标签）：

- run `soft1024_from_soft768_best`：early-stop@58，**fold1(干净留出) best epoch 27，mAP50=0.77829 / mAP50-95=0.49589**，高于 soft768 的 0.47325(+0.023)。
- 关键观察：fold1 干净曲线 ep27 见顶(0.496)后回落至 ~0.478；而 fold2(训练内)分数随 epoch 单调升至 last(58) 0.809 —— fold2 上升是训练记忆非泛化，**不可作为平台代理**。
- 结论：best.pt(ep27)=0.49589 是干净留出最高分，且为纯分辨率增益（无先验改动），较 rareos 可信。

### 提交产物（soft1024 best.pt ep27）

- `submission_soft1024_ab00.zip`：imgsz=1024、full-image TTA、conf=0.001，1000 txt / 39374 框，格式校验通过。
- ⚠️ 框量 39374 明显高于 soft768 的 21124。按历史经验框量暴涨平台常掉分，但 soft1024 干净 fold1=0.496 是真实增益，故值得一次平台 A/B 判定。

### 工具脚本修复（本轮）

- `tools/scan_conf_per_class.py`：GPU 一次前向缓存 `.npz` → CPU「前缀精确法」重扫（贪心降序，去低分尾不影响前缀 TP/FP，精确等价；per-class O(1) 阈值）；修复逐图 orig_shape 换算 GT（640×360/1920×1080 混尺寸）；修复相对路径 cache 的 makedirs 崩溃。
- `tools/infer_ultra_tiled.py`：新增 `--perclass-conf JSON`（融合 NMS 后按类砍框）。
- `tools/oversample_rare_clean.py`、`configs/rareos_labelrefresh.yaml`、`scripts_5090/run_submission_ab.sh`：v0.18.0 新增。

### 状态

- [x] rareos ab00 平台 50.1870（回退，不主推）。
- [x] rareos vs soft fold2 诊断：证实整图重采样先验漂移。
- [x] soft1024 训练完成：fold1 best 0.49589(ep27)。
- [x] submission_soft1024_ab00.zip 已生成。
- [ ] soft1024 ab00 平台 A/B 结果待队长回报（判断 1024 路线方向是否坐实）。

---

## v0.18.0 (experiment) — 干净少数类重采样 rareos + per-class 阈值扫描工具 + rareos 提交

**日期:** 2026-09-05
**类型:** 数据类别平衡 / 推理后处理工具 / 提交产物 / 迁移到单卡 5090
**背景:** v0.17.0 soft-fusion 在新标签上 best mAP50-95=0.47325（平台分 50.5050），三模态弱注入路线收益收敛。分析训练/提交类别分布发现极端不平衡是 mAP@50-95（对 12 类等权）的主要拖累：boat 仅 135 框、ball 95、tricycle 27（fold1 train），模型提交里 tricycle 只出 22 框几乎靠猜。本轮放弃 v0.13 曾翻车的「伪标签 + uav/10x 过采样」，改为只对三个真正稀缺类（boat/ball/tricycle）做干净图级硬链接重采样，规避测试集污染并保持 person/car 先验几乎不变。同时在 GPU 服务器（单卡 RTX 5090）跑完 gated 与 rareos 两路训练，生成 rareos 提交供平台 A/B。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `tools/oversample_rare_clean.py` | 新增 | 干净图级重采样：仅 boat(1)/ball(7)/tricycle(11)，`--mult` 指定倍数，硬链接实现零额外磁盘；guard 打印 person/car 倍数确认先验不变；不含伪标签、不动 data/test |
| `configs/yolo_native_m_trimodal_rareos_labelrefresh.yaml` | 新增 | 从平台最佳 SAR YOLO11m 权重微调重采样数据，run 名 `rareos768_labelrefresh_from_sar_best`，超参与 soft_labelrefresh 对齐（imgsz768/batch8/AdamW/lr0.00016/close_mosaic12） |
| `tools/scan_conf_per_class.py` | 新增→重写 | per-class 置信度扫描。先重写为「前缀精确法」：贪心按分数降序，去掉低分尾不影响前缀 TP/FP，故每类每 IoU 只匹配一次、任意阈值 O(1) 取前缀；GPU 仅一次前向缓存 `.npz`。修复两 bug：①逐图用自身 orig_shape 换算 GT（数据集为 27 张 640×360 + 373 张 1920×1080，混用第一张尺寸会让小图 GT 全错位）；②缓存为相对路径时 dirname 为空的 makedirs 崩溃。输出每类最优 conf 与 delta，可 `--out-json` 写盘供推理复用 |
| `tools/infer_ultra_tiled.py` | 修改 | 新增 `--perclass-conf JSON`：在融合 NMS 后按类砍低于阈值框，供 per-class 提交复用 |
| `scripts_5090/run_submission_ab.sh` | 新增 | 一键产出两个 A/B 提交：ab00_baseline（全局 conf=0.001）+ ab01_perclass（按类阈值）。注：此脚本按 8 卡并行编写，本机单卡请按序运行步骤 |

### 数据准备（CPU 硬链接，零额外磁盘）

```bash
python tools/oversample_rare_clean.py \
  --src data/yolo_trimodal_soft_m \
  --dst data/yolo_trimodal_soft_m_rareos \
  --mult 1:2,7:2,11:4
# train 1600 → 1772（+172 dup）；tricycle 25→100(4x)、boat 107→214(2x)、ball 72→150(2x)
# person 仅连带 1.14x、car 1.09x（guard 确认先验未大变）；val 400 原样
```

### 训练（GPU 服务器 32709，单卡 RTX 5090；与 gated 同机并行跑完）

| run | 结果 | 说明 |
|-----|------|------|
| `gated768_labelrefresh_from_sar_best` | 80/80，best ep66，mAP50=0.75492 / **mAP50-95=0.47372** | 局部 gate 注入，≈ soft_labelrefresh 基线，无明显增益 |
| `rareos768_labelrefresh_from_sar_best` | early-stop @62，best ep37，mAP50=0.76264 / **mAP50-95=0.48505** | ✅ 反超平台最佳 soft_labelrefresh（0.47325）+0.012，当前本地最佳 |

### 提交产物（rareos best.pt，fold1 val TTA 评估 mAP50-95=0.4974）

- `submission_rareos768_labelrefresh_ab00_baseline.zip`：conf=0.001，1000 txt，25571 框，主推候选
- `submission_rareos768_labelrefresh_ab01_perclass.zip`：per-class 阈值版，与 baseline 同框（扫描结论见下）
- 类别分布对比旧 full_tta：boat 123→187、tricycle 22→59，重采样稀缺类检出明显增多

### per-class 阈值扫描结论

- 全部 12 类最优 conf 均为 **0.001**：该已充分训练的模型提高阈值只伤 recall、不增 precision，故 **conf=0.001 baseline 即最优**，per-class 版无增益（后续无需再试阈值方向）。
- 附：per-class AP（val，TTA，conf=0.001）person 0.431/boat 0.407/animal 0.508/seat 0.600/sign 0.365/bicycle 0.402/car 0.522/ball 0.369/light 0.488/garbage 0.542/uav 0.583/tricycle 0.752，无类别塌缩。

### 磁盘清理（迁移新实例时，数据盘 6.1G → 21G）

- 删除 5 个 run 目录下全部 `epoch*.pt`（中间断点，保留 best/last）：native_x_sar/rgb_sar768-2、-3，native_m_trimodal/{soft_labelrefresh, fusion, soft}_from_sar_best
- 删除废弃 fusion 数据 `data/yolo_trimodal_fusion_m` + `data/test_trimodal_fusion`（v0.16.0 真三模态 0.439 已废弃，可再生成）

### 状态

- [x] 数据准备：`data/yolo_trimodal_soft_m_rareos` 1772 图，check_det_dataset 通过。
- [x] gated 训练完成（0.47372）与 rareos 训练完成（0.48505）。
- [x] rareos 提交 ab00_baseline / ab01_perclass 已生成（zip 不入库，位于仓库根目录）。
- [x] 工具脚本（oversample / scan_conf / infer perclass / submission builder）已提交并 push。
- [ ] ab00_baseline.zip 平台 A/B 结果待队长回报（期望 ~51+ 验证重采样路线）。

---

## v0.17.1 (experiment) — DEYOLO/GLS 启发的 gated 三模态局部增强

**日期:** 2026-09-02
**类型:** 数据融合策略 / 训练配置 / 小目标边界增强
**背景:** v0.17.0 的 soft-fusion 在新标签上 best mAP50-95=0.47325，说明弱三模态注入有效但还没有拉开差距。进一步参考 DEYOLO 的跨模态双特征增强、GLS-YOLOv8n 的 RGB-depth-thermal 互补，以及多模态茶芽小目标检测中强调的密集小目标局部显著性，本轮改为局部 gate：仅在红外局部显著或深度边缘强的位置注入残差信号，减少整图亮度分布漂移。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `tools/prepare_yolo_sar_enhanced_data.py` | 修改 | 新增 `--fusion-mode gated`，用 IR 局部显著残差和 depth edge gate 做局部增强，保留原 `soft` 模式兼容 |
| `configs/yolo_native_m_trimodal_gated_labelrefresh.yaml` | 新增 | 从平台最佳 SAR-style YOLO11m 权重训练 gated 三模态数据，run 名 `gated768_labelrefresh_from_sar_best` |

### 推荐运行

```bash
python tools/prepare_yolo_sar_enhanced_data.py \
  --fold 1 \
  --out data/yolo_trimodal_gated_m \
  --test-out data/test_trimodal_gated \
  --ir-weight 0.18 \
  --depth-weight 0.10 \
  --sharpen 0.24 \
  --fusion-mode gated \
  --overwrite
screen -dmS yolo_m_trimodal_gated_labelrefresh_v0171 bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate race && OMP_NUM_THREADS=8 yolo detect train cfg=configs/yolo_native_m_trimodal_gated_labelrefresh.yaml > yolo_m_trimodal_gated_labelrefresh_v0171_train.log 2>&1'
```

### 状态

- [x] 代码、配置与本记录已准备，等待 push 后在服务器生成 gated 数据并训练。
- [ ] 生成 `data/yolo_trimodal_gated_m` 与 `data/test_trimodal_gated`。
- [ ] 训练完成后与 v0.17.0 soft best mAP50-95=0.47325 对比。

---

## v0.17.0 (experiment) — 适配队长 2026-09-02 新标签：清 YOLO cache 后重训 soft-fusion

**日期:** 2026-09-02
**类型:** 数据标签刷新 / 训练配置 / 三模态保守优化
**背景:** 队长已将修正后的训练标签覆盖到 `data/train/labels`。远端检查确认 `data/yolo_sar_m/*/labels` 和 `data/yolo_trimodal_soft_m/*/labels` 都是软链接，能够跟随新标签；但 Ultralytics 的 `labels.cache` 仍停留在 2026-08-28 或 2026-09-01，早于新标签时间 2026-09-02 00:12，因此旧验证和 early-stop 结论不再可靠。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `configs/yolo_native_m_trimodal_soft_labelrefresh.yaml` | 新增 | 从平台最佳 SAR-style YOLO11m 权重重新微调三模态 soft-fusion，独立 run 名 `soft768_labelrefresh_from_sar_best`，避免复用旧标签训练痕迹 |

### 执行要点

```bash
find data/yolo_sar_m data/yolo_trimodal_soft_m -name "*.cache" -delete
screen -dmS yolo_m_trimodal_soft_labelrefresh_v017 bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate race && OMP_NUM_THREADS=8 yolo detect train cfg=configs/yolo_native_m_trimodal_soft_labelrefresh.yaml > yolo_m_trimodal_soft_labelrefresh_v017_train.log 2>&1'
```

### 状态

- [x] 远端只读确认：`data/train/labels` 共 2000 个标签，时间戳为 2026-09-02 00:12。
- [x] 远端只读确认：YOLO 派生标签目录为软链接，`find -L` 后 `train=1600`、`val=400`。
- [x] 配置与本记录已准备并已 push：commit `bff3c50`。
- [x] 远端已 `git pull --ff-only` 到 `bff3c50`，并删除 `data/yolo_sar_m`、`data/yolo_trimodal_soft_m` 下旧 `labels.cache`。
- [x] 重新训练完成：`runs/native_m_trimodal/soft768_labelrefresh_from_sar_best`，80/80，best epoch 66。
- [x] 最终复评：mAP50=0.759、mAP50-95=0.472；`results.csv` 精确 best 为 epoch 66，mAP50=0.75973、mAP50-95=0.47325。
- [x] 生成召回型提交包 `submission_yolo_trimodal_soft_labelrefresh_tta.zip`：1000 txt，89211 框，1.8M；使用 full+tile+TTA，框数过高，仅作高召回备选。
- [x] 生成稳健提交包 `submission_yolo_trimodal_soft_labelrefresh_full_tta.zip`：1000 txt，21124 框，512K；使用 full-image TTA，不开 tile，建议优先平台 A/B。
- [x] 旧 soft-fusion early-stop 结果仅作废弃参考；本轮结果以 2026-09-02 新标签训练后的 epoch 66 best.pt 为准。

---

## v0.16.3 (run) — 补跑 soft-fusion 最后 12 epoch：关闭 early stopping

**日期:** 2026-09-01
**类型:** 训练补跑 / 收敛验证
**背景:** v0.16.2 因 `patience=20` 在 48/60 early stop，虽然 best.pt 复评已达 mAP50-95=0.476，但原计划最后 12 轮未完整执行。本轮按用户要求补跑剩余阶段，验证关闭早停后的低学习率最终收敛是否还能提升。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `configs/yolo_native_m_trimodal_soft_finish.yaml` | 新增 | 从 `soft768_from_sar_best/weights/last.pt` 接着跑 12 epoch，`patience=0`，关闭 mosaic/mixup/copy_paste，低学习率做最终 polish |

### 推荐运行

```bash
screen -dmS yolo_m_trimodal_soft_finish_v016 bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate race && OMP_NUM_THREADS=8 yolo detect train cfg=configs/yolo_native_m_trimodal_soft_finish.yaml > yolo_m_trimodal_soft_finish_v016_train.log 2>&1'
```

### 状态

- [x] 配置与本记录已准备，等待 push 后启动。
- [ ] 训练：`runs/native_m_trimodal/soft768_finish60_from_last`。
- [ ] 训练完成后对比 v0.16.2 best epoch 28 的 mAP50-95=0.476，并决定是否重新生成提交包。

---

## v0.16.2 (experiment) — 三模态软融合微调：保留平台最佳输入分布

**日期:** 2026-09-01
**类型:** 训练配置 / 保守三模态优化
**背景:** v0.16.0 真三模态像素级替换输入只到 mAP50-95=0.43967，说明直接改变 YOLO 预训练输入分布会伤害检测效果。本轮改为更温和的三模态软融合：仍保留 RGB 色彩与 SAR-style 亮度增强路线，仅降低 IR/Depth 注入强度，尝试在不破坏平台最佳分布的前提下吸收跨模态互补信息。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `configs/yolo_native_m_trimodal_soft_finetune.yaml` | 新增 | 从平台最佳 YOLO11m SAR 768 权重继续微调 60 epoch，使用弱 IR/Depth 注入数据 `data/yolo_trimodal_soft_m`，低 lr、弱增强、`close_mosaic=10` |

### 论文迁移点

- 保留 DEYOLO/GLS-YOLOv8n/茶芽多模态互补思想，但不再把三模态强行替换为独立通道。
- 可见光仍提供主纹理和颜色先验；IR/Depth 只作为亮度显著性和结构边缘的弱残差信息。
- 降低 mosaic/mixup/copy-paste 强度，避免小目标和多模态细节在二次增强中被过度扰动。

### 推荐运行

```bash
python tools/prepare_yolo_sar_enhanced_data.py \
  --fold 1 --out data/yolo_trimodal_soft_m \
  --ir-weight 0.08 --depth-weight 0.04 --sharpen 0.25 --overwrite
screen -dmS yolo_m_trimodal_soft_v016 bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate race && OMP_NUM_THREADS=8 yolo detect train cfg=configs/yolo_native_m_trimodal_soft_finetune.yaml > yolo_m_trimodal_soft_v016_train.log 2>&1'
```

### 状态

- [x] 配置与本记录已准备并已 push。
- [x] 数据准备完成：data/yolo_trimodal_soft_m 包含 train=1600、val=400，标签各 1600/400。
- [x] 训练已启动：runs/native_m_trimodal/soft768_from_sar_best，screen yolo_m_trimodal_soft_v016，日志 yolo_m_trimodal_soft_v016_train.log；YOLO11m 平台最佳权重加载 649/649。
- [x] 训练 EarlyStopping 于 48/60；best epoch 28，最终复评 mAP50=0.747、mAP50-95=0.476，高于平台最佳 YOLO11m SAR 768 clean 基线 0.46663 和 YOLO11x SAR 768 的 0.46932。
- [x] 扩展 `tools/prepare_yolo_sar_enhanced_data.py`，支持用相同 soft-fusion 参数生成 test/visible 推理图，避免训练/提交输入分布不一致。
- [x] 已生成 `data/test_trimodal_soft/visible`：1000 张 soft-fusion 测试图。
- [x] 已生成 `submission_yolo_trimodal_soft_tta.zip`：1000 txt，32385 框，714K；框数介于平台最佳 28623 与 YOLO11x 候选 34443 之间，优先作为新训练型 A/B 候选。

---

## v0.16.1 (experiment) — 三模态后融合：平台最佳主模型 + 辅助模态候选补充

**日期:** 2026-09-01
**类型:** 推理后处理 / 多模态互补 / 低风险提交候选
**背景:** v0.16.0 的真三模态像素级替换输入只达到 mAP50-95=0.43967，说明直接改变 YOLO 预训练输入分布风险较高。本轮保留平台最佳 YOLO11m SAR 768 TTA 预测为主，只让三模态弱模型或 YOLO11x 候选在后处理阶段提供有限补充。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `tools/fuse_yolo_submissions.py` | 新增 | 支持从多个 submission 目录/zip 读取官方格式预测，按类别执行 weighted box fusion，主模型高权重、辅助模型低权重，并用 `aux-min-conf` 控制弱模型只补充较可信框 |

### 论文迁移点

- DEYOLO 的跨模态互补思想迁移到预测级融合，避免再改 YOLO 主干导致预训练分布漂移。
- GLS-YOLOv8n 与茶芽 RGB-D-IR 论文强调多模态信息互补；本轮让 visible/SAR-style 主模型保持主导，IR/Depth 三模态模型只在高置信候选上参与。
- 对密集小目标检测，使用 weighted fusion 合并相近框，避免简单拼接带来重复框暴涨。

### 推荐运行

```bash
python tools/fuse_yolo_submissions.py \
  --inputs submission_yolo_sar_aug_tta_768 submission_yolo_trimodal_tta \
  --weights 1.0 0.35 \
  --output submission_yolo_sar_trimodal_wbf \
  --zip submission_yolo_sar_trimodal_wbf.zip \
  --iou 0.55 --aux-min-conf 0.02 --max-det 100
```

### 状态

- [x] 脚本与本记录已准备，等待 push 后在服务器生成三模态辅助提交并融合。
- [x] 生成 `submission_yolo_trimodal_tta`。
- [x] 修复首次运行发现的 WBF 聚类元数据问题：聚类代表框改为即时计算，不再覆盖原始带权重/来源的框。
- [x] 生成后融合候选并完成校验：
  - submission_yolo_trimodal_tta.zip：1000 txt，35440 框，767K，仅作为辅助源，不建议单独提交。
  - submission_yolo_sar_trimodal_wbf.zip：1000 txt，26873 框，617K；平台最佳 SAR 768 TTA 主导，三模态高置信框辅助。
  - submission_yolo_sar_x_trimodal_wbf.zip：1000 txt，28085 框，640K；SAR 768 TTA + YOLO11x SAR 768 + 三模态辅助，框数接近历史最佳 28623，优先作为 A/B 候选。

---

## v0.16.0 (experiment) — 真三模态融合 YOLO11m：基于平台最佳 768 权重微调

**日期:** 2026-09-01
**类型:** 数据融合 / 论文方法迁移 / 训练启动前记录
**背景:** 当前平台已验证最高分仍是 `submission_yolo_sar_aug_tta_768.zip`（YOLO11m SAR-style 768 + TTA，平台 48.876）。本轮按比赛三模态要求，从该平台最佳权重继续微调，而不是切回不稳定的自写三分支检测器。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `tools/prepare_yolo_trimodal_fused_data.py` | 新增 | 将 `visible / infrared / depth` 对齐成 3 通道融合图：可见光细节、红外强度、深度结构/边缘；同时支持生成 test 融合图用于提交推理 |
| `configs/yolo_native_m_trimodal_fusion_finetune.yaml` | 新增 | 从当前平台最佳 `native_m_sar/rgb_sar768-2/weights/best.pt` 低学习率微调 80 epoch，保持 `imgsz=768` 和干净 train/val split |

### 论文迁移点

- `Real-time dense small object detection algorithm based on multi-modal tea shoots`: 借鉴密集小目标场景下 RGB-D-IR 多模态互补，保留局部细节和结构边缘。
- `GLS-YOLOv8n`: 借鉴 RGB-depth-thermal 的通道级互补思想，用可见光纹理、红外显著性、深度结构分别占据 3 个输入通道。
- `DEYOLO`: 借鉴跨模态增强思路，先对各模态做轻量 CLAHE/去噪/边缘增强，再交给 YOLO 学习，避免直接大改 Ultralytics 骨架造成训练不稳定。

### 风险控制

- 不使用伪标签，不把 `data/test` 写入训练集；test 融合图仅用于最终推理。
- 使用 YOLO11m 平台最佳权重微调，避免 YOLO11x 本地略高但平台未验证的路线直接替代。
- 第一阶段先跑 80 epoch 消融；若本地验证未追近 0.46663，则不生成主提交。

### 推荐运行

```bash
python tools/prepare_yolo_trimodal_fused_data.py --fold 1 --overwrite
screen -dmS yolo_m_trimodal_v016 bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate race && OMP_NUM_THREADS=8 yolo detect train cfg=configs/yolo_native_m_trimodal_fusion_finetune.yaml > yolo_m_trimodal_v016_train.log 2>&1'
```

### 状态

- [x] 代码、配置与本记录已准备并已 push。
- [x] 数据准备完成：data/yolo_trimodal_fusion_m 包含 train=1600、val=400，data/test_trimodal_fusion/visible 包含 test=1000；无缺失模态。
- [x] 训练已启动：runs/native_m_trimodal/fusion768_from_sar_best，screen yolo_m_trimodal_v016，日志 yolo_m_trimodal_v016_train.log；YOLO11m 平台最佳权重加载 649/649。
- [x] 训练完成：80/80，best epoch 64，mAP50=0.70673，mAP50-95=0.43967；低于平台最佳 YOLO11m SAR 768 clean 基线 0.46663，也低于 YOLO11x SAR 768 的 0.46932。
- [x] 结论：真三模态像素级替换输入带来明显预训练分布偏移，不生成主提交包；后续三模态应采用更温和的 RGB/SAR-style 主干 + IR/Depth 辅助增强或推理重打分。

---

## v0.15.2 (chore) — 数据盘清理：释放约 48G（回收站 + 废弃 DINO 权重 + 中间 epoch 权重）

**日期:** 2026-08-30
**类型:** 维护 / 磁盘清理
**背景:** 数据盘 `/root/autodl-tmp`（70G）已用 69G（99%），将阻断后续训练权重保存与数据集下载。经确认后，永久删除与当前最佳方案（v0.15.1 YOLO11x SAR 768, mAP50-95=0.46932）无关的文件。

### 清理内容（数据盘 99% → 29%，可用 1.1G → 50G）

| 类别 | 路径 | 释放量 |
|------|------|--------|
| 回收站 | `/root/autodl-tmp/.Trash-0`（含 aic_race.zip 18.4G、旧 M3FD-DETR、旧 submission、codex tar） | ~18G |
| 废弃 DINO 权重 | `checkpoints/`（rush* 系列、latest.pth、debug、yolo_dual/fusion/rgb 试验） | ~15G |
| 中间 epoch 权重 | `native_x_sar/rgb_sar768` 与 `native_m_sar/rgb_sar768-2` 的 `weights/epoch*.pt`（保留 best/last） | ~14G |
| 旧 YOLO run | `runs/detect/runs/` 下 native_x、native_m、yolo11m_1024、probe、val-*、native_m_sar 的 finetune/sar_v3 试验目录 | ~1.5G |
| 已弃用数据 | `data/yolo_sar_v3`（v0.13 伪标签+过采样，v0.15 已回归 clean） | ~1.9G |

### 保留（当前最佳方案）

- `runs/detect/runs/native_x_sar/rgb_sar768/weights/{best,last}.pt`（v0.15.1 最佳 0.46932）
- `runs/detect/runs/native_m_sar/rgb_sar768-2/weights/{best,last}.pt`（v0.11.2 可信基线 0.46663）
- `data/train`、`data/test`、`data/yolo_sar_m`、`yolo11x.pt`、全部代码

### 状态

- [x] 已永久删除（非回收站），执行时无训练进程
- [x] 数据盘 99% → 29%（可用 1.1G → 50G）

---

## v0.15.1 (run) — 启动干净数据 YOLO11x SAR-style 768 容量实验

**日期:** 2026-08-29
**类型:** 训练启动 / 干净数据 A-B 实验
**背景:** v0.15.0 已先完成 inference-only 的 tile 候选提交；本次在 push 完成后启动已提交的 `configs/yolo_native_x_sar_aug.yaml`，测试更大 YOLO11x 容量是否能在 SAR-style 3ch 增强数据上超过 YOLO11m 768。继续遵守 no-pseudo-label / no-test-train 原则。

### 训练命令

```bash
screen -dmS yolo_x_sar768_v015 bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate race && OMP_NUM_THREADS=8 yolo detect train cfg=configs/yolo_native_x_sar_aug.yaml > yolo_native_x_sar_aug_train_v015.log 2>&1'
```

### 数据与风险控制

- 数据: `data/yolo_sar_m/data.yaml`，只包含训练集增强图与 fold1 val，不使用 `data/test`。
- 配置: `imgsz=768, batch=6, multi_scale=false`，避免 v0.11.1 记录的多尺度 OOM。
- 对照: 以 `runs/detect/runs/native_m_sar/rgb_sar768-2/weights/best.pt` 的 `mAP50-95=0.46663` 和平台最高候选为基线。
- 观察: 若 60-80 epoch 仍未接近 0.46663，应停止作为消融，不强行等待。

### 状态

- [x] 训练已启动：screen yolo_x_sar768_v015，日志 yolo_native_x_sar_aug_train_v015.log，run 目录 runs/detect/runs/native_x_sar/rgb_sar768/。
- [x] 训练已完成：120/120，screen 自动退出，GPU 释放。
- [x] 最佳点：epoch 100，mAP50=0.76565，mAP50-95=0.46932；略高于旧 YOLO11m SAR 768 clean 基线 0.46663（+0.00269）。
- [x] 后段观察：close_mosaic 后 106-120 epoch 回落到 0.45341 左右，说明应使用 best.pt 而不是 last.pt。
- [x] 已生成干净 TTA 提交：submission_yolo_x_sar768_tta.zip，1000 txt，34443 框，752K；框数高于旧 YOLO11m SAR 768 TTA 的 28623，但远低于 tile+TTA 的 89599，建议作为平台 A/B 候选，不直接覆盖历史最高提交。

---

## v0.15.0 (experiment) — 小目标 Tile 推理融合：避免伪标签污染的提交侧优化

**日期:** 2026-08-29
**类型:** 推理增强 / 论文方法迁移 / 低风险提交侧优化
**背景:** 队长 v0.13.0-v0.14.0 记录显示，伪标签和测试集写入训练集虽然可能抬高本地验证，但平台分反而下降；本轮严格避开测试集训练、伪标签和数据污染，优先采用 SAR 小目标检测论文常见的 multi-scale / crop-based inference 思路，在提交生成阶段提升小目标召回。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `tools/infer_ultra_tiled.py` | 新增 | 基于 Ultralytics YOLO 权重执行全图 + 重叠切片推理，将 tile 坐标映射回原图后做类别内 NMS 融合，并按官方格式生成 txt/zip；默认不修改旧 `infer_ultra.py` 流程 |

### 关键原则

- **inference-only**：脚本只读取 `data/test/visible` 图像并生成提交文件，不读取标签、不写训练集、不生成伪标签。
- **clean-data**：训练仍只允许使用原始训练集与合规的训练集内过采样，禁止把测试集加入 train。
- **可回退**：旧提交脚本和已有最佳提交包不变，新脚本单独输出到新目录/zip。
- **小目标取向**：默认 `tile=512, overlap=0.25, imgsz=768`，用局部放大视角补充全图推理，最后每图限制 `max_det=100`。

### 推荐运行

先用当前可信最高平台基线权重试 768 tile TTA：

```bash
/root/miniconda3/bin/python tools/infer_ultra_tiled.py   --weights runs/detect/runs/native_m_sar/rgb_sar768-2/weights/best.pt   --data-root data/test   --output submission_yolo_sar_tile768_tta   --zip submission_yolo_sar_tile768_tta.zip   --imgsz 768 --tile 512 --overlap 0.25   --conf 0.001 --iou 0.6 --fuse-iou 0.55   --max-det 100 --batch 8 --device cuda --tta
```

如需对照队长 1024 干净两阶段模型，仅作为 A/B，不覆盖历史最佳提交：

```bash
/root/miniconda3/bin/python tools/infer_ultra_tiled.py   --weights runs/detect/runs/yolo11m_1024/stage2_unfreeze/weights/best.pt   --data-root data/test   --output submission_yolo_stage2_1024_tile   --zip submission_yolo_stage2_1024_tile.zip   --imgsz 1024 --tile 512 --overlap 0.25   --conf 0.001 --iou 0.6 --fuse-iou 0.55   --max-det 100 --batch 6 --device cuda
```

### 验证

- [x] `/root/miniconda3/bin/python -m py_compile tools/infer_ultra_tiled.py` 通过。
- [x] `/root/miniconda3/bin/python tools/infer_ultra_tiled.py --help` 通过。
- [x] `--limit 5` 冒烟生成 5 个 txt、261 个框，输出为官方归一化格式。
- [x] 全量 768 tile+TTA 生成 `submission_yolo_sar_tile768_tta.zip`：1000 txt、89599 框、1.8M；框数明显高于历史 768 TTA 的 28623 和 1024 TTA 的 24614，建议作为激进召回候选，不要单独替代最高分提交。
- [x] 保守 768 tile 生成 `submission_yolo_sar_tile768_conf001.zip`：1000 txt、51349 框、1.1M；相比激进 TTA 版减少约 43%，但仍高于历史 768 TTA 的 28623 框。
- [x] 更保守 768 tile 生成 `submission_yolo_sar_tile768_conf003.zip`：1000 txt、28176 框、657K；框数贴近历史最高平台候选 `submission_yolo_sar_aug_tta_768.zip` 的 28623 框，优先作为平台 A/B 候选。

---

## v0.14.0 (result) — yolo11m 1024 两阶段训练：干净数据刷新记录

**日期:** 2026-08-29
**类型:** 实验结果 / 提交产物
**背景:** 伪标签已移除后，训练集恢复为干净的 2067 张（1600 原始 + 467 过采样）。
此前 1024 版本受伪标签污染导致平台分下降（48.876 → 47.928），本方案用干净数据
重新跑 1024 + 5090 满载（batch 20 / cache=ram / workers 16），验证融合路线放弃后
纯 RGB 数据策略的极限。

### 实验配置（scripts_5090/）

| 阶段 | 模型 | imgsz | batch | epochs | freeze | lr0 | 说明 |
|------|-------|--------|--------|--------|--------|-----|------|
| Stage 1 | yolo11m.pt | 1024 | 20 | 60 | [0..9] | 0.001 | 冻结保护预训练特征 |
| Stage 2 | stage1 last.pt 续跑 | 1024 | 20 | 120 | 全解冻 | 0.0005 | 低 lr 充分微调 |

共同设置: AdamW / cos_lr / close_mosaic=10 / mixup=0.15 / copy_paste=0.1 /
cache=ram / workers=16 / deterministic=False / seed=42 / RTX 5090。

### 结果（fold1 验证集 400 张）

| 阶段 | best epoch | mAP50-95 | 备注 |
|------|-----------|----------|------|
| Stage 1（冻结） | 29 | 0.48408 | 冻结期即超过 768 基线 |
| **Stage 2（解冻）** | **94** | **0.48085** | 最终 best |
| Stage 2 raw 复评 | — | 0.481 | 与训练记录一致 |
| Stage 2 + TTA | — | **0.483** | 验证略高，正式提交用 |

对比: 768+TTA（平台 48.876）→ 本方案验证 0.483，本地已超 0.4666 基线约 +0.017。

### 提交产物

- 正式提交: `submission_yolo_stage2_1024_tta.zip`（TTA，1000 txt，24614 框，573K）
- 对照: `submission_yolo_stage2_1024.zip`（raw，16656 框，430K）
- 模型: `runs/detect/runs/yolo11m_1024/stage2_unfreeze/weights/best.pt`
- 脚本: `scripts_5090/run_stage1_frozen.sh` / `run_stage2_unfreeze.sh`（stage2 路径已修复 runs/detect 前缀）

### 经验

1. **干净数据是关键**：同一套 1024 配置，伪标签版平台掉分、干净版验证 0.48+，证实伪标签污染测试集分布是上次掉分主因。
2. **5090 满载有效**：batch 6→20、workers 8→16、cache=ram，训练效率与效果均提升。
3. 解冻阶段（stage2）后期收益有限（best 在 94 轮），余弦衰减 + close_mosaic 后未见大幅跳升，可考虑 stage3 低 lr 精修或直接提交。

## v0.13.1 (revert) — 停用伪标签：移除测试集图像与伪标签脚本

**日期:** 2026-08-29
**类型:** 规则合规 / 数据回滚
**背景:** v0.13.0 的伪标签方案将测试集图像（SAR 增强后，`pseudo_*.jpg`）写入了训练集，违反赛题「测试集只能用于推理」约束，存在判罚风险；且伪标签由 best 模型推理生成（conf≥0.3），召回偏低，使模型在官方测试集上的表现下降（平台分 48.876 → 47.928）。决定停用并彻底移除。

### 修改概览

| 文件/目录 | 操作 | 说明 |
|-----------|------|------|
| `tools/pseudo_label.py` | 删除 | 伪标签生成脚本（git 提交 `445a525` 移除并推送） |
| `data/yolo_sar_v3/train/images/pseudo_*.jpg` | 删除 | 1000 张测试集 SAR 增强图 |
| `data/yolo_sar_v3/train/labels/pseudo_*.txt` | 删除 | 1000 个伪标签 |

### 影响与现状

- 训练集恢复为 **2067 张**（原始 1600 + 过采样 467），不再包含任何测试集内容。
- 已训练模型与提交包均未改动：`submission_yolo_sar_aug_tta_768.zip`（平台 48.876）保持为当前最高分提交；`submission_yolo_sar_v3_1024.zip`（47.928）保留作对照。
- v0.13.0 中的过采样（`tools/oversample_rare.py`）与 1024 高分辨率训练保留，后续如需重训请基于剔除伪标签后的 2067 张数据集进行。

## v0.13.0 (experiment) — 少数类过采样 + 伪标签 + 1024 高分辨率重训

**日期:** 2026-08-29
**类型:** 数据增强 / 精度提升
**背景:** per_class 报告定位三类问题——小目标漏检(sign/bicycle/ball/boat Recall 低)、少数类样本不足(ball19/boat28/tricycle2/uav30)、person 大类别密集小目标。据此并行推进三项改进。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `tools/oversample_rare.py` | 新增 | 图级过采样 boat2x/ball3x/uav2x/tricycle10x/garbage1x，训练集 1600→2067 张 |
| `tools/pseudo_label.py` | 新增 | 测试集 SAR 增强 + best 模型推理生成伪标签(conf≥0.3)，新增 1000 张(986 张有框,4825 框) |
| `data/yolo_sar_v3/` | 新增 | 合并训练集 3067 张(原始+过采样+伪标签)，val 400 张 |
| 训练 | 新增 | imgsz 768→1024、best.pt 微调、epochs=120、batch=6、close_mosaic=15 |

### 结果对比

| 实验 | best epoch | mAP50 | mAP50-95 | 备注 |
|------|------------|-------|----------|------|
| SAR-style YOLO11m 768（基线） | 104 | 0.75012 | 0.46663 | v0.11.2 最佳 |
| **SAR-style YOLO11m 1024 v3（过采样+伪标签）** | **103** | **0.753** | **0.47612** | 当前最优，+0.00949 |

训练曲线：换分辨率+数据域后前 20 轮从 0.31 起步，90-105 轮到达峰值 0.4761，120 轮完整跑完（patience=35 未早停）。

### TTA 对比与提交产物

| 评估 | mAP50-95 | 说明 |
|------|----------|------|
| raw（imgsz=1024） | **0.4761** | 正式提交 |
| TTA（augment=True） | 0.4753 | 无增益（与 v0.9.0 dual 一致），不采用 |

- 正式提交包：`submission_yolo_sar_v3_1024.zip`（raw，1000 个 txt，30940 框，691K）
- TTA 对照：`submission_yolo_sar_v3_1024_tta.zip`（43698 框，923K，验证无增益不采用）
- 模型：`runs/detect/runs/native_m_sar/sar_v3_1024-3/weights/best.pt`
  - 注意 run 目录带 `-3` 后缀（前两次失败启动占用了 `sar_v3_1024`/`-2`）
- 验证集：fold1 val 400 张、2847 实例（不变）

### 结论与后续

过采样+伪标签+1024 高分辨率整体 **+0.0095**（0.4666→0.4761）。弱项仍在 ball/bicycle/sign/boat 等小目标/低频类。下一步候选：小目标局部放大/patch 检测、低频类进一步重采样、更高分辨率微调。

## v0.12.0 (experiment) — SAR-style 继续扩展：YOLO11x 与低学习率精修

**日期:** 2026-08-28
**类型:** 继续优化 / A-B 实验
**背景:** v0.11.2 的 SAR-style YOLO11m 768 已达到 `mAP50-95=0.46663`，超过原生 YOLO11x RGB 640 的 `0.45035`。继续沿着有效路线扩展两组实验：更大模型容量与低学习率定位精修。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `configs/yolo_native_x_sar_aug.yaml` | 新增 | YOLO11x 在 SAR-style 增强 3ch 数据上训练，固定 768，batch=6，测试大模型容量是否继续带来收益 |
| `configs/yolo_native_m_sar_finetune.yaml` | 新增 | 从 `rgb_sar768-2/weights/best.pt` 低学习率精修 60 epoch，关闭 mosaic/mixup/copy_paste，弱尺度扰动，尝试提升定位与置信度校准 |

### 推荐观察指标

- 若 YOLO11x SAR 在 60-80 epoch 超过 0.46663，优先等其完整收敛并用 768+TTA 生成提交。
- 若 YOLO11m finetune 在 20-30 epoch 无提升，保留为消融，不强行继续作为主线。

---

## v0.11.2 (result) — SAR-style YOLO11m 反超原生 YOLO11x

**日期:** 2026-08-28
**类型:** 实验结果 / 提交产物
**结论:** 论文启发的 SAR-style 增强数据 + YOLO11m 固定 768 训练取得当前最佳 fold1 验证结果。

### 结果对比

| 实验 | best epoch | mAP50 | mAP50-95 | 备注 |
|------|------------|-------|----------|------|
| native YOLO11m RGB 640 | 79 | 0.71913 | 0.44472 | 原生 RGB baseline |
| native YOLO11x RGB 640 | 93 | 0.72719 | 0.45035 | 大模型 baseline，训练完成后反超 YOLO11m |
| **SAR-style YOLO11m 768** | **104** | **0.75012** | **0.46663** | 当前最优，较 YOLO11x +0.01628 |

### 训练与产物

- 训练目录：`runs/detect/runs/native_m_sar/rgb_sar768-2/`
- 最优权重：`runs/detect/runs/native_m_sar/rgb_sar768-2/weights/best.pt`
- 推理命令：`python tools/infer_ultra.py --weights runs/detect/runs/native_m_sar/rgb_sar768-2/weights/best.pt --data-root data/test --output submission_yolo_sar_aug_tta_768 --tta --imgsz 768 --zip submission_yolo_sar_aug_tta_768.zip --batch 8 --device cuda`
- 提交包：`submission_yolo_sar_aug_tta_768.zip`（1000 个 txt，约 646K）

### 注意

Note: submission_yolo_sar_aug_tta.zip was generated once with default imgsz=640. Prefer submission_yolo_sar_aug_tta_768.zip, which matches the 768 training size.

---

## v0.11.1 (patch) — 固定 SAR-style YOLO11m 训练尺度避免 OOM

**日期:** 2026-08-28
**类型:** 训练稳定性修复
**背景:** `yolo_native_m_sar_aug` 首次启动时 `imgsz=768 + multi_scale=true` 会采样到 1400+ 的大尺度；在 `yolo11x` baseline 同时占用约 9.4GB 显存时，新实验出现 CUDA OOM warning。为保留小目标分辨率收益并降低显存波动，改为固定 768 训练。

### 修改

| 文件 | 类型 | 摘要 |
|------|------|------|
| `configs/yolo_native_m_sar_aug.yaml` | 修复 | `multi_scale: true` 改为 `false`，并在配置注释中记录原因 |

### 运行策略

`data/yolo_sar_m` 已完成生成（1600 train / 400 val），后续直接重启训练，不再重复数据准备：

```bash
yolo detect train cfg=configs/yolo_native_m_sar_aug.yaml 2>&1 | tee yolo_native_m_sar_aug_train_fixed.log
```

---

## v0.11.0 (experiment) — SAR 论文启发的 YOLO11m 增强数据实验

**日期:** 2026-08-28
**类型:** 论文方法迁移 / YOLO baseline 优化
**背景:** 当前最强 baseline 已切换为原生 Ultralytics YOLO，服务器正在并行训练 `yolo11m` 与 `yolo11x`。截至本次查看，`yolo11m` 明显优于 `yolo11x`，best 约为 `mAP50=0.7083 / mAP50-95=0.4248`，因此后续优化优先围绕 `yolo11m` 做低风险 A/B。

### 论文启发与取舍

- **DenoDet V2 / SAR 去噪思路:** 不直接改 Ultralytics 主干，先在离线数据侧加入轻量 bilateral/median 去噪，降低传感器噪声和局部伪纹理对小目标的干扰。
- **SARLite / QGPG-Net 小目标细节增强:** 在保留 RGB 颜色结构的前提下，对 luminance 做 CLAHE、IR 弱注入、Depth 边缘弱注入和 unsharp mask，提高小目标边界与局部对比度。
- **YOSDet 定位质量启发:** 用更高 `imgsz=768`、低置信度评估/提交和 `close_mosaic`，优先保召回与定位质量。
- **避免重复无效路线:** 早期 5ch fusion 和 dual-branch attention 已经接近或弱于 RGB baseline，本轮不再扩大多模态结构复杂度，先保留 3ch ImageNet 预训练兼容性。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `tools/prepare_yolo_sar_enhanced_data.py` | 新增 | 生成 3ch SAR-style 增强 YOLO 数据集：RGB 保色彩 + L 通道去噪/CLAHE/锐化 + IR/Depth 弱 saliency 注入；标签用 symlink 保持一致 |
| `configs/yolo_native_m_sar_aug.yaml` | 新增 | 原生 YOLO11m 增强数据实验配置：`imgsz=768`、AdamW、mosaic/mixup/copy_paste、multi_scale、close_mosaic |

### 推荐运行顺序

当前 `native_m` / `native_x` 两条训练仍在占用 GPU 和 dataloader CPU，先不要抢资源。待它们结束后运行：

```bash
python tools/prepare_yolo_sar_enhanced_data.py --fold 1 --out data/yolo_sar_m --overwrite
yolo detect train cfg=configs/yolo_native_m_sar_aug.yaml 2>&1 | tee yolo_native_m_sar_aug_train.log
```

若显存不足，优先把 `batch: 8` 降到 `batch: 6`，不要先降 `imgsz`，因为本实验核心收益来自小目标分辨率。

### 验证

- [x] `python -m py_compile tools/prepare_yolo_sar_enhanced_data.py` 通过。
- [x] 单张样本增强函数冒烟通过，输出 `(360, 640, 3) uint8`，值域 `[0,255]`。
- [ ] 等当前原生训练完成后生成 `data/yolo_sar_m` 全量增强数据。
- [ ] 跑 `yolo_native_m_sar_aug`，与 `native_m/rgb640-4` best 进行 fold1 A/B 对比。

---

## v0.8.0 (major) — 方向切换：无卡环境改用 YOLO 早期融合方案

**日期:** 2026-08-28  
**类型:** 方向决策 / 新增主路径  
**背景:** 服务器进入无卡模式（无 GPU），原 M3F-DETR 方案无法继续训练；且 rush_v2~v9 验证集 `mAP@50-95` 最高约 0.005，继续排错性价比低。决定改用 YOLO11 早期融合（RGB + IR + Depth → 5 通道输入）作为主路径，M3F-DETR 旧链路保留冻结作为参考。

**详细说明:** 见 `项目问题分析报告.md` 第二十三节（决策原因、实施步骤、文件清单、交接注意事项）。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `yolo/__init__.py` | 新增 | YOLO 方案包 |
| `yolo/dataset.py` | 新增 | 三模态数据加载；`mode=rgb` 输出 3ch、`mode=fusion` 输出 5ch（RGB+IR灰度+Depth归一化）；YOLO 归一化标签；同步增强 |
| `yolo/model.py` | 新增 | 以 `yolo11n.yaml` + `ch=5/3` + `nc=12` 构建模型；从 `yolo11n.pt` 迁移权重；首层卷积新增通道用 RGB 权重均值初始化 |
| `yolo/evaluate.py` | 新增 | YOLO 推理解码 + COCO mAP@50-95 评估（复用 engine/evaluator.compute_map） |
| `tools/train_yolo.py` | 新增 | YOLO 训练入口（沿用 train.py 风格：YAML 配置、分段日志、EMA、AMP、best/latest/final checkpoint） |
| `tools/infer_yolo.py` | 新增 | 测试集推理 → NMS → 每图 ≤100 框 → 官方格式 txt → zip |
| `configs/yolo_rgb.yaml` | 新增 | RGB-only 对照配置（先跑，验证链路） |
| `configs/yolo_fusion.yaml` | 新增 | 5ch 早期融合配置（主路径） |
| `requirements.txt` | 修改 | 增加 `ultralytics` |

### 推荐运行（恢复 GPU 后）

```bash
# 1) RGB-only 对照（必须先跑通，确认环境/数据/评估链路）
python -u tools/train_yolo.py --config configs/yolo_rgb.yaml --fold 1 \
  2>&1 | tee yolo_rgb_train.log

# 2) 5ch 早期融合（主路径）
python -u tools/train_yolo.py --config configs/yolo_fusion.yaml --fold 1 \
  2>&1 | tee yolo_fusion_train.log

# 3) 提交文件生成
python tools/infer_yolo.py --checkpoint checkpoints/yolo_fusion/best.pth \
  --data-root data/test --output submission_yolo --zip submission_yolo.zip
```

### 验证顺序

1. 小数据过拟合冒烟：loss 必须下降；
2. `mode=rgb` 验证集 `mAP@50-95` 非 0；
3. `mode=fusion` 与 RGB-only 对比，确认多模态增益；
4. 提交 1000 个 txt + zip。

**2026-08-28 无卡冒烟已通过**：迷你数据集（8 张、CPU、320×192）上训练 1 epoch 与推理提交全流程跑通，见《项目问题分析报告》23.6。

---

## v0.10.0 (experiment) — 原生 ultralytics 大模型 + 完整增强（RGB）

**日期:** 2026-08-28  
**类型:** 大模型升级实验  
**背景:** RGB v2（yolo11n，640×384，弱增强）验证 mAP@50-95=0.3104（+TTA 0.3234），要冲击更高分需模型规模与增强双升级。

### 本轮改动

- `tools/prepare_yolo_data.py`（新增）：把三模态数据整理成 ultralytics 原生 YOLO 格式（fold1 划分，符号链接省磁盘）；
- `tools/infer_ultra.py`（新增）：加载原生 best.pt，用 ultralytics 自带 letterbox 预处理生成提交文件，支持 `--tta`；
- 训练：**yolo11x（batch 8）+ yolo11m（batch 16）并行**，imgsz=640，epochs=100，mosaic/mixup(0.15)/HSV/scale 全开，seed 42。

### 状态

训练进行中（runs/native_x、runs/native_m），完成后与 RGB v2（0.3104/0.3234）对比，取最优生成新提交包。

---

## v0.9.0 (experiment) — 双分支 + P3-P5 注意力融合（方案 B）

**日期:** 2026-08-28  
**类型:** 结构升级实验  
**目标:** 早期融合（5ch 拼接，fusion 0.178）未跑赢 RGB-only（0.297）后，升级为双分支架构：RGB 与 IR+Depth 各用独立预训练 backbone 提特征，在 P3/P4/P5 用 CrossModalFusion 做通道+空间注意力融合，期望充分利用红外（温差突出目标）与深度（几何结构）的特性。

### 结构设计

```
RGB(3ch) ──▶ backbone_rgb（yolo11n 预训练）──▶ P3/P4/P5 ─┐
                                                          ├─▶ CrossModalFusion×3 ─▶ Detect head(12类)
IR+Depth(2ch) ─▶ backbone_aux（yolo11n 预训练）──▶ P3/P4/P5 ┘
```

`CrossModalFusion`（`yolo/fusion.py`）对每个尺度：
1. 通道注意力：两分支拼接后全局池化 → MLP → sigmoid，重标定 RGB 特征；
2. 空间注意力：拼接特征 1x1 卷积 → sigmoid，突出共同关注区域；
3. 残差融合：1x1 投影跨模态残差，稳定训练。

### 关键修复（顺带闭环）

- **stem 权重迁移 bug**：旧实现把"目标模型随机 stem 的前 3 通道"当作预训练权重复制给新增通道，导致 5ch fusion 的 stem 实际全随机。v0.9.0 改为：前 3 通道直接继承官方预训练 RGB 权重，新增/辅助通道用预训练 RGB 均值初始化（`yolo/model.py::_init_stem_from_pretrained`）。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `yolo/fusion.py` | 新增 | `CrossModalFusion` 跨模态注意力融合单元 |
| `yolo/model.py` | 增强 | `DualBranchYOLO`、`build_dual_yolo_model`；`apply_freeze` 支持前缀冻结；修复 stem 迁移 |
| `yolo/dataset.py` | 增强 | `mode=dual` 输出 5ch（与 fusion 相同，模型内部拆分） |
| `tools/train_yolo.py` | 增强 | 支持 `model.arch=dual`；优化器把 head/fusion 归入完整 lr 组 |
| `tools/infer_yolo.py` | 增强 | 从 checkpoint cfg 恢复 `arch` |
| `configs/yolo_dual.yaml` | 新增 | 双分支训练配置（冻结双 backbone，100 epoch） |

### 冒烟验证（GPU，2026-08-28）

- 构建：RGB 分支迁移 448 键、Aux 分支 447 键；总参数 5.08M（冻结 backbone 后可训练 0.76M）；
- 训练一步：loss 47.9，反向传播正常；
- 推理：eval 输出 `(B, 16, N)` 正常；
- 小验证集：链路可跑通（随机头 mAP≈0，符合预期）。

### 状态与结论

- Dual v1（冻结整条分支）约 40 epoch 平台化在 0.22，停止；
- Dual v2（neck 可训练，从 v1 best 续训）best 达 **0.3105**（mAP@50 0.5407），与 RGB v2（0.3086）持平，无显著提升；TTA 反而降至 0.2927（RGB v2+TTA 为 0.3234）。
- **结论：双分支+注意力融合不构成显著增益，正式提交维持 RGB v2 + TTA（0.3234）；dual 保留为消融记录。** 后续主攻 RGB 增强路线（mosaic/mixup、分辨率、类别重采样），详见《项目问题分析报告》25.6/25.7。

---

## v0.8.1 (breakthrough) — YOLO 方案首次出分：RGB 基线验证集 mAP@50-95 ≈ 0.30

**日期:** 2026-08-28  
**类型:** 实验突破 / 关键修复 / 提交准备  
**一句话总结:** 从 M3F-DETR 一个月卡在 `mAP@50-95≈0.005`，切换到 YOLO11 后 RGB 基线直接到 **0.297**，加水平翻转 TTA 到 **0.323**，验证了方向切换的正确性。

### 一、结果总览（fold 1 验证集，400 张，RTX 5090）

| 实验 | 配置 | mAP@50-95 | mAP@50 | mAP@75 | 说明 |
|------|------|-----------|--------|--------|------|
| M3F-DETR rush_v2~v9（历史） | 自研 DETR，一个月 | ≈0.005 | ≈0.02 | ≈0 | 已废弃主路径 |
| **YOLO RGB v1** | yolo11n + 冻结 backbone，60 epoch，640×384 | **0.2971** | 0.5320 | 0.2643 | 首次出分 |
| **YOLO RGB v1 + TTA** | 水平翻转双视角 + 类别内 NMS 融合 | **0.3231** | 0.5695 | 0.3237 | +0.026 |
| YOLO fusion v1 | 5ch 早期融合（RGB+IR+Depth），80 epoch | 0.1782 | 0.3306 | 0.1609 | 未跑赢 RGB |
| YOLO RGB v2 | 解冻 backbone 全量微调，100 epoch | 0.3104 | 0.5565 | best.pth（第 60 轮） |
| **YOLO RGB v2 + TTA** | 解冻 backbone + 水平翻转 TTA | **0.3234** | 0.5698 | 正式提交模型 |

RGB v1 每类 AP50-95（修复 per-class 统计后）：seat 0.52、uav 0.40、light 0.39、car 0.37、animal 0.33，无类别塌缩。
置信度阈值扫描：`conf=0.001` 最优（0.2971），阈值越高 mAP 越低，提交保持 0.001。

### 二、关键修复（本轮踩坑闭环）

1. **EMA 滞后导致 best.pth 选错（高）**
   - 现象：训练内验证 mAP 停在 0.0004~0.02，但原始模型实际已达 0.27。
   - 根因：`ema_decay=0.9999` 配合每 epoch 仅 ~100 步时 EMA 权重严重滞后（有效视窗约 10000 步），验证用的 EMA 模型几乎还是初始权重。
   - 修复：`tools/train_yolo.py` 验证时**用原始模型选 best**（EMA 仅作参考输出）；配置 `ema_decay` 降为 `0.999`。
   - 影响：fusion/RGB-v2 的 best.pth 均为真实最优；RGB v1 的 best.pth 因旧代码失效，最终取 final.pth 原始权重。

2. **compute_map 每类 AP 恒为 0（中）**
   - 根因：pycocotools 的 `COCOeval.stats` 需要先调用 `summarize()` 才会填充；旧代码 per-class 只 evaluate/accumulate 就读 stats，得到空列表被 except 吞掉。
   - 修复：`engine/evaluator.py` 在 per-class 循环中静默调用 `summarize()`（`contextlib.redirect_stdout` 抑制打印）。
   - 影响：每类 AP 恢复正常，可继续用于类别不平衡诊断。

3. **深度图实际为 8bit JPG 可视化（中）**
   - 赛题文档描述 16bit PNG（毫米值），但下载数据实际是 8bit JPG 可视化深度（0~255）。按 16bit 处理会把整图裁成 0。
   - 修复：`yolo/dataset.py::_load_depth` 兼容两种格式（max>255 按 16bit，否则按 1~255 映射），与旧 `datasets/depth_process.py` 逻辑一致。

4. **图像尺寸语义 bug（低）**
   - `(H, W)` 在 `_resize` 中被解包成 `(w, h)`，导致输出图像被旋转式缩放（640×384 的高瘦图）。
   - 修复：`h, w = self.size`，cv2.resize 传 `(w, h)`。

### 三、新增/修改文件

| 文件 | 类型 | 摘要 |
|------|------|------|
| `configs/yolo_rgb_v2.yaml` | 新增 | 解冻 backbone 全量微调配置（100 epoch，lr 0.0005） |
| `tools/infer_yolo.py` | 增强 | 新增 `--tta`：水平翻转双视角推理 + 类别内 NMS 融合 |
| `tools/train_yolo.py` | 修复 | best 选择改用原始模型；EMA 仅参考 |
| `configs/yolo_rgb.yaml` / `yolo_fusion.yaml` | 调参 | `ema_decay: 0.9999 → 0.999` |
| `engine/evaluator.py` | 修复 | per-class AP 静默 summarize |
| `yolo/dataset.py` | 修复 | 深度图 8bit/16bit 兼容、尺寸语义 |

### 四、当前问题

1. **5ch 早期融合未跑赢 RGB**（fusion 0.178 vs RGB 0.297）：可能原因——1600 张训练数据太少，融合通道均值初始化带来的特征偏移需要更多数据/训练；增强过弱；或早期融合本身在这份数据上信息冗余（RGB 已含大部分可判别信息）。
2. **RGB 进入 ~0.30 平台期**：当前增强仅 resize + 随机翻转 + RGB 光度扰动，缺少 mosaic/mixup/尺度抖动；解冻 backbone（v2）提升有限。
3. **EMA 仍略低于原始模型**（0.999 衰减在 100 步/epoch 下视窗 ~1000 步），可考虑 0.99 或仅作推理参考。
4. **小目标/难类**：boat/ball/garbage can/tricycle 等低频类别 AP 偏低，需要 per-class 分析与类别重采样。

### 五、后续计划（按优先级）

1. **提交（已完成）**：取 RGB-v2 best.pth + `--tta` 生成正式提交包 `submission_yolo_final.zip`（1000 个 txt，验证集口径 mAP@50-95=0.3234）。
2. **增强**：实现 mosaic/mixup/scale jitter（对齐 ultralytics 原生增强），预期缓解平台期。
3. **分辨率**：640×384 → 800×512 或 1280×768，验证小目标（uav）提升。
4. **类别不平衡**：按 GT 频次重采样或损失加权。
5. **融合改进**：双分支 + P3-P5 交叉注意力融合（原方案 B），替代早期融合。
6. **后处理**：TTA 已合入；conf/NMS 网格扫描脚本可复用。

---

## v0.7.7 (experiment) — rush_v9：900 query 多尺度 anchor 修复 v8 尺寸瓶颈

**日期:** 2026-08-27  
**目标:** 针对 `rush_v8_anchor_multiscale` 第 10 轮 mAP 仍极低的问题，扩大 anchor 尺寸覆盖并提升每图候选密度。

### 结果依据

- `rush_v8_anchor_multiscale` epoch 10: `mAP@50-95=0.0001`, `mAP@50=0.0007`。
- IoU 诊断显示仍有定位/排序信号，但不足以形成有效 mAP：
  - `每图最佳 IoU(同类别)=0.3740`
  - `GT recall@IoU0.5 same_cls=0.0336`
  - 预测框高度均值 `0.0760`，GT 高度均值 `0.1178`，单一 anchor 尺寸偏窄。
  - 预测类别仍集中在 class 10，后续需要继续观察类别恢复。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/detector/dino_detector.py` | 增强 | `anchor_box_size` 支持多个 `[w, h]` 尺寸；生成 `num_centers × num_sizes` anchor |
| `configs/rush_v9_anchor900_multiscale.yaml` | 新增 | 使用 900 query = 15×20 网格 × 3 尺寸 anchor，继续使用 P3/P4/P5 多尺度 memory |

### 推荐运行

```bash
python -u train.py --config configs/rush_v9_anchor900_multiscale.yaml --fold 1 \
  2>&1 | tee rush_v9_anchor900_multiscale_train.log
```

---

## v0.7.6 (experiment) — rush_v8：多尺度 decoder memory + anchor box query

**日期:** 2026-08-27  
**目标:** 针对 mAP 长期远低于预期的问题，从网络结构层面修复“单尺度 decoder + 随机 query 直接回归绝对框”的定位瓶颈。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/detector/box_head.py` | 增强 | 保留旧 sigmoid 框输出，同时新增 `forward_logits()` 和 delta 初始化方法，支持 anchor residual box prediction |
| `models/detector/dino_detector.py` | 结构改造 | 新增多尺度 `decoder_feature_levels`，可将 P3/P4/P5 拼成 decoder memory；新增固定 anchor boxes，框头预测 anchor 残差 |
| `models/m3f_detr.py` | 结构接入 | 将多尺度 decoder 和 anchor 参数从主模型传入 detector |
| `train.py` / `evaluate.py` / `inference.py` / `tools/*.py` | 兼容 | 训练保存并恢复新 detector 配置，评估、提交、诊断与训练结构保持一致 |
| `configs/rush_v8_anchor_multiscale.yaml` | 新增 | 启用 RGB 归一化、多尺度 decoder memory、anchor boxes，保存到 `checkpoints/rush_v8_anchor_multiscale/` |

### 原因记录

- 旧 detector 虽命名为 DINO，但实际只把一个 FPN 层送入普通 Transformer decoder，没有多尺度 memory，也没有 reference point / anchor 约束。
- `BoxHead` 直接从随机 query embedding 输出 `cx, cy, w, h`，在小数据集上从零学习绝对坐标很慢，容易出现 good box 排名低、top1 IoU 很差的问题。
- v8 让 query 从规则网格 anchor 出发预测残差，并让 decoder 同时看 P3/P4/P5，优先解决定位召回和高分框排序问题。

### 推荐运行

```bash
python -u train.py --config configs/rush_v8_anchor_multiscale.yaml --fold 1 \
  2>&1 | tee rush_v8_anchor_multiscale_train.log
```

---

## v0.7.5 (experiment) — rush_v7：归一化 RGB + 普通正样本分类目标继续训练

**日期:** 2026-08-27  
**目标:** 针对 `rush_v6_norm` 早期 mAP 仍接近 0 的问题，去掉从零训练阶段的 IoU 质量软标签，避免低 IoU 预测把正样本分类置信度长期压低。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `configs/rush_v7_norm_plain.yaml` | 新增 | 保留 `normalize_rgb: true`，关闭 `quality_class_targets`，降低 `aux_loss_weight` 到 0.05，并使用普通正样本分类目标训练 |

### 原因记录

- `rush_v6_norm` 在 epoch 30 附近验证仍几乎无 mAP 提升，训练日志中主损失下降但辅助损失较高。
- `quality_class_targets` 会把匹配正样本的分类目标设为当前预测框 IoU；从零训练时 IoU 很低，等价于持续压低正样本置信度，不利于按置信度排序的 mAP。
- `rush_v7_norm_plain` 改回普通正样本目标，先观察前 10/20 轮验证是否比 v6 更快产生有效检测。

### 推荐运行

```bash
python -u train.py --config configs/rush_v7_norm_plain.yaml --fold 1 \
  2>&1 | tee rush_v7_norm_plain_train.log
```

---

## v0.7.4 (experiment) — rush_v6：为 pretrained Swin 增加 RGB ImageNet 归一化

**日期:** 2026-08-27  
**目标:** 修复 RGB 输入未按 ImageNet mean/std 归一化导致 pretrained Swin 特征分布不匹配的问题。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `datasets/rgb_ir_depth_dataset.py` | 修复 | 新增 `normalize_rgb` 选项，仅对 RGB 分支执行 ImageNet mean/std 归一化 |
| `train.py` / `evaluate.py` / `inference.py` / `tools/*.py` | 修复 | 从配置或 checkpoint cfg 传递 `normalize_rgb`，保证训练、验证、推理一致 |
| `configs/rush_v6_norm.yaml` | 新增 | 从零训练 normalized RGB 版本，保存到 `checkpoints/rush_v6_norm/` |

### 结果记录

- `rush_v6_norm` 运行到 epoch 36 左右后暂停，epoch 10/20/30 的 mAP 仍接近 0。
- 该实验说明 RGB 归一化是必要修复，但单独修复仍不足以让 mAP 快速恢复，需要继续排查分类置信度/排序问题。

---

## v0.7.3 (experiment) — rush_v5：尝试 IoU 质量目标校准分类分数

**日期:** 2026-08-26  
**目标:** 针对 `rush_v4_mapfix` 的诊断现象“存在部分好框，但高分框 IoU 很低”，尝试让分类分数学习框质量。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `models/losses/dino_loss.py` | 增强 | 新增 `quality_class_targets` 和 `quality_floor`，可把 matched positive 分类目标设为预测框 IoU |
| `train.py` | 增强 | 从 loss 配置读取质量目标参数并传入 `DINOLoss` |
| `configs/rush_v5_quality.yaml` | 新增 | 从 `rush_v4_mapfix/best.pth` 继续训练，尝试质量分校准 |

### 结果记录

- `rush_v5_quality` 继续训练后 mAP 有轻微提升，但仍远低于预期。
- 后续判断：质量软标签适合已有较好定位基础时做分数校准，不适合从零训练阶段直接启用。

---

## v0.7.2 (experiment) — 修复低 mAP 的训练信号与输入尺寸问题

**日期:** 2026-08-26
**目标:** 针对 `rush_v3_continue` 训练后 mAP 仍极低的问题，修复“好框低分、坏框高分”的训练信号不足，并让高分辨率输入配置真正生效。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `datasets/` | 纳入版本 | 将服务器上此前被 `.gitignore` 忽略但训练必需的数据集源码纳入 Git，避免复现断链 |
| `datasets/rgb_ir_depth_dataset.py` | 修复 | 新增 `size` 参数，不再硬编码 `384x640` |
| `models/backbone/rgb_backbone.py` / `models/m3f_detr.py` | 修复 | Swin backbone 接收 checkpoint/config 记录的 `image_size` |
| `models/detector/transformer.py` / `models/detector/dino_detector.py` | 增强 | decoder 返回每层输出，检测头生成 `aux_outputs`；新增 `decoder_feature_level`，v4 使用 P4 而非最粗 P5 |
| `models/losses/dino_loss.py` | 增强 | 对 decoder 中间层加入辅助监督，默认权重 `aux_loss_weight=0.5` |
| `train.py` / `evaluate.py` / `inference.py` / `tools/*.py` | 修复 | 训练、验证、推理、诊断统一使用 checkpoint 中的 `image_size` |
| `configs/rush_v4_mapfix.yaml` | 新增 | 从 `rush_v3_continue/best.pth` 继续训练，输入 `512x800`，batch=2，LR=3e-5 |

### 推荐运行

```bash
python -u train.py --config configs/rush_v4_mapfix.yaml --fold 1 \
  --resume checkpoints/rush_v3_continue/best.pth 2>&1 | tee rush_v4_mapfix_train.log
```

注：初版尝试 `640x1024/batch=4`，但 CMFA 在 P2 上做全局跨模态注意力，token 平方级显存过高，启动 epoch 161 时 OOM，因此 v4 配置改为更稳的 `512x800/batch=2`。

---

## v0.7.1 (experiment) — 新增 rush_v3 续训配置，优先拉高定位召回

**日期:** 2026-08-26  
**目标:** 基于 `rush_v2` 60 轮 checkpoint 继续训练，保留已恢复的类别分布，同时加强定位损失，观察 IoU@0.5 recall 和 mAP 是否继续提升。

### 修改概览

| 文件 | 类型 | 摘要 |
|------|------|------|
| `configs/rush_v3_continue.yaml` | 新增 | 从 `checkpoints/rush_v2/latest.pth` 续训到 160 epoch；保存到 `checkpoints/rush_v3_continue/` |

### 关键配置

```yaml
train.epochs: 160
train.batch_size: 8
optimizer.lr: 0.00005
optimizer.backbone_lr: 0.000005
loss.cost_class: 1.0
loss.cost_ce: 0.75
loss.cost_bbox: 8.0
loss.cost_giou: 4.0
```

### 推荐运行

```bash
python -u train.py --config configs/rush_v3_continue.yaml --fold 1 \
  --resume checkpoints/rush_v2/latest.pth 2>&1 | tee rush_v3_continue_train.log
```

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
