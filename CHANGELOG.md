# 代码修改记录 (CHANGELOG)

> 项目: M3F-DETR | 基于: `项目问题分析报告.md` (第四版)

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
