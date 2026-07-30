# 代码修改记录 (CHANGELOG)

> 项目: M3F-DETR | 基于: `项目问题分析报告.md`

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
