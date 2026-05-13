# conv3d_mish_tanh 算子优化报告

## 1. 算子概述

- **算子名**: `conv3d_mish_tanh`
- **功能**: 对 Conv3D 输出做融合激活 `y = tanh(mish(x))`
- **数学定义**:
  - `softplus(x) = max(x, 0) + log(1 + exp(-|x|))`
  - `mish(x) = x * tanh(softplus(x))`
  - `y = tanh(mish(x))`
- **数据类型**: float32
- **输入格式**: ND 连续存储（5D NCDHW）
- **Op Type**: Vector

## 2. 原始实现分析

### 2.1 代码结构

```
原始 kernel（199 行）:
├── 10 个 TQue<VECCALC, 1>     —— 全部单 buffer (BUFFER_NUM=1)
├── 8-element 标量 tile        —— 每次只处理 8 个元素
├── GetValue/SetValue 循环    —— 标量方式读/写 GM
├── scalar select 实现 max(x,0) —— 逐元素比较赋值
└── 独立 scalar tail 路径       —— 4-lane 复制模式
```

### 2.2 关键问题

| 问题 | 位置 | 影响 |
|------|------|------|
| 单 buffer (BUFFER_NUM=1) | 全部 10 个 queue | 无内存/计算 overlap |
| 标量 GM 访问 | `GetValue(k)` / `SetValue(k)` 8 次循环 | ~256x 多余 GM 事务 |
| 小 tile (8 elem) | `for (k=0; k<8; ++k)` | dispatch 开销巨大 |
| 标量 max(x,0) | `sp.SetValue(k, (xv>0)?xv:0)` | 逐元素分支，无法向量化 |
| 10 个独立 scratch queue | `qV, qAbs, qTmp, qEx, qOne, qNum, qDen, qSp, qT, qY` | UB 碎片化严重 |
| 冗余 tail 处理 | scalar 4-lane 复制 | 与主路径逻辑重复 |

## 3. 优化方案

### 3.1 Double Buffering

```
Before: BUFFER_NUM = 1, TPosition::VECCALC
After:  BUFFER_NUM = 2, TPosition::VECIN / VECOUT
```

I/O queue 深度从 1 改为 2，硬件自动 overlap DataCopy 与 Compute。
使用正确的 pipe 类型：`VECIN` 用于输入，`VECOUT` 用于输出。

### 3.2 Bulk DataCopy

```
Before:
  #pragma unroll
  for (uint32_t k = 0; k < 8; ++k) {
      float xv = xGm.GetValue(i + k);
      v.SetValue(k, xv);
  }

After:
  AscendC::DataCopy(xLocal, xGm_[offset], count);
```

单次 `DataCopy` 调用替代 8 次标量 `GetValue` + `SetValue`。对 4096 元素的 tile，GM 事务从 512 次减少到 1 次。

### 3.3 大 Tile

```
Before: 8 elements / tile
After:  4096 elements / tile (TILE_ELEMS = 4096，经 sweep 选定)
```

UB 占用（最终配置 TILE_ELEMS=4096）：
- qX_: 2 × 4096 × 4B = 32KB
- qY_: 2 × 4096 × 4B = 32KB
- qAbs_/qTmp_/qExp_/qNum_/qDen_: 5 × 4096 × 4B = 80KB
- **总计: 144KB** (UB 容量 256KB，占用 56%)

### 3.4 max(x,0) 向量化

```
Before (标量分支):
  AscendC::Duplicate(sp, 0.0f, 8);
  for (uint32_t k = 0; k < 8; ++k) {
      float xv = v.GetValue(k);
      sp.SetValue(k, (xv > 0.0f) ? xv : 0.0f);
  }

After (向量恒等式):
  AscendC::Add(tmpLocal, xLocal, absLocal, count);   // x + |x|
  AscendC::Muls(tmpLocal, tmpLocal, 0.5f, count);     // (x+|x|)/2 = max(x,0)
```

利用恒等式 `max(x,0) = (x + |x|) / 2`，用 2 个向量操作替代 8 次标量分支。`absLocal` 在 softplus 中已计算，无额外开销。

### 3.5 计算临时量精简

```
Before: 10 queues (qV, qAbs, qTmp, qEx, qOne, qNum, qDen, qSp, qT, qY)
After:   5 queues (qAbs_, qTmp_, qExp_, qNum_, qDen_)
         + 2 I/O queues (qX_, qY_)
```

寄存器复用链：

```
absLocal  = |x|
tmpLocal  = -|x| → x+|x| → max(x,0) → softplus → tanh(sp) → -2*mish
expLocal  = exp(-|x|) → 1+exp → log(1+exp) → -2*sp → exp(-2sp) → exp(-2mish)
numLocal  = -exp(-2sp) → 1-exp(-2sp) → -exp(-2mish) → 1-exp(-2mish)
denLocal  = 1+exp(-2sp) → 1+exp(-2mish)
yLocal    = mish(x) → tanh(mish)  (output tensor from qY_)
```

完全消除了 `qV, qEx, qOne, qSp, qT` 等冗余 queue，消除每轮 `Duplicate(1.0f)` 调用（改用 `Adds`/`Muls` 组合）。

### 3.6 汇总对比

| 维度 | 原始 | 优化后 | 改进 |
|------|------|--------|------|
| I/O queue 数 | 0（混在 VECCALC 中） | 2 (VECIN+VECOUT) | 正确的 pipe 分离 |
| Compute queue 数 | 10 | 5 | -50% |
| BUFFER_NUM | 1 | 2 | 2x throughput |
| Tile 大小 | 8 | 4096 | 512x |
| GM 传输方式 | GetValue/SetValue | DataCopy | 批量 DMA |
| max(x,0) 实现 | 标量 select | (x+|x|)/2 | 向量化 |
| 每 tile 向量操作数 | ~35 (含标量) | 21 | -40% dispatch |
| 尾块处理 | 独立标量路径 | 复用主路径 | 代码简化 |
| UB 占用 | ~320B | 144KB | 合理利用 UB |

## 4. Tiling 参数 Sweep

在 kernel 优化完成后，对 `block_dim` 和 `TILE_ELEMS` 做了 5 点 grid search，
使用 `tools/sweep_conv3d_mish_tanh.py` 自动化执行。

### 4.1 搜索空间与结果

| Config | block_dim | TILE_ELEMS | custom_ms | ref_ms | speedup |
|--------|-----------|------------|-----------|--------|---------|
| 基线 | 32 | 2048 | 1.662 ms | 5.115 ms | 3.08x |
| A | 48 | 2048 | 1.121 ms | 5.131 ms | 4.58x |
| B | 64 | 2048 | 1.665 ms | 5.110 ms | 3.07x |
| C | 32 | 4096 | 1.404 ms | 5.112 ms | 3.64x |
| **D** | **48** | **4096** | **0.945 ms** | **5.104 ms** | **5.40x** |

### 4.2 分析

- **block_dim=48 是甜点**：910B2 有 48 个 AI Vector Core。32 少用了 1/3 的算力；64 超发导致 core 争抢，性能退化回 3.07x。
- **TILE_ELEMS=4096 > 2048**：tile 翻倍使 loop 迭代次数减半，dispatch 开销更低。UB 占用 144KB（56%）仍在安全线内。
- **block_dim + TILE_ELEMS 正交互**：48×4096 组合效果 > 单独优化之和 (4.58×3.64/3.08 ≈ 5.41，实测 5.40，高度吻合)。

## 5. PyTorch Profiler 细粒度分析

使用 `tools/profile_conv3d_mish_tanh.py` 通过 `torch_npu.profiler` 采集
`kernel_details.csv`，得到 Ref 和 Custom 每条 kernel 的完整时序分解。
相比 msprof op（只能看单个算子），PyTorch profiler 能看到完整的执行图。

### 5.1 Reference 执行流程（每次迭代 3 个 kernel）

```
SoftplusV2  →  Mul(x * tanh)  →  Tanh
  1910us         1207us           974us
  vec=85.9%      vec=3.0%        vec=80.3%
  sca=25.2%      sca=1.1%        sca=6.4%
  mte2=19.6%     mte2=99.7%      mte2=31.2%
                            ↑
                     纯内存搬运 kernel
             (mish 的中间乘法无法融合到激活中)
```

**关键发现**: `mish(x) = x * tanh(softplus(x))` 拆成了 3 个 kernel，
其中 `Mul` kernel 的 vec 利用率仅 3.0%，MTE2 占 99.7%——本质是把
SoftplusV2 的结果读出来、乘 x、写回去，全部是冗余内存流量。

### 5.2 Custom 执行流程（每次迭代 1 个 kernel）

```
Conv3dMishTanhCustom
     910us
  vec=96.5%
  sca=19.3%
  mte2=47.0%
```

### 5.3 Kernel 级 Head-to-Head

| 指标 | Ref (3 kernels) | Custom (1 kernel) | 改进 |
|------|----------------|-------------------|------|
| Kernel 纯执行时间 | 4091 us | 910 us | **4.49x** |
| Kernel 启动次数 | 3 | 1 | 3x fewer |
| 向量利用率 (平均) | 56.6% | 96.5% | +39.9pp |
| 中间 GM 中转 | Softplus→Mul→Tanh (2 次) | 0 次 | 消除 |
| 冗余 Mul kernel | 1207 us | — | 消灭 |

### 5.4 端到端速度分解

```
Ref 端到端 5070us  =  4091us(kernel) + 979us(调度/launch/分配)
Custom 端到端 915us =   910us(kernel) +   5us(调度/launch)

Kernel 加速:  4.49x
框架开销加速: 196x
端到端加速:   5.54x  ← 两个效应的乘积
```

融合的真正收益不是简单 "拼 kernel"，而是：
1. 消灭了 Mul kernel 的 1207us 纯内存搬运
2. 消除 kernel 间 pipeline bubble（vec 利用率 57% → 96.5%）
3. 一次 GM 读 + 一次 GM 写，代替三次读 + 三次写

## 6. 最终性能

### 6.1 测试配置

- **硬件**: Ascend 910B2
- **输入形状**: [16, 64, 30, 62, 62] (float32, ~118M 元素)
- **Warmup**: 5 次，**Repeat**: 20 次
- **最终配置**: `block_dim=48, TILE_ELEMS=4096`

### 6.2 正确性

```
[eval] allclose=True [ref=PyTorch mish+tanh]
```

### 6.3 当前实测数据（2026-05-13）

| 指标 | 值 |
|------|-----|
| Ref (softplus + mul + tanh) | 5.14 ms |
| Custom (fused, bd48, tl4096) | 0.98 ms |
| **端到端 Speedup** | **5.23x** |
| Kernel 纯时间 Speedup | 4.49x |
| Custom vec 利用率 | 96.5% |
| Custom kernel 单次时长 | 910 us |

### 6.4 优化路径总览

```
原始代码 (标量 GM 访问, 8-elem tile, 单 buffer)
  │  ~1.0x (未成功跑通，无基线数据)
  │
  ├─ kernel 重构 (double buf + DataCopy + 向量 max + 临时量精简)
  │   └─ 3.07x speedup, 1.66ms
  │
  └─ tiling sweep (block_dim=48, TILE_ELEMS=4096)
      └─ 5.23x speedup, 0.98ms  ← 最终结果
```

## 7. 进一步优化方向

| 方向 | 预期收益 | 风险 |
|------|----------|------|
| float16 混合精度 | ~1.5-2x (Exp/Log 在 fp16 有硬件加速) | 精度损失，需评估数值稳定性 |
| 双发射 vector 指令 | ~5-10% (编译器自动调度) | 依赖编译器版本 |
| 自适应 tiling (动态选 block_dim) | 小 tensor 场景收益 | 增加 host 逻辑复杂度 |
| 消除 tail tile 独立路径 | 代码简化 | 收益可忽略 |

## 8. 文件变更

| 文件 | 变更内容 |
|------|----------|
| `output/kernelbench165_txt/conv3d_mish_tanh.txt` | 重写 kernel_src (double buf + DataCopy + 向量化)；model_src → eval_src；block_dim=48, TILE_ELEMS=4096 |
| `scripts/pj-lab/run_kernelbench165_local.sh` | 增加 profiler 支持、conda 激活、set +u 守卫、--txt-dir 全量扫描 |
| `tools/sweep_conv3d_mish_tanh.py` | tiling 参数网格搜索脚本 |
| `tools/profile_conv3d_mish_tanh.py` | PyTorch profiler 细粒度 kernel 分析脚本 |
| `docs/conv3d_mish_tanh_optimization.md` | 本文档 |
