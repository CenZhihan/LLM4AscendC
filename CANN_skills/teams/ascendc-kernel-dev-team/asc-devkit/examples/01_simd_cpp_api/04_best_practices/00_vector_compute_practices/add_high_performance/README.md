# Add性能调优样例

## 概述

本样例以加法为例，介绍基于静态Tensor方式编程的性能调优方法。整个调优过程分为七个步骤（case 0-6），逐步展示从标量运算到向量运算、从单核到多核、从基础实现到深度优化的完整调优路径。

**优化路径**：
- Case 0: 单核标量版本（基准）
- Case 1: 单核向量版本
- Case 2: 多核均匀切分 + 小块搬运
- Case 3: 多核均匀切分 + 大块搬运
- Case 4: 多核均匀切分 + 双缓冲优化
- Case 5: 多核均匀切分 + 双缓冲 + L2Cache bypass
- Case 6: 多核均匀切分 + 双缓冲 + L2Cache bypass + 避免Bank Conflict

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── 00_add_high_performance
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│   │   └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   ├── add.asc                 // Ascend C样例实现（包含7个优化case）
```

## 样例描述

- **样例功能**：

  样例实现的是固定shape为8192×8192的两个矩阵相加。

  Add的计算公式为：

$$
  z = x + y
$$

  - x：输入，形状为[8192, 8192]，数据类型为half；
  - y：输入，形状为[8192, 8192]，数据类型为half；
  - z：输出，形状为[8192, 8192]，数据类型为half；

## 样例实现

### 性能指标说明

**表1 AI Core 性能指标字段说明表**
| 字段名 | 字段含义 |
|:---:|:---|
|Task Duration(μs)|Task整体耗时，包含调度到加速器的时间、加速器上的执行时间以及响应结束时间。|
|aiv_time|Task在AI Vector Core上的理论执行时间，单位为μs。|
| aiv_vec_time(μs) | vec类型指令（向量类运算指令）耗时，单位μs。 |
| aiv_vec_ratio | vec类型指令（向量类运算指令）的cycle数在total cycle数中的占用比。 |
| aiv_scalar_time(μs) | scalar类型指令（标量类运算指令）耗时，单位μs。 |
| aiv_scalar_ratio | scalar类型指令（标量类运算指令）的cycle数在total cycle数中的占用比。 |
| aiv_mte2_time(μs) | mte2类型指令（GM->UB搬运类指令）耗时，单位μs。 |
| aiv_mte2_ratio | mte2类型指令（GM->UB搬运类指令）的cycle数在total cycle数中的占用比。 |
| aiv_mte3_time(μs) | mte3类型指令（UB->GM搬运类指令）耗时，单位μs。 |
| aiv_mte3_ratio | mte3类型指令（UB->GM搬运类指令）的cycle数在total cycle数中的占用比。 |

### Case 0: 单核标量版本（基准程序）

**实现方式**：参考 `KernelAdd::ProcessScalar()` 函数实现

基准程序实现了`half`类型的两组输入数据的加法，采用`for`循环`scalar`运算的方式进行计算。

**关键代码**：
```cpp
for (uint32_t i = 0; i < curLen; i++) {
      float xVal = (float)xLocal.GetValue(i);
      float yVal = (float)yLocal.GetValue(i);
      zLocal.SetValue(i, (half)(xVal + yVal));
    }
```

**样例配置**：
- 单核标量运算
- `dataCopyLen = 4096` 为每次搬运的数据量元素个数
- 单次搬运的数据量为 4096 * 2B = 8192 Byte，单次scalar处理的数据量为 4 Byte

**性能数据**：

| Task Duration(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1239734.12 | 1239733.62 | 0.015 | 0% | 1233756.661 | 99.5% | 5889.469 | 0.005 | 2426.024 | 0.002 |

**优化效果分析**：
- 端到端耗时：**1239734.12μs**（约1.24秒）
- 标量指令耗时：1233756.661μs，占比 **99.5%**
- 向量指令耗时：0.015μs，占比接近0%
- 性能瓶颈：标量运算串行执行，无法利用硬件并行能力，该场景仅作为Add运算性能对比样例，在实际业务场景中不建议用户使用Scalar运算

**原理说明**：
- 标量运算每次只能处理1个数据元素，需要逐元素循环
- AI Core的硬件优势在于向量/矩阵并行计算，标量运算无法发挥硬件能力

**性能优化建议**：
> ⚠️ **避免标量循环，使用向量指令**
> 
> 在Ascend C编程中，应避免使用`for`循环配合`GetValue/SetValue`的标量运算。使用`AscendC::Add`等向量指令可带来数量级的性能提升。

---

### Case 1: 单核向量版本

**实现方式**：参考 `KernelAdd::ProcessSingle()` 函数实现

将标量运算转换为向量运算，使用`AscendC::Add`向量指令替代标量循环，大幅提升计算效率

**关键代码**：
```cpp
AscendC::Add(zLocal, xLocal, yLocal, curLen);
```

**样例配置**：
- 单核运算
- `dataCopyLen = 4096` 为每次搬运的数据量元素个数
- 单次搬运操作`DataCopy`的数据量为 8192 Byte
- 单次`Add`处理两个输入`Tensor`，处理的总数据量为 16384 Byte


**性能数据**：

| Task Duration(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 6906.26 | 6905.68 | 761.649 | 0.11 | 337.964 | 0.049 | 6007.112 | 0.87 | 2705.84 | 0.392 |

**优化效果分析**：
- 端到端性能：6906.26μs，相比Case 0提升 **99.5%**
- 标量指令耗时：从1233756.66μs降至337.964μs，减少 **99.97%**
- 向量指令耗时：761.649μs，占比11.1%
- 数据搬运耗时：6007.112μs，占比87%，搬运流水串行提升

**原理说明**：
- 向量指令单次可处理多个数据元素（本例中单次处理 4096*2 个half元素）
- 向量单元的并行计算能力远超标量单元
- 但数据搬运成为瓶颈，说明计算速度已经快于数据供给，单核时数据搬运的请求量不足，导致带宽未用满

**性能优化建议**：
> 💡 **使用向量指令替代标量循环**
> 
> 使用`AscendC::Add`、`AscendC::Mul`等向量API替代逐元素的标量循环，可充分利用AI Core的向量计算单元，获得百倍以上的性能提升。

> 💡 **不建议用户仅使用单核**

**下一步优化方向**：
- 数据搬运（MTE2）占比87%，成为主要瓶颈
- 需要通过多核并行和增大搬运粒度来提升带宽利用率

---

### Case 2: 多核均匀切分 + 小块搬运

**实现方式**：参考 `KernelAdd::Process()` 函数实现

开启多核并行计算，将8192×8192的矩阵切分到多个AIV Core上并行处理，采用均匀切分策略。

**样例配置**：
- 行方向切分6份，列方向切分8份，将数据均匀切分至48个核运算
- `dataCopyLen = 4096` 为每次搬运的数据量元素个数
- 单次搬运操作`DataCopy`的数据量为 4096 * 2B = 8192 Byte
- 单次`Add`处理两个输入`Tensor`，处理的总数据量为 16384 Byte

**关键代码**：
```cpp
// 均匀切分计算每个核处理的行列数
uint32_t baseCoreN = totalN / splitN;
uint32_t remainderN = totalN % splitN;
if (blockIdxN < remainderN) {
    actualCoreN = baseCoreN + 1;  // 均匀分配余数
}
```

**性能数据**：

| Task Duration(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 312.22 | 306.29 | 15.897 | 0.052 | 8.446 | 0.028 | 223.329 | 0.729 | 54.891 | 0.179 |

**优化效果分析**：
- 端到端任务耗时为312.22μs，相比Case 1提升 **95.5%**
- 数据搬运MTE2耗时为223.329μs，占比72.9%

**原理说明**：
- 48个AIV Core并行处理，理论上可获得48倍加速
- 实际加速比22.1倍，低于理论值的原因：
  - 数据搬运仍是瓶颈（mte2占比72.9%）
- 均匀切分确保各核负载均衡，避免出现部分核空闲、部分核繁忙的负载不均问题

**性能优化建议**：
> 💡 **充分利用多核并行，采用均匀切分策略**
> 
> 1. 将数据均匀切分到多个AI Core，实现并行计算
> 2. 使用均匀切分策略（余数分配到前几个核），确保负载均衡
> 3. 切分粒度需考虑：核数、数据量、UB空间大小

**下一步优化方向**：
- MTE2占比72.9%，MTE3占比17.9%，搬运仍是瓶颈
- 计算仅占5.2%，说明"计算快、搬运慢"
- 可通过增大单次搬运数据量提升带宽利用率
---

### Case 3: 多核均匀切分 + 大块数据搬运

**实现方式**：参考 `KernelAdd::Process()` 函数实现

为了充分利用带宽资源，增大搬运指令的数据量。

**样例配置**：
- 行方向切分6份，列方向切分8份，将数据均匀切分至48个核运算
- `dataCopyLen = 16384` 为每次切分的数据量元素个数（4倍于Case2）
- 单次搬运操作`DataCopy`的数据量为 32678 Byte
- 单次`Add`处理两个输入`Tensor`，处理的总数据量为 65536 Byte


**性能数据**：

| Task Duration(μs)  | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 270.76 |266.22 | 12.853 | 0.048 | 4.949 | 0.019 | 188.331 | 0.707 | 54.936 | 0.206 |

**优化效果分析**：
- 端到端性能：270.76μs，相比Case 2提升 **15.3%**
- 通过增大单次搬运的数据量，MTE2耗时从223.329μs降至188.331μs，减少 **15.7%**

**原理说明**：
- 增大单次搬运数据量可减少搬运次数
- 每次搬运的启动开销固定，搬运更多数据可摊薄开销
- 更大的连续数据块可更好地利用内存带宽
- 但数据量受限于UB空间大小（本例UB需容纳x、y、z三份数据）

**性能优化建议**：
> 💡 **增大单次数据搬运量，减少搬运次数**
> 
> 1. 在UB空间允许的范围内，尽量增大`dataCopyLen`
> 2. 使用连续的大块数据搬运，避免频繁的小数据块搬运
> 3. 需平衡UB空间使用和搬运效率


> ⚠️ **注意：dataCopyLen并非越大越好**
> 
> 在Case 3的基础上，如果进一步增大dataCopyLen（如从16384增至16512），端到端性能基本持平（271.5μs vs 270.76μs，+0.27%）。建议结合数据总量、UB空间、对齐要求综合考虑，确定最优的dataCopyLen值。

---

### Case 4: 双缓冲优化

**实现方式**：参考 `KernelAdd::ProcessDoubleBuffer()` 函数实现

采用双缓冲（Double Buffer）技术，实现数据搬运与计算的流水线并行，隐藏内存访问延迟。

**关键代码**：
```cpp
// Ping-Pong双缓冲地址
static constexpr uint32_t xAddrPing = 0;
static constexpr uint32_t yAddrPing = MAX_DATA_COPY_LEN * sizeof(half);
static constexpr uint32_t zAddrPing = yAddrPing + MAX_DATA_COPY_LEN * sizeof(half);
static constexpr uint32_t xAddrPong = zAddrPing + MAX_DATA_COPY_LEN * sizeof(half);
static constexpr uint32_t yAddrPong = xAddrPong + MAX_DATA_COPY_LEN * sizeof(half);
static constexpr uint32_t zAddrPong = yAddrPong + MAX_DATA_COPY_LEN * sizeof(half);

// 双缓冲流水线：交替使用两个事件ID和两组缓冲区
for (uint32_t loopIdx = 0; loopIdx < totalBlocks; loopIdx++) {
    int32_t eventID = (loopIdx % 2 == 0 ? EVENT_ID0 : EVENT_ID1);
    AscendC::LocalTensor<half> &xLocal = (loopIdx % 2 == 0 ? xPing : xPong);
    // ... 数据搬运和计算，使用对应的eventID同步
    AscendC::Add(zLocal, xLocal, yLocal, curLen);
}
```

**样例配置**：
- 行方向切分6份，列方向切分8份，将数据均匀切分至48个核运算
- `dataCopyLen = 16384` 为每次切分的数据量元素个数
- 单次搬运操作`DataCopy`的数据量为 32678 Byte
- 单次`Add`处理两个输入`Tensor`，处理的总数据量为 65536 Byte
- 将待处理的数据一分为二，由此数据的进出搬运和Vector计算实现并行执行

**内存布局**：

```
UB内存分配（双缓冲）：
┌──────────────┐
│  xPing       │  0x00000
│  16384*2B    │
├──────────────┤
│  yPing       │  0x08000 (32768)
│  16384*2B    │
├──────────────┤
│  zPing       │  0x10000 (65536)
│  16384*2B    │
├──────────────┤
│  xPong       │  0x18000 (98304)
│  16384*2B    │
├──────────────┤
│  yPong       │  0x20000 (131072)
│  16384*2B    │
├──────────────┤
│  zPong       │  0x28000 (163840)
│  16384*2B    │
└──────────────┘
```

**性能数据**：

| Task Duration(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 265.96 | 261.15 | 12.853 | 0.049 | 4.332 | 0.017 | 251.088 | 0.961 | 95.478 | 0.336 |

**优化效果分析**：
- 端到端性能：265.96μs，相比Case 3提升 **1.8%**
- MTE2耗时从188.331μs增至251.088μs（+33.3%），MTE3耗时从54.936μs增至95.478μs（+73.8%），此时从串行的纯读带宽变成混合读写带宽，因此耗时增加，用户应需更多地关注端到端耗时的减少
- 由于开启了双缓冲，在流水中搬运和计算并行执行，隐藏数据搬运时间并降低Vector指令的等待时间

**原理说明**：
- **Ping-Pong机制**：
  - Ping缓冲区进行计算时，Pong缓冲区进行数据搬运
  - 交替执行，实现计算与搬运的流水线并行，如下图所示

    <img src="figure/DoubleBuffer.png" width="50%">

**性能优化建议**：
> 💡 **使用双缓冲实现搬运与计算并行**
> 
> 1. 当计算与搬运时间相近时，双缓冲收益最大
> 2. 需要足够的UB空间（约2倍单缓冲空间）
> 3. 使用独立的Event ID管理两组缓冲区的同步

**下一步优化方向**：
- 双缓冲的收益有限，说明瓶颈在搬运速度本身
- 可尝试L2 Cache优化来提升搬运效率
---

### Case 5: 双缓冲 + L2 Cache bypass

**实现方式**：参考 `KernelAdd::ProcessDoubleBufferL2Bypass()` 函数实现（内部调用 `ProcessDoubleBuffer()`，区别在于 `Init` 时设置 `enableL2Bypass=true`）

在双缓冲基础上，对于只需要载入一次的数据量可以设置L2 Cache bypass，直接从GM载入UB。

**关键代码**：
```cpp
// 在Init时设置L2 Cache bypass
if (enableL2Bypass) {
    xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
}
// ProcessDoubleBufferL2Bypass内部调用ProcessDoubleBuffer
```

**样例配置**：
- 行方向切分6份，列方向切分8份，将数据均匀切分至48个核运算
- `dataCopyLen = 16384` 为每次切分的数据量元素个数
- 单次搬运操作`DataCopy`的数据量为 32678 Byte
- 单次`Add`处理两个输入`Tensor`，处理的总数据量为 65536 Byte
- 将待处理的数据一分为二，由此数据的进出搬运和Vector计算实现并行执行

**L2 Cache策略**：
- xGm：禁用L2 Cache（一次性读取）
- yGm：禁用L2 Cache（一次性读取）

**性能数据**：

| Task Duration(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 187.1| 183.62 | 12.853 | 0.07 | 5.416 | 0.029 | 171.163 | 0.932 | 81.061 | 0.441 |

**优化效果分析**：
- 端到端性能：187.1μs，相比Case 4提升 **29.7%**
- MTE2耗时：从251.088μs降至171.163μs，减少 **31.8%**
- MTE3耗时：从95.478μs降至81.061μs，减少 **15.1%**
- 向量指令耗时：12.853μs，保持不变

**原理说明**：
- **L2 Cache的作用**：
  - L2 Cache是AI Core和HBM之间的缓存层
  - 重复访问的数据可从L2 Cache读取，速度更快
- **流式访问特点**：
  - Add算子的输入数据只读取一次，不存在数据复用
  - 由于本样例场景数据量较大，超出L2 Cache大小，导致数据写回操作，从而引起额外耗时

**性能优化建议**：
> 💡 **合理采用L2 Cache bypass**
> 
> 1. 对于只读取一次的输入数据（如本例的x、y），设置`SetL2CacheHint(CACHE_MODE_DISABLE)`
> 2. 对于需要重复访问的数据（如卷积的权重），保留L2 Cache
> 3. 建议用户按照实测数据进行配置优化，在实际的模型和训练场景中，需要结合上下游算子进行合理配置

**下一步优化方向**：
- 搬运效率已提升，但向量指令效率仍有优化空间
- 可尝试优化UB内存布局避免Bank Conflict
---

### Case 6: 双缓冲 + L2 Cache bypass + 避免Bank Conflict

**实现方式**：参考 `KernelAdd::ProcessDoubleBufferBankConflict()` 函数实现

在双缓冲+L2 Cache bypass基础上，优化内存地址布局，避免UB（Unified Buffer）的Bank Conflict，实现最优性能。

**关键代码**：
```cpp
// 设置L2 Cache bypass
xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
yGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

// 优化后的地址布局（避免Bank Conflict）
static constexpr uint32_t xAddrPingBC = 0;
static constexpr uint32_t yAddrPingBC = BANK_CONFLICT_DATA_COPY_LEN * sizeof(half);
static constexpr uint32_t xAddrPongBC = MAX_DATA_COPY_LEN * sizeof(half) * 2;
static constexpr uint32_t yAddrPongBC = xAddrPongBC + BANK_CONFLICT_DATA_COPY_LEN * sizeof(half);
static constexpr uint32_t zAddrPingBC = MAX_DATA_COPY_LEN * sizeof(half) * 4;
static constexpr uint32_t zAddrPongBC = zAddrPingBC + BANK_CONFLICT_DATA_COPY_LEN * sizeof(half);
```

**样例配置**：
- 行方向切分6份，列方向切分8份，将数据均匀切分至48个核运算
- `dataCopyLen = 16256` 为每次切分的数据量元素个数
- 单次搬运操作`DataCopy`的数据量为 32512 Byte
- 单次`Add`处理两个输入`Tensor`，处理的总数据量为 65204 Byte
- 将待处理的数据一分为二，由此数据的进出搬运和Vector计算实现并行执行

**内存布局优化**：

未优化前的UB Bank内存布局（即case5）
<img src="figure/UBBankConflict.png" width="100%">

可以看到这样同时存在一个bank内的读写冲突，一个bankgroup内的读读冲突以及写写冲突。

优化后的UB Bank内存布局
<img src="figure/UBBankConflictResolution.png" width="90%">

由于vec指令一拍读取256B的数据（即同时读取8个block的数据），如上图xping、yping的起始地址正好错开了256B，有效消解了ub bank冲突。

**Bank Conflict详解**：
- UB分为多个Bank Group，同时读写同一Bank Group会导致冲突
- 通过调整dataCopyLen（16384→16256）使数据起始地址偏移
- 确保vec指令一拍访存的数据分布在不同的Bank Group

**性能数据**：

| Task Duration(μs) | aiv_time(μs) | aiv_vec_time(μs) | aiv_vec_ratio | aiv_scalar_time(μs) | aiv_scalar_ratio | aiv_mte2_time(μs) | aiv_mte2_ratio | aiv_mte3_time(μs) | aiv_mte3_ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 183 | 178.32 | 7.012 | 0.039 | 11.332 | 0.064 | 169.776 | 0.952 | 43.292 | 0.243 |

**优化效果分析**：
- 端到端性能：183μs，相比Case 5提升 **2.2%**
- 向量指令耗时：从12.853μs降至7.012μs，减少 **45.4%**
- MTE3耗时：从81.061μs降至43.292μs，减少 **46.6%**

**原理说明**：
- **Bank Conflict问题**：
  - UB（Unified Buffer）分为多个Bank Group
  - 向量指令一次读写的数据如果落在同一个Bank，会产生读写冲突
  - 向量指令一次读/写的数据如果落在同一个Bank Group，会产生读读冲突或写写冲突
  - Bank Conflict会导致访存串行化，降低向量指令效率
- **解决方法**：
  - 减小dataCopyLen（16384→16256），使数据起始地址产生偏移
  - 重新设计内存布局，确保同一时刻访问的数据分布在不同Bank

**性能优化建议**：
> 💡 **优化UB内存布局，避免Bank Conflict**
> 
> 1. 当`aiv_vec_time`异常高时，可能存在Bank Conflict
> 2. 通过调整dataCopyLen或内存布局偏移，使数据分布在不同Bank Group 
> 4. Bank Conflict优化对vector-bound场景收益明显

**最终性能总结**：
- 相比基准Case 0：性能提升 **6774倍**（1239734.12μs → 183μs）
- 相比单核向量Case 1：性能提升 **37.7倍**（6906.26μs → 183μs）


---

## 性能对比总结

下表为本样例在Atlas A2训练系列产品上运行的性能数据对比：

| Case | 优化策略 | 核数 | dataCopyLen | Task Duration(μs) | aiv_vec_time(μs) | 理论vector耗时(μs) | 读带宽(TB/S) | 读写混合带宽(TB/S) | 端到端耗时相对Case 0 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 单核标量（基准） | 1 | 4096 | 1239734.12 | NA | 283.405 | 0.0456 | NA | 1x |
| 1 | 单核向量 | 1 | 4096 | 6906.26 | 761.649 | 283.405 | 0.0447 | NA | 179.6x |
| 2 | 多核均匀切分 | 48 | 4096 | 312.22 | 15.897 | 5.904 | 1.2020 | NA | 3971x |
| 3 | 增大搬运粒度 | 48 | 16384 | 270.76 | 12.853 | 5.904 | 1.4253 | NA | 4581x |
| 4 | 双缓冲 | 48 | 16384 | 265.96 | 12.853 | 5.904 | NA | 1.604 | 4661x |
| 5 | L2 Cache bypass | 48 | 16384 | 187.1 | 12.853 | 5.904 | NA | 2.353 | 6626x |
| 6 | Bank Conflict优化 | 48 | 16256 | 183 | 7.012 | 5.904 | NA | 2.372 | 6774x |

表中的“理论vector耗时”表示在当前核数配置下，仅考虑Vector计算本身时的理论执行时间。本样例的性能数据在Atlas A2训练系列产品上运行，该处理器每cycle处理128个half数据，主频为1.85GHz。理论vector耗时的计算公式为
$$
T_{\text{theory}} = \frac{M \times N}{128 \times 1.85 \times 10^9 \times \text{核数}}
$$

例如，48 核场景下：
$$
T_{\text{theory}} = \frac{8192 \times 8192}{128 \times 1.85 \times 10^9 \times 48} = \frac{67108864}{1.13664 \times 10^{13}} \approx 5.904 \times 10^{-6} \text{ s} = 5.904 \text{ μs}
$$

可以看到 Case 6 的 aiv_vec_time 为 7.012 μs，已经很接近 48 核场景下的理论耗时。

Case 0-3 未开启双缓冲，数据搬运串行执行，使用读带宽衡量性能。从 Case 4 开始开启双缓冲，此时 mte2 利用率较高，读写行为并行发生，因此读写混合带宽按总读写数据量除以 $T_{mte2}$ 进行估算。其中，读带宽的计算公式为：
$$
BW_{read} = \frac{D_{read}}{T_{mte2}}
$$

读写混合带宽的计算公式为：
$$
BW_{rw} = \frac{D_{read} + D_{write}}{T_{mte2}}
$$

式中：
- $D_{read} = M \times N \times sizeof(half) \times 2$ 为总读数据量（x和y两个输入矩阵）
- $D_{write} = M \times N \times sizeof(half)$ 为总写数据量（z输出矩阵）
- $T_{mte2}$ 为 aiv_mte2_time（GM→UB搬运耗时，μs）
- 开启双缓冲后，mte2 与 mte3 流水并行；从本样例数据看，Case 4-6 的 mte2 利用率分别为 96.1%、93.2%、95.2%，因此这里采用 $T_{mte2}$ 作为主路径时间进行估算

以 Case 3 为例（$M=N=8192$，$T_{mte2}=188.331\mu s$）：
$$
BW_{read} = \frac{8192 \times 8192 \times 2 \times 2}{188.331 \times 10^{-6}} = \frac{268435456}{188.331 \times 10^{-6}} \approx 1.425 \times 10^{9} \text{ B/s} \approx 1.425 \text{ TB/s}
$$

以 Case 6 为例（$T_{mte2}=169.776\mu s$）：
$$
BW_{rw} = \frac{8192 \times 8192 \times (2+1) \times 2}{169.776 \times 10^{-6}} = \frac{402653184}{169.776 \times 10^{-6}} \approx 2.372 \times 10^{12} \text{ B/s} \approx 2.372 \text{ TB/s}
$$

之所以读写混合带宽会高于 1.8 TB/s，是因为这里统计的不是纯读带宽，而是“读 + 写”的混合带宽。开启双缓冲后，读写流水是并行的；同时，z 写数据时会命中 L2 Cache，而 L2 的带宽很高，所以写的带宽较高。因此，在以 mte2 作为读写混合时间来估算时，分子统计了读写总数据量，最终得到的混合带宽会高于 1.8 TB/s。


**优化要点总结**：

| 优化手段 | 核心原理 | 适用场景 |
|:---|:---:|:---|
| 标量→向量 | 向量指令并行处理多个元素 | 所有计算密集型算子 |
| 单核→多核 | 多核并行，负载均衡 | 大数据量场景 |
| 增大搬运粒度 | 减少搬运次数，摊薄启动开销 | 搬运密集型场景 |
| 双缓冲 | 搬运与计算流水线并行 | 计算与搬运时间相近 |
| L2 Cache bypass | 避免Cache污染，减少开销 | 流式访问（只读一次） |
| Bank Conflict优化 | 优化内存布局，避免访存冲突 | Vector-bound场景 |

---

## 编译运行

### 切换Case

在 cmake 编译时通过 `-DCASE_TYPE=N` 指定要编译的 case：

```bash
cmake -DCASE_TYPE=6 ..   # 编译 case 6（可替换为0-6）
```

各 case 说明：
- `0`: 单核标量版本
- `1`: 单核向量版本
- `2`: 多核均匀切分 (dataCopyLen=4096)
- `3`: 多核均匀切分 (dataCopyLen=16384)
- `4`: 双缓冲优化
- `5`: 双缓冲 + L2Cache bypass
- `6`: 双缓冲 + L2Cache bypass + 避免Bank Conflict

### 编译执行

在本样例根目录下执行如下步骤，编译并执行算子：

- **配置环境变量**  
  请根据当前环境上CANN开发套件包的[安装方式](../../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - 默认路径，非root用户安装CANN软件包
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - 指定路径install_path，安装CANN软件包
    ```bash
    source ${install_path}/cann/set_env.sh
    ```
    
- **样例执行**
  ```bash 
  mkdir -p build && cd build;   # 创建并进入build目录
  cmake -DNPU_ARCH=2201 -DCASE_TYPE=6 ..;make -j;  # 编译指定case（NPU_ARCH: 2201=A3, 3510=A5；CASE_TYPE: 0-6）
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                           # 执行（使用编译时指定的case）
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  
  执行结果如下，说明精度对比成功。
  ```bash
  error ratio: 0.0000, tolerance:0.0001
  test pass!
  ```

### 性能分析

使用 `msprof` 工具获取详细性能数据：

```bash
msprof ./demo   # 分析性能
```

当前目录下会生成PROF_前缀的文件夹，`mindstudio_profiler_output`目录保存Host和各个Device的性能数据汇总，性能数据分析推荐查看该目录下文件

```bash
PROF_xxxx_XXXXXX
├── device_{id}
└── host
└── mindstudio_profiler_log
└── mindstudio_profiler_output    # 保存Host和各个Device的性能数据汇总
    ├── msprof_*.json
    ├── xx_*.csv
    └── README.txt
```
查看具体的性能分析结果：
```
# 查看Task Duration 以及各项数据
cat ./PROF_*/mindstudio_profiler_output/op_summary_*.csv
```
