# 基于现有 tiling_calc / tiling_validate 的目标工具实现方案

## 1. 目标

你要的目标工具，不应该重新发明一套 tiling 体系，而应该建立在仓库里已经存在的两层能力之上：

1. `tiling_calc` 已经负责做数值 tiling 候选或规划态路由。
2. `tiling_validate` 已经负责做硬件约束校验。

目标工具建议定位为一个新的高层工具，例如：

- `tiling_budget_codegen`

它的职责不是替代现有 `tiling_calc` / `tiling_validate`，而是把两者向上封装成一个完整的规划器，补齐以下当前缺口：

- UB 占用预算分配
- alignment-aware tile 搜索
- `block_dim` / `tile_length` / `loop_count` / `tail_length` 统一输出
- `TPipe` / `TQue` 申请代码生成
- 双缓冲启停策略
- tail 拆分建议

最终这个工具应该输出一份既能喂给 LLM，也能直接嵌入 kernel 代码的结构化结果。

---

## 2. 现有实现里已经可以直接复用的部分

### 2.1 节点层现状

现有节点文件：

- `generator/agent/nodes/tiling_calc.py`
- `generator/agent/nodes/tiling_validate.py`

这两层主要做三件事：

1. 从 `state.tool_choice_json.args` 和 `current_query` 合并输入。
2. 调用 `TilingRetriever` 执行真实计算或校验。
3. 把结果格式化成稳定的文本摘要，方便 agent 后续引用。

因此，目标工具也应该沿用同样模式：

- 节点层只做输入解析、结果格式化、日志记录。
- 真正的算法实现继续放在 `generator/agent/retrievers/` 下。

### 2.2 检索器层现状

核心入口在：

- `generator/agent/retrievers/tiling_retriever.py`

当前已经做了这些事情：

1. 根据 `op_type` / `op_name` / `category` 选择 tiling 路线。
2. 支持通用 elementwise / broadcast / conversion / reduction。
3. 返回两类状态：
   - `numeric_ok`：已经得到可校验的数值 tiling。
   - `planner_ok`：只得到规划态路由，尚不是完整数值 tiling。
4. 对非数值结果，`tiling_validate` 会自动跳过，不伪造输入。

这套状态机值得完整保留，因为它能明确区分：

- 当前是不是已经拿到了可执行 tiling
- 当前只是拿到了策略方向

### 2.3 当前数值约束能力

校验逻辑在：

- `generator/agent/retrievers/tiling_validation.py`

当前已经覆盖：

1. `repeat_times <= 255`
2. 对齐约束
   - reduction 使用 32B 对齐
   - 其他场景使用 256B 对齐
3. UB 容量约束
4. `block_num` 上限约束
5. `tile_length > 0`
6. 低 UB 利用率 / 接近 UB 上限等 warning

因此，目标工具不应该自己复制一套 validator，而应该在候选结果确定后，继续复用现有 `validate_tiling_params()`。

### 2.4 当前数值 tiling 里可直接复用的公式

通用 tiling 在：

- `generator/agent/retrievers/tiling_generic.py`

里面已经定义了几条关键规则：

1. 每元素字节数来自 `dtype_bytes(dtype)`。
2. 双缓冲默认通过 `double_buffer_slots = 2` 进入 UB 预算。
3. `tile_length` 会被对齐到 256B 对齐粒度。
4. `repeat_times` 按 dtype 推出：
   - 2B 及以下：每次 repeat 128 个元素
   - 4B：每次 repeat 64 个元素
   - 更大：每次 repeat 32 个元素
5. 若 `repeat_times > 255`，则主动缩小 tile。

这部分完全可以作为目标工具的初始候选生成器。

---

## 3. 当前实现和目标工具之间的差距

你要求的目标输出是：

- `block_dim`
- `tile_length`
- `loop_count`
- `tail_length`
- UB 占用估算与预算分配表
- 标准化 C++ `Init()` 代码片段
- tiling 策略建议

现有工具还缺下面几项：

1. 缺少统一的高层输入 schema
   - 现在主要围绕 `total_elements`、`input_shape`、`output_shape`、`reduction_axes`。
   - 还没有面向“多输入/多输出 tensor + pipeline stage”的建模。

2. 缺少 UB 预算明细表
   - 当前只有一个总量字段 `ub_usage_bytes`。
   - 没有分 stage、分 queue、分 depth 的预算拆分。

3. 缺少代码生成
   - 当前只返回数字，不产出 `TPipe::InitBuffer(...)` 模板。

4. 缺少双缓冲策略决策
   - 当前 generic tiling 默认按双缓冲预算估算。
   - 但没有输出“适不适合开双缓冲，哪些 stage 应该开，哪些 stage 不该开”。

5. 缺少 tail 处理建议
   - 现有结果里有 `tail_num_last_core`。
   - 但没有把它转成“是否拆 tail、是否需要 pad、是否独立处理尾块”的建议。

---

## 4. 推荐实现方式

## 4.1 不改旧工具语义，新增一个高层工具

建议新增以下文件：

- `generator/agent/nodes/tiling_budget_codegen.py`
- `generator/agent/retrievers/tiling_budget_codegen.py`
- `generator/agent/tests/test_tiling_budget_codegen.py`

同时接入：

- `generator/agent/agent_state.py`
- `generator/agent/builtin_tools.py`
- `generator/agent/agent_config.py`

不要把这套能力直接塞进 `tiling_calc.py`。原因很直接：

1. `tiling_calc` 当前语义很清晰，就是“给定 workload 生成 tiling 候选”。
2. 目标工具比它更高一层，已经包含预算、代码生成、策略建议。
3. 混在一起会导致已有测试语义漂移。

### 4.2 工具层级关系

推荐关系如下：

1. `tiling_calc`
   - 负责生成 baseline tiling 候选。
2. `tiling_validate`
   - 负责验证数值候选是否满足底层硬件约束。
3. `tiling_budget_codegen`
   - 负责基于 baseline 候选做预算规划、stage 拆分、代码生成、最终建议。

也就是说，目标工具内部应当显式复用：

1. `TilingRetriever.compute_tiling()`
2. `validate_tiling_params()`

---

## 5. 输入设计

## 5.1 推荐输入 schema

建议这个工具强制使用结构化 `args`，不要依赖 query fallback。

```json
{
  "op_name": "fused_add_relu",
  "op_type": "elementwise",
  "chip": "DAV_2201",
  "total_shape": [8388608],
  "dtype": "float16",
  "input_tensor_count": 2,
  "output_tensor_count": 1,
  "ub_total_bytes": 196608,
  "ub_reserved_bytes": 4096,
  "enable_double_buffer": true,
  "pipeline_stages": [
    {
      "stage_name": "in_x",
      "position": "VECIN",
      "buffer_role": "input",
      "per_tile_elements": 1,
      "depth": 2
    },
    {
      "stage_name": "in_y",
      "position": "VECIN",
      "buffer_role": "input",
      "per_tile_elements": 1,
      "depth": 2
    },
    {
      "stage_name": "out_z",
      "position": "VECOUT",
      "buffer_role": "output",
      "per_tile_elements": 1,
      "depth": 2
    },
    {
      "stage_name": "tmp_relu_mask",
      "position": "VECCALC",
      "buffer_role": "temp",
      "per_tile_elements": 1,
      "depth": 1
    }
  ]
}
```

### 5.2 字段解释

| 字段 | 必填 | 说明 |
|------|------|------|
| `op_name` | 否 | 算子名，用于分类和日志 |
| `op_type` | 是 | `elementwise` / `broadcast` / `reduction` / `conversion` |
| `chip` | 否 | 默认 `DAV_2201` |
| `total_shape` | 建议必填 | 用于推导 `total_elements` |
| `dtype` | 是 | 统一 dtype；后续可扩展为每 stage 单独 dtype |
| `input_tensor_count` | 是 | 输入 tensor 数量 |
| `output_tensor_count` | 是 | 输出 tensor 数量 |
| `ub_total_bytes` | 否 | 若不填则走现有默认 UB |
| `ub_reserved_bytes` | 否 | 给 framework / 临时控制信息预留 |
| `enable_double_buffer` | 是 | 是否尝试双缓冲 |
| `pipeline_stages` | 是 | 阶段定义 |

### 5.3 `pipeline_stages` 建议字段

| 字段 | 必填 | 说明 |
|------|------|------|
| `stage_name` | 是 | 阶段名 |
| `position` | 是 | `VECIN` / `VECOUT` / `VECCALC` |
| `buffer_role` | 是 | `input` / `output` / `temp` / `workspace` |
| `per_tile_elements` | 是 | 每个 tile 需要的元素倍数，通常输入输出是 1，融合中间量可能 > 1 |
| `depth` | 是 | 期望队列深度 |
| `fixed_bytes` | 否 | 固定字节数，用于与 tile 无关的控制 buffer |
| `dtype` | 否 | 不填时继承全局 dtype |
| `enable_double_buffer` | 否 | 允许该 stage 单独关闭双缓冲 |

这里的关键点是：

- `per_tile_elements` 用来描述与 `tile_length` 正相关的 buffer。
- `fixed_bytes` 用来描述与 `tile_length` 无关的 buffer。

两者同时支持，工具才足够通用。

---

## 6. 输出设计

## 6.1 推荐输出结构

```json
{
  "status": "ok",
  "block_dim": 32,
  "tile_length": 4096,
  "loop_count": 64,
  "tail_length": 2048,
  "tail_num_last_core": 6144,
  "repeat_times": 32,
  "ub_usage_bytes": 98304,
  "ub_usage_pct": 50.0,
  "ub_budget_table": [
    {
      "stage_name": "in_x",
      "position": "VECIN",
      "bytes_per_buffer": 8192,
      "depth": 2,
      "total_bytes": 16384,
      "reason": "tile_length * sizeof(dtype)"
    }
  ],
  "init_code": "...",
  "strategy_suggestions": [
    "tail_length is not alignment-safe; use split tail path with DataCopyPad",
    "double buffer is beneficial because loop_count >= 2 and UB headroom is sufficient"
  ],
  "validation": {
    "is_valid": true,
    "errors": [],
    "warnings": []
  }
}
```

### 6.2 字段与现有结果的映射

| 目标工具字段 | 现有字段来源 |
|-------------|-------------|
| `block_dim` | `block_num` |
| `tile_length` | `tile_length` |
| `loop_count` | 新增计算 |
| `tail_length` | 新增计算 |
| `tail_num_last_core` | `tail_num_last_core` |
| `repeat_times` | `repeat_times` |
| `ub_usage_bytes` | `ub_usage_bytes` |
| `ub_usage_pct` | `ub_usage_pct` |
| `validation` | `validate_tiling_params()` |

---

## 7. 核心算法设计

## 7.1 第一步：归一化 workload

先做与 `tiling_calc` 一致的输入归一化：

1. 从 `total_shape` 计算 `total_elements`。
2. 若是 reduction / broadcast / conversion，则同时保留形状信息。
3. 统一 dtype 名称。
4. 统一 chip 名称。

建议保留一个独立函数：

```python
def normalize_budget_codegen_request(args: dict) -> NormalizedPlanRequest:
    ...
```

这个函数只负责：

- 解析
- 填默认值
- 报缺项

不要在这里直接做 tiling 搜索。

## 7.2 第二步：调用现有 `tiling_calc` 生成 baseline 候选

内部直接调用：

```python
seed = tiling_retriever.compute_tiling(...)
```

处理原则：

1. 若 `seed.status == numeric_ok`
   - 直接拿它的 `block_num`、`tile_length`、`repeat_times` 作为起点。

2. 若 `seed.status == planner_ok`
   - 保留其 `strategy_kind`、`stage_summaries`。
   - 然后进入“预算搜索模式”，继续寻找真正的数值 tile。

3. 若 `seed` 是 unsupported
   - 直接返回失败，并指出缺的 structured inputs。

这样做的好处是：

- 不会破坏现有分类器和策略器。
- 新工具只补“预算与代码生成层”。

## 7.3 第三步：alignment-aware tile 搜索

这里必须完全继承当前 validator 的约束口径，否则新工具算出的 tile 可能过不了旧 validator。

### 对齐规则

沿用现有逻辑：

- reduction: 32B 对齐
- 其他: 256B 对齐

```python
alignment_bytes = 32 if operator_class == "reduction" else 256
elements_per_alignment = alignment_bytes // dtype_bytes(dtype)
```

### repeat 规则

沿用现有逻辑：

```python
if elem_size <= 2:
    elements_per_repeat = 128
elif elem_size <= 4:
    elements_per_repeat = 64
else:
    elements_per_repeat = 32

repeat_times = ceil_div(tile_length, elements_per_repeat)
```

### 搜索顺序

推荐从大到小搜索 `tile_length`，因为目标通常是：

- 提高 UB 利用率
- 降低循环次数
- 保证 repeat 和对齐合法

伪代码：

```python
candidate = align_down(seed_tile_length or max_feasible_tile, elements_per_alignment)
while candidate >= elements_per_alignment:
    repeat_times = ceil_div(candidate, elements_per_repeat)
    if repeat_times > 255:
        candidate -= elements_per_alignment
        continue

    budget = estimate_ub_budget(candidate, ...)
    if budget.total_bytes > usable_ub_bytes:
        candidate -= elements_per_alignment
        continue

    validation = validate_tiling_params(...)
    if validation.is_valid:
        accept candidate
        break

    candidate -= elements_per_alignment
```

`usable_ub_bytes` 建议定义为：

```python
usable_ub_bytes = ub_total_bytes - ub_reserved_bytes
```

## 7.4 第四步：`block_dim` 搜索

现有 generic tiling 已经会给出一个 `block_num`。新工具不需要完全推翻它，但建议增加一层修正搜索。

### 搜索范围

沿用现有 validator 的 block 上限：

- `DAV_2201`, `DAV_1001`, `DAV_2002`, `DAV_3002`: 最大 32
- 其他扩展 chip: 最大 64

### 目标函数

建议优先级：

1. 尽量提高 core 利用率
2. 避免每核 workload 太小
3. 尽量减少最后一核严重失衡

可用一个简单评分函数：

```python
score =
    occupancy_score
    - tail_penalty
    - tiny_workload_penalty
    + tile_efficiency_bonus
```

工程上不需要复杂启发式。第一版可以直接这样做：

1. 以 `seed.block_num` 为默认值。
2. 在 `seed.block_num` 附近做局部搜索，例如 `[-4, +4]`。
3. 选出满足预算且 `loop_count >= 1` 的最优值。

## 7.5 第五步：计算 `loop_count` 和 `tail_length`

在 block 维度和 tile 确定后，统一用下面公式输出：

```python
num_per_core = ceil_div(total_elements, block_dim)
loop_count = ceil_div(num_per_core, tile_length)
tail_length = num_per_core - (loop_count - 1) * tile_length if loop_count > 0 else 0
```

如果你想与当前 `tiling_calc` 的切分完全一致，也可以保留：

- `num_per_core`
- `tail_num_last_core`

并额外输出：

- `last_core_loop_count`
- `last_core_tail_length`

但为了符合你的目标输出，建议统一主输出为：

- `block_dim`
- `tile_length`
- `loop_count`
- `tail_length`

再把更细的末核信息作为补充字段。

## 7.6 第六步：UB 预算估算与预算表生成

这是目标工具新增的核心能力。

### 预算公式

对每个 stage：

```python
stage_elem_size = dtype_bytes(stage.dtype or global_dtype)
dynamic_bytes = stage.per_tile_elements * tile_length * stage_elem_size
raw_bytes = dynamic_bytes + stage.fixed_bytes
aligned_bytes = align_up(raw_bytes, alignment_bytes)
effective_depth = stage.depth
stage_total_bytes = aligned_bytes * effective_depth
```

总 UB：

```python
ub_total = sum(stage_total_bytes for stage in stages) + ub_reserved_bytes
```

### 建议输出表字段

| 列名 | 说明 |
|------|------|
| `stage_name` | 阶段名 |
| `position` | `VECIN/VECOUT/VECCALC` |
| `bytes_per_buffer` | 单 buffer 大小 |
| `depth` | 队列深度 |
| `total_bytes` | 该 stage 总占用 |
| `alignment_bytes` | 对齐粒度 |
| `formula` | 计算说明 |
| `notes` | 特殊说明 |

### 预算输出示例

| stage_name | position | bytes_per_buffer | depth | total_bytes | formula |
|------------|----------|------------------|-------|-------------|---------|
| `in_x` | `VECIN` | 8192 | 2 | 16384 | `tile_length * sizeof(fp16)` |
| `in_y` | `VECIN` | 8192 | 2 | 16384 | `tile_length * sizeof(fp16)` |
| `out_z` | `VECOUT` | 8192 | 2 | 16384 | `tile_length * sizeof(fp16)` |
| `tmp_relu_mask` | `VECCALC` | 8192 | 1 | 8192 | `tile_length * sizeof(fp16)` |
| `reserved` | `N/A` | 4096 | 1 | 4096 | `framework reserve` |

## 7.7 第七步：双缓冲策略

双缓冲不是简单地把所有 depth 都改成 2，而应该有决策逻辑。

### 什么时候建议开启

建议同时满足：

1. `enable_double_buffer == true`
2. `loop_count >= 2`
3. 输入 / 输出 stage 存在可重叠的搬运与计算
4. 双缓冲后的 UB 预算仍然通过

### 什么时候建议关闭

任一条件成立即可建议关闭：

1. `loop_count == 1`
2. 双缓冲会导致 tile 缩得太小
3. 双缓冲后 UB 利用率低且收益有限
4. temp buffer 很大，导致实际吞吐下降

### 建议规则

对不同 stage 使用不同默认策略：

| stage role | 默认 depth |
|------------|-----------|
| `input` | 2 if enabled else 1 |
| `output` | 2 if enabled else 1 |
| `temp` | 默认 1 |
| `workspace` | 默认 1 |

这里建议非常保守：

- 默认只对 `VECIN` / `VECOUT` 开双缓冲。
- `VECCALC` 临时 buffer 默认单缓冲。

这和仓库里现有示例一致，例如 leaky relu 模板使用：

- 输入队列 depth=2
- 输出队列 depth=2
- 计算临时队列 depth=1

---

## 8. `TPipe` / `TQue` 代码生成设计

## 8.1 为什么建议默认只生成 `TQue`

你要求的是“可直接放入 `Init()` 的 `TPipe/TQue` 申请代码”。

建议第一版代码生成只输出 `TQue` 风格的 staged allocation，原因有两个：

1. 仓库里现成 kernel 模板就是这种风格。
2. 当前仓库的修复规则里明确提醒过，不要在生成代码里误用 `TBuf` 的 `InitBuffer` 形态。

因此，目标工具默认生成：

- `VECIN` / `VECOUT` / `VECCALC` 都用 `TQue`
- 临时单缓冲也用 `depth=1` 的 `TQue`

等后续验证稳定后，再考虑增加 `TBuf` 作为可选输出模式。

## 8.2 成员声明模板

建议代码生成输出两段：

1. 类成员声明
2. `Init()` 内的 `pipe_.InitBuffer(...)`

### 成员声明模板

```cpp
AscendC::TPipe pipe_;
AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueX_;
AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueY_;
AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueueZ_;
AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQueue_;
```

### `Init()` 申请模板

```cpp
pipe_.InitBuffer(inQueueX_, 2, tileLength_ * sizeof(T));
pipe_.InitBuffer(inQueueY_, 2, tileLength_ * sizeof(T));
pipe_.InitBuffer(outQueueZ_, 2, tileLength_ * sizeof(T));
pipe_.InitBuffer(tmpQueue_, 1, tileLength_ * sizeof(T));
```

### 生成规则

对每个 stage：

1. `position` 映射到 `AscendC::TPosition::*`
2. `depth` 直接进入模板参数和 `InitBuffer` 第二个参数
3. buffer size 取预算表中的 `bytes_per_buffer`

即：

```python
declaration = f"AscendC::TQue<AscendC::TPosition::{pos}, {depth}> {name}_;"
init_line = f"pipe_.InitBuffer({name}_, {depth}, {bytes_per_buffer});"
```

## 8.3 融合算子场景建议

若输入输出 tensor 数量是动态的，建议工具支持批量命名：

- `inQueue0_`, `inQueue1_`, ...
- `outQueue0_`, `outQueue1_`, ...
- `tmpQueue0_`, `tmpQueue1_`, ...

这样生成器更容易与模板对接。

---

## 9. tail 策略建议

当前工具里没有把 tail 处理建议显式结构化。目标工具建议新增以下规则。

### 9.1 不拆 tail

当满足以下任一条件时，可以不拆：

1. `tail_length == 0`
2. `tail_length == tile_length`
3. `tail_length` 满足对齐要求

### 9.2 建议拆 tail

以下场景建议输出 `split_tail=true`：

1. `tail_length < elements_per_alignment`
2. `tail_length` 不满足对齐要求
3. reduction 最后一块需要单独 `DataCopyPad`
4. 双缓冲路径要求固定 tile，而尾块太小会破坏流水节奏

### 9.3 建议文本模板

```text
tail_length is not alignment-safe for the selected dtype; generate a dedicated tail path or use DataCopyPad.
```

或者：

```text
tail_length is aligned and large enough; reuse the main loop body with a final short iteration.
```

---

## 10. 结果数据结构建议

建议新增 dataclass，而不是复用 `TilingParamsResult` 直接硬塞字段。

```python
@dataclass
class UBBudgetItem:
    stage_name: str
    position: str
    bytes_per_buffer: int
    depth: int
    total_bytes: int
    alignment_bytes: int
    formula: str
    notes: str = ""


@dataclass
class TQueueCodegenResult:
    declarations: list[str]
    init_lines: list[str]
    full_init_code: str


@dataclass
class TilingBudgetCodegenResult:
    status: str
    block_dim: int | None
    tile_length: int | None
    loop_count: int | None
    tail_length: int | None
    tail_num_last_core: int | None
    repeat_times: int | None
    ub_usage_bytes: int | None
    ub_usage_pct: float | None
    ub_budget_table: list[UBBudgetItem]
    init_code: str
    strategy_suggestions: list[str]
    validation_status: str
    validation_errors: list[str]
    validation_warnings: list[str]
    seed_strategy_kind: str = ""
    seed_status: str = ""
```

这样可以避免污染现有 `TilingParamsResult`，也方便单测。

---

## 11. 节点层实现建议

`generator/agent/nodes/tiling_budget_codegen.py` 建议遵循与现有节点相同的接口风格：

```python
def tiling_budget_codegen_node(
    state: GeneratorAgentState,
    tiling_retriever: TilingRetriever = None,
) -> Dict[str, Any]:
    ...
```

### 节点职责

1. 从 `state.tool_choice_json.args` 读取结构化输入。
2. 调用新的 planner：

```python
planner.plan_budget_and_codegen(...)
```

3. 生成稳定的摘要文本，格式建议类似：

```text
TILING_BUDGET_CODEGEN_SUMMARY
summary_version=1
status=ok
block_dim=32
tile_length=4096
loop_count=64
tail_length=2048
ub_usage_bytes=98304
strategy_suggestions=[...]
init_code=...
```

4. 把结构化结果存入 state，例如：

- `tiling_budget_codegen_result`
- `tiling_budget_codegen_results`

---

## 12. 校验闭环设计

目标工具内部应该分两层校验。

### 12.1 数值校验

直接复用：

```python
validate_tiling_params(
    {
        "operator_class": operator_class,
        "tile_length": tile_length,
        "repeat_times": repeat_times,
        "ub_usage_bytes": ub_usage_bytes,
        "block_num": block_dim,
        "dtype": dtype,
    },
    chip=chip,
)
```

### 12.2 规划校验

这是新增的逻辑，建议补以下检查：

1. `sum(stage.total_bytes) + ub_reserved_bytes == ub_usage_bytes`
2. `depth == 2` 只在允许双缓冲的 stage 上启用
3. `loop_count == 1` 时给出关闭双缓冲建议
4. `tail_length` 不对齐时给出 split-tail 建议
5. `init_code` 中生成的 depth 与预算表一致

也就是说，目标工具最终的结果中应同时带：

- 底层硬件校验结果
- 上层规划一致性校验结果

---

## 13. 单测设计

建议新增测试文件：

- `generator/agent/tests/test_tiling_budget_codegen.py`

至少覆盖下面几类 case。

### 13.1 基础 elementwise 双缓冲

验证：

1. 能产出 `block_dim` / `tile_length` / `loop_count` / `tail_length`
2. 输入输出 queue depth 为 2
3. temp queue depth 为 1

### 13.2 UB 不足导致降 tile

验证：

1. planner 会缩小 `tile_length`
2. `validation.is_valid == true`
3. 预算表总和不超 UB

### 13.3 尾块不对齐

验证：

1. 输出 `split tail` 建议
2. 或输出 `DataCopyPad` 建议

### 13.4 关闭双缓冲

验证：

1. 输入输出 depth 都退回 1
2. UB 使用量显著下降

### 13.5 reduction 场景

验证：

1. 对齐粒度走 32B
2. planner 能正确继承 `tiling_calc` 的 reduction 结果

---

## 14. 推荐落地顺序

建议分三步实现，不要一次做完全部特性。

### Phase 1

先实现最小可用版本：

1. elementwise only
2. 单 dtype
3. `pipeline_stages` 仅支持 `per_tile_elements + depth`
4. 只生成 `TQue`
5. 只做 `block_dim` / `tile_length` / `loop_count` / `tail_length` + UB 表 + `InitBuffer` 代码

### Phase 2

扩展到：

1. broadcast
2. reduction
3. conversion
4. 分 stage 独立 dtype
5. planner_ok 到 numeric candidate 的桥接搜索

### Phase 3

再增加：

1. 更复杂的 tail 拆分方案
2. `TBuf` 可选输出模式
3. 更细的 pipeline overlap 建模
4. 更复杂的 fused operator budget 模型

---

## 15. 一个可以直接照着实现的最小版本

如果只做第一版，我建议把主流程写成下面这样：

```python
def plan_budget_and_codegen(request: NormalizedPlanRequest) -> TilingBudgetCodegenResult:
    seed = tiling_retriever.compute_tiling(
        total_elements=request.total_elements,
        dtype=request.dtype,
        op_type=request.op_type,
        ub_capacity_bytes=request.ub_total_bytes,
        op_name=request.op_name,
        chip=request.chip,
    )

    if not seed.supported:
        return unsupported_result_from_seed(seed)

    alignment_bytes = 32 if seed.operator_class == "reduction" else 256
    elem_size = dtype_bytes(request.dtype)
    usable_ub_bytes = request.ub_total_bytes - request.ub_reserved_bytes
    candidate_tile = align_down(seed.tile_length or request.total_elements, alignment_bytes // elem_size)

    best = None
    while candidate_tile > 0:
        budget = build_budget_table(
            tile_length=candidate_tile,
            alignment_bytes=alignment_bytes,
            stages=request.pipeline_stages,
            global_dtype=request.dtype,
            reserved_bytes=request.ub_reserved_bytes,
        )
        repeat_times = compute_repeat_times(candidate_tile, elem_size)
        validation = validate_tiling_params(
            {
                "status": "numeric_ok",
                "operator_class": seed.operator_class,
                "tile_length": candidate_tile,
                "repeat_times": repeat_times,
                "ub_usage_bytes": budget.total_bytes,
                "block_num": seed.block_num,
                "dtype": request.dtype,
            },
            chip=request.chip,
        )
        if budget.total_bytes <= usable_ub_bytes and validation.is_valid:
            best = (candidate_tile, budget, validation)
            break
        candidate_tile -= alignment_bytes // elem_size

    if best is None:
        return no_feasible_budget_result(...)

    tile_length, budget, validation = best
    block_dim = seed.block_num or 1
    num_per_core = ceil_div(request.total_elements, block_dim)
    loop_count = ceil_div(num_per_core, tile_length)
    tail_length = num_per_core - (loop_count - 1) * tile_length
    init_code = render_tque_init_code(budget.stages)
    suggestions = build_strategy_suggestions(...)

    return build_success_result(...)
```

这版已经能满足你列出的核心输出。

---

## 16. 结论

最合适的实现方式不是改写现有 `tiling_calc` / `tiling_validate`，而是在它们之上新增一个“预算 + 代码生成”层。

一句话概括这个目标工具应该怎么做：

- 用 `tiling_calc` 产出初始 tiling 候选
- 用新的 budget planner 做对齐感知 tile 搜索和 UB 分配
- 用 `tiling_validate` 做底层硬件校验
- 最后生成 `TPipe/TQue` 代码和 tail / double-buffer 策略建议

这样实现的优点是：

1. 最大化复用当前仓库已有能力。
2. 不破坏现有工具语义和测试。
3. 输出刚好补齐你要的 UB 预算表和标准化代码片段。
4. 后续可以很自然扩展到更复杂的 reduction / fused operator。