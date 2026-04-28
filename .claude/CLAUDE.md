## Agent4AscendKernel 实验模式

> **生效条件**：以下实验模式指令仅在用户明确要求进入实验模式、或明确表达"评估工具效果""做工具实验""测试工具"等意图时生效。

### 身份

你是 Agent4AscendKernel 实验框架的主 Agent（研究者），负责系统性评估 `generator/agent/nodes` 下各工具对 AscendC Kernel 生成质量的影响。

### 实验目标

通过控制变量法，对比不同工具子集对 Kernel 生成正确性（编译通过、精度达标）的贡献，产出量化的实验报告。

### 核心原则

1. **控制变量**：每次实验只改变一个变量（启用的工具子集），其他条件（模型、温度、case 列表）保持不变。
2. **两轮迭代**：每个 case 必须经历两轮（iter1: 生成->评测; iter2: 评测反馈->再生成->最终评测）。
3. **最大轮数**：每次 iter 中 Tool Orchestrator 最多选择 5 轮工具。
4. **难度覆盖**：每个实验配置至少覆盖简单、中等、复杂三类算子。
5. **工具边界**：Tool Orchestrator 的工具列表中**严禁**包含 Claude Code 自带的通用工具（glob、grep、read、web_search 等）。主 Agent 在执行层面可以使用 Read/Bash 来驱动节点，但不能将这些原生能力暴露给 Tool Orchestrator。

### 可用工具清单（仅 generator/agent/nodes 下实现）

| 工具 key | 对应节点文件 | 说明 | 典型用途 |
|---------|-------------|------|---------|
| `kb_query` | `kb_query.py` | 知识库查询 | AscendC API 文档检索 |
| `web_search` | `web_search.py` | 网络搜索 | 在线资料补充 |
| `code_rag` | `code_rag.py` | 代码 RAG 检索 | 相关代码片段语义检索 |
| `code_search_snippet` | `code_search_snippet.py` | 结构化代码搜索 | asc-devkit 示例代码块检索 |
| `env_check_env` | `env_check.py` | 环境检查-环境 | 编译器、CANN 版本检查 |
| `env_check_npu` | `env_check.py` | 环境检查-NPU | NPU 设备状态检查 |
| `env_check_api` | `env_check.py` | 环境检查-API | API 可用性检查 |
| `kb_shell_search` | `kb_shell_search.py` | KB Shell 搜索 | Knowledge/ 目录文档 grep 搜索 |
| `api_lookup` | `api_lookup.py` | API 签名查询 | 查询具体 API 的签名和参数 |
| `api_constraint` | `api_constraint.py` | API 约束检查 | 查询 API 的使用限制 |
| `api_alternative` | `api_alternative.py` | API 替代查询 | 查找功能等价的替代 API |
| `tiling_calc` | `tiling_calc.py` | Tiling 计算 | 自动计算 tiling 参数 |
| `tiling_validate` | `tiling_validate.py` | Tiling 验证 | 验证 tiling 策略的正确性 |
| `npu_arch` | `npu_arch.py` | NPU 架构查询 | 查询芯片架构能力和限制 |
| `code_style` | `code_style_check.py` | 代码风格检查 | 检查代码是否符合规范 |
| `security_check` | `security_check.py` | 安全检查 | 检查潜在安全问题 |
| `ascend_search` | `ascend_search.py` | Ascend 文档搜索 | 在线 Ascend 文档搜索 |
| `ascend_fetch` | `ascend_fetch.py` | Ascend 文档获取 | 获取指定 URL 的 Ascend 文档内容 |
| `registered_tool` | `registered_tool.py` | 注册工具 | 调用已注册的自定义工具 |

### Case 难度分级

从 `vendor/mkb/dataset.py` 的 category 推断：

- **简单**：`activation`（如 gelu, relu, sigmoid, softmax, leaky_relu, hardtanh, elu, swish）— 元素级单算子，无复杂内存布局，tiling 直接
- **中等**：`broadcast`（如 add_bias_broadcast, clamp_broadcast）、`reduce`（如 reduce_sum）、`normalization`（如 layer_norm）、`matmul`（如 matmul_add）、`attention`（如 multi_head_attention, scaled_dot_product_attention）— 涉及多维广播、规约、tiling 策略、多 buffer 管理
- **复杂**：`convolution`（各种 conv2d/conv3d/transposed/pointwise/depthwise）、`fuse`（多算子融合如 conv2d+bn+activation）、`arch`（完整模型子图如 resnet_basic_block, transformer_block）— 复杂内存访问模式、workspace 管理、多核协同、融合优化

每个实验配置至少从每个难度等级中选 1-2 个 case，建议总共 4-6 个 case。

### 单次 Case 实验流程

```
Step 0: 任务准备
  - 从 vendor/mkb/dataset.py 读取算子名称和 category
  - 读取 vendor/mkb/reference/{category}/{op}.py 作为原始 PyTorch 架构
  - 读取固定的 one-shot 示例文件 `generator/prompts/ascendc_new_model_leaky_relu.py`（统一使用 leaky_relu 作为示例）
  - 用 generator/prompt_generators/prompt_utils.py 的 ascendc_template() 构建完整任务 prompt
  - 明确声明本实验启用的工具子集

Step 1-5: Iter 1 工具收集（最多 5 轮）
  - 调用 Tool Orchestrator Subagent
    * Prompt 复用 generator/agent/nodes/choose_tool.py 中 _build_tool_selection_prompt() 的构建逻辑
    * 传入：用户任务、已收集结果摘要、可用工具列表、剩余轮数
  - Tool Orchestrator 从启用的工具子集中选择一个工具（或 ANSWER）
  - 主 Agent 执行选中的工具节点（通过 Python 调用 generator/agent/nodes/ 下对应函数）
  - 将工具结果追加到状态，返回给 Tool Orchestrator
  - 循环直到输出 ANSWER 或达到最大轮数（5 轮）

Step 6: Iter 1 Kernel 生成
  - 收集所有工具结果，按 generator/agent/nodes/answer.py 中 _format_retrieved_content() 的格式组织证据
  - 调用 Kernel Expert Subagent
    * Prompt 复用 generator/prompt_generators/prompt_utils.py 的 ASCENDC_PROBLEM_STATEMENT + ASCENDC_OUTPUT_FORMAT_NO_EXAMPLE + ascendc_template
    * 传入：原始任务 prompt + 所有收集到的工具结果证据
  - Kernel Expert 输出单个 Python 代码块，内含 6 个字符串变量

Step 7: Iter 1 结果保存
  - txt: output/caseStudy/{experiment_tag}/{op}.txt
  - json: output/caseStudy/{experiment_tag}/{op}_trace.json（记录工具调用日志、思考过程、生成内容）

Step 8: Iter 1 评测
  - 运行 eval_operator.py 评测（见下方评测方法）
  - 收集编译状态、精度状态、性能数据、错误日志

Step 9-13: Iter 2 修复迭代（最多 5 轮）
  - 将 Iter 1 的评测结果（编译错误信息、精度偏差、失败类型）作为额外上下文追加
  - 重新调用 Tool Orchestrator（可再次选择工具获取修复信息，如 api_constraint 针对编译错误）
  - 调用 Kernel Expert 重新生成代码

Step 14: Iter 2 结果保存
  - txt: 覆盖保存到同一目录（output/caseStudy/{experiment_tag}/{op}.txt）
  - json: 追加第二轮记录到 {op}_trace.json

Step 15: Iter 2 最终评测
  - 运行 eval_operator.py
  - 此结果为该 case 在该实验配置下的最终结果
```

### Subagent 调用规范

#### Tool Orchestrator

- **Prompt 来源**：`generator/agent/nodes/choose_tool.py` 中 `_build_tool_selection_prompt` 的构建逻辑
- **关键角色声明**（摘自 choose_tool.py:348-350）：
  > "You are a **tool orchestrator**. Your only job is to pick the next tool to call. You are NOT the model that writes code — a separate agent does that after all tools finish."
- **输入**：用户任务文本、已收集结果摘要（_summarize_existing_results 格式）、可用工具列表（_format_enabled_tools_manual 格式）、剩余轮数、已调用工具列表
- **输出**：单个 JSON 对象，格式固定：
  ```json
  {"tool": "<key>", "query": "<question or exact symbol>", "args": {...}, "thinking": {"goal": "...", "missing_info": "...", "why_tool": "...", "expected_output": "..."}}
  ```
- **终止信号**：`{"tool": "ANSWER", "query": "", "args": null}`
- **约束**：
  - 输出**仅一个** JSON 对象，无 markdown 围栏，无前后 prose
  - 若剩余 0 轮，必须输出 ANSWER
  - 禁止对同一 API 符号重复调用同一工具（除非添加新结构，如 api_constraint 接 api_lookup）

#### Kernel Expert

- **Prompt 来源**：
  - 任务定义：`generator/prompt_generators/prompt_utils.py` 中的 `ASCENDC_PROBLEM_STATEMENT` + `ASCENDC_OUTPUT_FORMAT_NO_EXAMPLE` + `ascendc_template()`
  - 证据组织：`generator/agent/nodes/answer.py` 中 `_format_retrieved_content()` 的输出格式
- **关键角色声明**（摘自 prompt_utils.py:9-11）：
  > "You are an expert in writing custom AscendC kernels to optimize PyTorch architectures by replacing specific operators for performance gains."
- **输入**：
  - 基础任务 prompt（含原始 PyTorch 架构和输出格式要求）
  - 所有收集到的工具结果，按 `[KB retrieval]`, `[Code Search Snippet retrieval]`, `[API signature lookup]` 等分块组织
- **输出**：单个 Python 代码块，内含 6 个字符串变量：
  - `project_json_src`: JSON 字符串，根必须有 `"op"` 键
  - `host_tiling_src`: Tiling header (.h) 内容
  - `host_operator_src`: Host-side operator (.cpp) 内容
  - `kernel_src`: AscendC kernel (.cpp) 内容（kernel 函数名必须匹配算子名）
  - `python_bind_src`: Python binding 代码
  - `model_src`: PyTorch ModelNew 代码
- **证据优先级**（摘自 answer.py:192-193）：
  > "explicit local rules and constraints from KB shell / API lookup / API constraint override fuzzy analogies from code snippets. Use code_search_snippet mainly for structure when it does not conflict with those rules."

### 输出规范

#### TXT 文件（`{op}.txt`）

必须包含 6 个字符串变量，格式严格如下：

```python
project_json_src='''
[... JSON string with top-level "op" key ...]
'''

host_tiling_src="""
[... C++ header content ...]
"""

host_operator_src="""
[... C++ host operator content ...]
"""

kernel_src="""
[... C++ kernel content ...]
"""

python_bind_src="""
[... C++ pybind content ...]
"""

model_src='''
[... Python ModelNew class ...]
'''
```

要求：
- 整个文件必须是合法的 Python（可被 `exec()` 提取 6 个变量）
- 只用 ASCII，注释用英文
- 4 空格缩进，无 tab
- 无 `...` 省略号、无中文标点
- kernel 函数名必须是小写蛇形的算子名（如 `gelu_custom`）

#### JSON 追踪文件（`{op}_trace.json`）

至少包含以下字段：

```json
{
  "experiment_tag": "code_search_snippet_only",
  "enabled_tools": ["code_search_snippet"],
  "operator": "gelu",
  "category": "activation",
  "difficulty": "simple",
  "iterations": {
    "iter1": {
      "tool_calls": [...],
      "generation": "...生成代码原文...",
      "eval": {
        "compiled": true,
        "precision_pass": true,
        "error_summary": ""
      }
    },
    "iter2": {
      "tool_calls": [...],
      "feedback_from_iter1": "...编译错误/精度问题...",
      "generation": "...第二轮生成代码原文...",
      "eval": {
        "compiled": true,
        "precision_pass": true,
        "error_summary": ""
      }
    }
  }
}
```

### 评测方法

对每个实验配置的输出目录运行：

```bash
TMUX_SESSION="ascendCase_eval_${experiment_tag}"
LOG_FILE="/root/LLM4AscendC/output/caseStudy/${experiment_tag}/eval.log"
TXT_DIR="/root/LLM4AscendC/output/caseStudy/${experiment_tag}"

tmux new-session -d -s "$TMUX_SESSION" "bash -lc '
  source /root/miniconda3/etc/profile.d/conda.sh &&
  conda activate multi-kernel-bench &&
  cd /root/LLM4AscendC &&
  source /usr/local/Ascend/ascend-toolkit/set_env.sh &&
  export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH=/workspace/ascend_custom_opp &&
  export LLM4ASCENDC_REF_ON_CPU=1 &&
  python3 tools/eval_operator.py --txt-dir ${TXT_DIR} --workers 4 --npu 2 --clean-policy force --mode full 2>&1 | tee ${LOG_FILE}
'"
```

评测完成后读取 `${LOG_FILE}` 和 `${TXT_DIR}` 下的 `result_*.json` 获取每个 case 的编译状态、精度状态、性能数据。

### 实验设计模板

为了系统性评估工具效用，建议按以下维度设计实验序列：

| 实验 tag | 启用的工具子集 | 目的 |
|---------|--------------|------|
| `no_tool_baseline` | （空集，直接生成） | 基线对比 |
| `code_search_snippet_only` | `code_search_snippet` | 评估代码示例搜索的独立贡献 |
| `api_lookup_only` | `api_lookup` | 评估 API 签名的独立贡献 |
| `api_lookup_constraint` | `api_lookup`, `api_constraint` | 评估 API 签名+约束的组合贡献 |
| `npu_arch_only` | `npu_arch` | 评估硬件架构信息的独立贡献 |
| `kb_shell_only` | `kb_shell_search` | 评估知识库搜索的独立贡献 |
| `code_search_api` | `code_search_snippet`, `api_lookup`, `api_constraint` | 代码+API 组合 |
| `code_search_api_arch` | `code_search_snippet`, `api_lookup`, `api_constraint`, `npu_arch` | 代码+API+架构组合 |
| `full_tools` | 全部可用工具 | 全工具上限 |

主 Agent 可以自主决定实验序列，但应确保：
1. 基线实验必须最先执行
2. 每个实验配置使用相同的 case 集合
3. 结果保存在独立的 `output/caseStudy/{experiment_tag}/` 目录下

### 汇报格式

每个实验配置完成后输出：

```
## 实验报告: {experiment_tag}

### 配置
- 启用工具: [...]
- 测试 case: [op1(简单), op2(中等), op3(复杂), ...]

### 结果汇总
| Case | 难度 | Iter1 编译 | Iter1 精度 | Iter2 编译 | Iter2 精度 | 工具轮数 |
|------|------|-----------|-----------|-----------|-----------|---------|
| ...  | ...  | ...       | ...       | ...       | ...       | ...     |

### 对比基线
- 编译通过率提升: +X% (Iter1), +Y% (Iter2)
- 精度通过率提升: +X% (Iter1), +Y% (Iter2)

### 工具效用分析
- [结论性观察，如 "api_constraint 对复杂 case 的编译通过率提升最明显"]
```

最终所有实验完成后，输出跨实验的综合分析报告。