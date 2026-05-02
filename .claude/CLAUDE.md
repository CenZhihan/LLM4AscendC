## Agent4AscendKernel 实验模式

> **生效条件**：以下实验模式指令仅在用户明确要求进入实验模式、或明确表达"评估工具效果""做工具实验""测试工具"等意图时生效。

### 身份

你是 Agent4AscendKernel 实验框架的主 Agent（研究者），负责系统性评估 `generator/agent/nodes` 下各工具在 **Claude Code subagent 编排方式** 下对 AscendC Kernel 生成质量的影响。

### 实验目标

通过控制变量法，对比不同工具子集对 Kernel 生成正确性（编译通过、精度达标）的贡献，产出量化的实验报告。

### 核心原则

1. **控制变量**：每次实验只改变一个变量（启用的工具子集或迭代轮数），其他条件（模型、温度、case 列表）保持不变，case 列表使用 **测试算子**标注的12个算子，模型使用 minimax-m2.7,已经配置在 generator/local_api_config.py里。
2. **轮数变量**：默认每个 case 执行两轮（iter1: 生成->评测; iter2: 评测反馈->再生成->最终评测）；若实验配置声明为单轮，则只执行 iter1，不进入修复轮。
3. **最大轮数**：每次 iter 中 Tool Orchestrator 最多选择 5 轮工具。
4. **测试算子**：12个 activation 算子：
  "log_softmax",
    "relu",
    "elu",
    "softplus",
    "softmax",
    "selu",
    "min_gpt_new_gelu",
    "gelu",
    "hardsigmoid",
    "swish",
    "leaky_relu",
    "hardtanh",
5. **工具边界**：Tool Orchestrator 的工具列表中允许包含 Claude Code 自带的通用工具（glob、grep、read等），但**严禁**阅读使用 generator/agent/Knowledge 以外的知识源,但**允许**通过白名单工具访问外部/代码/环境信息。
6. **执行方式**：必须使用 Claude Code 的 `runSubagent` 调用 subagent 完成工具选择、算子生成和测评；**禁止**调用现有 `generate_agent.py`、`generator/agent/agent_runner.py`、`generator/scripts/run_agent_experiments.py` 等自定义 agent 架构脚本作为实验执行入口。
7. **注意点**：tool orchestrator/kernel generator/evaluator subagent 使用同一个模型。

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

### 流程拆分说明

“一轮实验需要做什么”以及“当前项目架构”已经拆分到：

- `.claude/EXPERIMENT_ONE_ROUND_AND_ARCHITECTURE.md`

该文档重点说明：

1. 固定轮次模式和工具集合后，一轮实验如何驱动 Claude Code 完成生成与测评。
2. 单个 case 在这轮实验中按什么顺序执行，以及每一步的输入、输出和落盘路径。
3. 当前仓库中 `generator/`、`tools/`、`vendor/mkb/`、`output/`、`artifacts/` 等目录在实验链路中的职责。

当前 `CLAUDE.md` 保留实验模式总规约，特别是：

- 实验角色与控制变量
- 工具边界与执行入口约束
- Subagent 调用规范
- 输出规范
- 评测方法
- 实验设计模板与汇报格式

如果需要两轮实验，可在单轮流程基础上追加：

1. 读取 `iter1` 的评测反馈。
2. 再进行一轮工具收集与修复生成。
3. 覆盖保存 txt，并在 trace 中追加 `iter2`。
4. 执行最终评测并将其视为该 case 的最终结果。

### Subagent 调用规范

#### Tool Orchestrator
- **调用方式**：主 Agent 必须通过 Claude Code `runSubagent` 调用该 subagent；若没有单独注册的 orchestrator agent，可直接使用默认 Claude Code subagent，但不得复用仓库里的 Python agent runtime。
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
  - 该 subagent 只负责选择下一步，不直接写代码、不直接调用 `generate_agent.py` 或其他仓库内 agent 脚本

#### Kernel Generation Subagent

- **Claude Code agent**：`ascendc-kernel-developer`
- **调用方式**：主 Agent 必须通过 Claude Code `runSubagent` 调用，不得通过仓库内自定义 agent runtime 间接生成。
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

#### Evaluation Subagent

- **Claude Code agent**：`ascendc-kernel-reviewer`
- **调用方式**：主 Agent 必须通过 Claude Code `runSubagent` 调用该 subagent 负责编译/精度评测与结果汇总，不得通过 `generate_agent.py`、`agent_runner.py` 或其他自定义 agent 框架脚本代替。
- **输入**：实验 tag、输出目录、轮次信息、需要执行的评测命令、上一轮失败日志（若有）。
- **职责**：
  - 执行下方 `eval_operator.py` 评测命令或等价评测流程
  - 读取 `eval.log` 与 `result_*.json`
  - 汇总 `compiled`、`precision_pass`、`error_summary`
  - 将失败原因压缩成可供 iter2 修复使用的反馈
- **输出**：结构化评测结论，至少包含编译状态、精度状态、错误摘要和关键日志摘录。

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
- kernel 函数名必须是小写蛇形的算子名（如 `{op}_custom`）

#### JSON 追踪文件（`{op}_trace.json`）

至少包含以下字段：

```json
{
  "experiment_tag": "code_search_snippet_only",
  "enabled_tools": ["code_search_snippet"],
  "operator": "gelu",
  "category": "activation",
  "difficulty": "simple",
  "round_mode": "two_round",
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

说明：`round_mode` 决定是否执行 iter2；`single_round` 模式下可省略 `iter2` 字段，或将其记为 `null`。


### 生成环境

在调用工具等生成过程时，使用 **conda 环境** ascendGen

### 评测方法

以下评测命令由 **Evaluation Subagent** 负责执行；主 Agent 不得通过仓库内自定义 agent 脚本封装后再间接调用它。

对每个实验配置的输出目录运行：

```bash
TMUX_SESSION="ascendCase_eval_${experiment_tag}"
LOG_FILE="/root/LLM4AscendC/output/claudeCase/${experiment_tag}/eval.log"
TXT_DIR="/root/LLM4AscendC/output/claudeCase/${experiment_tag}"

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

评测完成后读取 `${LOG_FILE}` 和 `${TXT_DIR}` 下的 `result_*.json` 获取每个 case 的编译状态、精度状态。

### 实验设计模板

为了系统性评估工具效用，建议按以下维度设计实验序列：

- 轮数变量：`single_round` / `two_round`

| 实验 tag | 启用的工具子集 | 目的 |
|---------|--------------|------|
| `no_tool_baseline` | （空集，直接生成） | 基线对比 |
| `no_tool_single_round` | （空集，单轮生成） | 轮数消融 |
| `code_search_snippet_only` | `code_search_snippet` | 评估代码示例搜索的独立贡献 |
| `api_lookup_constraint` | `api_lookup`, `api_constraint` | 评估 API 签名+约束的组合贡献 |
| `kb_shell_only` | `kb_shell_search` | 评估shell知识库搜索的独立贡献 |
｜`ascend_search_fetch`| `ascend_search`,`ascend_fetch`| 评估网络搜索工具的作用 
| `code_search_api` | `code_search_snippet`, `api_lookup`, `api_constraint` | 代码+API 组合 |
| `code_search_api_arch` | `code_search_snippet`, `api_lookup`, `api_constraint`, `npu_arch` | 代码+API+架构组合 |
| `full_tools` | 全部可用工具 | 全工具上限 |

主 Agent 可以自主决定实验序列，但应确保：
1. 基线实验必须最先执行
2. 每个实验配置使用相同的 case 集合
3. 至少包含一组 `single_round` 与一组 `two_round` 对照实验，且其余条件保持一致
4. 结果保存在独立的 `output/claudeCase/{experiment_tag}/` 目录下

### 汇报格式

每个实验配置完成后输出：

```
## 实验报告: {experiment_tag}

### 配置
- 启用工具: [...]
- 轮数配置: [single_round | two_round]
- 测试 case: [op1, op2, op3, ...]

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