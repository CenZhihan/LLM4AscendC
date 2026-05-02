# 一轮实验执行说明与当前项目架构

本文档从 `.claude/CLAUDE.md` 中拆出两部分内容：

1. 一轮实验需要做什么
2. 当前 `LLM4AscendC` 项目的工作流架构与目录职责

## 一轮实验需要做什么

这里的“一轮实验”不是指单个 case 的 `iter1`。

这里的“一轮实验”指的是：

1. 先固定本次实验的轮次模式和工具集合。
2. 再让 Claude Code 按该配置对目标 case 执行代码生成。
3. 最后对生成结果执行测评并汇总结果。

换句话说，“一轮实验”描述的是一次完整的实验配置执行过程；而 `iter1` / `iter2` 是该实验内部、单个 case 可能经历的生成与修复回合。

可以把两者区分为：

- 实验级：先定 `round_mode`、`enabled_tools`、`experiment_tag`、`case` 集合，然后跑生成与测评。
- case 级：在某个具体算子上，按配置执行 `iter1`，必要时再执行 `iter2`。

### 输入准备

在开始一轮实验前，需要先固定以下输入：

- 实验 tag：用于区分本次实验配置，例如 `no_tool_baseline`、`code_search_snippet_only`
- 轮数模式：`single_round` 或 `two_round`
- 工具子集：本轮允许 Tool Orchestrator 选择的工具列表
- 目标 case 集合：来自 `vendor/mkb/dataset.py` 的合法算子列表

在这个定义下，一轮实验首先固定“轮次模式 + 工具集合”，然后对目标 case 集合逐个执行下面的 case 级流程。

### 实验级总流程

从实验配置视角看，一轮实验可以概括成下面四步：

1. 确定实验配置：`experiment_tag`、`round_mode`、`enabled_tools`、测试 case 集合。
2. 对 case 集合逐个调用 Claude Code 完成代码生成。
3. 对生成出的 txt 结果执行评测。
4. 汇总该实验配置下的编译率、精度通过率和失败原因。

下面的 Step 0 到 Step 8 说明的是“单个 case 在一轮实验中的执行步骤”。

### Step 0: 对单个 case 组装任务

先把生成算子所需的最小上下文准备好：

1. 从 `vendor/mkb/dataset.py` 读取算子名、类别和必要元信息。
2. 读取 `vendor/mkb/reference/{category}/{op}.py`，拿到原始 PyTorch 参考实现。
3. 读取固定 one-shot 示例 `generator/prompts/ascendc_new_model_leaky_relu.py`。
4. 使用 `generator/prompt_generators/prompt_utils.py` 中的 `ascendc_template()` 组装基础任务 prompt。
5. 在 prompt 中显式声明本轮启用的工具子集。

这一阶段的产物是“可直接给 Tool Orchestrator 和 Kernel Generation Subagent 使用的基础任务文本”。

### Step 1-5: 工具收集

这一阶段的目标不是直接写代码，而是最多进行 5 轮信息收集。

每一轮都按下面的闭环执行：

1. 通过 `runSubagent` 调用 Tool Orchestrator。
2. Prompt 构造逻辑复用 `generator/agent/nodes/choose_tool.py` 中 `_build_tool_selection_prompt()` 的规则。
3. 输入给 orchestrator 的上下文包括：
   - 当前用户任务
   - 已收集结果摘要
   - 启用工具列表
   - 剩余轮数
   - 已调用工具列表
4. Orchestrator 返回一个 JSON：

```json
{"tool": "<key>", "query": "<question or exact symbol>", "args": {...}, "thinking": {...}}
```

5. 主 Agent 执行该工具查询。
6. 把工具返回结果追加回状态。
7. 再次调用 Tool Orchestrator，直到：
   - 返回 `{"tool": "ANSWER", "query": "", "args": null}`
   - 或达到 5 轮上限

这一阶段的产物是“证据集合”，典型内容包括：API 签名、KB 片段、代码示例、环境检查结果、架构限制等。

### Step 6: 生成 Kernel

工具收集结束后，进入第一次代码生成：

1. 按 `generator/agent/nodes/answer.py` 中 `_format_retrieved_content()` 的格式整理证据。
2. 通过 `runSubagent` 调用 `ascendc-kernel-developer`。
3. 输入内容由两部分组成：
   - Step 0 组装的基础任务 prompt
   - Step 1-5 收集到的结构化证据
4. Subagent 输出一个 Python 代码块，内含 6 个字符串变量：
   - `project_json_src`
   - `host_tiling_src`
   - `host_operator_src`
   - `kernel_src`
   - `python_bind_src`
   - `model_src`

这一阶段的产物是“该 case 的完整 txt 内容”。

### Step 7: 保存产物

将该 case 的第一次生成结果落盘：

- 文本产物：`output/claudeCase/{experiment_tag}/{op}.txt`
- 追踪产物：`output/claudeCase/{experiment_tag}/{op}_trace.json`

`trace.json` 至少应记录：

- 实验 tag
- 启用工具列表
- 算子名与类别
- `iter1.tool_calls`
- `iter1.generation`

### Step 8: 执行评测

该 case 的第一次评测由 `ascendc-kernel-reviewer` 负责执行与汇总：

1. 调用 Evaluation Subagent。
2. 输入实验 tag、输出目录、当前轮次信息，以及评测命令。
3. 评测实质上会运行 `tools/eval_operator.py`，读取：
   - `eval.log`
   - `result_*.json`
4. 输出结构化结论，至少包含：
   - `compiled`
   - `precision_pass`
   - `error_summary`
   - 关键日志摘录

推荐评测命令如下：

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

### 一轮实验的结束条件

当下面四项都完成时，这一轮实验结束：

1. 已固定本轮实验配置：`round_mode`、`enabled_tools`、`experiment_tag`、测试 case 集合。
2. 目标 case 已按配置完成代码生成。
3. 目标 case 已完成测评，并生成对应的 txt、trace、log 和 result 文件。
4. 已能对该实验配置输出汇总结果。

如果实验模式是 `single_round`，则单个 case 只经历一次生成与测评。
如果实验模式是 `two_round`，则单个 case 在本轮实验内部还会额外经历 `iter2` 修复与最终评测。

### 一轮实验核对清单

- 是否固定了实验 tag、工具子集、case 列表和轮数模式
- 是否明确“一轮实验”是实验配置级流程，而不是单个 case 的 `iter1`
- 是否只从允许的知识源与白名单工具取信息
- 是否通过 `runSubagent` 调用了 orchestrator / developer / reviewer
- 是否没有调用仓库内旧的 Python agent runtime 作为执行入口
- 是否生成了合法的 `{op}.txt`
- 是否生成了包含 `iter1` 记录的 `{op}_trace.json`
- 是否完成了目标 case 的测评并拿到结构化结果
- 是否能够对该实验配置输出汇总结论

## 当前项目架构

当前仓库的主链路可以概括为：

`数据集与参考实现 -> Prompt 组装 -> 工具编排 -> Kernel 生成 -> txt 落盘 -> 评测物化 -> 日志与结果汇总`

### 目录分层

```text
LLM4AscendC/
├── .claude/
│   ├── CLAUDE.md
│   └── EXPERIMENT_ONE_ROUND_AND_ARCHITECTURE.md
├── generator/
├── tools/
├── vendor/mkb/
├── output/
├── artifacts/
├── ascend_custom_opp/
├── CANN_skills/
├── scripts/
├── eval_logs/
└── leaky_relu/
```

### 核心模块职责

#### `.claude/`

- `CLAUDE.md`：实验模式总规约，定义角色、原则、工具边界、Subagent 约束、输出格式、评测方法和实验汇报格式。
- `EXPERIMENT_ONE_ROUND_AND_ARCHITECTURE.md`：实验配置级流程说明，以及单个 case 在其中如何执行。

#### `generator/`

这是生成侧主模块，负责把算子需求变成 AscendC 代码或 Agent 工作流。

- `prompt_generators/`：拼接 prompt 的策略实现。
- `prompts/`：one-shot 或参考提示样例。
- `agent/`：多轮 Agent 逻辑。
- `agent/nodes/`：choose_tool、answer 等节点级编排逻辑。
- `agent/retrievers/`：KB、在线文档、代码检索、环境检查等具体检索实现。
- `rag/`：代码 RAG 索引与检索。
- `scripts/`：历史生成脚本与实验脚本。
- `local_api_config.py`：本地模型与接口配置。

在本实验模式里，`generator/` 主要承担两件事：

1. 提供 prompt 与节点逻辑的“规范来源”
2. 提供工具定义、检索实现和历史参考

#### `tools/`

这是评测和工程物化侧主模块。

- `eval_operator.py`：统一评测入口。
- `txt_operator.py`：把 txt 内容物化成算子工程时会用到的辅助逻辑。
- `common/`：环境、路径、日志等公共能力。
- `pybind_template/`：构建 pybind 扩展时使用的模板。
- 其余脚本用于生成、审计、批量运行和辅助检查。

在实验主链路里，`tools/eval_operator.py` 是生成结果进入真实编译与精度验证的桥梁。

#### `vendor/mkb/`

这是数据集和参考实现来源。

- `dataset.py`：合法算子名、类别等元数据。
- `reference/`：按类别划分的 PyTorch 参考实现。
- `correctness.py`：正确性对比逻辑。
- `mkb_eval_config.py`：精度验证相关配置。

实验中的 case 选择、参考模型读取、正确性对比都依赖这里。

#### `output/`

这是生成结果的主输出区。

- `output/claudeCase/`：当前实验模式建议使用的输出目录。
- 其他子目录保存历史实验、不同模型或不同策略的生成结果。

对本实验而言，`output/claudeCase/{experiment_tag}/` 是每个实验配置的独立结果根目录。

#### `artifacts/`

这是评测阶段产生的物化产物区，通常包含：

- `_txt_staging/`：由 txt 临时展开的工程 staging 目录。
- 每个算子的 build/workspace/logs/state/result 文件。

可以把 `output/` 理解为“生成结果入口”，把 `artifacts/` 理解为“评测过程与落地产物”。

#### `ascend_custom_opp/`

这是自定义 OPP 安装路径。

- `vendors/`：默认安装位置。
- `_parallel_w0` 到 `_parallel_w3`：并行评测时给不同 worker 隔离使用的安装根目录。

评测时通过 `LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH` 指向这里，避免并发安装相互覆盖。

#### `CANN_skills/`

这是额外的技能库与 Agent 资源目录，提供 AscendC 开发相关知识、模板与团队资产。

在当前实验模式中，它更像“辅助知识资产库”，不是主执行入口。

#### `scripts/`

包含各类实验、批处理和本地运行脚本，适合人工操作或旧流程复现。

当前实验模式明确要求：不要把这些旧脚本作为本实验的主执行入口。

#### `eval_logs/`

保存历史汇总报告、失败原因拆解和重评测结果，适合做跨实验分析。

#### `leaky_relu/`

这是一个按单算子组织的工作目录，通常用于局部 case 开发、样例物化、日志和状态保存，适合做单算子调试参考。

### 当前实验模式下的推荐链路

对于本仓库，建议把目录职责理解成下面这条链：

1. `vendor/mkb/` 提供算子名和参考实现。
2. `generator/` 负责构造 prompt、编排工具、调用生成 Subagent。
3. 生成结果写入 `output/claudeCase/{experiment_tag}/`。
4. `tools/eval_operator.py` 从 `output/claudeCase/...` 读取 txt 并物化到 `artifacts/`。
5. 编译安装过程会使用 `ascend_custom_opp/`。
6. 最终日志、状态和结果 JSON 落在 `artifacts/` 与 `output/claudeCase/...`。

### 与实验文档的对应关系

- 如果你要看“实验模式允许什么、不允许什么、Subagent 怎么调用”，看 `.claude/CLAUDE.md`。
- 如果你要看“固定轮次模式和工具集合后，Claude Code 如何完成生成与测评，以及当前仓库各目录在流程里扮演什么角色”，看本文档。