# LLM4AscendC

## 1. 项目是做什么的

**LLM4AscendC** 面向在 **昇腾 NPU（Ascend 910 系列等）** 上开发与验证 **自定义 AscendC 算子** 的流程：把 LLM 或工具链产出的算子描述（通常为 **MKB 风格的 `*.txt` 打包**）落地为可编译工程，经 **CANN `msopgen` 脚手架 → CMake 编译 → 自定义 OPP 安装 → PyTorch/pybind 扩展**，最后在 Python 中与 **MKB 参考实现（`vendor/mkb/reference/`）** 对比，检查 **编译是否成功、数值是否正确**。

典型用途包括：

- 对 `output/<op_key>.txt` 等单算子包做 **一键 full 评测**；
- 仅 **编译安装**（`build-only`）或在上次构建不变时 **只跑正确性**（`eval-only`）；
- 批量对某目录下多个 `*.txt` 串行评测（`--txt-dir`）。

**算子键 `op_key`** 必须与 `vendor/mkb/dataset.py` 中登记的名称一致，且 **`*.txt` 的文件名（不含扩展名）等于该 `op_key`**，以便绑定 MKB 参考与评测逻辑。

---

## 2. 如何评测算子

### 2.1 入口命令

在项目根目录（本仓库下的 `LLM4AscendC/`）执行：

```bash
python3 tools/eval_operator.py <必选其一：算子来源> [可选参数]
```

**运行环境**：完整流水线依赖 **CANN / msopgen / NPU**，须在已配置昇腾环境的机器上执行。常见做法是在宿主机用 **`docker exec -it <容器ID> bash`** 进入侧载了本仓库的容器（示例容器 ID 可与脚本 `scripts/run_matmul17_local.sh`、`scripts/run_matmul_gpt4o_parallel3.sh` 注释中一致），再 `cd` 到容器内的 `LLM4AscendC` 根目录后运行上述命令。

**算子来源（三选一，必选）**

| 参数 | 含义 |
|------|------|
| `--op <DIR>` | 已 materialize 的算子目录，例如 `operators/xxx` 或绝对路径。目录内需含 `operator.json`、`op_host/`、`op_kernel/`、`eval/spec.py` 等。 |
| `--txt <PATH>` | 单个 **MKB 风格 txt 包** 路径（如 `output/gelu.txt`）。脚本会写入 `artifacts/.../_txt_staging/`，并生成临时算子目录再跑流水线；若 txt 位于 `output/<批次目录>/...`，则产物会归档到 `artifacts/<批次目录>/...`。 |
| `--txt-dir <DIR>` | 目录内所有 `*.txt` **按文件名排序后** 评测；默认 **单进程顺序** 执行。可用 `--workers N`（`N>1`）改为 **多进程任务队列**：各 worker 从队列取下一个 `*.txt`，**谁空谁取**（等价）。每个文件的 **stem = MKB `op_key`**。某次失败会记录错误并继续其余任务（返回码非 0）。 |

**常用示例**

```bash
# 对单个 txt 做完整流水线（默认）
python3 tools/eval_operator.py --txt output/gelu.txt

# 强制全量重编译（见下文 clean-policy）
python3 tools/eval_operator.py --txt output/gelu.txt --mode full --clean-policy force

# 只编译与安装，不跑 NPU 正确性
python3 tools/eval_operator.py --txt output/gelu.txt --mode build-only

# 在已有成功构建且指纹未变的前提下，只跑 eval
python3 tools/eval_operator.py --txt output/gelu.txt --mode eval-only

# 批量目录（串行，默认）
python3 tools/eval_operator.py --txt-dir output/kernelbench165_txt --clean-policy smart

# 批量目录（并行 5 worker；须设置 LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH，见下文）
export LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH="/path/to/writable/ascend_custom_opp"
python3 tools/eval_operator.py --txt-dir output/gpt-5_selected_shot --workers 5 --mode full

# 并行且多卡分散（例如 7 worker、3 张物理卡：device_id = worker_id % 3）
python3 tools/eval_operator.py --txt-dir output/gpt-5_selected_shot --workers 7 --npu 3 --mode full
```

### 2.2 可选参数一览

| 参数 | 默认值 | 可选值 | 含义与适用场景 |
|------|--------|--------|----------------|
| `--mode` | `full` | `full` / `build-only` / `eval-only` | **`full`**：msopgen → 编译 → 安装 OPP → 构建并安装 pybind wheel → 运行 `eval/spec.py`（正确性或与参考对比）。**`build-only`**：只做到安装 wheel 并写 `state/`，不执行 eval；适合 CI 只验证能否编过。**`eval-only`**：跳过构建，直接跑 eval；要求 `artifacts/<op_key>/state/installed.json` 已存在，且源码指纹与上次构建一致，否则会报错要求先 `full` 或 `build-only`。适合反复调试验证逻辑而不重复长时间编译。 |
| `--clean-policy` | `force` | `force` / `smart` | **`force`**：每次进入构建阶段前 **删除** `artifacts/<op_key>/workspace` 与 `pybind`，保证从干净目录重编；结果可复现、磁盘与耗时开销较大。**`smart`**：仅当算子目录 **内容指纹**相对上次构建发生变化时，才清理并重编；指纹未变则复用已有构建；若期望复用但从未成功安装过，会报错提示改用 `force`。适合迭代小改或批量任务中减少重复编译。 |
| `--workers` | `1` | `1` … `32` | **仅与 `--txt-dir` 合用**。`1` 表示与过去行为一致的单进程顺序评测。`N>1` 时使用 `multiprocessing` 的 **spawn** 启动 `N` 个子进程，共享任务队列，依次处理目录中的 `*.txt`。**必须**设置 `LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH`：每个 worker 将自定义 OPP 安装到该路径下的 **`_parallel_w0` … `_parallel_w{N-1}`**，避免多进程同时写入同一 OPP 根目录导致安装冲突。 |
| `--npu` | `1` | `1` … `8` | **仅当 `--txt-dir` 且 `--workers` > 1 时生效**（`--workers` 为 `1` 时忽略）。使用 `K` 张物理 NPU（索引 `0..K-1`），每个 worker 在子进程内设置 `ASCEND_VISIBLE_DEVICES = (worker_id % K)`，使 `eval/spec.py` 中的 `torch.device("npu:0")` 映射到对应物理卡，减轻同卡多进程导致的 **NPU OOM** 或超时波动。默认 `K=1` 表示所有 worker 仍只看到物理卡 0（与未传 `--npu` 时一致）。 |

**说明**：

- 若 `--txt/--txt-dir` 的路径位于 `output/` 下，则会将评测产物与 staging **按 output 的相对路径镜像归档**到 `artifacts/<output相对路径>/...`，用于区分不同批次（例如 `gpt-5_selected_shot`、其下的 `_batch_first10` 等）。
- 若 `--txt/--txt-dir` 不在 `output/` 下，则按“野生”方式落在 `artifacts/` 根下（保持兼容）。
- `--txt` / `--txt-dir` 会清理并重建对应的 staging 目录，避免旧内容干扰：
  - 单个 txt：`artifacts/<group可选>/_txt_staging/<op_key>/`
  - 批量目录：`artifacts/<group可选>/_txt_staging/<op_key>/`

**并行 `--txt-dir`（`--workers` > 1）补充说明**：

- **产物路径**：与串行相同，仍为 `artifacts/<group可选>/<op_key>/…`；不同算子由不同 worker 处理，**不会**因并行覆盖彼此的 `result_*.json`（同一 `op_key` 不会在队列中出现两次）。
- **OPP**：见上表 `--workers`；子目录名为 `_parallel_w0` … `_parallel_w{N-1}`，位于 `LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH` 所指向目录的 **下**（若该路径为 `/a/opp`，则 worker 0 使用 `/a/opp/_parallel_w0`）。
- **NPU / 显存（`--npu K`）**：当 `--workers` > 1 时，可为 `--npu` 指定参与轮询的物理卡数量 `K`（默认 `1`）。映射公式：**`device_id = worker_id % K`**（例如 `--workers 7 --npu 3`：worker 0/3/6 → 物理卡 0，1/4 → 卡 1，2/5 → 卡 2）。子进程在跑流水线前设置 `ASCEND_VISIBLE_DEVICES=device_id`，与 `eval/spec.py` 中固定的 `npu:0` 配合使用。若 `K < workers`，仍会有同卡多进程，需结合显存与稳定性自行调参。`K=1` 时等价于所有 worker 绑定到物理卡 0。
- **pip**：各算子 pybind 包名一般不同，并发 `pip install` 多数可接受；若遇偶发安装冲突，可改为较小 `--workers` 或后续在工具链侧加锁（本版未实现）。

**手动验证建议（并行 + 分卡）**：

1. **基线**：`--workers=3` 且不传 `--npu`（默认 `npu=1`），与单进程顺序跑对比失败类型与 `artifacts/<group>/<op_key>/result_*.json` 是否一致。
2. **分卡**：在有多张卡的环境下使用 `--workers=3 --npu=3`，对比 **NPU out of memory**、`507014`（aicore timeout）等错误出现频率是否较「三进程同卡」明显下降。
3. **检查产物**：每个算子的评测结果在 **`artifacts/<group可选>/<op_key>/result_*.json`**；各步骤日志在 **`artifacts/<group可选>/<op_key>/logs/`**（如 `*-06-eval.log`）。并行时终端会打印 `[w<id>] ASCEND_VISIBLE_DEVICES=...` 便于确认绑定。

示例：

- `--txt output/gpt-5_selected_shot/relu.txt` → `artifacts/gpt-5_selected_shot/relu/` 与 `artifacts/gpt-5_selected_shot/_txt_staging/relu/`
- `--txt-dir output/gpt-5_selected_shot/_batch_first10` → `artifacts/gpt-5_selected_shot/_batch_first10/<op_key>/...`

### 2.3 流水线在做什么（`full` / `build-only` 的构建部分）

1. **指纹**：对算子目录做内容指纹，供 `smart` 策略判断是否重编。  
2. **msopgen**：根据 `operator.json` 中的 IR 生成 AscendC 工程。  
3. **覆盖源码**：将 `op_host/`、`op_kernel/` 覆盖到生成树。  
4. **build.sh**：编译并打 **custom OPP**（`.run`）。  
5. **安装 OPP**：默认安装到可写路径（见 `tools/common/env.py` 中的 `ascend_custom_opp_path`），并 **source** 对应 `set_env.bash` 以便运行时加载。  
6. **pybind**：基于模板生成扩展、`bdist_wheel` 并 `pip install`。  
7. **eval**（仅 `full`）：在独立子进程中执行 `eval/spec.py`，通常会加载 MKB reference 与自定义 `ModelNew` 做对比。

各步骤日志位于 **`artifacts/<group可选>/<op_key>/logs/`**，文件名带时间戳与序号，例如 `*-01-msopgen.log` … `*-06-eval.log`。

### 2.4 结果输出

- **`artifacts/<group可选>/<op_key>/result_<op_key>.json`**：汇总 `compiled`、`correctness`、`correctness_info`、指纹与日志路径等。  
- 失败时终端会打印 `[done] FAILED` 并附 summary 路径；成功为 `[done] OK`。

### 2.5 环境变量说明

本仓库**直接读取或改写**的环境变量如下（不含 CANN 工具链内部大量 `ASCEND_*`，那些由 `source set_env.sh` 等注入）。

#### 使用者可自行设置（可选）

| 变量 | 典型取值 | 作用 |
|------|----------|------|
| `LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH` | 绝对路径，可写 | 覆盖 `env.py` 中默认的 `ascend_custom_opp_path`，将自定义 OPP 安装到该目录。**并行 `--txt-dir`（`--workers`>1）时必填**，且每个 worker 会使用其下的 `_parallel_w<id>` 子目录，避免多进程安装互踩。 |
| `LLM4ASCENDC_REF_ON_CPU` | `1` / `true` / `yes` / `on` | MKB **参考模型 `Model`** 在 **CPU** 上跑，自定义算子仍在 NPU（`vendor/mkb/correctness.py`）。用于对照：NPU 上参考若因图模式/融合与 PyTorch 不一致，可开此项再比一次。 |

#### 由 `eval_operator.py` 在评测子进程中自动设置（一般不必手改）

| 变量 | 作用 |
|------|------|
| `LLM4ASCENDC_OP_MODULE` | 当前算子对应的 pybind 模块名（与安装的 wheel 一致）。 |
| `LLM4ASCENDC_OP_DIR` | 当前算子目录绝对路径。 |
| `LLM4ASCENDC_ROOT` | 项目根目录绝对路径。 |
| `PYTHONPATH` | 首部追加 `LLM4ASCENDC_ROOT`，保证 `vendor.mkb` 等可导入。 |

#### 由 `tools/common/env.py` 的 `build_subprocess_env()` 在子进程中增强

| 变量 | 作用 |
|------|------|
| `LD_LIBRARY_PATH` | 在原有路径前**追加**昇腾 **driver** 库路径（默认见 `EnvConfig.driver_libs`），保证运行时能加载驱动相关 `.so`。 |
| `ASCEND_CUSTOM_OPP_PATH` | 若 `EnvConfig.ascend_custom_opp_path` 非空（默认 `/workspace/ascend_custom_opp`），则设为该路径，用于找到自定义 OPP 安装内容。 |

路径类默认值（`ascend_set_env.sh`、conda、`ascend_custom_opp_path` 等）写在 **`EnvConfig` 数据类**里，换机器时如与本地不符，应**改 `tools/common/env.py`** 或后续如改为读环境变量再覆盖（当前代码以代码内默认值为主）。

#### pybind 构建时（`tools/pybind_template/setup.py`，通常由流水线注入）

| 变量 | 作用 |
|------|------|
| `CUSTOM_OP_NAME` | 扩展模块名；流水线通过 `artifacts/.../pybind/.build_env.json` 等与指纹一并写入，一般无需在 shell 里设。 |
| `CUSTOM_OP_VERSION` | wheel 版本串。同上。 |
| `USE_NINJA` | 设为 `1` 时启用 Ninja 编译扩展；可选。 |

#### 与昇腾/CANN 相关、本仓库不定义但运行前需满足

- 需在 shell 中 **`source $ASCEND_HOME_PATH/ascend-toolkit/set_env.sh`**（或你机器上的等价路径）；`ASCEND_HOME_PATH`、`PATH`、`PYTHONPATH` 等由 CANN 脚本统一配置。
- 自定义 OPP 安装后，通常还需 **`source <ascend_custom_opp_path>/vendors/customize/bin/set_env.bash`**（`eval_operator` 通过 `shell_prefix()` 在子命令里自动拼接）。

正确性试验次数与随机种子见 **`vendor/mkb/mkb_eval_config.py`**（如 `num_correct_trials`、`seed_num`）。

### 2.6 其他运行提示

- 需要 **CANN / Ascend 工具链**、`msopgen`、可编译 AscendC 的环境；**`EnvConfig` 中的路径**需与当前机器一致。  
- **`artifacts/`** 体积大，为生成物；若纳入 Git，建议加入 **`.gitignore`**。

---

## 3. LLM 算子生成

### 3.1 概述

**`generator/`** 模块使用大语言模型（LLM）根据算子描述自动生成 AscendC 算子代码。支持多种提示策略与检索增强（RAG），并内置基于 LangGraph 的多轮智能 Agent 工作流。

生成器与评测流水线共享同一份算子注册表（`vendor/mkb/dataset.py`），目前覆盖约 180 个算子，分布在 15 个类别中：activation、broadcast、convolution、arch、fuse、loss、math、matmul、normalization、optimizer、pooling、index、resize、reduce、attention。

### 3.2 入口命令

```bash
# 构建 RAG 代码索引（首次使用前执行一次）
python3 tools/generate_operator.py --build-index --code-dir ascendCode

# 使用 RAG 策略生成算子
python3 tools/generate_operator.py --model deepseek-chat --strategy rag --categories all --workers 4
```

**常用参数**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--model` | `deepseek-chat` | 使用的 LLM 模型名称（需在 `api_config.py` 或环境变量中配置 API 地址与密钥） |
| `--strategy` | `rag` | 提示策略：`rag`（检索增强）、`add_shot`（添加示例）、`selected_shot`（按类别选择示例） |
| `--categories` | `all` | 算子类别，逗号分隔或 `all` 表示全部生成 |
| `--workers` | `1` | 并行生成的 worker 数量 |
| `--build-index` | — | 构建 RAG 代码索引（与 `--code-dir` 配合使用） |
| `--code-dir` | — | 待索引的 AscendC 代码库目录 |

### 3.3 提示策略（`generator/prompt_generators/`）

| 策略 | 说明 |
|------|------|
| `rag` | 从 RAG 索引中检索与目标算子最相似的代码片段，将其作为上下文注入提示词。适合 LLM 缺乏某类算子编写经验时使用。 |
| `add_shot` | 从示例库中挑选少量典型算子代码作为 few-shot 示例附加到提示中。 |
| `selected_shot` | 根据算子类别从预筛选的示例池中挑选最相关的 few-shot 示例。相比 `add_shot` 更精准，但需要为每个类别准备足够的示例。 |

### 3.4 RAG 模块（`generator/rag/`）

RAG 模块负责将外部代码库（C++ 头文件与实现）索引为向量数据库，并在生成时检索相似代码片段。

- **索引构建**：扫描 `--code-dir` 指定目录下的 `.cpp` 与 `.h` 文件，按固定大小分块，使用 BGE-M3 模型计算嵌入向量后写入 ChromaDB 持久化存储（`generator/rag/index/`）。
- **检索**：生成时根据算子名称、类别与查询字符串构造复合查询，在 ChromaDB 中执行向量相似度检索，返回 top-k 代码片段。
- **依赖**：BGE-M3 嵌入模型（默认路径 `generator/models/BAAI/bge-m3`）、ChromaDB。

### 3.5 LangGraph Agent（`generator/agent/`）

Agent 是一个基于 LangGraph StateGraph 的多轮工作流，LLM 在每轮自主决定下一步动作：查询知识库、搜索网页、检索代码库、检查环境兼容性，或直接生成答案。

#### 3.5.1 工具类型

| 工具 | 标识 | 数据源 | 说明 |
|------|------|--------|------|
| 知识库查询 | `KB` | ChromaDB 中的 Ascend C API 文档 | 查询本地索引的昇腾 API 文档片段 |
| 网页搜索 | `WEB` | DuckDuckGo（`ddgs` 包） | 搜索技术文档、博客与教程 |
| 代码检索 | `CODE_RAG` | `generator/rag/` 索引 | 检索代码库中的相似实现 |
| 环境检查 | `ENV_CHECK` | CANN 头文件与运行时环境 | 检查 CANN 版本、NPU 设备状态、API 兼容性 |

每种工具均以 Retriever 类封装，提供 `is_available()` 与 `retrieve(query)` 两个标准方法，便于在节点中统一调用。

#### 3.5.2 Agent 模式

通过 `ToolType` 的冻结集合（`AgentToolMode`）指定启用的工具。常用模式：

| 模式字符串 | 含义 |
|------------|------|
| `no_tool` | 不使用工具，LLM 直接回答 |
| `kb_only` | 仅使用知识库 |
| `web_only` | 仅使用网页搜索 |
| `code_rag_only` | 仅使用代码检索 |
| `kb_and_web` | 知识库 + 网页搜索 |
| `kb_and_code_rag` | 知识库 + 代码检索 |
| `web_and_code_rag` | 网页搜索 + 代码检索 |
| `all` | 启用全部三种工具（KB、WEB、CODE_RAG） |

可通过 `frozenset` 自由组合，例如同时启用所有工具和 `ENV_CHECK`：

```python
from generator.agent.agent_config import ALL, ToolType
mode = frozenset({*ALL, ToolType.ENV_CHECK})
```

#### 3.5.3 Agent 工作流

```
entry
  │
  ├─ (无工具时) → answer → END
  └─ (有工具时) → choose_tool
                    │
                    ├─ KB ──→ kb_query ──→ choose_tool (循环)
                    ├─ WEB ──→ web_search ──→ choose_tool (循环)
                    ├─ CODE_RAG ──→ code_rag ──→ choose_tool (循环)
                    ├─ ENV_CHECK ──→ env_check ──→ choose_tool (循环)
                    └─ ANSWER ──→ answer → END
```

- **`entry`**：入口节点，根据是否启用工具决定走向。
- **`choose_tool`**：LLM 根据当前已有信息选择下一步动作（KB / WEB / CODE_RAG / ENV_CHECK / ANSWER）。每轮输出一行动作标识和一行英文查询。
- **检索节点**：执行对应工具的 `retrieve(query)`，将结果追加到 Agent 状态中。
- **`answer`**：综合所有检索结果，由 LLM 生成最终算子代码。
- 最大查询轮数由 `agent_max_query_rounds` 控制（默认 3 轮），达到上限后强制进入 `answer`。

#### 3.5.4 使用示例

```python
from generator.agent.agent_builder import create_agent
from generator.agent.agent_state import create_initial_state
from generator.prompt_generators.rag_prompt_generator import rag_prompt

# 构建初始状态
state = create_initial_state(
    base_prompt=rag_prompt("gelu", "activation"),
    op_name="gelu",
    category="activation",
    strategy_name="rag",
)

# 创建并调用 Agent
agent = create_agent("all")  # 或 "kb,web,code_rag,env_check"
result = agent.invoke(state)

# 获取生成结果
generated_code = result["messages"][-1].content
tool_log = result.get("tool_calls_log", [])
```

#### 3.5.5 Agent LLM 配置

Agent 使用独立于生成器的 LLM 配置，优先级为：

1. 环境变量 `XI_AI_API_KEY` + `XI_AI_BASE_URL` + `XI_AI_MODEL`
2. `generator/utils/api_config.py` 中的 `API_KEY` / `BASE_URL` / `MODEL`
3. 默认回退到 `deepseek-chat`（`https://api.deepseek.com/v1`）

### 3.6 配置与依赖

| 文件 | 作用 |
|------|------|
| `generator/config.py` | 生成器与 Agent 的配置（模型路径、RAG 参数、Agent 参数等） |
| `generator/utils/api_config.py`（可选） | LLM API 密钥、基地址、模型名称 |
| `tools/common/env.py` | CANN/NPU 环境配置（与评测流水线共享） |
| `requirements.txt` | Python 依赖列表 |

---

## 4. 相关路径速查

| 路径 | 说明 |
|------|------|
| `tools/eval_operator.py` | 统一评测入口 |
| `tools/generate_operator.py` | LLM 算子生成入口 |
| `generator/` | 生成模块：提示策略、RAG 检索、LangGraph Agent |
| `generator/agent/` | LangGraph 智能 Agent（KB、WEB、CODE_RAG、ENV_CHECK 工具） |
| `generator/rag/` | RAG 代码索引与嵌入检索（ChromaDB + BGE-M3） |
| `generator/prompt_generators/` | 提示策略实现（rag、add_shot、selected_shot 等） |
| `vendor/mkb/reference/` | MKB PyTorch 参考实现（按类别分子目录） |
| `vendor/mkb/dataset.py` | 合法 `op_key` 与类别映射 |
| `vendor/mkb/correctness.py` | 正确性对比模板 |
| `output/*.txt` | 示例/批次的 MKB 风格算子 txt 包 |
| `CANN_skills/` | CANN Skills 子模块（昇腾开发工作流的可复用技能/Agent） |
| `artifacts/<group可选>/<op_key>/` | 每算子构建产物、日志与结果 JSON（`<group可选>` 为 output 下相对路径镜像归档） |
