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

项目提供两套生成入口：

#### （1）`tools/generate_operator.py` — 基于 RAG 提示

```bash
# 构建 RAG 代码索引（首次使用前执行一次）
python3 tools/generate_operator.py --build-index --code-dir ascendCode

# 使用 RAG 策略生成算子
python3 tools/generate_operator.py --model deepseek-chat --strategy rag --categories all --workers 4
```

**常用参数**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--model` | `deepseek-chat` | 使用的 LLM 模型名称 |
| `--strategy` | `rag` | 提示策略：`rag`（检索增强）、`add_shot`（添加示例）、`selected_shot`（按类别选择示例） |
| `--categories` | `all` | 算子类别列表，或 `all` 表示全部生成 |
| `--workers` | `4` | 并行生成的 worker 数量 |
| `--build-index` | — | 构建 RAG 代码索引（与 `--code-dir` 配合使用） |
| `--code-dir` | `ascendCode` | 待索引的 AscendC 代码库目录 |

#### （2）`generator/scripts/generation/generate_agent.py` — 基于 LangGraph Agent

```bash
# Agent + Code RAG only 生成全部算子
python3 generator/scripts/generation/generate_agent.py \
  --tool-mode code_rag_only --strategy add_shot --categories all --workers 4

# Agent + 仅使用 102 个已验证算子（kernelbench102）
python3 generator/scripts/generation/generate_agent.py \
  --tool-mode code_rag_only --strategy add_shot --kernelbench102 --workers 4

# Agent + KB + Code RAG 组合
python3 generator/scripts/generation/generate_agent.py \
  --tool-mode kb_and_code_rag --strategy add_shot --categories activation --workers 4

# 指定模型（覆盖 local_api_config 中的 XI_AI_MODEL），仅 KB，activation ∩ kernelbench102
python3 generator/scripts/generation/generate_agent.py \
  --model gpt-4o --tool-mode kb_only --strategy one_shot \
  --categories activation --kernelbench102 --workers 4 --runs 1
```

**Agent 专用参数**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--tool-mode` | `no_tool` | 工具模式：上列预置名，或 **逗号分隔** 的小写工具键（如 `kb,code_rag,env_check_env`）及已注册插件名；由 `parse_tool_mode` 解析 |
| `--strategy` | `add_shot` | Prompt 策略名称，由 `prompt_generators/` 注册表提供 |
| `--kernelbench102` | — | 只生成 `generator/kernelbench102_ops.py` 中登记的 102 个已验证算子 |
| `--categories` | `all` | 算子类别列表 |
| `--workers` | `4` | 并行 worker 数量 |
| `--start-from` | — | 从指定算子开始生成（断点续传） |
| `--runs` | `1` | 运行轮数 |
| `--model` | — | 仅覆盖模型名；**优先级高于** `generator/local_api_config.py` 中的 `XI_AI_MODEL` / `MODEL` |
| `--output-dir` | — | 自定义输出根目录；不设则按解析后的模型名自动分目录（见下节「默认输出」） |

> **注意**：若 `--tool-mode` 包含 `code_rag`，需提前构建 RAG 索引（同 `tools/generate_operator.py --build-index`）。**Agent 的 LLM** 仅从 `generator/local_api_config.py` 读取密钥与 Base URL（见 §3.5.5），**不再**使用 `USE_API_CONFIG` 或 `XI_AI_*` 环境变量作为 Agent 入口的配置来源。

#### （3）Python API — 自定义工具组合

除 CLI 预置名外，可用 **`frozenset[str]`**（`AgentToolMode`）列出要启用的工具键；键名一律为 **小写 snake_case**，与 LLM 输出的 JSON `tool` 字段一致。

```python
from generator.agent import (
    generate_kernel_with_agent,
    KernelGenerationTask,
    parse_tool_mode,
)

from generator.agent.agent_config import KB_ONLY, CODE_RAG_ONLY

# 使用 frozenset 自定义工具组合（字符串键）
custom_mode = frozenset({"kb", "code_rag", "env_check_env"})

task = KernelGenerationTask(
    language="ascendc",
    op="gelu",
    strategy_name="add_shot",
    category="activation",
)
result = generate_kernel_with_agent(task, custom_mode)
print(result.generated_code)
```

也支持直接传入字符串，内部由 `parse_tool_mode` 解析（可包含已在 `tool_registry` 注册的插件名）：

```python
result = generate_kernel_with_agent(task, "kb,code_rag,env_check_env")
result = generate_kernel_with_agent(task, parse_tool_mode("kb,code_rag,env_check_env"))
```

内置工具键一览（与 `generator/agent/agent_config.py` 中 `BUILTIN_TOOL_NAMES` 一致）：

| 键 (`tool`) | 说明 |
|-------------|------|
| `kb` | 知识库查询（ChromaDB Ascend C API 文档） |
| `web` | 网页搜索（DuckDuckGo / `ddgs`） |
| `code_rag` | 代码检索（`generator/rag/` 索引） |
| `env_check_env` | 环境概览（CANN、工具链） |
| `env_check_npu` | NPU 设备查询 |
| `env_check_api` | API 头文件存在性检查 |
| `kb_shell_search` | 知识库内 shell/grep 式搜索 |
| `api_lookup` | API 签名与文档查询 |
| `api_constraint` | API 参数与平台约束检查 |
| `api_alternative` | API 替代方案 |
| `tiling_calc` | Tiling 参数计算 |
| `tiling_validate` | Tiling 参数校验 |
| `npu_arch` | NPU 架构 / UB 容量等 |
| `code_style` | Ascend C 代码风格检查 |
| `security_check` | 不安全模式扫描 |

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

Agent 基于 LangGraph `StateGraph`：在 `build_agent_app` 入口会 **`get_tool_registry().clear()`**，再为本图的 `tool_mode` **仅注册**出现的内置工具（`RegisteredToolSpec` + 薄 `handler` 包装现有节点）以及快照恢复的插件；拓扑为 **`entry` → `choose_tool` → (`answer` | `tool_dispatch` → `choose_tool`)**。

**与 `eval_operator.py` 的关系**：Agent 与脚本生成路径将算子 `.txt` 写到 `output/`（或 `--output-dir`）；评测仍用 `tools/eval_operator.py --txt …` / `--txt-dir …` 指向这些文件，二者只通过产物路径衔接，无代码耦合。

#### 3.5.1 工具类型与数据源

| 工具 | JSON `tool` 键 | 数据源 | 说明 |
|------|----------------|--------|------|
| 知识库查询 | `kb` | ChromaDB Ascend C API 文档 | 本地 API 文档片段；每次调用由 `kb_query_node` 取 **top_k=3** 条片段写入状态 |
| 网页搜索 | `web` | DuckDuckGo（`ddgs`） | 技术文档与博客 |
| 代码检索 | `code_rag` | `generator/rag/` | 相似内核实现 |
| 环境概览 | `env_check_env` | CANN / 运行时 | 工具链与版本 |
| NPU 查询 | `env_check_npu` | `npu-smi` 等 | 设备状态 |
| API 头文件检查 | `env_check_api` | CANN 头文件 | 符号是否存在 |
| KB Shell 搜索 | `kb_shell_search` | 打包知识库目录 | grep/find 式搜索 |
| API 查找 | `api_lookup` | 结构化 API 文档 | 签名与限制 |
| API 约束 | `api_constraint` | 同上 | 对齐、repeatTimes 等 |
| API 替代 | `api_alternative` | 同上 | 备选 API |
| Tiling 计算 | `tiling_calc` | 推导规则 | 分块参数 |
| Tiling 验证 | `tiling_validate` | 推导规则 | 合法性检查 |
| NPU 架构 | `npu_arch` | 内置芯片表 | UB、宏、特性 |
| 代码风格 | `code_style` | 规则集 | 风格问题 |
| 安全检查 | `security_check` | 规则集 | 危险模式 |

底层仍以各 **Retriever** 实现为主；**`tool_dispatch`** 节点根据 `state["next_action"]`（小写键或 `ANSWER`）在 **同一 Registry** 上查找并调用 `handler`。

实现落点简述：**`generator/agent/nodes/`** 中各节点负责一次工具调用的编排（读状态、调 retriever、写回结果）；**`generator/agent/retrievers/`** 承担具体检索与环境探测逻辑。**`builtin_tools.py`** 为内置工具组装注册表元数据并把 `handler` 指到上述节点。选工具、写答案、KB 英译子请求等 **凡进入 LLM API 的文案当前为英文**。

#### 3.5.2 Agent 模式

`AgentToolMode` 即 `frozenset[str]`（小写工具键，外加可选插件键）。常用 **字符串** 预置名由 `parse_tool_mode` 解析：

| 模式字符串 | 含义 |
|------------|------|
| `no_tool` | 不使用工具，直接 `answer` |
| `kb_only` / `web_only` / `code_rag_only` | 单工具 |
| `kb_and_web` / `kb_and_code_rag` / `web_and_code_rag` | 两两组合 |
| `all` | 预置子集（`kb` + `web` + `code_rag` + 三个 `env_check_*`） |

任意扩展组合示例：

```python
from generator.agent.agent_config import ALL

mode = frozenset({*ALL, "kb_shell_search", "api_lookup"})
```

#### 3.5.3 Agent 工作流（统一调度）

```
entry
  │
  ├─ (no_tool) ──────────────────────────────→ answer → END
  └─ (有工具) → choose_tool
                    │
                    ├─ (query_round_count ≥ MAX_QUERY_ROUNDS) → answer → END
                    ├─ (JSON 解析失败或 tool 不在本图 tool_mode) → choose_tool（自环，burn 一轮）
                    ├─ tool = ANSWER ─────────────────────────→ answer → END
                    └─ 否则 → tool_dispatch（Registry handler）→ choose_tool
```

- **`choose_tool`**：向 LLM 发送 **全英文** 说明与 few-shot（含 CANN / API 快速迭代、**优先用已启用工具**获取最新与最规范用法、工具检索有效的编排指引）；模型必须只输出一个 JSON 对象（见下节）。不再发起「二次 JSON 修复」模型调用。
- **`tool_dispatch`**：按 Registry 调用对应 `handler`（内部仍为各 `*_node`），写回各 `*_results` / `tool_calls_log`；`query_round_count` 由各工具节点与 `choose_tool` 的 burn 逻辑按原语义更新。
- **`answer`**：根据 `messages` 与聚合检索结果生成最终代码。
- 最大轮数：`generator/agent/agent_state.py` 中 **`MAX_QUERY_ROUNDS`**（默认 3）；达到上限后 **`choose_tool` 侧** 也会强制要求输出 `ANSWER`。

#### 3.5.3.1 JSON 工具选择与解析失败（ToolChoiceV1）

- 字段：`"tool"`（**小写**内置键、已注册插件键、或 **`ANSWER`**）、`"query"`（字符串）、`"args"`（对象或 `null`）。
- 若 **无法解析 JSON** 或 **`tool` 不在当前 `tool_mode`**：`query_round_count += 1`（与跑完一次工具等价 burn），`tool_choice_parse_failed=True`，向 **`tool_choice_error_log`** 追加一条（含 `raw_model_output`、错误说明、时间戳等），路由 **回到 `choose_tool`**；**不会**立刻进入 `ANSWER`。
- 批跑报告：`generate_kernel_with_agent` 返回的 `report` / 落盘的 `{op}_report.json` 中含 **`tool_choice_parse_errors`**（对 `tool_choice_error_log` 的摘要，含截断后的原始输出）。

#### 3.5.4 使用示例

**CLI 方式**（推荐，最简单）：

```bash
# 使用 Agent + Code RAG 生成单个算子
python3 generator/scripts/generation/generate_agent.py \
  --tool-mode code_rag_only --strategy add_shot \
  --categories activation --workers 4
```

**默认输出目录**（未传 `--output-dir` 时）：`output/ascendc/<model_slug>/agent_<tool_mode_string>/<strategy>/run<N>/`。其中 `<model_slug>` 由最终采用的模型名经路径安全化得到（与 `--model` 或配置文件中的 `XI_AI_MODEL` 一致，例如 `gpt-4o`）；`<tool_mode_string>` 为 `tool_mode_to_string` 的结果（如 `kb_only`、`kb,code_rag`）。传 `--output-dir` 则完全使用该路径，不再自动插入 `<model_slug>` 等分段。

**Python API 方式**（适合脚本调用）：

```python
from generator.agent import (
    generate_kernel_with_agent,
    KernelGenerationTask,
    parse_tool_mode,
)

task = KernelGenerationTask(
    language="ascendc",
    op="gelu",
    strategy_name="add_shot",
    category="activation",
)
result = generate_kernel_with_agent(task, "code_rag_only")
print(result.generated_code)

custom_mode = frozenset({"kb", "code_rag", "env_check_env"})
result = generate_kernel_with_agent(task, custom_mode)
```

#### 3.5.5 Agent LLM 配置

**`generate_agent.py` CLI** 与 **`generate_kernel_with_agent(..., llm_config=None)`** 解析配置的方式如下（与 `tools/generate_operator.py` 使用的 `generator/llm_config.py` **相互独立**）：

1. **命令行** `--model`：若传入非空字符串，则仅 **覆盖模型名** `model` 字段。
2. **`generator/local_api_config.py`**（须自行从 `generator/local_api_config.example.py` 复制并填写）：读取 `XI_AI_API_KEY` 或 `OPENAI_API_KEY`、`XI_AI_BASE_URL` 或 `OPENAI_API_BASE`（未填则回退为 `https://api-2.xi-ai.cn/v1`，见 `generator/agent/agent_config.py`）、`XI_AI_MODEL` 或 `MODEL`。`api_key` 缺失，或未传 `--model` 且配置里也没有模型名时，会报错退出；**不再**通过 `USE_API_CONFIG`、`XI_AI_*` 环境变量或 DeepSeek 默认兜底为 Agent 提供密钥。

在代码中传入 **`llm_config=dict`** 时，将 **直接使用** 该字典（需自行包含 `api_key`、`base_url`、`model`），不再读取上述文件。

#### 3.5.6 在线文档检索工具试跑（未接入 Agent 路由）

为便于先观察在线文档检索返回结构，仓库提供了两类 retriever 与一个合并测试脚本：

- `generator/agent/retrievers/ascend_docs_search_retriever.py`：在线文档搜索（hiascend）
- `generator/agent/retrievers/ascend_docs_fetch_retriever.py`：按 URL 抓取并结构化提取正文/API/代码/表格
- `tools/test_ascend_docs_tools.py`：`search` / `fetch` / `chain` 三模式本地试跑

示例：

```bash
# 只搜索（推荐中文关键词）
python3 tools/test_ascend_docs_tools.py \
  --mode search --keyword "AscendC::DataCopy 对齐限制" \
  --doc_type API --page_size 5 --version 8.5.0

# 只抓取
python3 tools/test_ascend_docs_tools.py \
  --mode fetch --url "https://www.hiascend.com/document/detail/zh/..." \
  --extract_code --extract_tables --max_content_chars 8000

# 链式：先搜索再抓 top2，并保存 JSON
python3 tools/test_ascend_docs_tools.py \
  --mode chain --keyword "Ascend C 临时内存申请" \
  --fetch_topk 2 --extract_code --print_json \
  --save_json /tmp/ascend_docs_chain.json
```

依赖：`requirements-generation.txt` 中已列出 `requests` 与 `beautifulsoup4`。

### 3.6 配置与依赖

| 文件 | 作用 |
|------|------|
| `generator/config.py` | 生成器与 Agent 的配置（模型路径、RAG 参数、Agent 参数等） |
| `generator/agent/agent_config.py` | `AgentToolMode`（`frozenset[str]`）、`parse_tool_mode`、预置模式常量、LLM 配置 |
| `generator/kernelbench102_ops.py` | 预定义的 102 个通过数值正确性验证的算子列表 |
| `api_config.py`（可选） | LLM API 密钥、基地址、模型名称 |
| `tools/common/env.py` | CANN/NPU 环境配置（与评测流水线共享） |
| `requirements.txt` | Python 依赖列表 |

---

## 4. 相关路径速查

| 路径 | 说明 |
|------|------|
| `tools/eval_operator.py` | 统一评测入口 |
| `tools/generate_operator.py` | LLM 算子生成入口（RAG 策略） |
| `generator/scripts/generation/generate_agent.py` | Agent 生成入口（LangGraph 多轮工作流） |
| `generator/agent/` | LangGraph Agent（Registry + `tool_dispatch` 单调度） |
| `generator/agent/builtin_tools.py` | 按 `tool_mode` 注册内置 `RegisteredToolSpec` |
| `generator/agent/tool_registry.py` | 进程内工具注册表（内置 + `register_tool` 插件） |
| `generator/agent/nodes/` | 各工具节点（编排层，调用 `retrievers/`） |
| `generator/agent/retrievers/` | KB / Web / Code RAG / 环境检查等具体检索与探测实现 |
| `generator/agent/retrievers/ascend_docs_search_retriever.py` | 在线 Ascend 文档搜索 retriever（hiascend） |
| `generator/agent/retrievers/ascend_docs_fetch_retriever.py` | 在线 Ascend 文档详情抓取与结构化解析 retriever |
| `generator/agent/agent_config.py` | 工具键常量、`parse_tool_mode`、`tool_mode_to_string`、Agent LLM 本地文件加载 |
| `generator/agent/_example_prompts_relu_kb/` | `choose_tool` / `answer` 侧提示样例快照（与线上一致时宜同步更新） |
| `tools/test_ascend_docs_tools.py` | 在线文档工具本地试跑脚本（search/fetch/chain） |
| `generator/rag/` | RAG 代码索引与嵌入检索（ChromaDB + BGE-M3） |
| `generator/prompt_generators/` | 提示策略实现（rag、add_shot、selected_shot 等） |
| `generator/kernelbench102_ops.py` | 102 个通过数值验证的算子列表 |
| `vendor/mkb/reference/` | MKB PyTorch 参考实现（按类别分子目录） |
| `vendor/mkb/dataset.py` | 合法 `op_key` 与类别映射 |
| `vendor/mkb/correctness.py` | 正确性对比模板 |
| `output/*.txt` | 示例/批次的 MKB 风格算子 txt 包 |
| `CANN_skills/` | CANN Skills 子模块（昇腾开发工作流的可复用技能/Agent） |
| `artifacts/<group可选>/<op_key>/` | 每算子构建产物、日志与结果 JSON（`<group可选>` 为 output 下相对路径镜像归档） |
