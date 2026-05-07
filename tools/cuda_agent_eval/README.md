# CUDA-Agent 风格多算子评测（独立入口）

本目录与 [`tools/eval_cuda_agent_operator.py`](../eval_cuda_agent_operator.py) 提供 **与单算子 MKB 流程并行** 的评测入口：参考模型来自 **CUDA-Agent-Ops-6K** 数据集行的 PyTorch `code`（快照为 `eval/reference_code.py`），自定义实现仍为完整 AscendC 工程 + pybind，数值比对复用 [`vendor/mkb/correctness.execute_template`](../../vendor/mkb/correctness.py)。

**未改动**：[`tools/eval_operator.py`](../../tools/eval_operator.py)（单算子主入口与 `_execute_pipeline` 行为不变）。

**扩展**：[`tools/txt_operator.py`](../../tools/txt_operator.py) 新增 **`materialize_cuda_agent_operator_from_txt`**（不要求 MKB `get_ref_py_path`），以及共享写入函数 **`_materialize_bundle_to_operator_dir`**。通过 **`eval_cuda_agent_operator.py --txt`** 材料化时：

- 仅有 **六块**（必填五段 + **`model_src`**、**无 `eval_src`**）时，会自动生成 **`eval/spec.py`**（从 **`eval/reference_code.py`** 读 golden，与 **`eval/model_new.py`** 一起做 `execute_template`），因此 **只需命令行 `--dataset-path` + `--row-index`** 即可选用 jsonl 中对应行的 PyTorch 参考，**不必**在 txt 里再抄一份 `eval_src`。
- 若 bundle **含 `eval_src`**：仍按 bundle 写入 **`eval/spec.py`**；若同时有 **`model_src`**，仍写入 **`eval/model_new.py`**。

---

## Python 与环境

多算子评测与单算子 **`eval_operator.py` 相同**：完整 **`full`** 流水线依赖 **CANN、`torch_npu`、NPU**，请在已配置昇腾栈的 conda 中运行。

与本仓库文档一致时，可先加载 conda 并激活 **`czh_environ`**（路径按你本机安装位置修改），例如：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate czh_environ
```

依赖安装顺序见仓库根目录 [`requirements-generation.txt`](../../requirements-generation.txt)（其中说明了「仅生成」与「与评测同一 conda」两种用法）；若与评测共用环境，应按 README §1.5.3 / `requirements-generation.txt` 保持 **`torch` / `numpy` 版本与 CANN 配套一致**，避免混装 CPU CUDA wheel。

评测脚本本身不限定解释器路径；激活上述环境后使用其中的 **`python3`** 调用即可。

---

## 产物目录

- **默认根目录**：仓库根下 **`artifacts_cuda_agent/`**（与单算子使用的 **`artifacts/`** 并列，避免混淆）。
- **覆盖方式**（任选其一）：
  - 环境变量 **`LLM4ASCENDC_CUDA_AGENT_ART_ROOT`** 指向绝对路径；
  - 命令行 **`--art-root`**。

单任务布局与单算子流水线一致（`workspace/`、`generated/`、`pybind/`、`logs/`、`state/`、`result_<op_key>.json`），并额外写入 **`meta_task.json`**（见下文）。

---

## 命令行入口（与 `eval_operator.py` 对齐）

**算子来源二选一**（与单算子一致）：

- **`--txt <PATH>`**：每次运行会先 **删除** `artifacts_cuda_agent/<可选output分组>/_txt_staging/<txt文件名不含后缀>/`，再用 **`materialize_cuda_agent_operator_from_txt`** 生成 **`operator/`**；无需事先手写材料化目录。
- **`--op <DIR>`**：直接使用已有材料化目录。

产物根：默认 **`artifacts_cuda_agent/`**（txt 在仓库 **`output/`** 下时，会镜像单算子逻辑拼接 **分组子路径**，规则与 `eval_operator.py` 使用的 **`_artifact_group_rel_from_txt_path`** 相同）。

**清理**：透传 **`--clean-policy force|smart`** 到 **`_execute_pipeline`**，对该 **`op_key`** 下的 **`workspace/`、`pybind/`** 等与单算子一致；**staging** 仅在每次 **`--txt`** 运行开头整目录重建（等价于单算子对 `_txt_staging` 的处理）。

```bash
cd LLM4AscendC
python3 tools/eval_cuda_agent_operator.py --help
```

### 常用参数

| 参数 | 说明 |
|------|------|
| **`--txt`** | **与 `--op` 互斥**。MKB 风格 bundle；**stem = `op_key`**；材料化到 **`.../_txt_staging/<stem>/operator/`**。 |
| **`--op`** | **与 `--txt` 互斥**。已展开算子目录。 |
| `--mode` | `full`（默认） / `build-only` / `eval-only`。 |
| `--clean-policy` | `force`（默认） / `smart`。 |
| `--art-root` | 覆盖本次产物根（txt 模式下仍会在其下写 `_txt_staging`）。 |
| `--dataset-path` | CUDA-Agent-Ops-6K **`.parquet` 或 `.jsonl`**（须与 **`--row-index`** 成对）。 |
| `--row-index` | 写入 **`eval/reference_code.py`**。 |
| `--check-reference-only` | 只做参考校验（可选快照），不跑编译/NPU。 |

**`full` / `eval-only`**：需要 **`eval/reference_code.py`**；若本地尚无，请 **`--dataset-path` + `--row-index`**，或事先放好该文件。**`build-only`** 不要求参考文件。

**与单算子生成对齐**：批量生成的 txt 若只有 **六块**（无第七段 `eval_src`），用本脚本 **`--txt`** 材料化即可；**`--dataset-path` / `--row-index`** 指定 jsonl 行后，评测使用的 golden 即为该行 **`code`** 快照到 **`reference_code.py`** 的内容。

### 示例（推荐：仅 txt + 数据集行）

```bash
python3 tools/eval_cuda_agent_operator.py \
  --txt output/cuda_agent_ops_6k/ca6k_rh_fused.txt \
  --dataset-path data/external/CUDA-Agent-Ops-6K/cuda_agent_ops_6k.jsonl \
  --row-index 55 \
  --mode full --clean-policy force
```

### 示例（已有材料化目录）

```bash
python3 tools/eval_cuda_agent_operator.py --op output/cuda_agent_ops_6k/ca6k_rh_fused_operator \
  --dataset-path data/external/CUDA-Agent-Ops-6K/cuda_agent_ops_6k.jsonl --row-index 55
```

读取 parquet 需要 **`datasets`** 或 **`pyarrow`**；仅用 **`cuda_agent_ops_6k.jsonl`** 时无需二者。

---

## `eval/spec.py` 约定（多算子）

参考 [`eval_spec_example.py`](eval_spec_example.py)（可复制为算子目录下的 **`eval/spec.py`**）：

1. 从 **`eval/reference_code.py`** 执行得到 **`Model`、`get_inputs`、`get_init_inputs`**（与 `execute_template` 约定一致）。
2. 通过环境变量 **`LLM4ASCENDC_OP_MODULE`** 导入 pybind 扩展（由现有 `run_eval` 注入，与单算子一致）。
3. 从 **`eval/model_new.py`** 加载 **`ModelNew`**（手写 oracle 或后续生成产物），内部使用 **`custom_ops_lib`**。
4. 调用 **`execute_template`**，对 **整条前向的最终输出** 做 `allclose`（与单算子评测语义相同：**整体一致**，非逐算子中间结果）。

`op_key` 建议在独立命名空间内命名（例如计划中的 **`ca6k_00042`**），**无需** 登记 [`vendor/mkb/dataset.py`](../../vendor/mkb/dataset.py)。

---

## `meta_task.json`

在使用 **`--dataset-path`** 与 **`--row-index`** 时，会在 **`artifacts_cuda_agent/<op_key>/meta_task.json`** 写入任务元数据，便于批量实验追溯：

- `schema`: `cuda_agent_meta_task_v1`
- `row_index`、`ops`、`data_source`、`parquet_path`
- `code_sha256`：快照后参考源码的 SHA256

该文件用于记录与追溯；**默认不参与** `op_dir` 指纹（指纹仍以材料化源码为准）。若希望「改参考即强制重编」，请确保 **`eval/reference_code.py`** 已纳入材料化目录并参与指纹。

---

## 本目录模块说明

| 文件 | 作用 |
|------|------|
| [`constants.py`](constants.py) | 默认产物根、`LLM4ASCENDC_CUDA_AGENT_ART_ROOT`、`ca6k_{index:05d}` 命名建议、文件名常量。 |
| [`meta.py`](meta.py) | `meta_task.json` 载荷与写入。 |
| [`dataset_snapshot.py`](dataset_snapshot.py) | 读取数据集行、写入 `reference_code.py`、参考符号检查。 |
| [`eval_spec_example.py`](eval_spec_example.py) | 标准 **`eval/spec.py`** 示例实现。 |

---

## 与批量任务的关系

并行或「6000 行 → 多目录」的批量调度 **未** 在本仓库单独实现；当前约定为：**每个材料化算子目录调用一次** `eval_cuda_agent_operator.py`。后续若增加队列/worker，应对齐 [`eval_operator.py`](../eval_operator.py) 中 `--txt-dir` 与 `LLM4ASCENDC_ASCEND_CUSTOM_OPP_PATH` 的多进程策略。

---

## 评测语义提示（融合与归因）

CUDA-Agent-Ops-6K 中多为 **多算子融合** 的 `Model`；本 harness 的比对与 **`execute_template` 相同，均为最终输出级**。若自定义实现将多个算子 **融合优化**，只要数值与参考一致即通过；**失败时通常无法从单次结果自动归因到某一个子算子**，除非另行增加中间张量校验等扩展手段。
