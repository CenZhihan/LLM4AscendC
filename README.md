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

**算子来源（三选一，必选）**

| 参数 | 含义 |
|------|------|
| `--op <DIR>` | 已 materialize 的算子目录，例如 `operators/xxx` 或绝对路径。目录内需含 `operator.json`、`op_host/`、`op_kernel/`、`eval/spec.py` 等。 |
| `--txt <PATH>` | 单个 **MKB 风格 txt 包** 路径（如 `output/gelu.txt`）。脚本会写入 `artifacts/_txt_staging/`，并生成临时算子目录再跑流水线。 |
| `--txt-dir <DIR>` | 目录内所有 `*.txt` **按文件名排序后依次** 评测；每个文件的 **stem = MKB `op_key`**。某次失败会记录错误并继续下一个（返回码非 0）。 |

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

# 批量目录
python3 tools/eval_operator.py --txt-dir output/kernelbench165_txt --clean-policy smart
```

### 2.2 可选参数一览

| 参数 | 默认值 | 可选值 | 含义与适用场景 |
|------|--------|--------|----------------|
| `--mode` | `full` | `full` / `build-only` / `eval-only` | **`full`**：msopgen → 编译 → 安装 OPP → 构建并安装 pybind wheel → 运行 `eval/spec.py`（正确性或与参考对比）。**`build-only`**：只做到安装 wheel 并写 `state/`，不执行 eval；适合 CI 只验证能否编过。**`eval-only`**：跳过构建，直接跑 eval；要求 `artifacts/<op_key>/state/installed.json` 已存在，且源码指纹与上次构建一致，否则会报错要求先 `full` 或 `build-only`。适合反复调试验证逻辑而不重复长时间编译。 |
| `--clean-policy` | `force` | `force` / `smart` | **`force`**：每次进入构建阶段前 **删除** `artifacts/<op_key>/workspace` 与 `pybind`，保证从干净目录重编；结果可复现、磁盘与耗时开销较大。**`smart`**：仅当算子目录 **内容指纹**相对上次构建发生变化时，才清理并重编；指纹未变则复用已有构建；若期望复用但从未成功安装过，会报错提示改用 `force`。适合迭代小改或批量任务中减少重复编译。 |

**说明**：`--txt` / `--txt-dir` 会清理并重建 `artifacts/_txt_staging/`（单 txt 时为 staging 根；batch 时为每个 stem 子目录），避免旧内容干扰。

### 2.3 流水线在做什么（`full` / `build-only` 的构建部分）

1. **指纹**：对算子目录做内容指纹，供 `smart` 策略判断是否重编。  
2. **msopgen**：根据 `operator.json` 中的 IR 生成 AscendC 工程。  
3. **覆盖源码**：将 `op_host/`、`op_kernel/` 覆盖到生成树。  
4. **build.sh**：编译并打 **custom OPP**（`.run`）。  
5. **安装 OPP**：默认安装到可写路径（见 `tools/common/env.py` 中的 `ascend_custom_opp_path`），并 **source** 对应 `set_env.bash` 以便运行时加载。  
6. **pybind**：基于模板生成扩展、`bdist_wheel` 并 `pip install`。  
7. **eval**（仅 `full`）：在独立子进程中执行 `eval/spec.py`，通常会加载 MKB reference 与自定义 `ModelNew` 做对比。

各步骤日志位于 **`artifacts/<op_key>/logs/`**，文件名带时间戳与序号，例如 `*-01-msopgen.log` … `*-06-eval.log`。

### 2.4 结果输出

- **`artifacts/<op_key>/result_<op_key>.json`**：汇总 `compiled`、`correctness`、`correctness_info`、指纹与日志路径等。  
- 失败时终端会打印 `[done] FAILED` 并附 summary 路径；成功为 `[done] OK`。

### 2.5 环境变量说明

本仓库**直接读取或改写**的环境变量如下（不含 CANN 工具链内部大量 `ASCEND_*`，那些由 `source set_env.sh` 等注入）。

#### 使用者可自行设置（可选）

| 变量 | 典型取值 | 作用 |
|------|----------|------|
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

## 3. 相关路径速查

| 路径 | 说明 |
|------|------|
| `tools/eval_operator.py` | 统一评测入口 |
| `vendor/mkb/reference/` | MKB PyTorch 参考实现（按类别分子目录） |
| `vendor/mkb/dataset.py` | 合法 `op_key` 与类别映射 |
| `vendor/mkb/correctness.py` | 正确性对比模板 |
| `output/*.txt` | 示例/批次的 MKB 风格算子 txt 包 |
| `artifacts/<op_key>/` | 每算子构建产物、日志与结果 JSON |
