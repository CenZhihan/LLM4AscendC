# 多轮修复记忆机制设计（第一阶段）

**目标**：在 AscendC 算子智能体多轮生成—评测—修复流程中，把「跨轮可验证的改进」沉淀为可检索经验，并在后续 attempt 中注入上下文，减少同类 API / 构建错误反复出现。

**范围**：仅做**修复型记忆**（记录「如何变好」）；不做「规避型」泛化负面清单（留待后续）。与现有多轮脚本逻辑对齐：每轮有评测结果（`compiled` / `correctness`）、修复用错误摘要、以及相邻两轮代码可比。

---

## 1. 记忆载体与单条结构

**载体**：以 **JSON Lines（`.jsonl`）** 为主存——便于校验字段、追加写入、后续换 embedding 检索而不推翻格式。若需人类可读展示，可由同目录脚本渲染为 Markdown，**不以大段 Markdown 为唯一真源**。

**单条记忆（证据包 + 叙述）必填要素**：


| 要素     | 说明                                                                                   |
| ------ | ------------------------------------------------------------------------------------ |
| 标识与版本  | `memory_id`、`schema_version`，便于演进与去重。                                                |
| 实验上下文  | `op`、`category`、`tool_mode`、`strategy`、`eval_mode` 等随条记录；**manifest 不再按 tool_mode/eval_mode 预筛**，选条时由 `query_text` 提供当前配置供模型判断相关性。                  |
| 分层标签   | `tier`：`A` 或 `B`；`confidence`：`high`（仅 A）/ `medium`（B）。                              |
| 状态转移   | 上轮与本轮的 `compiled` / `correctness` 对比（机器可读、不可由 LLM 改写）。                               |
| 失败阶段   | `failure_stage_before` / `failure_stage_after`：由评测产物中的阶段信息（如构建链各阶段、数值评测阶段）规则映射得到粗标签。 |
| 错误锚点   | 各轮短锚点（从 `correctness_info` 与关键日志尾部抽取的稳定子串），用于检索与人工核对。                                |
| 代码侧证据  | 可选：代码摘要或 hash，用于否定「无实质改动却写记忆」。                                                       |
| 证据引用   | 相对路径或 run 内定位：相邻两轮产物、结果 JSON、修复上下文文件等。                                               |
| 自然语言经验 | **English** conditional: **When** [trigger] **do not** [bad] **; instead** [good]; no absolutes.            |


**Review 模型职责**：在**规则已判定可写**之后，仅生成/润色 **`natural_language`（英文一句）**；输入含相邻两轮 **日志信号摘要** 与 **unified diff（截断）**；若与客观 `transition` 矛盾则丢弃本条。

---

## 2. 写入时机与门槛

**挂接位置（逻辑）**：第 `k` 轮评测结束、已持久化本轮结果与修复上下文之后；**开始第 `k+1` 轮生成之前**。此时同时具备第 `k-1` 轮与第 `k` 轮的错误与代码，便于对比（`k = 1` 无上一轮，一般不写跨轮记忆，见下）。**Review 调用前**由 `code_diff.format_attempt_code_diff` 对相邻两轮完整 txt 产物做 **unified diff**（长度有上限），与评测信号摘要一并送入总结模型。

**总否定（任一满足则不写入主库）**：

- 本轮无可靠评测状态（生成失败、评测链路未产出有效 `compiled`/`correctness`、纯基础设施类失败如网关/超时且无稳定构建锚点）。
- 与上一轮相比**无实质代码变更**。
- Review 输出不满足模板或与客观转移不一致。

**Tier A（`confidence = high`，优先沉淀）**——`k ≥ 2`，且满足其一：

- `compiled`：非成功 → 成功。
- 在已可编译前提下，`correctness`：未通过 → 通过。

**Tier B（`confidence = medium`，条件更严）**——`k ≥ 2`，整体仍未通过（例如数值仍未过），且**不满足** Tier A，但**同时**满足：

- **阶段推进**：`failure_stage_before` 与 `failure_stage_after` 不同，且符合预先约定的「向更可诊断/更浅层失败推进」白名单（实现时固定表，避免主观）。
- **锚点变化**：规范化后的前后锚点不相同（排除仅换行、路径前缀等伪变化）。
- **代码变更**：前后代码摘要或 hash 不同。

**不写**：仅日志措辞变化、客观状态与阶段均未改善、或无法凑齐证据包字段。

---

## 3. 利用时机与方式

**时机**：**每一次**即将开始新一轮生成调用**之前**（含 `attempt = 1` 与 `attempt ≥ 2`）。

- 首轮：查询可依赖 `op`、`category`、`tool_mode` 等（`query_text` 中仍会带上 `tool_mode`/`eval_mode` 供选条模型参考）；无上一轮错误文本时偏「类级」经验。
- 次轮及以后：查询以**当前将注入的修复摘要**（错误锚点、阶段、`correctness_info` 片段）为主，提高相关性。

**方式**：从全局 `.jsonl` 的 **尾部窗口**（默认最多约 500 条）生成 manifest 短行；**不再**按 `tool_mode` / `eval_mode` 与当前 run 做硬过滤，以便跨工具配置复用（选条模型结合 `query_text` 自行判断是否适用）。以固定区块拼入生成侧上下文（与现有修复提示并列），每条展示宜短：层级、转移一行、条件式经验、关键锚点。

**降权或不注入**：`tier = B` 且与当前错误锚点重叠过低；或 schema 版本过旧。

---

## 4. 工程注意（简要）

- **并行多算子**：全局追加需**并发安全**（文件锁、或每进程写临时队列再合并）。
- **记忆库位置**：默认在仓库根目录下的 **`repair_memory/`**（与代码包 `generator/repair_memory/` 区分）；与单次 `run_dir` 解耦，便于跨实验复用；条目中仍保留 `tool_mode`/`eval_mode` 等元数据供展示与选条模型参考，**manifest 不再按二者硬过滤**。旧路径 `artifacts/repair_memory/` 已废弃，请迁移数据或设置 `LLM4ASCENDC_REPAIR_MEMORY_ROOT`。
- **演进**：字段带 `schema_version`；后续可将「检索 LLM」替换为 embedding + 向量库，不改变 Tier 与否定门槛定义。

---

## 6. 实现落地（与 Claude manifest 思路对齐）

**代码位置**：`generator/repair_memory/`（`schema`、`tier_gate`、`failure_stage`、`anchors`、`inbox`、`merge`、`manifest`、`inject`、`select`、`review_llm`、`pipeline`）；多轮挂接在 `generator/scripts/run_agent_multi_rounds.py` 与 `generator/scripts/run_agent_cuda_agent_multi_rounds.py`；Agent 状态字段 `retrieved_repair_memories` / `eval_mode` 在 `generator/agent/agent_state.py`、`agent_runner.py`，注入在 `nodes/choose_tool.py` 与 `nodes/answer.py`。

**存储布局**（默认根目录 `<REPO_ROOT>/repair_memory/`，可用环境变量覆盖）：

- `canonical/repair_memories.jsonl`：合并后的真源（每行一条完整 JSON）。
- `inbox/<run_slug>/mem_<uuid>.jsonl`：子进程**仅**写入**单行**的独立文件（避免多进程争用同一文件）。
- `inbox/<run_slug>/merged/`：已成功合并入 canonical 的收件副本（便于审计）。

**并行写入策略**：子进程不直接 append canonical；写入 inbox 后调用 `merge_run_inbox`（`fcntl` 独占锁写 canonical，再移动收件文件）。整批实验结束时主进程再执行一次 merge，收敛遗留文件。`run_slug` 由 `run_dir` 相对仓库根路径规范化得到，使同一 output run 共用同一 inbox 桶。

**检索**：从 canonical 尾部窗口生成 **manifest 短行**（`id`、op、category、`tool_mode`、`tier`、锚点、摘要；**不按**当前 run 的 `tool_mode`/`eval_mode` 预筛），选条 LLM 返回 `memory_ids`，再按 id 从尾部窗口取全文片段注入（窗口外旧 id 可能暂不可见，可调 `max_records`）。

**环境变量**：

- `LLM4ASCENDC_REPAIR_MEMORY=0`：关闭写入与检索注入。
- `LLM4ASCENDC_REPAIR_MEMORY_ROOT`：覆盖记忆根目录。

**工具脚本**：`generator/scripts/render_repair_memory_manifest.py` 从 canonical 单向生成 `repair_memory_manifest.txt`（调试/展示）。

**Agent report**：每轮生成结束后，`{op}_report.json` 中增加 **`repair_memories_applied`** 数组：与选条顺序一致，每条含 `memory_id`、`tier`、`natural_language`（及 `transition`、锚点等），便于分析检索是否命中、记忆是否被实际注入。

**平台说明**：合并依赖 Linux `fcntl`；非 Linux 环境合并为 no-op（inbox 仍会累积，可拷贝至 Linux 再合并）。

---

## 5. 小结


| 维度  | 要点                                                                     |
| --- | ---------------------------------------------------------------------- |
| 形式  | JSONL 单条 = 证据包 + 条件式自然语言；Markdown 仅作展示可选。                              |
| 写入  | 仅修复型；Tier A/B 由规则判定；Review 在日志摘要之外可见 **相邻两轮 txt 的 unified diff（有上限）**，便于写出针对具体改动的条件句。 |
| 利用  | 每轮生成前检索 n 条注入；manifest **跨 tool_mode/eval_mode** 复用；控制长度。 |


*文档版本：与「第一阶段：修复型 + Tier A/B」讨论一致，实现细节以代码落地为准。*

---

## 7. 附录：形态说明、Prompt 实录与选条机制

以下与目录 **`generator/agent/_example_prompts_memory/`** 中的示例文件一一对应，便于答辩或给学长展示时直接打开 txt / json。

##### 1. 单条记忆在 canonical 里长什么样？

每条在 **`repair_memory/canonical/repair_memories.jsonl`** 中占一行，为 **一条完整 JSON 对象**（证据字段 + 一条 **英文**条件式 `natural_language`：`When ... do not ...; instead ...`）。字段含义见正文第 1 节表格。

**示例（已排版，便于阅读；真源为单行 JSONL）**：见同目录 **`example_one_memory_record.json`**（从真实 run 抽取的 hardsigmoid Tier A 记录结构）。

##### 2. 给「选条」LLM 看的「摘要」长什么样？

**不是**把整段 `natural_language` 原样塞进 manifest：实现上由 **`generator/repair_memory/manifest.py::build_manifest_lines`** 为每条记忆生成 **一行**，形如：

`id=<uuid>\top=<op>\tcategory=<cat>\ttool_mode=<tm>\ttier=<A|B>\tanchor=<短锚点>\tsummary=<natural_language 单行截断约160字>>`

多条 manifest 用换行拼接成 `manifest_text`。**多行示例**：**`example_manifest_lines.txt`**。

##### 3. 总结记忆（Review LLM）的完整 prompt

写入前由 **`generator/repair_memory/review_llm.py::generate_repair_natural_language`** 调用：一条 **system** + 一条 **user**（英文指令；含 Tier、`Previous/Current attempt signals`、以及可选 **unified diff**；各信号块最长约 6000 字符，diff 默认最长约 12000 字符）。

**完整实录（占位内容；请以仓库内最新代码为准）**：**`review_memory_llm_messages_example.txt`**。

##### 4. 选条 LLM 的完整 prompt

由 **`generator/repair_memory/select.py::select_memory_ids`** 调用：一条 **system**（要求只返回 `{"memory_ids":[...]}`）+ 一条 **user**（`Select up to N` + `=== Manifest ===` + `=== Current query ===`）。

**完整实录**：**`select_memory_ids_llm_messages_example.txt`**。

##### 5. 选条机制是什么？（结合选条 prompt）

1. **入口**：多轮脚本在 **`attempt_id >= 2`** 且开启 **`--use-repair-memory`** 时，在生成前调用 **`build_retrieval_block_for_attempt`**（`inject.py`）。
2. **Manifest**：从 **`repair_memory/canonical/repair_memories.jsonl` 尾部窗口** 读入记录，**仅按 `schema_version` 过滤** 后打成 **每行一条的短 manifest**（见上文第 2 点）；**不再**按当前 run 的 `tool_mode`/`eval_mode` 预筛，以便其它工具配置下沉淀的经验也可进入候选；文件 mtime/size 变化会刷新缓存，避免重复读盘解析。
3. **Query**：由当前 **`op` / `category` / `tool_mode` / `eval_mode` / `attempt_id`** 与 **上一轮修复用原始日志**（`repair_error_logs_raw` 截断）拼成 **`query_text`**。
4. **选条**：单独一次 chat completion，模型在 **「不确定则不选」** 的 system 约束下，从 manifest 里选出至多 **`max_n`（默认 5）** 个 **`memory_id`**；返回需能被正则 **`\{[\s\S]*\}`** 截出并 **`json.loads`** 成功。
5. **取全文**：用返回的 id 在尾部窗口的完整记录里 **按 id 查表**；找不到的 id 丢弃。
6. **注入**：将命中记录的 **`natural_language` + tier + transition + anchors_after** 格式化为 **`format_injection_block`** 文本，写入 agent state 的 **`retrieved_repair_memories`**，供 **工具路由（choose_tool）** 与 **最终写代码（answer）** 两段 user 文本拼接使用。

##### 6. 融入检索记忆后的「算子生成」完整 prompt（写代码阶段）

最终 AscendC 产物主要由 **`generator/agent/nodes/answer.py::answer_node`** 在 **`attempt_id > 1`** 时拼接：**system** 固定为 `You are a helpful assistant.`，**user** 内含「修复约束 + `Original user instruction`（即 `base_prompt`）+ 上一轮日志 + 上一轮全文代码 + 工具检索块 + **`Retrieved repair memories`** 块」。

**完整结构实录（含占位符说明）**：**`answer_node_generation_with_memories_example.txt`**。

> 说明：**首轮 attempt1** 若已有检索块，记忆会以较短模板出现在 user 中；**工具编排轮**中记忆通过 **`choose_tool.py::_extract_user_question`** 拼进 `Task specification`，与 answer 阶段为 **不同 API 调用**，展示材料可按需两段都保留。

##### 7. 本附录文件清单

| 文件 | 用途 |
| --- | --- |
| `example_one_memory_record.json` | 单条 canonical 记忆（排版 JSON） |
| `example_manifest_lines.txt` | 选条 LLM 所见的 manifest 行示例 |
| `review_memory_llm_messages_example.txt` | Review LLM messages 实录 |
| `select_memory_ids_llm_messages_example.txt` | 选条 LLM messages 实录 |
| `answer_node_generation_with_memories_example.txt` | 写代码 LLM messages 实录（含记忆块） |