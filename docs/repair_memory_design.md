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
| 实验上下文  | `op`、`category`、`tool_mode`、`strategy`、`eval_mode` 等，检索时过滤，避免跨配置误用。                  |
| 分层标签   | `tier`：`A` 或 `B`；`confidence`：`high`（仅 A）/ `medium`（B）。                              |
| 状态转移   | 上轮与本轮的 `compiled` / `correctness` 对比（机器可读、不可由 LLM 改写）。                               |
| 失败阶段   | `failure_stage_before` / `failure_stage_after`：由评测产物中的阶段信息（如构建链各阶段、数值评测阶段）规则映射得到粗标签。 |
| 错误锚点   | 各轮短锚点（从 `correctness_info` 与关键日志尾部抽取的稳定子串），用于检索与人工核对。                                |
| 代码侧证据  | 可选：代码摘要或 hash，用于否定「无实质改动却写记忆」。                                                       |
| 证据引用   | 相对路径或 run 内定位：相邻两轮产物、结果 JSON、修复上下文文件等。                                               |
| 自然语言经验 | **条件式**三要素：**当** [触发条件，含阶段/锚点] **时**，**不要** [无效做法] **，应** [有效做法]；禁止绝对化表述。            |


**Review 模型职责**：在**规则已判定可写**之后，仅生成/润色 `自然语言经验` 与锚点文案；若与客观 `transition` 矛盾则丢弃本条。

---

## 2. 写入时机与门槛

**挂接位置（逻辑）**：第 `k` 轮评测结束、已持久化本轮结果与修复上下文之后；**开始第 `k+1` 轮生成之前**。此时同时具备第 `k-1` 轮与第 `k` 轮的错误与代码，便于对比（`k = 1` 无上一轮，一般不写跨轮记忆，见下）。

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

- 首轮：查询可依赖 `op`、`category`、`tool_mode` 等；无上一轮错误文本时偏「类级」经验。
- 次轮及以后：查询以**当前将注入的修复摘要**（错误锚点、阶段、`correctness_info` 片段）为主，提高相关性。

**方式**：从全局 `.jsonl` 中选取 **Top-n（建议 3～5）** 条，过滤 `tool_mode` / `eval_mode` 等与当前 run 不一致的条目；以固定区块拼入生成侧上下文（与现有修复提示并列），每条展示宜短：层级、转移一行、条件式经验、关键锚点。

**降权或不注入**：`tier = B` 且与当前错误锚点重叠过低；或 schema 版本过旧。

---

## 4. 工程注意（简要）

- **并行多算子**：全局追加需**并发安全**（文件锁、或每进程写临时队列再合并）。
- **记忆库位置**：与单次 `run_dir` 解耦的仓库级或用户级路径，便于跨实验复用；条目中带配置标签以便过滤。
- **演进**：字段带 `schema_version`；后续可将「检索 LLM」替换为 embedding + 向量库，不改变 Tier 与否定门槛定义。

---

## 6. 实现落地（与 Claude manifest 思路对齐）

**代码位置**：`generator/repair_memory/`（`schema`、`tier_gate`、`failure_stage`、`anchors`、`inbox`、`merge`、`manifest`、`inject`、`select`、`review_llm`、`pipeline`）；多轮挂接在 `generator/scripts/run_agent_multi_rounds.py` 与 `generator/scripts/run_agent_cuda_agent_multi_rounds.py`；Agent 状态字段 `retrieved_repair_memories` / `eval_mode` 在 `generator/agent/agent_state.py`、`agent_runner.py`，注入在 `nodes/choose_tool.py` 与 `nodes/answer.py`。

**存储布局**（默认根目录 `artifacts/repair_memory/`，可用环境变量覆盖）：

- `canonical/repair_memories.jsonl`：合并后的真源（每行一条完整 JSON）。
- `inbox/<run_slug>/mem_<uuid>.jsonl`：子进程**仅**写入**单行**的独立文件（避免多进程争用同一文件）。
- `inbox/<run_slug>/merged/`：已成功合并入 canonical 的收件副本（便于审计）。

**并行写入策略**：子进程不直接 append canonical；写入 inbox 后调用 `merge_run_inbox`（`fcntl` 独占锁写 canonical，再移动收件文件）。整批实验结束时主进程再执行一次 merge，收敛遗留文件。`run_slug` 由 `run_dir` 相对仓库根路径规范化得到，使同一 output run 共用同一 inbox 桶。

**检索**：从 canonical 尾部窗口生成 **manifest 短行**（`id`、op、category、`tool_mode`、`tier`、锚点、摘要），选条 LLM 返回 `memory_ids`，再按 id 从尾部窗口取全文片段注入（窗口外旧 id 可能暂不可见，可调 `max_records`）。

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
| 写入  | 仅修复型；Tier A = 编译或数值客观变好；Tier B = 阶段推进 + 锚点变 + 代码变；硬规则守门 + Review 只写叙述。 |
| 利用  | 每轮生成前检索 n 条注入；按配置与锚点过滤；控制长度。                                           |


*文档版本：与「第一阶段：修复型 + Tier A/B」讨论一致，实现细节以代码落地为准。*