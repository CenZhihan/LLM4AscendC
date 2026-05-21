## **自动记忆与索引概述**

Claude Code 的“记忆”是一套基于本地文件的永久存储机制：agent运行时，可以将关于用户、反馈或项目的关键信息保存到指定目录，这些信息在后续会话中会加载并影响行为。核心由以下部分组成：

- **自动记忆目录**：每个项目都有一个独立的内存目录，默认位于 `~/.claude/projects/<项目>/memory/`。其中包含多个主题文件和一个索引文件 `MEMORY.md`。
- **记忆类型**：源码中限定了四种记忆类型：`user`（用户信息）、`feedback`（对 AI 行为的指导）、`project`（不易从代码推断的项目上下文）和 `reference`（外部资源或文档），通过常量 `MEMORY_TYPES` 定义 。
- **索引文件限制**：为了防止上下文爆炸，索引 `MEMORY.md` 被限制为 **200 行**并且总字节数不超过 25 KB，这些限制由常量 `MAX_ENTRYPOINT_LINES` 与 `MAX_ENTRYPOINT_BYTES` 定义 。超出限制时会截断旧条目，并在结尾插入警告 。
- **自动化维护**：系统会在空闲时运行整理任务，将每日日志提炼为主题文件，删除重复或过时信息，并生成简洁的索引。未公开的 `KAIROS` 守护进程和 Auto Dream 模块协助完成这些任务。

这一机制旨在提供跨会话的长期记忆，避免重复学习，并减少上下文窗口浪费。
**
### **内存目录和索引文件管理（****`memdir.ts`**)


源码 `memdir.ts` 包含多项与记忆目录管理相关的函数：

- **入口文件常量**：`ENTRYPOINT_NAME` 定义索引文件名为 `MEMORY.md`，`MAX_ENTRYPOINT_LINES` 与 `MAX_ENTRYPOINT_BYTES` 限定索引长度 。
- **截断索引**：函数 `truncateEntrypointContent()` 在读取索引时检查行数和字节数，若超出限制则裁剪内容并附加警告，提醒用户压缩条目长度 。
- **目录创建**：`ensureMemoryDirExists()` 在构建提示时确保内存目录存在；如果权限错误则记录日志但不阻塞主流程 。
- **构建提示文本**：`buildMemoryLines()` 生成说明如何保存记忆的提示文字，包括：每条记忆应写入独立文件、使用 YAML frontmatter (`title/description/type`)、按语义组织并避免重复；然后在索引文件中插入指向该文件的一行 。函数还提醒索引加载进上下文时会截断行数 。`buildMemoryPrompt()` 在此基础上加载现有索引内容（若存在）并插入到提示中 。
- **每日日志模式**：对于长会话或助手模式，`buildAssistantDailyLogPrompt()` 指示助理将新的记忆附加到按日期命名的日志文件 (`logs/YYYY/MM/YYYY‑MM‑DD.md`)，而非直接写入 `MEMORY.md`，然后由后台进程在夜间将日志汇总到索引 。
- **搜索历史内容**：`buildSearchingPastContextSection()` 给出在需要回忆历史时如何使用 `grep` 或嵌入式搜索工具检索记忆目录及会话日志的指导 。

这些函数显示，自动记忆通过文本约定与索引限制维护结构化的长久记录，并在上下文中合理呈现。


### **记忆路径与环境控制（****`paths.ts`****）**

文件 `paths.ts` 定义了如何确定自动记忆存储的位置，并提供安全检查：

- **是否启用记忆**：`isAutoMemoryEnabled()` 根据环境变量与设置文件判断是否启用自动记忆，例如 `CLAUDE_CODE_DISABLE_AUTO_MEMORY=true` 会完全关闭该功能 。
- **计算存储目录**：`getMemoryBaseDir()` 返回默认配置目录（通常为 `~/.claude`），而 `getAutoMemPath()` 根据项目根目录生成路径 `<memoryBase>/projects/<sanitized‑git‑root>/memory/` 。此函数还允许通过 `CLAUDE_COWORK_MEMORY_PATH_OVERRIDE` 或用户设置覆盖默认路径  。
- **路径验证**：内部函数 `validateMemoryPath()` 检查候选路径是否为绝对路径、长度足够、不包含网络路径和空字节，防止路径遍历或不安全的根目录 。
- **每天日志路径和索引**：`getAutoMemDailyLogPath()` 构建每日日志文件的路径 `logs/YYYY/MM/YYYY‑MM‑DD.md` ，`getAutoMemEntrypoint()` 返回索引 `MEMORY.md` 的绝对路径 。
- **目录内检查**：`isAutoMemPath()` 判断一个绝对路径是否位于自动记忆目录下，以避免越界写入 。

通过这些函数，自动记忆在不同环境下都能可靠地定位目录，同时减少路径注入风险。

### **扫描与筛选记忆（****`memoryScan.ts`和`findRelevantMemories.ts`****）**

保存的记忆会被扫描并在处理用户请求时有选择地加载：

- **扫描目录**：`scanMemoryFiles()` 遍历内存目录中所有 `.md` 文件（排除索引 `MEMORY.md`），读取每个文件前 30 行的 frontmatter，并返回文件名、绝对路径、修改时间以及描述与类型 。为了效率，最多返回最近修改的 200 个文件 。
- **格式化清单**：`formatMemoryManifest()` 将这些记忆头格式化为“一行一条”的清单，用于作为输入给选择模型 。
- **选择相关记忆**：`findRelevantMemories()` 获取扫描结果后调用 `selectRelevantMemories()`，并通过预定义的系统提示 `SELECT_MEMORIES_SYSTEM_PROMPT` 让 Sonnet 模型选择与用户查询最相关的至多 5 个记忆文件  。规则要求若不确定某条记忆是否有用，则宁愿不选 。

这说明 Claude Code 在记忆检索阶段并未使用向量搜索，而是通过简单的元数据列表和外部大模型的判断来决定加载哪些记忆。


### **记忆类型与不保存内容（****`memoryTypes.ts`****）**

为了规范记忆的结构，`memoryTypes.ts` 定义了四种记忆类型并提供解析函数。源码中还包含用于生成提示的文本常量，如 `TYPES_SECTION_INDIVIDUAL`、`WHAT_NOT_TO_SAVE_SECTION` 等，提醒助理：

- **不要保存可从代码中推断的内容**，例如代码模式、架构结构、Git 历史或调试解决方案 。
- **记忆应附带说明**：何时保存、如何使用，以及不应存入负面评判或偏见。

这些约束帮助模型在编写记忆时遵循统一格式，避免噪声和重复。



### **记忆陈旧度处理（****`memoryAge.ts`****）**

陈旧记忆容易导致模型引用过时信息，源码提供了一些帮助函数：

- `memoryAgeDays()` 计算记忆修改时间距离现在的天数 。
- `memoryAge()` 将天数转换成 “today / yesterday / N days ago” 的人类友好描述 。
- `memoryFreshnessText()` 根据天龄返回警告文本，提醒这条记忆可能已过时，需要核对当前代码或信息 。

这些函数可以在呈现记忆时添加系统提示，让模型在引用旧记忆时更加谨慎。

