### API Tool 改造执行计划

#### 目标

提升 `api_lookup` 和 `api_constraint` 的可用性，减少误导性结果，让工具输出能直接服务代码生成和二轮修复。

#### 问题归纳

1. `api_lookup` 目前会把注释、公式或普通命中行误当成签名，导致返回内容不可直接用于生成。
2. `api_constraint` 在关键信息缺失时仍可能给出“符合约束”的假阳性，误导后续生成。
3. 当前约束检查以通用规则为主，缺少 `ReduceSum`、`DataCopy`、`Duplicate`、`Cast` 这类高频 API 的专用校验。
4. 计划必须分阶段推进，先修可靠性和输出语义，再扩展场景化能力。

#### Phase 1：可靠性修复

##### api_lookup

1. 签名提取改为“声明优先”。
2. 显式跳过注释、公式、宏说明、markdown 表格等非声明行。
3. 支持从头文件中拼接多行函数声明，而不是只取命中行。
4. 如果只有文档命中或注释命中，返回空签名，并显式附带低置信度标记。
5. 为结果增加结构化元数据：
	* `match_kind`: `builtin_knowledge` / `header_decl` / `doc_excerpt` / `not_found`
	* `confidence`: `high` / `medium` / `low`
	* `is_actionable`: 是否足够支撑代码生成

##### api_constraint

1. 在兼容现有 `is_compliant` 布尔语义的前提下，新增三态字段 `compliance_status`：
	* `pass`
	* `fail`
	* `insufficient_context`
2. 当缺少关键参数时，不再输出“符合约束”，而是输出 `insufficient_context`。
3. 补充高价值专用检查：
	* `ReduceSum`: `dst`/`workspace` 是否别名，workspace 信息是否充足
	* `DataCopy` / `DataCopyPad`: 搬运方向、`count * sizeof(dtype)` 对齐、pad 参数完整性
	* `Duplicate`: `count`、`repeat_times`、`mask` 信息是否充足
	* `Cast`: 源/目标 dtype 是否明确且合法

#### Phase 2：生成友好的 API 契约

1. 为 `api_lookup` 增加更适合生成消费的结构化字段：
	* 支持 dtype
	* 临时 buffer / workspace 需求
	* 是否要求独立 workspace
	* 常见调用风格：`queue + DataCopy + Compute` 或 `repeatTime + mask`
	* 常见 companion API
2. 这部分只在 Phase 1 稳定后推进，不与可靠性修复绑在同一轮实现中。

#### Phase 3：按场景推荐 API

1. 单独提供“按场景推荐 API”的能力，不直接塞进当前 `api_lookup`。
2. 优先方案：新增场景化工具或在 code search 层提供 pattern 检索，再由 agent 组合使用。
3. 不在本轮和 `api_lookup` / `api_constraint` 强耦合实现，避免职责混乱。

#### 输入契约

`api_constraint` 不继续依赖“从 query 猜参数”作为主路径。主路径改为结构化字段输入，query 解析只作为兜底。

第一阶段至少支持以下字段：

* `api_name`
* `count`
* `dtype`
* `repeat_times`
* `ub_usage_bytes`
* `ub_capacity_bytes`
* `is_gm_to_ub`
* `workspace_alias`
* `workspace_size_bytes`
* `mask`
* `src_dtype`
* `dst_dtype`
* `pad_size`

#### 验收标准

1. `api_lookup` 不再把注释或公式误报为签名。
2. `api_constraint` 在关键字段缺失时返回 `insufficient_context`，不再假阳性通过。
3. 新增单元测试覆盖上述行为。
4. 通过针对性回归验证：至少对 `gelu`、`layer_norm` 的相关 API 查询行为进行回归检查。

#### 本轮执行范围

本轮只执行 Phase 1：

1. 重构 `api_lookup` 的签名提取与结果元数据。
2. 为 `api_constraint` 增加 `compliance_status` 和高价值专用检查。
3. 补充并运行单元测试。