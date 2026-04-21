task(
    description = "执行 softmax0312 算子设计",
    prompt = """
你是 Ascend C 算子设计专家。请为 {operator_name} 执行以下步骤：

**详细步骤指导**：见 [phase1-design-subagent.md](phase1-design-subagent.md)

## 步骤概览
1. 需求检查（算子名称、数学公式、输入输出规格、精度要求）
2. 查询 NPU 架构（调用 /ascendc-npu-arch）
3. 获取设计指导（调用 /ascendc-tiling-design）
4. 确认 API 用法（调用 /ascendc-api-best-practices）
5. API 可行性验证（查阅官方文档、填写映射表）
6. 分支规划（判断简单/复杂算子、规划分支场景）
7. 输出设计文档（输出到 {operator_name}/docs/design.md）

输出要求：完整的设计文档，包含每个分支的核心伪代码（2.4节）
""",
    subagent_type = "general"
)
