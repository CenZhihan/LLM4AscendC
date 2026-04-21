task(
    description = "评估 softmax0312 设计文档",
    prompt = """
你是 Ascend C 算子设计文档评审专家。请评估 {operator_name} 的设计文档。

**设计文档**：`ops/{operator_name}/docs/design.md`

**详细评估标准**：见 [phase1-evaluation-subagent.md](phase1-evaluation-subagent.md)

**准出条件**：总分 >= 8.5
""",
    subagent_type = "general"
)
