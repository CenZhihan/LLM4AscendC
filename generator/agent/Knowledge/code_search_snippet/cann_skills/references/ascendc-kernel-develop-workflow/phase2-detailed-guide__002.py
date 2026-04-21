branches = ["ar_fullload", "ar_colsplit", "ara_fullload", "ara_rowsplit"]
completed = []
max_iterations = 3
operator_name = "softmax0309"

for branch in branches:
    # ===== 步骤 1: 调度 通用 Agent A 实现代码 =====
    impl_result = task(
        description=f"实现 {branch}",
        prompt=f"""
你是一位 Ascend C 算子开发专家。请实现 {branch} 分支。

**⚠️ 重要**：
- **只负责实现代码，不负责代码审查**
- 代码审查将由主 Agent 调度独立的 通用 Agent B 完成

**任务**（包含步骤 6 审视代码）：
1. 步骤 0: 读取环境报告
2. 步骤 1: 读取设计文档
3. 步骤 2: 查阅参考文档
4. 步骤 3-4: 创建配置文件
5. 步骤 5: 实现代码
6. 步骤 6: 审视代码（自审查，参考 code-review-checklist.md）⚠️ **强制执行**
7. 步骤 7: 编译验证
8. 步骤 8: 测试验证

**返回内容（⚠️ 必须包含以下所有字段，否则验收直接失败）**：

0. **审视代码结果** ⛔️ **必须有，缺少此项将导致验收失败**
