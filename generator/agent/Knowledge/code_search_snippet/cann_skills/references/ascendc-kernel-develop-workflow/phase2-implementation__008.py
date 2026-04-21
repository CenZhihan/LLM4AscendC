# ✅ 正确：通用 Agent A 审视代码 + 通用 Agent B 复跑验收 + 新 通用 Agent 修复

# 步骤 1: 通用 Agent A 完成所有步骤（0-8）
impl_result = task(
    description="实现代码",
    prompt="实现步骤 0-8（实现+审视代码+编译+测试）→ 返回完整报告"
)

# 步骤 2: 通用 Agent B 验收（审查+复跑测试）
review_result = task(
    description="代码验收",
    prompt="审查代码质量 + 复跑测试用例验证结果 → 返回验收报告"
)

# 步骤 3: 根据评分决定后续
if review_result['总分'] >= 8.5 and review_result['测试通过']:
    # 验收通过
    pass
else:
    # 验收失败：启动【新 通用 Agent】修复（不恢复旧 通用 Agent）
    impl_result = task(
        description="修复代码",
        prompt=f"""
你是一位 Ascend C 算子开发专家。请根据验收反馈修复代码。

**⚠️ 重要**：
- 先阅读设计文档：ops/{operator_name}/docs/design.md
- 针对具体问题进行修复，不要盲目修改

**验收评分**：{review_result['总分']}/10

**必须修改的问题**：
{review_result.get('问题列表', '无')}

**任务**：
1. 步骤 0: 读取环境报告
2. 步骤 1: 读取设计文档
3. 针对问题修改代码
4. 步骤 7: 编译验证
5. 步骤 8: 测试验证

**返回内容**：
1. 修改说明
2. 编译结果
3. 测试结果
"""
    )
