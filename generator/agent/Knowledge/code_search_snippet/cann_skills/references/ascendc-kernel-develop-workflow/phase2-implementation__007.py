# ❌ 错误：通用 Agent B 只看报告不复跑测试
# 问题：无法验证测试结果真实性

review_result = task(
    description="代码审查",
    prompt="审查代码质量 → 给出评分"
)
# 没有复跑测试用例验证
