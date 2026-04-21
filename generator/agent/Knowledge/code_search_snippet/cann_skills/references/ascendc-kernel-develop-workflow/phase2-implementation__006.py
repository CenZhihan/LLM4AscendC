# ❌ 错误：通用 Agent A 跳过步骤 6 审视代码
# 问题：代码质量问题未提前发现，编译失败率高

impl_result = task(
    description="实现代码",
    prompt="实现步骤 0-5 → 直接编译测试 → 返回"
)
# 缺少步骤6审视代码
