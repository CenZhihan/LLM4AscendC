// 测试用例命名规范：Test{OperatorName}{Scenario}
TEST_F({OperatorName}Test, TestScenario_{description}) {
    // 1. 准备输入参数
    int param1 = value1;
    int param2 = value2;

    // 2. 创建输入tensor
    auto x1 = CreateTensor({M, K}, dtype1);
    auto x2 = CreateTensor({K, N}, dtype2);

    // 3. 设置特殊参数（如有）
    // quantScale = xxx;

    // 4. 执行算子
    auto output = ExecuteOperator(x1, x2, param1, param2);

    // 5. 验证结果
    EXPECT_EQ(output.shape(), expected_shape);
    // 或 EXPECT_NEAR(...) 用于浮点数比较
}
