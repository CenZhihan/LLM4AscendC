// 异常用例（最先编写）
TEST_F(l2_abs_test, case_anullptr_input) {
    auto out = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAbs, INPUT((aclTensor*)nullptr), OUTPUT(out));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

// 正常用例
TEST_F(l2_abs_test, case_abs_for_float_type) {
    auto self = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2.0, 2.0);
    auto out = TensorDesc(self).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnAbs, INPUT(self), OUTPUT(out));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}
