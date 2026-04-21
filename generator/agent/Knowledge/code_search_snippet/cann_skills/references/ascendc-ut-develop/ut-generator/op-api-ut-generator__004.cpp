// 单输入单输出
auto ut = OP_API_UT(aclnnAbs, INPUT(self), OUTPUT(out));

// 多输入带Scalar
auto ut = OP_API_UT(aclnnAdd, INPUT(self, other, alpha), OUTPUT(out));

// nullptr测试
auto ut = OP_API_UT(aclnnAbs, INPUT((aclTensor*)nullptr), OUTPUT(out));
