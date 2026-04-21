// TensorDesc 构造
auto tensor = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

// OP_API_UT 宏
auto ut = OP_API_UT(aclnnAbs, INPUT(self), OUTPUT(out));
