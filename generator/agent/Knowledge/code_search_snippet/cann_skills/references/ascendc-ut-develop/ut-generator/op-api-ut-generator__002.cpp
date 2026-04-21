// 基本构造
auto tensor = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

// 链式调用
auto tensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND)
              .ValueRange(-2.0, 2.0)
              .Precision(0.0001, 0.0001);

// 非连续内存
auto tensor = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
