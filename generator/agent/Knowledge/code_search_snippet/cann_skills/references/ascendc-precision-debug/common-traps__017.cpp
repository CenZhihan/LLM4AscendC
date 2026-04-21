// ❌ 错误：8 字节非对齐
DataCopy(indicesGm, indicesLocal, 2);  // 2 * 4 = 8B

// ✅ 正确：使用 DataCopyPad
DataCopyExtParams p{1, rowsThisCore * sizeof(int32_t), 0, 0};
DataCopyPad(indicesGm, indicesLocal, p);
