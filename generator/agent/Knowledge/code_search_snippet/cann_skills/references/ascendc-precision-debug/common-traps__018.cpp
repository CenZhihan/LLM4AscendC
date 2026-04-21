// ❌ 避免
outGm.SetValue(0, 10);

// ✅ 推荐
LocalTensor<T> tmp = buf.Get<T>();
tmp.SetValue(0, value);
DataCopyPad(dstGm, tmp, {1, sizeof(T), 0, 0});
