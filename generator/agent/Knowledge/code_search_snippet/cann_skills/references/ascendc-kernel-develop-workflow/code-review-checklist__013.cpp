// ✅ 正确：EnQue/DeQue 同步
void CopyIn() {
    LocalTensor<T> x = inQueue.AllocTensor<T>();
    DataCopy(x, xGm, count);
    inQueue.EnQue(x);
}
void Compute() {
    LocalTensor<T> x = inQueue.DeQue<T>();
    // ... 计算 ...
    inQueue.FreeTensor(x);
}

// ❌ 错误：缺少同步
LocalTensor<T> x = allocator.Alloc<T, 64>();
DataCopy(x, xGm, count);
Compute(x);  // ⛔️ 数据可能未就绪
