// ❌ 错误
LocalTensor<T> x = allocator.Alloc<T, 64>();
DataCopy(x, xGm, count);
Cast<half, int8_t>(xHalf, x, ...);  // ⛔️ 数据可能未就绪

// ✅ 正确：EnQue/DeQue
void CopyIn() {
    LocalTensor<T> x = inQueue.AllocTensor<T>();
    DataCopy(x, xGm, count);
    inQueue.EnQue(x);
}
void Compute() {
    LocalTensor<T> x = inQueue.DeQue<T>();  // 等待数据就绪
    Cast<half, int8_t>(xHalf, x, ...);
    inQueue.FreeTensor(x);
}

// ⚠️ 调试验证：临时加 PipeBarrier
DataCopy(x, xGm, count);
PipeBarrier<PIPE_ALL>();  // 若结果正确则确认是同步问题
