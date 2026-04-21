// 打印 tensor 内容
AscendC::LocalTensor<T> xLocal = inQueue.DeQue<T>();
DumpTensor(xLocal, 0, 128);  // 打印前128个元素
