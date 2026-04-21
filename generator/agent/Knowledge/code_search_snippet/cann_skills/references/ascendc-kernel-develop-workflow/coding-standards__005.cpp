// ❌ 错误：当数据长度不是 32 字节的倍数时会出错
AscendC::DataCopy(xLocal, xGm, dataLength);  // 危险！
AscendC::DataCopy(yGm, yLocal, dataLength);   // 危险！
