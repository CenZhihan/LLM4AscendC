// ✅ 正确：统一使用 DataCopyPad
AscendC::DataCopyPadParams padParams;
AscendC::DataCopyPad(xLocal, xGm, 
    {1, static_cast<uint16_t>(dataBytes), 0, 0}, padParams);
AscendC::DataCopyPad(yGm, yLocal, 
    {1, static_cast<uint16_t>(dataBytes), 0, 0});
