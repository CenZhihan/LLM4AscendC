/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include "kernel_operator.h"
// #include "api_check/kernel_cpu_check.h"

using namespace std;
using namespace AscendC;

struct TestSpmBufferParams {
    uint32_t dataSize;
    uint32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, int32_t);
};

class TestSpmBufferSuite : public testing::Test, public testing::WithParamInterface<TestSpmBufferParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

template<typename T>
void MainNormalSpmBuffer(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;

    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);
    tpipe.InitSpmBuffer(dataSize);
    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();
    tpipe.WriteSpmBuffer(inputLocal, dataSize);
    tpipe.ReadSpmBuffer(inputLocal, dataSize);
    tpipe.WriteSpmBuffer(inputLocal, dataSize, 0);
    tpipe.ReadSpmBuffer(inputLocal, dataSize, 0);
    DataCopy(dstGlobal, inputLocal, dataSize);
    DataCopy(srcGlobal, inputLocal, dataSize);
}

INSTANTIATE_TEST_CASE_P(TEST_SPM_BUFFER, TestSpmBufferSuite,
    ::testing::Values(TestSpmBufferParams { 64, 4, MainNormalSpmBuffer<int32_t> },
    TestSpmBufferParams { 64, 2, MainNormalSpmBuffer<int16_t> },
    TestSpmBufferParams { 64, 4, MainNormalSpmBuffer<float> },
    TestSpmBufferParams { 64, 2, MainNormalSpmBuffer<half> },
    TestSpmBufferParams { 64, 1, MainNormalSpmBuffer<int8_t> },
    TestSpmBufferParams { 64, 1, MainNormalSpmBuffer<uint8_t> }));

TEST_P(TestSpmBufferSuite, TestSpmBufferCases)
{
    AscendC::SetGCoreType(2);
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.typeSize];
    uint8_t dstGm[param.dataSize * param.typeSize];
    param.cal_func(srcGm, dstGm, param.dataSize);
    for (uint32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(srcGm[i], 0x00);
        EXPECT_EQ(dstGm[i], 0x00);
    }
    AscendC::SetGCoreType(0);
}