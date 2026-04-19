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
// #include "model/model_factory.h"
#include "common.h"
using namespace std;
using namespace AscendC;

struct BrcbTestParams {
    uint32_t srcLen;
    uint32_t dataBitSize;
    uint32_t stride;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t, uint32_t);
};

class BrcbTestSuite : public testing::Test, public testing::WithParamInterface<BrcbTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

template <typename T>
void TestVbrcbFunc(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm, __gm__ uint32_t length, __gm__ uint32_t stride)
{
    TPipe tpipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<T> outputGlobal;
    uint32_t srcLen = length;
    uint32_t dstLen = srcLen * (32 / sizeof(T)) * 2;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), srcLen);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dstLen);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcLen * sizeof(T));
    LocalTensor<T> input0Local = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dstLen * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    for (int32_t i = 0; i < srcLen; ++i) {
        input0Local.SetValue(i, i);
    }

    AscendCUtils::SetMask<uint8_t>(256);
    DataCopy(input0Local, input0Global, srcLen);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    BrcbRepeatParams params { stride, 8 };
    uint8_t repeat = srcLen / 8;
    Brcb(outputLocal, input0Local, repeat, params);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(outputGlobal, outputLocal, dstLen);
    pipe_barrier(PIPE_ALL);
}

INSTANTIATE_TEST_CASE_P(TEST_BRCB, BrcbTestSuite,
    ::testing::Values(BrcbTestParams { 16, 2, 1, TestVbrcbFunc<uint16_t> },
    BrcbTestParams { 32, 2, 1, TestVbrcbFunc<uint16_t> }, BrcbTestParams { 128, 2, 2, TestVbrcbFunc<uint16_t> },
    BrcbTestParams { 16, 4, 2, TestVbrcbFunc<uint32_t> }, BrcbTestParams { 32, 4, 1, TestVbrcbFunc<uint32_t> },
    BrcbTestParams { 64, 4, 1, TestVbrcbFunc<uint32_t> }));

TEST_P(BrcbTestSuite, BrcbTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.srcLen * param.dataBitSize] = {0};
    const uint32_t dstLen = param.srcLen * (32 / param.dataBitSize) * 2;
    uint8_t dstGm[dstLen * param.dataBitSize] = {0};
    param.cal_func(dstGm, srcGm, param.srcLen, param.stride);
    for (size_t i = 0; i < dstLen; ++i) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
    return;
}