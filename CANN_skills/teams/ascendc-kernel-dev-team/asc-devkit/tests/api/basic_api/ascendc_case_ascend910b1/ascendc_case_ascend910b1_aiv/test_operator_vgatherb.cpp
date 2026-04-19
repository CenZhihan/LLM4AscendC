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

struct GatherbTestParams {
    uint32_t srcLen;
    uint32_t offsetLen;
    uint32_t dataBitSize;
    void (*CalFunc)(uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t);
};

class GatherbTestSuite : public testing::Test, public testing::WithParamInterface<GatherbTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

template <typename T>
void test_vgatherb_func(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm,
    __gm__ uint8_t* __restrict__ offsetGm, __gm__ uint32_t srcLen, __gm__ uint32_t offsetLen)
{
    TPipe tpipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<uint32_t> offsetGlobal;
    GlobalTensor<T> outputGlobal;
    uint32_t dstLen = srcLen;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), srcLen);
    offsetGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(offsetGm), offsetLen);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dstLen);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcLen * sizeof(T));
    LocalTensor<T> input0Local = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, offsetLen * sizeof(uint32_t));
    LocalTensor<uint32_t> offsetLocal = tbuf1.Get<uint32_t>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dstLen * sizeof(T));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    for (int32_t i = 0; i < srcLen; ++i) {
        input0Local.SetValue(i, 1);
    }
    for (int32_t i = 0; i < offsetLen; ++i) {
        offsetLocal.SetValue(i, 0);
    }

    AscendCUtils::SetMask<uint8_t>(256);
    DataCopy(input0Local, input0Global, srcLen);
    DataCopy(offsetLocal, offsetGlobal, offsetLen);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    GatherRepeatParams params { 1, 8 };
    uint8_t repeat = srcLen * sizeof(T) / 256;
    Gatherb(outputLocal, input0Local, offsetLocal, repeat, params);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(outputGlobal, outputLocal, dstLen);
    pipe_barrier(PIPE_ALL);
}

INSTANTIATE_TEST_CASE_P(TEST_VGATHERB, GatherbTestSuite,
    ::testing::Values(GatherbTestParams { 128, 16, 2, test_vgatherb_func<uint16_t> },
    GatherbTestParams { 256, 32, 4, test_vgatherb_func<uint32_t> },
    GatherbTestParams { 128, 16, 4, test_vgatherb_func<uint32_t> },
    GatherbTestParams { 512, 32, 2, test_vgatherb_func<uint16_t> }));

TEST_P(GatherbTestSuite, GatherbTestCase)
{
    auto param = GetParam();
    uint8_t dstGm[param.srcLen * param.dataBitSize] = {0};
    uint8_t srcGm[param.srcLen * param.dataBitSize] = {0};
    uint8_t offsetGm[param.offsetLen * 4] = {0};
    param.CalFunc(dstGm, srcGm, offsetGm, param.srcLen, param.offsetLen);
    for (size_t i = 0; i < param.srcLen; ++i) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
    return;
}