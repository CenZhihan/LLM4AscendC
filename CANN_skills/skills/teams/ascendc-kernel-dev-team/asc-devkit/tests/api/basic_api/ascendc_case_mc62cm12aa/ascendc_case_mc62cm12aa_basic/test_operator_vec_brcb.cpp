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
#define private public
#define protect public
#include "kernel_operator.h"
// #include "model/model_factory.h"
#include "common.h"
using namespace std;
using namespace AscendC;

struct BrcbTestParams {
    uint32_t srcLen;
    uint32_t dataBitSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t);
};

class BrcbTestSuite : public testing::Test, public testing::WithParamInterface<BrcbTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

template <typename T>
void test_vbrcb_func(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm, __gm__ uint32_t length)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    uint32_t srcLen = length;
    uint32_t dstLen = srcLen * (32 / sizeof(T));
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), srcLen);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dstLen);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcLen * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dstLen * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    for (int32_t i = 0; i < srcLen; ++i) {
        inputLocal.SetValue(i, i);
    }

    AscendCUtils::SetMask<uint8_t>(256);
    DataCopy(inputLocal, inputGlobal, srcLen);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    SetMaskCount();
    BrcbRepeatParams params { 1, 8 };
    uint8_t repeat = srcLen / 8;
    Brcb(outputLocal, inputLocal, repeat, params);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(outputGlobal, outputLocal, dstLen);
    pipe_barrier(PIPE_ALL);
}

INSTANTIATE_TEST_CASE_P(TEST_BRCB, BrcbTestSuite,
    ::testing::Values(
        BrcbTestParams { 32, 1, test_vbrcb_func<uint8_t> },
        BrcbTestParams { 32, 1, test_vbrcb_func<int8_t> },
        BrcbTestParams { 16, 2, test_vbrcb_func<uint16_t> },
        BrcbTestParams { 16, 2, test_vbrcb_func<half> },
        BrcbTestParams { 16, 2, test_vbrcb_func<bfloat16_t> },
        BrcbTestParams { 32, 2, test_vbrcb_func<int16_t> }, 
        BrcbTestParams { 128, 2, test_vbrcb_func<uint16_t> },
        BrcbTestParams { 16, 4, test_vbrcb_func<int32_t> },
        BrcbTestParams { 16, 4, test_vbrcb_func<float> }, 
        BrcbTestParams { 32, 4, test_vbrcb_func<uint32_t> },
        BrcbTestParams { 64, 4, test_vbrcb_func<uint32_t> },
        BrcbTestParams { 64, 8, test_vbrcb_func<uint64_t> },
        BrcbTestParams { 64, 8, test_vbrcb_func<int64_t> }
    )
);

TEST_P(BrcbTestSuite, BrcbTestCase)
{
    auto param = GetParam();
    uint8_t src_gm[param.srcLen * param.dataBitSize] = {0};
    const uint32_t dstLen = param.srcLen * (32 / param.dataBitSize);
    uint8_t dstGm[dstLen * param.dataBitSize] = {0};
    param.cal_func(dstGm, src_gm, param.srcLen);
    for (size_t i = 0; i < dstLen; ++i) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
    return;
}