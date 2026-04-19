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

using namespace std;
using namespace AscendC;

enum TestMode {
    LEVEL2_MODE,
    LEVEL0_BIT_MODE,
    LEVEL0_COUNT_MODE,
};

template <typename T1, typename T2>
void VecCast(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm,
    __gm__ int32_t dataSize, RoundMode roundMode, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<T1> inputGlobal;
    GlobalTensor<T2> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T1*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T2*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T1));
    LocalTensor<T1> inputLocal = tbuf.Get<T1>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T2));
    LocalTensor<T2> outputLocal = tbuf1.Get<T2>();

    for (int32_t i = 0; i < dataSize; ++i) {
        inputLocal.SetValue(i, 1);
    }

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if (testMode == TestMode::LEVEL0_BIT_MODE) {
        uint32_t repeatSize = 256 / max(sizeof(T1), sizeof(T2));
        uint8_t repeatTimes = dataSize / repeatSize;
        uint64_t mask[2] = {0};
        if (repeatSize == 128) {
            mask[0] = 0xffffffffffffffff;
            mask[1] = 0xffffffffffffffff;
        } else if (repeatSize == 64) {
            mask[0] = 0xffffffffffffffff;
        } else {
            mask[0] = 0xffffffff;
        }
        UnaryRepeatParams repeatParams(1, 1, sizeof(T2) * repeatSize / 32, sizeof(T1) * repeatSize / 32);
        Cast(outputLocal, inputLocal, roundMode, mask, repeatTimes, repeatParams);
    } else if (testMode == TestMode::LEVEL0_COUNT_MODE) {
        uint32_t repeatSize = 256 / max(sizeof(T1), sizeof(T2));
        uint8_t repeatTimes = dataSize / repeatSize;
        uint64_t mask = repeatSize;
        UnaryRepeatParams repeatParams(1, 1, sizeof(T2) * repeatSize / 32, sizeof(T1) * repeatSize / 32);
        Cast(outputLocal, inputLocal, roundMode, mask, repeatTimes, repeatParams);
    } else {
        Cast(outputLocal, inputLocal, roundMode, dataSize);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

struct CastTestParams {
    int32_t dataSize;
    int32_t srcDatabitSize;
    int32_t dstDatabitSize;
    RoundMode roundMode;
    TestMode testMode;
    void (*cal_func)(uint8_t*, uint8_t*, int32_t, RoundMode, TestMode);
};

class CastSimpleTestsuite : public testing::Test, public testing::WithParamInterface<CastTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(CastSimpleTestCase, CastSimpleTestsuite,
    ::testing::Values(
        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<uint8_t, half> },
        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int8_t, half> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<half, float> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<half, int32_t> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<bfloat16_t, float> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<bfloat16_t, int32_t> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int16_t, float> },
        CastTestParams { 256, 4, 8, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<float, int64_t> },
        CastTestParams { 256, 4, 8, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int32_t, int64_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<half, uint8_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<half, int8_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<float, half> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<float, bfloat16_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<float, int16_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<int32_t, half> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int32_t, int16_t> },
        CastTestParams { 256, 8, 4, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<int64_t, float> },
        CastTestParams { 256, 8, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int64_t, int32_t> },
        CastTestParams { 256, 2, 2, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<half, int16_t> },
        CastTestParams { 256, 2, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int16_t, half> },
        CastTestParams { 256, 4, 4, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<float, float> },
        CastTestParams { 256, 4, 4, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<float, int32_t> },
        CastTestParams { 256, 4, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int32_t, float> },
        CastTestParams { 256, 1, 2, RoundMode::CAST_RINT, TestMode::LEVEL0_BIT_MODE, VecCast<uint8_t, half> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_FLOOR, TestMode::LEVEL0_BIT_MODE, VecCast<half, float> },
        CastTestParams { 256, 4, 8, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<int32_t, int64_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_ROUND, TestMode::LEVEL0_BIT_MODE, VecCast<half, uint8_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_TRUNC, TestMode::LEVEL0_BIT_MODE, VecCast<float, half> },
        CastTestParams { 256, 8, 4, RoundMode::CAST_ODD, TestMode::LEVEL0_BIT_MODE, VecCast<int64_t, int32_t> },

        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<uint8_t, half> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<half, float> },
        CastTestParams { 256, 4, 8, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<int32_t, int64_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<half, uint8_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<float, half> },
        CastTestParams { 256, 8, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<int64_t, int32_t> },
        CastTestParams { 256, 1, 2, RoundMode::CAST_RINT, TestMode::LEVEL0_COUNT_MODE, VecCast<uint8_t, half> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_FLOOR, TestMode::LEVEL0_COUNT_MODE, VecCast<half, float> },
        CastTestParams { 256, 4, 8, RoundMode::CAST_CEIL, TestMode::LEVEL0_COUNT_MODE, VecCast<int32_t, int64_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_ROUND, TestMode::LEVEL0_COUNT_MODE, VecCast<half, uint8_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_TRUNC, TestMode::LEVEL0_COUNT_MODE, VecCast<float, half> },
        CastTestParams { 256, 8, 4, RoundMode::CAST_ODD, TestMode::LEVEL0_COUNT_MODE, VecCast<int64_t, int32_t> },

        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<uint8_t, half> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<half, float> },
        CastTestParams { 256, 4, 8, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<int32_t, int64_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<half, uint8_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<float, half> },
        CastTestParams { 256, 8, 4, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<int64_t, int32_t> },
        CastTestParams { 256, 1, 2, RoundMode::CAST_RINT, TestMode::LEVEL2_MODE, VecCast<uint8_t, half> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_FLOOR, TestMode::LEVEL2_MODE, VecCast<half, float> },
        CastTestParams { 256, 4, 8, RoundMode::CAST_CEIL, TestMode::LEVEL2_MODE, VecCast<int32_t, int64_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_ROUND, TestMode::LEVEL2_MODE, VecCast<half, uint8_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_TRUNC, TestMode::LEVEL2_MODE, VecCast<float, half> },
        CastTestParams { 256, 8, 4, RoundMode::CAST_ODD, TestMode::LEVEL2_MODE, VecCast<int64_t, int32_t> }));

TEST_P(CastSimpleTestsuite, CastSimpleTestCase)
{
    TPipe tpipe;
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.srcDatabitSize];
    uint8_t dstGm[param.dataSize * param.dstDatabitSize];

    param.cal_func(dstGm, srcGm, param.dataSize, param.roundMode, param.testMode);
    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}