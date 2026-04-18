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
    LEVEL0_NORM_MODE,
    LEVEL0_BIT_MODE,
    LEVEL0_FMIX_NORM_MODE,
    LEVEL0_FMIX_BIT_MODE,
    LEVEL2_COUNTER_MODE,
    LEVEL2_FMIX_COUNTER_MODE,
};

template <typename SRC_T, typename DST_T>
void VecTernary(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm,
    __gm__ int32_t dataSize, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<SRC_T> src;
    GlobalTensor<DST_T> dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ SRC_T*>(srcGm), dataSize);
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ DST_T*>(dstGm), dataSize);

    LocalTensor<SRC_T> src_ub;
    TBuffAddr tbuf0;
    tbuf0.logicPos = (uint8_t)TPosition::VECCALC;
    src_ub.SetAddr(tbuf0);
    src_ub.InitBuffer(0, dataSize);

    LocalTensor<DST_T> dst_ub;
    TBuffAddr tbuf1;
    tbuf1.logicPos = (uint8_t)TPosition::VECCALC;
    dst_ub.SetAddr(tbuf1);
    dst_ub.InitBuffer(dataSize * sizeof(DST_T) * 2, dataSize);

    for (int32_t i = 0; i < dataSize; ++i) {
        src_ub.SetValue(i, 1);
    }
    for (int32_t i = 0; i < dataSize; ++i) {
        dst_ub.SetValue(i, 2);
    }

    DataCopy(src_ub, src, dataSize);
    DataCopy(dst_ub, dst, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    SRC_T scalar = 3;
    if (testMode == TestMode::LEVEL0_NORM_MODE || testMode == TestMode::LEVEL0_FMIX_NORM_MODE) {
        uint32_t repeatSize = 256 / max(sizeof(SRC_T), sizeof(DST_T));
        uint8_t repeatTimes = dataSize / repeatSize;
        uint64_t mask = repeatSize - 2;
        UnaryRepeatParams repeatParams(1, 1, sizeof(DST_T) * repeatSize / 32, sizeof(SRC_T) * repeatSize / 32);
        Axpy(dst_ub, src_ub, scalar, mask, repeatTimes, repeatParams);
    } else if (testMode == TestMode::LEVEL0_BIT_MODE || testMode == TestMode::LEVEL0_FMIX_BIT_MODE) {
        uint32_t repeatSize = 256 / max(sizeof(SRC_T), sizeof(DST_T));
        uint8_t repeatTimes = dataSize / repeatSize;
        uint64_t mask[2] = {0};
        if (repeatSize == 128) {
            mask[0] = 0xfffffffffffffff5;
            mask[1] = 0xfffffffffffffff5;
        } else if (repeatSize == 64) {
            mask[0] = 0xfffffffffffffff5;
        }
        UnaryRepeatParams repeatParams(1, 1, sizeof(DST_T) * repeatSize / 32, sizeof(SRC_T) * repeatSize / 32);
        Axpy(dst_ub, src_ub, scalar, mask, repeatTimes, repeatParams);
    } else if (testMode == TestMode::LEVEL2_COUNTER_MODE || testMode == TestMode::LEVEL2_FMIX_COUNTER_MODE) {
        uint64_t calCount = dataSize - 1;
        Axpy(dst_ub, src_ub, scalar, calCount);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(dst, dst_ub, dataSize);
    pipe_barrier(PIPE_ALL);
}

struct TernaryTestParams {
    int32_t dataSize;
    uint32_t srcDatabitSize;
    uint32_t dstDatabitSize;
    TestMode testMode;
    void (*cal_func)(uint8_t*, uint8_t*, int32_t, TestMode);
};

class TernarySimpleTestsuite : public testing::Test, public testing::WithParamInterface<TernaryTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TernarySimpleTestCase, TernarySimpleTestsuite,
    ::testing::Values(
        TernaryTestParams { 256, 2, 2, TestMode::LEVEL0_NORM_MODE, VecTernary<half, half> },
        TernaryTestParams { 128, 2, 2, TestMode::LEVEL0_NORM_MODE, VecTernary<half, half> },
        TernaryTestParams { 256, 4, 4, TestMode::LEVEL0_NORM_MODE, VecTernary<float, float> },
        TernaryTestParams { 128, 4, 4, TestMode::LEVEL0_NORM_MODE, VecTernary<float, float> },

        TernaryTestParams { 256, 2, 2, TestMode::LEVEL0_BIT_MODE, VecTernary<half, half> },
        TernaryTestParams { 256, 4, 4, TestMode::LEVEL0_BIT_MODE, VecTernary<float, float> },
        TernaryTestParams { 128, 2, 2, TestMode::LEVEL0_BIT_MODE, VecTernary<half, half> },
        TernaryTestParams { 128, 4, 4, TestMode::LEVEL0_BIT_MODE, VecTernary<float, float> },

        TernaryTestParams { 256, 2, 4, TestMode::LEVEL0_FMIX_NORM_MODE, VecTernary<half, float> },
        TernaryTestParams { 256, 2, 4, TestMode::LEVEL0_FMIX_NORM_MODE, VecTernary<half, float> },

        TernaryTestParams { 256, 2, 4, TestMode::LEVEL0_FMIX_BIT_MODE, VecTernary<half, float> },
        TernaryTestParams { 128, 2, 4, TestMode::LEVEL0_FMIX_BIT_MODE, VecTernary<half, float> },

        TernaryTestParams { 256, 2, 2, TestMode::LEVEL2_COUNTER_MODE, VecTernary<half, half> },
        TernaryTestParams { 256, 4, 4, TestMode::LEVEL2_COUNTER_MODE, VecTernary<float, float> },

        TernaryTestParams { 256, 2, 4, TestMode::LEVEL2_FMIX_COUNTER_MODE, VecTernary<half, float> },
        TernaryTestParams { 256, 2, 4, TestMode::LEVEL2_FMIX_COUNTER_MODE, VecTernary<half, float> }
    ));

TEST_P(TernarySimpleTestsuite, TernarySimpleTestCase)
{
    TPipe tpipe;
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.srcDatabitSize];
    uint8_t dstGm[param.dataSize * param.dstDatabitSize];

    param.cal_func(dstGm, srcGm, param.dataSize, param.testMode);
    // for (int32_t i = 0; i < param.dataSize; i++) {
    //     EXPECT_EQ(dstGm[i], 0x00);
    // }
}