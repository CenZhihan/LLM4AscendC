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

template <typename SrcT, typename DstT>
void VecTernary(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm,
    __gm__ int32_t dataSize, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<SrcT> src;
    GlobalTensor<DstT> dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(srcGm), dataSize);
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(SrcT));
    LocalTensor<SrcT> srcUb = tbuf.Get<SrcT>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(DstT));
    LocalTensor<DstT> dstUb = tbuf1.Get<DstT>();

    for (int32_t i = 0; i < dataSize; ++i) {
        srcUb.SetValue(i, 1);
    }
    for (int32_t i = 0; i < dataSize; ++i) {
        dstUb.SetValue(i, 2);
    }

    DataCopy(srcUb, src, dataSize);
    DataCopy(dstUb, dst, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    SrcT scalar = 3;
    if (testMode == TestMode::LEVEL0_NORM_MODE || testMode == TestMode::LEVEL0_FMIX_NORM_MODE) {
        uint32_t repeatSize = 256 / max(sizeof(SrcT), sizeof(DstT));
        uint8_t repeatTimes = dataSize / repeatSize;
        uint64_t mask = repeatSize - 2;
        UnaryRepeatParams repeatParams(1, 1, sizeof(DstT) * repeatSize / 32, sizeof(SrcT) * repeatSize / 32);
        Axpy(dstUb, srcUb, scalar, mask, repeatTimes, repeatParams);
    } else if (testMode == TestMode::LEVEL0_BIT_MODE || testMode == TestMode::LEVEL0_FMIX_BIT_MODE) {
        uint32_t repeatSize = 256 / max(sizeof(SrcT), sizeof(DstT));
        uint8_t repeatTimes = dataSize / repeatSize;
        uint64_t mask[2] = {0};
        if (repeatSize == 128) {
            mask[0] = 0xfffffffffffffff5;
            mask[1] = 0xfffffffffffffff5;
        } else if (repeatSize == 64) {
            mask[0] = 0xfffffffffffffff5;
        }
        UnaryRepeatParams repeatParams(1, 1, sizeof(DstT) * repeatSize / 32, sizeof(SrcT) * repeatSize / 32);
        Axpy(dstUb, srcUb, scalar, mask, repeatTimes, repeatParams);
    } else if (testMode == TestMode::LEVEL2_COUNTER_MODE || testMode == TestMode::LEVEL2_FMIX_COUNTER_MODE) {
        uint64_t calCount = dataSize - 1;
        Axpy(dstUb, srcUb, scalar, calCount);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(dst, dstUb, dataSize);
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
    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}