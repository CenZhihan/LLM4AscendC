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
void VecMulAddDst(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm, __gm__ uint8_t* __restrict__ src1Gm,
    __gm__ int32_t dataSize, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<SRC_T> src0;
    GlobalTensor<SRC_T> src1;
    GlobalTensor<DST_T> dst;
    src0.SetGlobalBuffer(reinterpret_cast<__gm__ SRC_T*>(src0Gm), dataSize);
    src1.SetGlobalBuffer(reinterpret_cast<__gm__ SRC_T*>(src1Gm), dataSize);
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ DST_T*>(dstGm), dataSize);

    LocalTensor<SRC_T> src0_ub;
    LocalTensor<SRC_T> src1_ub;
    TBuffAddr tbuf0;
    tbuf0.logicPos = (uint8_t)TPosition::VECCALC;
    src0_ub.SetAddr(tbuf0);
    src0_ub.InitBuffer(0, dataSize);
    TBuffAddr tbuf1;
    tbuf1.logicPos = (uint8_t)TPosition::VECCALC;
    src1_ub.SetAddr(tbuf1);
    src1_ub.InitBuffer(dataSize * sizeof(SRC_T), dataSize);

    LocalTensor<DST_T> dst_ub;
    TBuffAddr tbuf2;
    tbuf2.logicPos = (uint8_t)TPosition::VECCALC;
    dst_ub.SetAddr(tbuf2);
    dst_ub.InitBuffer(dataSize * sizeof(SRC_T) * 2, dataSize);

    DataCopy(src0_ub, src0, dataSize);
    DataCopy(src1_ub, src1, dataSize);
    DataCopy(dst_ub, dst, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    uint32_t repeatSize = 256 / max(sizeof(SRC_T), sizeof(DST_T));
    uint8_t repeatTimes = dataSize / repeatSize;
    BinaryRepeatParams repeatParams(1, 1, 1, sizeof(DST_T) * repeatSize / 32, sizeof(SRC_T) * repeatSize / 32,  sizeof(SRC_T) * repeatSize / 32);
    if (testMode == TestMode::LEVEL0_NORM_MODE || testMode == TestMode::LEVEL0_FMIX_NORM_MODE) {
        uint64_t mask = repeatSize - 2;
        MulAddDst(dst_ub, src0_ub, src1_ub, mask, repeatTimes, repeatParams);
    } else if (testMode == TestMode::LEVEL0_BIT_MODE || testMode == TestMode::LEVEL0_FMIX_BIT_MODE) {
        uint64_t mask[2] = {0};
        if (repeatSize == 128) {
            mask[0] = 0xfffffffffffffff5;
            mask[1] = 0xfffffffffffffff5;
        } else if (repeatSize == 64) {
            mask[0] = 0xfffffffffffffff5;
        }
        MulAddDst(dst_ub, src0_ub, src1_ub, mask, repeatTimes, repeatParams);
    } else if (testMode == TestMode::LEVEL2_COUNTER_MODE || testMode == TestMode::LEVEL2_FMIX_COUNTER_MODE) {
        uint64_t calCount = dataSize - 1;
        MulAddDst(dst_ub, src0_ub, src1_ub, calCount);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(dst, dst_ub, dataSize);
    pipe_barrier(PIPE_ALL);
}

struct MulAddDstTestParams {
    int32_t dataSize;
    uint32_t srcDatabitSize;
    uint32_t dstDatabitSize;
    TestMode testMode;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, int32_t, TestMode);
};

class MulAddDstSimpleTestsuite : public testing::Test, public testing::WithParamInterface<MulAddDstTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(MulAddDstSimpleTestCase, MulAddDstSimpleTestsuite,
    ::testing::Values(
        MulAddDstTestParams { 256, 2, 2, TestMode::LEVEL0_NORM_MODE, VecMulAddDst<half, half> },
        MulAddDstTestParams { 128, 2, 2, TestMode::LEVEL0_NORM_MODE, VecMulAddDst<half, half> },
        MulAddDstTestParams { 256, 4, 4, TestMode::LEVEL0_NORM_MODE, VecMulAddDst<float, float> },
        MulAddDstTestParams { 128, 4, 4, TestMode::LEVEL0_NORM_MODE, VecMulAddDst<float, float> },

        MulAddDstTestParams { 256, 2, 2, TestMode::LEVEL0_BIT_MODE, VecMulAddDst<half, half> },
        MulAddDstTestParams { 128, 2, 2, TestMode::LEVEL0_BIT_MODE, VecMulAddDst<half, half> },
        MulAddDstTestParams { 256, 4, 4, TestMode::LEVEL0_BIT_MODE, VecMulAddDst<float, float> },
        MulAddDstTestParams { 128, 4, 4, TestMode::LEVEL0_BIT_MODE, VecMulAddDst<float, float> },

        MulAddDstTestParams { 256, 2, 4, TestMode::LEVEL0_FMIX_NORM_MODE, VecMulAddDst<half, float> },
        MulAddDstTestParams { 256, 2, 4, TestMode::LEVEL0_FMIX_NORM_MODE, VecMulAddDst<half, float> },

        MulAddDstTestParams { 256, 2, 4, TestMode::LEVEL0_FMIX_BIT_MODE, VecMulAddDst<half, float> },
        MulAddDstTestParams { 128, 2, 4, TestMode::LEVEL0_FMIX_BIT_MODE, VecMulAddDst<half, float> },

        MulAddDstTestParams { 256, 2, 2, TestMode::LEVEL2_COUNTER_MODE, VecMulAddDst<half, half> },
        MulAddDstTestParams { 256, 4, 4, TestMode::LEVEL2_COUNTER_MODE, VecMulAddDst<float, float> },

        MulAddDstTestParams { 256, 2, 4, TestMode::LEVEL2_FMIX_COUNTER_MODE, VecMulAddDst<half, float> },
        MulAddDstTestParams { 256, 2, 4, TestMode::LEVEL2_FMIX_COUNTER_MODE, VecMulAddDst<half, float> }
    ));

TEST_P(MulAddDstSimpleTestsuite, MulAddDstSimpleTestCase)
{
    TPipe tpipe;
    auto param = GetParam();
    uint8_t src0Gm[param.dataSize * param.srcDatabitSize];
    uint8_t src1Gm[param.dataSize * param.srcDatabitSize];
    uint8_t dstGm[param.dataSize * param.dstDatabitSize];

    param.cal_func(dstGm, src0Gm, src1Gm, param.dataSize, param.testMode);
}