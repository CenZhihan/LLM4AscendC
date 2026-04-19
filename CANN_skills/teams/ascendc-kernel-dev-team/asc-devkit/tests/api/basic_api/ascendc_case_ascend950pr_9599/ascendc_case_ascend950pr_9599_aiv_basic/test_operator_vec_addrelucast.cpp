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
void VecAddReluCast(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm, __gm__ uint8_t* __restrict__ src1Gm,
    __gm__ int32_t dataSize, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<SrcT> src0;
    GlobalTensor<SrcT> src1;
    GlobalTensor<DstT> dst;
    src0.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(src0Gm), dataSize);
    src1.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(src1Gm), dataSize);
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(dstGm), dataSize);

    LocalTensor<SrcT> src0Ub;
    LocalTensor<SrcT> src1Ub;
    TBuffAddr tbuf0;
    tbuf0.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    src0Ub.SetAddr(tbuf0);
    src0Ub.InitBuffer(0, dataSize);
    TBuffAddr tbuf1;
    tbuf1.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    src1Ub.SetAddr(tbuf1);
    src1Ub.InitBuffer(dataSize * sizeof(SrcT), dataSize);

    LocalTensor<DstT> dstUb;
    TBuffAddr tbuf2;
    tbuf2.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    dstUb.SetAddr(tbuf2);
    dstUb.InitBuffer(dataSize * sizeof(SrcT) * 2, dataSize);

    DataCopy(src0Ub, src0, dataSize);
    DataCopy(src1Ub, src1, dataSize);
    DataCopy(dstUb, dst, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    uint32_t repeatSize = 256 / max(sizeof(SrcT), sizeof(DstT));
    uint8_t repeatTimes = dataSize / repeatSize;
    BinaryRepeatParams repeatParams(1, 1, 1, sizeof(DstT) * repeatSize / 32, sizeof(SrcT) * repeatSize / 32,  sizeof(SrcT) * repeatSize / 32);
    if (testMode == TestMode::LEVEL0_NORM_MODE || testMode == TestMode::LEVEL0_FMIX_NORM_MODE) {
        uint64_t mask = repeatSize - 2;
        AddReluCast(dstUb, src0Ub, src1Ub, mask, repeatTimes, repeatParams);
    } else if (testMode == TestMode::LEVEL0_BIT_MODE || testMode == TestMode::LEVEL0_FMIX_BIT_MODE) {
        uint64_t mask[2] = {0};
        if (repeatSize == 128) {
            mask[0] = 0xfffffffffffffff5;
            mask[1] = 0xfffffffffffffff5;
        } else if (repeatSize == 64) {
            mask[0] = 0xfffffffffffffff5;
        }
        AddReluCast(dstUb, src0Ub, src1Ub, mask, repeatTimes, repeatParams);
    } else if (testMode == TestMode::LEVEL2_COUNTER_MODE || testMode == TestMode::LEVEL2_FMIX_COUNTER_MODE) {
        uint64_t calCount = dataSize - 1;
        AddReluCast(dstUb, src0Ub, src1Ub, calCount);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(dst, dstUb, dataSize);
    pipe_barrier(PIPE_ALL);
}

template <typename SrcT, typename DstT>
void VecAddReluCastTensorTrait(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm, __gm__ uint8_t* __restrict__ src1Gm,
    __gm__ int32_t dataSize, TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<TensorTrait<SrcT>> src0;
    GlobalTensor<TensorTrait<SrcT>> src1;
    GlobalTensor<TensorTrait<DstT>> dst;
    src0.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(src0Gm), dataSize);
    src1.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(src1Gm), dataSize);
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(dstGm), dataSize);

    LocalTensor<TensorTrait<SrcT>> src0Ub;
    LocalTensor<TensorTrait<SrcT>> src1Ub;
    TBuffAddr tbuf0;
    tbuf0.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    src0Ub.SetAddr(tbuf0);
    src0Ub.InitBuffer(0, dataSize);
    TBuffAddr tbuf1;
    tbuf1.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    src1Ub.SetAddr(tbuf1);
    src1Ub.InitBuffer(dataSize * sizeof(SrcT), dataSize);

    LocalTensor<TensorTrait<DstT>> dstUb;
    TBuffAddr tbuf2;
    tbuf2.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    dstUb.SetAddr(tbuf2);
    dstUb.InitBuffer(dataSize * sizeof(SrcT) * 2, dataSize);

    DataCopy(src0Ub, src0, dataSize);
    DataCopy(src1Ub, src1, dataSize);
    DataCopy(dstUb, dst, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    uint32_t repeatSize = 256 / max(sizeof(SrcT), sizeof(DstT));
    uint8_t repeatTimes = dataSize / repeatSize;
    BinaryRepeatParams repeatParams(1, 1, 1, sizeof(DstT) * repeatSize / 32, sizeof(SrcT) * repeatSize / 32,  sizeof(SrcT) * repeatSize / 32);
    if (testMode == TestMode::LEVEL0_NORM_MODE || testMode == TestMode::LEVEL0_FMIX_NORM_MODE) {
        uint64_t mask = repeatSize - 2;
        AddReluCast(dstUb, src0Ub, src1Ub, mask, repeatTimes, repeatParams);
    } else if (testMode == TestMode::LEVEL0_BIT_MODE || testMode == TestMode::LEVEL0_FMIX_BIT_MODE) {
        uint64_t mask[2] = {0};
        if (repeatSize == 128) {
            mask[0] = 0xfffffffffffffff5;
            mask[1] = 0xfffffffffffffff5;
        } else if (repeatSize == 64) {
            mask[0] = 0xfffffffffffffff5;
        }
        AddReluCast(dstUb, src0Ub, src1Ub, mask, repeatTimes, repeatParams);
    } else if (testMode == TestMode::LEVEL2_COUNTER_MODE || testMode == TestMode::LEVEL2_FMIX_COUNTER_MODE) {
        uint64_t calCount = dataSize - 1;
        AddReluCast(dstUb, src0Ub, src1Ub, calCount);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(dst, dstUb, dataSize);
    pipe_barrier(PIPE_ALL);
}

struct AddReluCastTestParams {
    int32_t dataSize;
    uint32_t srcDatabitSize;
    uint32_t dstDatabitSize;
    TestMode testMode;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, int32_t, TestMode);
};

class AddReluCastSimpleTestsuite : public testing::Test, public testing::WithParamInterface<AddReluCastTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(AddReluCastSimpleTestCase, AddReluCastSimpleTestsuite,
    ::testing::Values(
        AddReluCastTestParams { 256, 4, 2, TestMode::LEVEL0_NORM_MODE, VecAddReluCast<float, half> },
        AddReluCastTestParams { 128, 2, 1, TestMode::LEVEL0_NORM_MODE, VecAddReluCast<half, int8_t> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL0_NORM_MODE, VecAddReluCast<int16_t, int8_t> },

        AddReluCastTestParams { 256, 4, 2, TestMode::LEVEL0_BIT_MODE, VecAddReluCast<float, half> },
        AddReluCastTestParams { 128, 2, 1, TestMode::LEVEL0_BIT_MODE, VecAddReluCast<half, int8_t> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL0_BIT_MODE, VecAddReluCast<int16_t, int8_t> },

        AddReluCastTestParams { 256, 4, 2, TestMode::LEVEL0_FMIX_NORM_MODE, VecAddReluCast<float, half> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL0_FMIX_NORM_MODE, VecAddReluCast<half, int8_t> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL0_FMIX_NORM_MODE, VecAddReluCast<int16_t, int8_t> },

        AddReluCastTestParams { 256, 4, 2, TestMode::LEVEL0_FMIX_BIT_MODE, VecAddReluCast<float, half> },
        AddReluCastTestParams { 128, 2, 1, TestMode::LEVEL0_FMIX_BIT_MODE, VecAddReluCast<half, int8_t> },
        AddReluCastTestParams { 128, 2, 1, TestMode::LEVEL0_FMIX_BIT_MODE, VecAddReluCast<int16_t, int8_t> },

        AddReluCastTestParams { 256, 4, 2, TestMode::LEVEL2_COUNTER_MODE, VecAddReluCast<float, half> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL2_COUNTER_MODE, VecAddReluCast<half, int8_t> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL2_COUNTER_MODE, VecAddReluCast<int16_t, int8_t> },

        AddReluCastTestParams { 256, 4, 2, TestMode::LEVEL2_FMIX_COUNTER_MODE, VecAddReluCast<float, half> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL2_FMIX_COUNTER_MODE, VecAddReluCast<half, int8_t> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL2_FMIX_COUNTER_MODE, VecAddReluCast<int16_t, int8_t> },

        //===--------------------[ TensorTrait Case ]---------------------===//
        AddReluCastTestParams { 256, 4, 2, TestMode::LEVEL0_NORM_MODE, VecAddReluCastTensorTrait<float, half> },
        AddReluCastTestParams { 128, 2, 1, TestMode::LEVEL0_NORM_MODE, VecAddReluCastTensorTrait<half, int8_t> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL0_NORM_MODE, VecAddReluCastTensorTrait<int16_t, int8_t> },

        AddReluCastTestParams { 256, 4, 2, TestMode::LEVEL0_BIT_MODE, VecAddReluCastTensorTrait<float, half> },
        AddReluCastTestParams { 128, 2, 1, TestMode::LEVEL0_BIT_MODE, VecAddReluCastTensorTrait<half, int8_t> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL0_BIT_MODE, VecAddReluCastTensorTrait<int16_t, int8_t> },

        AddReluCastTestParams { 256, 4, 2, TestMode::LEVEL0_FMIX_NORM_MODE, VecAddReluCastTensorTrait<float, half> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL0_FMIX_NORM_MODE, VecAddReluCastTensorTrait<half, int8_t> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL0_FMIX_NORM_MODE, VecAddReluCastTensorTrait<int16_t, int8_t> },

        AddReluCastTestParams { 256, 4, 2, TestMode::LEVEL0_FMIX_BIT_MODE, VecAddReluCastTensorTrait<float, half> },
        AddReluCastTestParams { 128, 2, 1, TestMode::LEVEL0_FMIX_BIT_MODE, VecAddReluCastTensorTrait<half, int8_t> },
        AddReluCastTestParams { 128, 2, 1, TestMode::LEVEL0_FMIX_BIT_MODE, VecAddReluCastTensorTrait<int16_t, int8_t> },

        AddReluCastTestParams { 256, 4, 2, TestMode::LEVEL2_COUNTER_MODE, VecAddReluCastTensorTrait<float, half> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL2_COUNTER_MODE, VecAddReluCastTensorTrait<half, int8_t> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL2_COUNTER_MODE, VecAddReluCastTensorTrait<int16_t, int8_t> },

        AddReluCastTestParams { 256, 4, 2, TestMode::LEVEL2_FMIX_COUNTER_MODE, VecAddReluCastTensorTrait<float, half> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL2_FMIX_COUNTER_MODE, VecAddReluCastTensorTrait<half, int8_t> },
        AddReluCastTestParams { 256, 2, 1, TestMode::LEVEL2_FMIX_COUNTER_MODE, VecAddReluCastTensorTrait<int16_t, int8_t> }
    ));

TEST_P(AddReluCastSimpleTestsuite, AddReluCastSimpleTestCase)
{
    TPipe tpipe;
    auto param = GetParam();
    uint8_t src0Gm[param.dataSize * param.srcDatabitSize];
    uint8_t src1Gm[param.dataSize * param.srcDatabitSize];
    uint8_t dstGm[param.dataSize * param.dstDatabitSize];

    param.cal_func(dstGm, src0Gm, src1Gm, param.dataSize, param.testMode);
}