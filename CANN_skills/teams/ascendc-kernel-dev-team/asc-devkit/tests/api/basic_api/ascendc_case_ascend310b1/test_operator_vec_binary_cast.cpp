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
#include "mockcpp/mockcpp.hpp"

using namespace std;
using namespace AscendC;

enum CastFunc {
    ADDRELUCAST,
    SUBRELUCAST,
    MULCAST,
};

template <typename T1, typename T2>
void VecFusedCast(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm1, __gm__ uint8_t* __restrict__ srcGm2, 
    __gm__ uint32_t dataSize, CastFunc castFunc)
{
    TPipe tpipe;
    GlobalTensor<T1> inputGlobal1;
    GlobalTensor<T1> inputGlobal2;
    GlobalTensor<T2> outputGlobal;
    inputGlobal1.SetGlobalBuffer(reinterpret_cast<__gm__ T1*>(srcGm1), dataSize);
    inputGlobal2.SetGlobalBuffer(reinterpret_cast<__gm__ T1*>(srcGm2), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T2*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T1));
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T1));
    LocalTensor<T1> inputlocal1 = tbuf1.Get<T1>();
    LocalTensor<T1> inputlocal2 = tbuf2.Get<T1>();

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T2));
    LocalTensor<T2> outputLocal = tbuf.Get<T2>();

    for (int32_t i = 0; i < dataSize; ++i) {
        inputlocal1.SetValue(i, 1);
        inputlocal2.SetValue(i, 1);
    }

    DataCopy(inputlocal1, inputGlobal1, dataSize);
    DataCopy(inputlocal2, inputGlobal2, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    uint32_t repeatSize = 256 / max(sizeof(T1), sizeof(T2));
    uint8_t repeatTimes = dataSize / repeatSize;
    uint64_t mask[2] = {0};
    if (repeatSize == 128) {
        mask[0] = 0xffffffffffffffff;
        mask[1] = 0xffffffffffffffff;
    } else if(repeatSize == 64) {
        mask[0] = 0xffffffffffffffff;
    } else {
        mask[0] = 0xffffffff;
    }
    uint64_t mask1 = repeatSize;
    BinaryRepeatParams repeatParams(1, 1, 1, sizeof(T2) * repeatSize / 32, sizeof(T1) * repeatSize / 32, sizeof(T1) * repeatSize / 32);

    if (castFunc == CastFunc::ADDRELUCAST) {
        AddReluCast(outputLocal, inputlocal1, inputlocal2, mask, repeatTimes, repeatParams);
        AddReluCast(outputLocal, inputlocal1, inputlocal2, mask1, repeatTimes, repeatParams);
        AddReluCast(outputLocal, inputlocal1, inputlocal2, dataSize);
    } else if (castFunc == CastFunc::SUBRELUCAST) {
        SubReluCast(outputLocal, inputlocal1, inputlocal2, mask, repeatTimes, repeatParams);
        SubReluCast(outputLocal, inputlocal1, inputlocal2, mask1, repeatTimes, repeatParams);
        SubReluCast(outputLocal, inputlocal1, inputlocal2, dataSize);
    } else {
        MulCast(outputLocal, inputlocal1, inputlocal2, mask, repeatTimes, repeatParams);
        MulCast(outputLocal, inputlocal1, inputlocal2, mask1, repeatTimes, repeatParams);
        MulCast(outputLocal, inputlocal1, inputlocal2, dataSize);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

struct BinaryCastTestParams {
    uint32_t dataSize;
    int32_t srcDatabitSize;
    int32_t dstDatabitSize;
    CastFunc castFunc;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint32_t, CastFunc);
};

class BinaryCastSimpleTestsuite : public testing::Test, public testing::WithParamInterface<BinaryCastTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(BinaryCastSimpleTestCase, BinaryCastSimpleTestsuite,
    ::testing::Values(
        BinaryCastTestParams { 256, 4, 2, CastFunc::ADDRELUCAST, VecFusedCast<float, half> },
        BinaryCastTestParams { 256, 4, 2, CastFunc::ADDRELUCAST, VecFusedCast<half, int8_t> },
        BinaryCastTestParams { 256, 4, 2, CastFunc::SUBRELUCAST, VecFusedCast<float, half> },
        BinaryCastTestParams { 256, 4, 2, CastFunc::SUBRELUCAST, VecFusedCast<half, int8_t> },
        BinaryCastTestParams { 256, 4, 2, CastFunc::MULCAST, VecFusedCast<half, uint8_t> },
        BinaryCastTestParams { 256, 4, 2, CastFunc::MULCAST, VecFusedCast<half, int8_t> },
        BinaryCastTestParams { 256, 4, 2, CastFunc::ADDRELUCAST, VecFusedCast<double, half> },
        BinaryCastTestParams { 256, 4, 2, CastFunc::SUBRELUCAST, VecFusedCast<double, half> },
        BinaryCastTestParams { 256, 4, 2, CastFunc::MULCAST, VecFusedCast<double, int8_t> }
    ));

TEST_P(BinaryCastSimpleTestsuite, BinaryCastSimpleTestCase)
{
    TPipe tpipe;
    auto param = GetParam();
    uint8_t srcGm1[param.dataSize * param.srcDatabitSize];
    uint8_t srcGm2[param.dataSize * param.srcDatabitSize];
    uint8_t dstGm[param.dataSize * param.dstDatabitSize];
    MOCKER(raise).stubs().will(returnValue(static_cast<int>(0)));
    param.cal_func(dstGm, srcGm1, srcGm2, param.dataSize, param.castFunc);
    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}