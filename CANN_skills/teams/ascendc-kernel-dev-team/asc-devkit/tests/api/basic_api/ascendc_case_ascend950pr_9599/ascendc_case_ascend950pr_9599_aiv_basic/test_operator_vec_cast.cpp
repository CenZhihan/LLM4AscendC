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
    using SrcT = typename Conditional<IsSameType<T1, hifloat8_t>::value||
        IsSameType<T1, fp8_e5m2_t>::value||IsSameType<T1, fp8_e4m3fn_t>::value, uint8_t, T1>::type;
    using DstT = typename Conditional<IsSameType<T2, hifloat8_t>::value||
        IsSameType<T2, fp8_e5m2_t>::value||IsSameType<T2, fp8_e4m3fn_t>::value, uint8_t, T2>::type;
    GlobalTensor<SrcT> inputGlobal;
    GlobalTensor<DstT> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<SrcT>*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<DstT>*>(dstGm), dataSize);
    LocalTensor<T1> inputLocal;
    TBuffAddr tbuf0;
    tbuf0.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    inputLocal.SetAddr(tbuf0);
    inputLocal.InitBuffer(0, dataSize);

    LocalTensor<T2> outputLocal;
    TBuffAddr tbuf1;
    tbuf1.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    outputLocal.SetAddr(tbuf1);
    outputLocal.InitBuffer(dataSize * sizeof(PrimT<T1>), dataSize);

    for (int32_t i = 0; i < dataSize; ++i) {
        inputLocal.SetValue(i, 1);
    }
    if constexpr (IsSameType<T1, hifloat8_t>::value||
        IsSameType<T1, fp8_e5m2_t>::value||IsSameType<T1, fp8_e4m3fn_t>::value) {
        LocalTensor<uint8_t> tmp = inputLocal.template ReinterpretCast<uint8_t>();
        DataCopy(tmp, inputGlobal, dataSize);
    } else {
        DataCopy(inputLocal, inputGlobal, dataSize);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if (testMode == TestMode::LEVEL0_BIT_MODE) {
        uint32_t repeatSize = 256 / max(sizeof(PrimT<T1>), sizeof(PrimT<T2>));
        uint8_t repeatTimes = dataSize / repeatSize;
        uint64_t mask[2] = {0};
        uint64_t counterMask[2] = {144, 0};
        if (repeatSize == 128) {
            mask[0] = 0xffffffffffffffff;
            mask[1] = 0xffffffffffffffff;
        } else if(repeatSize == 64) {
            mask[0] = 0xffffffffffffffff;
        } else {
            mask[0] = 0xffffffff;
        }
        UnaryRepeatParams repeatParams(1, 1, sizeof(PrimT<T2>) * repeatSize / 32, sizeof(PrimT<T1>) * repeatSize / 32);
        Cast(outputLocal, inputLocal, roundMode, mask, repeatTimes, repeatParams);

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<T1, MaskMode::COUNTER>(0, 144);
        Cast<T2, T1, false>(outputLocal, inputLocal, roundMode, AscendC::MASK_PLACEHOLDER_LIST, repeatTimes, repeatParams);
        AscendC::ResetMask();
        Cast(outputLocal, inputLocal, roundMode, counterMask, repeatTimes, repeatParams);
        AscendC::SetMaskNorm();
    } else if (testMode == TestMode::LEVEL0_COUNT_MODE) {
        uint32_t repeatSize = 256 / max(sizeof(PrimT<T1>), sizeof(PrimT<T2>));
        uint8_t repeatTimes = dataSize / repeatSize;
        uint64_t mask = repeatSize;
        UnaryRepeatParams repeatParams(1, 1, sizeof(PrimT<T2>) * repeatSize / 32, sizeof(PrimT<T1>) * repeatSize / 32);
        Cast(outputLocal, inputLocal, roundMode, mask, repeatTimes, repeatParams);
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<T1, MaskMode::COUNTER>(0, 144);
        Cast<T2, T1, false>(outputLocal, inputLocal, roundMode, AscendC::MASK_PLACEHOLDER, repeatTimes, repeatParams);
        AscendC::ResetMask();
        Cast(outputLocal, inputLocal, roundMode, mask, repeatTimes, repeatParams);
        AscendC::SetMaskNorm();
    } else {
        Cast(outputLocal, inputLocal, roundMode, dataSize);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    if constexpr (IsSameType<T2, hifloat8_t>::value||
        IsSameType<T2, fp8_e5m2_t>::value||IsSameType<T2, fp8_e4m3fn_t>::value) {
        LocalTensor<uint8_t> tmpOut = outputLocal.template ReinterpretCast<uint8_t>();
        DataCopy(outputGlobal, tmpOut, dataSize);
    } else {
        DataCopy(outputGlobal, outputLocal, dataSize);
    }

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
        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<uint8_t, uint16_t> },
        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int8_t, int16_t> },
        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int8_t, half> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<uint16_t, uint32_t> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int16_t, uint32_t> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int16_t, int32_t> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<half, float> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<bfloat16_t, float> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<bfloat16_t, int32_t> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int16_t, float> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<half, uint8_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<uint16_t, uint8_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int16_t, uint8_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<half, int8_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<uint32_t, uint16_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int32_t, uint16_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<uint32_t, int16_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<float, half> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<float, bfloat16_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<float, int16_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int32_t, int16_t> },
        CastTestParams { 256, 2, 2, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<half, int16_t> },
        CastTestParams { 256, 2, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int16_t, half> },
        CastTestParams { 256, 4, 4, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<float, float> },
        CastTestParams { 256, 4, 4, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<float, int32_t> },
        CastTestParams { 256, 4, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int32_t, float> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_FLOOR, TestMode::LEVEL0_BIT_MODE, VecCast<float, half> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<float, half> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_TRUNC, TestMode::LEVEL0_BIT_MODE, VecCast<float, half> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_HYBRID, TestMode::LEVEL0_BIT_MODE, VecCast<half, hifloat8_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_ROUND, TestMode::LEVEL0_BIT_MODE, VecCast<half, hifloat8_t> },
        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<hifloat8_t, half> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<uint32_t, uint8_t> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int32_t, uint8_t> },
        CastTestParams { 256, 1, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<uint8_t, uint32_t> },
        CastTestParams { 256, 1, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int8_t, int32_t> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_HYBRID, TestMode::LEVEL0_BIT_MODE, VecCast<float, hifloat8_t> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_ROUND, TestMode::LEVEL0_BIT_MODE, VecCast<float, hifloat8_t> },
        CastTestParams { 256, 1, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<hifloat8_t, float> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_RINT, TestMode::LEVEL0_BIT_MODE, VecCast<float, fp8_e5m2_t> },
        CastTestParams { 256, 1, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<fp8_e5m2_t, float> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_RINT, TestMode::LEVEL0_BIT_MODE, VecCast<float, fp8_e4m3fn_t> },
        CastTestParams { 256, 1, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<fp8_e4m3fn_t, float> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<half, int32_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<int32_t, half> },
        CastTestParams { 256, 4, 8, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<float, int64_t> },
        CastTestParams { 256, 4, 8, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int32_t, int64_t> },
        CastTestParams { 256, 8, 4, RoundMode::CAST_RINT, TestMode::LEVEL0_BIT_MODE, VecCast<int64_t, float> },
        CastTestParams { 256, 8, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<int64_t, int32_t> },

        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<uint8_t, half> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<half, float> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<half, uint8_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<float, half> },
        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<uint8_t, uint16_t> },
        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<int8_t, int16_t> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<uint16_t, uint32_t> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<int16_t, uint32_t> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<int16_t, int32_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<uint16_t, uint8_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<int16_t, uint8_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<uint32_t, uint16_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<int32_t, uint16_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<uint32_t, int16_t> },
        CastTestParams { 256, 2, 2, RoundMode::CAST_CEIL, TestMode::LEVEL0_COUNT_MODE, VecCast<half, int16_t> },
        CastTestParams { 256, 2, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<int16_t, half> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_FLOOR, TestMode::LEVEL0_COUNT_MODE, VecCast<float, half> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_CEIL, TestMode::LEVEL0_COUNT_MODE, VecCast<float, half> },
        CastTestParams { 256, 4, 4, RoundMode::CAST_TRUNC, TestMode::LEVEL0_COUNT_MODE, VecCast<float, float> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_TRUNC, TestMode::LEVEL0_COUNT_MODE, VecCast<float, half> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_HYBRID, TestMode::LEVEL0_COUNT_MODE, VecCast<half, hifloat8_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_ROUND, TestMode::LEVEL0_COUNT_MODE, VecCast<half, hifloat8_t> },
        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<hifloat8_t, half> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<uint32_t, uint8_t> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<int32_t, uint8_t> },
        CastTestParams { 256, 1, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<uint8_t, uint32_t> },
        CastTestParams { 256, 1, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<int8_t, int32_t> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_HYBRID, TestMode::LEVEL0_COUNT_MODE, VecCast<float, hifloat8_t> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_ROUND, TestMode::LEVEL0_COUNT_MODE, VecCast<float, hifloat8_t> },
        CastTestParams { 256, 1, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<hifloat8_t, float> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_RINT, TestMode::LEVEL0_COUNT_MODE, VecCast<float, fp8_e5m2_t> },
        CastTestParams { 256, 1, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<fp8_e5m2_t, float> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_RINT, TestMode::LEVEL0_COUNT_MODE, VecCast<float, fp8_e4m3fn_t> },
        CastTestParams { 256, 1, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<fp8_e4m3fn_t, float> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_CEIL, TestMode::LEVEL0_COUNT_MODE, VecCast<half, int32_t> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_CEIL, TestMode::LEVEL0_COUNT_MODE, VecCast<int32_t, half> },
        CastTestParams { 256, 4, 8, RoundMode::CAST_CEIL, TestMode::LEVEL0_COUNT_MODE, VecCast<float, int64_t> },
        CastTestParams { 256, 4, 8, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<int32_t, int64_t> },
        CastTestParams { 256, 8, 4, RoundMode::CAST_RINT, TestMode::LEVEL0_COUNT_MODE, VecCast<int64_t, float> },
        CastTestParams { 256, 8, 4, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<int64_t, int32_t> },

        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<uint8_t, half> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<half, float> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<half, uint8_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<float, half> },
        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<uint8_t, uint16_t> },
        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<int8_t, int16_t> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<uint16_t, uint32_t> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<int16_t, uint32_t> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<int16_t, int32_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<uint16_t, uint8_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<int16_t, uint8_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<uint32_t, uint16_t> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<int32_t, uint16_t> },
        CastTestParams { 256, 2, 2, RoundMode::CAST_CEIL, TestMode::LEVEL2_MODE, VecCast<half, int16_t> },
        CastTestParams { 256, 2, 2, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<int16_t, half> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_FLOOR, TestMode::LEVEL2_MODE, VecCast<float, half> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_CEIL, TestMode::LEVEL2_MODE, VecCast<float, half> },
        CastTestParams { 256, 4, 4, RoundMode::CAST_TRUNC, TestMode::LEVEL2_MODE, VecCast<float, float> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_TRUNC, TestMode::LEVEL2_MODE, VecCast<float, half> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<uint32_t, int16_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_HYBRID, TestMode::LEVEL2_MODE, VecCast<half, hifloat8_t> },
        CastTestParams { 256, 2, 1, RoundMode::CAST_ROUND, TestMode::LEVEL2_MODE, VecCast<half, hifloat8_t> },
        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<hifloat8_t, half> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_HYBRID, TestMode::LEVEL2_MODE, VecCast<float, hifloat8_t> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<uint32_t, uint8_t> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<int32_t, uint8_t> },
        CastTestParams { 256, 1, 4, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<uint8_t, uint32_t> },
        CastTestParams { 256, 1, 4, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<int8_t, int32_t> },
        CastTestParams { 256, 1, 4, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<hifloat8_t, float> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_ROUND, TestMode::LEVEL2_MODE, VecCast<float, hifloat8_t> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_RINT, TestMode::LEVEL2_MODE, VecCast<float, fp8_e5m2_t> },
        CastTestParams { 256, 1, 4, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<fp8_e5m2_t, float> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_RINT, TestMode::LEVEL2_MODE, VecCast<float, fp8_e4m3fn_t> },
        CastTestParams { 256, 1, 4, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<fp8_e4m3fn_t, float> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_CEIL, TestMode::LEVEL2_MODE, VecCast<half, int32_t> },
        CastTestParams { 256, 2, 4, RoundMode::CAST_CEIL, TestMode::LEVEL2_MODE, VecCast<int32_t, half> },
        CastTestParams { 256, 4, 8, RoundMode::CAST_CEIL, TestMode::LEVEL2_MODE, VecCast<float, int64_t> },
        CastTestParams { 256, 4, 8, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<int32_t, int64_t> },
        CastTestParams { 256, 8, 4, RoundMode::CAST_RINT, TestMode::LEVEL2_MODE, VecCast<int64_t, float> },
        CastTestParams { 256, 8, 4, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<int64_t, int32_t> },

        // TensorTrait
        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_BIT_MODE, VecCast<TensorTrait<uint8_t>, TensorTrait<half>> },
        CastTestParams { 256, 4, 4, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<TensorTrait<float>, TensorTrait<int32_t>> },
        CastTestParams { 256, 2, 2, RoundMode::CAST_CEIL, TestMode::LEVEL0_BIT_MODE, VecCast<TensorTrait<half>, TensorTrait<int16_t>> },

        CastTestParams { 256, 2, 2, RoundMode::CAST_CEIL, TestMode::LEVEL0_COUNT_MODE, VecCast<TensorTrait<half>, TensorTrait<int16_t>> },
        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL0_COUNT_MODE, VecCast<TensorTrait<uint8_t>, TensorTrait<half>> },

        CastTestParams { 256, 2, 2, RoundMode::CAST_CEIL, TestMode::LEVEL2_MODE, VecCast<TensorTrait<half>, TensorTrait<int16_t>> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_FLOOR, TestMode::LEVEL2_MODE, VecCast<TensorTrait<float>, TensorTrait<half>> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_HYBRID, TestMode::LEVEL2_MODE, VecCast<TensorTrait<float>, TensorTrait<hifloat8_t>> },
        CastTestParams { 256, 1, 2, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<TensorTrait<uint8_t>, TensorTrait<half>> },
        CastTestParams { 256, 8, 4, RoundMode::CAST_RINT, TestMode::LEVEL2_MODE, VecCast<TensorTrait<int64_t>, TensorTrait<float>> },
        CastTestParams { 256, 4, 1, RoundMode::CAST_ROUND, TestMode::LEVEL2_MODE, VecCast<TensorTrait<float>, TensorTrait<hifloat8_t>> },
        CastTestParams { 256, 4, 2, RoundMode::CAST_TRUNC, TestMode::LEVEL0_COUNT_MODE, VecCast<TensorTrait<float>, TensorTrait<half>> }

        ));

TEST_P(CastSimpleTestsuite, CastSimpleTestCase)
{
    TPipe tpipe;
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.srcDatabitSize];
    uint8_t dstGm[param.dataSize * param.dstDatabitSize];

    param.cal_func(dstGm, srcGm, param.dataSize, param.roundMode, param.testMode);
}
