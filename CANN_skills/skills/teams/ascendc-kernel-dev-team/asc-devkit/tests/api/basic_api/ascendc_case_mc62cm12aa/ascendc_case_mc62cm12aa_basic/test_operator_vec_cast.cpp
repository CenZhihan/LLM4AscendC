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
    GlobalTensor<SrcT> input_global;
    GlobalTensor<DstT> output_global;
    input_global.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(srcGm), dataSize);
    output_global.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(dstGm), dataSize);
    LocalTensor<T1> input_local;
    TBuffAddr tbuf0;
    tbuf0.logicPos = (uint8_t)TPosition::VECCALC;
    input_local.SetAddr(tbuf0);
    input_local.InitBuffer(0, dataSize);

    LocalTensor<T2> output_local;
    TBuffAddr tbuf1;
    tbuf1.logicPos = (uint8_t)TPosition::VECCALC;
    output_local.SetAddr(tbuf1);
    output_local.InitBuffer(dataSize * sizeof(PrimT<T1>), dataSize);

    for (int32_t i = 0; i < dataSize; ++i) {
        input_local.SetValue(i, 1);
    }
    if constexpr (IsSameType<T1, hifloat8_t>::value||
        IsSameType<T1, fp8_e5m2_t>::value||IsSameType<T1, fp8_e4m3fn_t>::value) {
        LocalTensor<uint8_t> tmp = input_local.template ReinterpretCast<uint8_t>();
        DataCopy(tmp, input_global, dataSize);
    } else {
        DataCopy(input_local, input_global, dataSize);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if (testMode == TestMode::LEVEL0_BIT_MODE) {
        uint32_t repeatSize = 256 / max(sizeof(T1), sizeof(T2));
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
        UnaryRepeatParams repeatParams(1, 1, sizeof(T2) * repeatSize / 32, sizeof(T1) * repeatSize / 32);
        Cast(output_local, input_local, roundMode, mask, repeatTimes, repeatParams);

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<T1, MaskMode::COUNTER>(0, 144);
        Cast<T2, T1, false>(output_local, input_local, roundMode, AscendC::MASK_PLACEHOLDER_LIST, repeatTimes, repeatParams);
        AscendC::ResetMask();
        Cast(output_local, input_local, roundMode, counterMask, repeatTimes, repeatParams);
        AscendC::SetMaskNorm();
    } else if (testMode == TestMode::LEVEL0_COUNT_MODE) {
        uint32_t repeatSize = 256 / max(sizeof(T1), sizeof(T2));
        uint8_t repeatTimes = dataSize / repeatSize;
        uint64_t mask = repeatSize;
        UnaryRepeatParams repeatParams(1, 1, sizeof(T2) * repeatSize / 32, sizeof(T1) * repeatSize / 32);
        Cast(output_local, input_local, roundMode, mask, repeatTimes, repeatParams);
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<T1, MaskMode::COUNTER>(0, 144);
        Cast<T2, T1, false>(output_local, input_local, roundMode, AscendC::MASK_PLACEHOLDER, repeatTimes, repeatParams);
        AscendC::ResetMask();
        Cast(output_local, input_local, roundMode, mask, repeatTimes, repeatParams);
        AscendC::SetMaskNorm();
    } else {
        Cast(output_local, input_local, roundMode, dataSize);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    if constexpr (IsSameType<T2, hifloat8_t>::value||
        IsSameType<T2, fp8_e5m2_t>::value||IsSameType<T2, fp8_e4m3fn_t>::value) {
        LocalTensor<uint8_t> tmpOut = output_local.template ReinterpretCast<uint8_t>();
        DataCopy(output_global, tmpOut, dataSize);
    } else {
        DataCopy(output_global, output_local, dataSize);
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
        CastTestParams { 256, 8, 4, RoundMode::CAST_NONE, TestMode::LEVEL2_MODE, VecCast<int64_t, int32_t> }
        ));

TEST_P(CastSimpleTestsuite, CastSimpleTestCase)
{
    TPipe tpipe;
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.srcDatabitSize];
    uint8_t dstGm[param.dataSize * param.dstDatabitSize];

    param.cal_func(dstGm, srcGm, param.dataSize, param.roundMode, param.testMode);
    // for (int32_t i = 0; i < param.dataSize; i++) {
    //     EXPECT_EQ(dstGm[i], 0x00);
    // }
}
