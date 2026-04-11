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

using TTI8 = TensorTrait<int8_t>;
using TTI16 = TensorTrait<int16_t>;
using TTI32 = TensorTrait<int32_t>;
using TTI64 = TensorTrait<int64_t>;
using TTU8 = TensorTrait<uint8_t>;
using TTU16 = TensorTrait<uint16_t>;
using TTU32 = TensorTrait<uint32_t>;
using TTU64 = TensorTrait<uint64_t>;
using TTHalf = TensorTrait<half>;
using TTFloat = TensorTrait<float>;
using TTBF16 = TensorTrait<bfloat16_t>;

enum TestMode {
    LEVEL3,
    LEVEL2,
    LEVEL0_BIT_MODE,
    LEVEL0_COUNT_MODE,
};

class TEST_COMPARE : public testing::Test {
protected:
    void SetUp()
    {
        SetGCoreType(2);
    }
    void TearDown()
    {
        SetGCoreType(0);
    }
};

template <typename T, TestMode TestMode, bool withDst>
void main_vec_compare_demo(__gm__ uint8_t* __restrict__ selMaskGm, __gm__ uint8_t* __restrict__ src0Gm,
    __gm__ uint8_t* __restrict__ src1Gm, CMPMODE cmpMode, uint32_t dataSize, uint32_t selMaskSize)
{
    TPipe tpipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<T> input1Global;
    GlobalTensor<uint8_t> selMaskGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(src0Gm), dataSize);
    input1Global.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(src1Gm), dataSize);
    selMaskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(selMaskGm), dataSize);

    LocalTensor<T> input0Local;
    TBuffAddr tbuf;
    tbuf.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    input0Local.SetAddr(tbuf);
    input0Local.InitBuffer(0, dataSize);

    LocalTensor<T> input1Local;
    TBuffAddr tbuf1;
    tbuf1.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    input1Local.SetAddr(tbuf1);
    input1Local.InitBuffer(input0Local.GetSize() * sizeof(PrimT<T>), dataSize);

    LocalTensor<uint8_t> selMaskLocal;
    TBuffAddr tbuf2;
    tbuf2.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    selMaskLocal.SetAddr(tbuf2);
    selMaskLocal.InitBuffer(input0Local.GetSize() * sizeof(PrimT<T>) + input1Local.GetSize() * sizeof(PrimT<T>), selMaskSize);

    DataCopy(input0Local, input0Global, dataSize);
    DataCopy(input1Local, input1Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if constexpr(withDst) {
        if constexpr (TestMode == LEVEL2) {
            Compare(selMaskLocal, input0Local, input1Local, cmpMode, dataSize);
        } else if constexpr (TestMode == LEVEL0_BIT_MODE) {
            uint8_t repeatTime = dataSize * sizeof(PrimT<T>) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask[2];
            uint64_t counterMask[2] = {144, 0};
            if (sizeof(PrimT<T>) == 2) {
                mask[0] = UINT64_MAX;
                mask[1] = UINT64_MAX;
            } else if (sizeof(PrimT<T>) == 4) {
                mask[0] = UINT64_MAX;
                mask[1] = 0;
            }
            Compare(selMaskLocal, input0Local, input1Local, cmpMode, mask, repeatTime, { 0, 1, 1, 0, 8, 8 });
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<PrimT<T>, MaskMode::COUNTER>(0, 144);
            Compare<T, uint8_t ,false>(selMaskLocal, input0Local, input1Local, cmpMode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 1, 0, 8, 8 });
            AscendC::ResetMask();
            Compare(selMaskLocal, input0Local, input1Local, cmpMode, counterMask, repeatTime, { 0, 1, 1, 0, 8, 8 });
            AscendC::SetMaskNorm();
        } else if constexpr (TestMode == LEVEL0_COUNT_MODE) {
            uint8_t repeatTime = dataSize * sizeof(PrimT<T>) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask = 0;
            if (sizeof(PrimT<T>) == 2) {
                mask = 128;
            } else if (sizeof(PrimT<T>) == 4) {
                mask = 64;
            }
            Compare(selMaskLocal, input0Local, input1Local, cmpMode, mask, repeatTime, { 0, 1, 1, 0, 8, 8 });
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<PrimT<T>, MaskMode::COUNTER>(0, 144);
            Compare<T, uint8_t ,false>(selMaskLocal, input0Local, input1Local, cmpMode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 1, 0, 8, 8 });
            AscendC::ResetMask();
            Compare(selMaskLocal, input0Local, input1Local, cmpMode, mask, repeatTime, { 0, 1, 1, 0, 8, 8 });
            AscendC::SetMaskNorm();
        } else if constexpr (TestMode == LEVEL3) {
            if (cmpMode == CMPMODE::EQ) {
                selMaskLocal = input0Local == input1Local;
            } else if (cmpMode == CMPMODE::GE) {
                selMaskLocal = input0Local >= input1Local;
            }else if (cmpMode == CMPMODE::GT) {
                selMaskLocal = input0Local > input1Local;
            }else if (cmpMode == CMPMODE::LE) {
                selMaskLocal = input0Local <= input1Local;
            }else if (cmpMode == CMPMODE::LT) {
                selMaskLocal = input0Local < input1Local;
            }else if (cmpMode == CMPMODE::NE) {
                selMaskLocal = input0Local != input1Local;
            }
        }
    } else {
        uint8_t repeatTime = dataSize * sizeof(PrimT<T>) / ONE_REPEAT_BYTE_SIZE;
        uint64_t mask[2];
        uint64_t counterMask[2] = {144, 0};
        uint64_t normMask[2] = {0, 0};
        if (sizeof(PrimT<T>) == 2) {
            mask[0] = UINT64_MAX;
            mask[1] = UINT64_MAX;
            normMask[0] = 127;  // max is 128
        } else if (sizeof(PrimT<T>) == 4) {
            mask[0] = UINT64_MAX;
            mask[1] = 0;
            normMask[0] = 60;  // max is 64
        }
        Compare(input0Local, input1Local, cmpMode, mask, { 0, 1, 1, 0, 8, 8 });
        GetCmpMask(selMaskLocal);
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<PrimT<T>, MaskMode::COUNTER>(0, 144);
        Compare<T, false>(input0Local, input1Local, cmpMode, AscendC::MASK_PLACEHOLDER_LIST, { 0, 1, 1, 0, 8, 8 });
        GetCmpMask(selMaskLocal);
        AscendC::ResetMask();
        Compare(input0Local, input1Local, cmpMode, counterMask, { 0, 1, 1, 0, 8, 8 });
        GetCmpMask(selMaskLocal);
        AscendC::SetMaskNorm();

        Compare(input0Local, input1Local, cmpMode, normMask[0], { 0, 1, 1, 0, 8, 8 });
        GetCmpMask(selMaskLocal);
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<PrimT<T>, MaskMode::COUNTER>(0, 144);
        Compare<T, false>(input0Local, input1Local, cmpMode, AscendC::MASK_PLACEHOLDER, { 0, 1, 1, 0, 8, 8 });
        GetCmpMask(selMaskLocal);
        AscendC::ResetMask();
        Compare(input0Local, input1Local, cmpMode, counterMask[0], { 0, 1, 1, 0, 8, 8 });
        GetCmpMask(selMaskLocal);
        AscendC::SetMaskNorm();
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(selMaskGlobal, selMaskLocal, selMaskSize);

    pipe_barrier(PIPE_ALL);
}

#define VEC_CMP_TESTCASE(dataType, relationOp, testMode, withDst)                                                \
    TEST_F(TEST_COMPARE, Compare##dataType##relationOp##testMode##withDst##Case)                                 \
    {                                                                                                      \
        uint32_t dataSize = 256;                                                                          \
        uint32_t selMaskSize = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));                    \
        uint8_t input0Gm[dataSize * sizeof(PrimT<dataType>)];                                                  \
        uint8_t input1Gm[dataSize * sizeof(PrimT<dataType>)];                                                  \
        uint8_t outputGm[dataSize];                                                                      \
                                                                                                           \
        main_vec_compare_demo<dataType, testMode, withDst>(outputGm, input0Gm, input1Gm, CMPMODE::relationOp, dataSize, \
            selMaskSize);                                                                     \
                                                                                                           \
        for (uint32_t i = 0; i < selMaskSize; i++) {                                                     \
            EXPECT_EQ(outputGm[i], 0x00);                                                                 \
        }                                                                                                  \
    }

VEC_CMP_TESTCASE(uint8_t, LT, LEVEL2, true);
VEC_CMP_TESTCASE(int8_t, LT, LEVEL2, true);
VEC_CMP_TESTCASE(float, LT, LEVEL2, true);
VEC_CMP_TESTCASE(half, LT, LEVEL2, true);
VEC_CMP_TESTCASE(int16_t, LT, LEVEL2, true);
VEC_CMP_TESTCASE(bfloat16_t, LT, LEVEL2, true);
VEC_CMP_TESTCASE(int32_t, LT, LEVEL2, true);
VEC_CMP_TESTCASE(uint32_t, LT, LEVEL2, true);
VEC_CMP_TESTCASE(uint16_t, LT, LEVEL2, true);
VEC_CMP_TESTCASE(int64_t, LT, LEVEL2, true);
VEC_CMP_TESTCASE(uint64_t, LT, LEVEL2, true);
VEC_CMP_TESTCASE(uint8_t, GT, LEVEL2, true);
VEC_CMP_TESTCASE(int8_t, GT, LEVEL2, true);
VEC_CMP_TESTCASE(float, GT, LEVEL2, true);
VEC_CMP_TESTCASE(half, GT, LEVEL2, true);
VEC_CMP_TESTCASE(int16_t, GT, LEVEL2, true);
VEC_CMP_TESTCASE(bfloat16_t, GT, LEVEL2, true);
VEC_CMP_TESTCASE(int32_t, GT, LEVEL2, true);
VEC_CMP_TESTCASE(uint32_t, GT, LEVEL2, true);
VEC_CMP_TESTCASE(uint16_t, GT, LEVEL2, true);
VEC_CMP_TESTCASE(int64_t, GT, LEVEL2, true);
VEC_CMP_TESTCASE(uint64_t, GT, LEVEL2, true);
VEC_CMP_TESTCASE(uint8_t, EQ, LEVEL2, true);
VEC_CMP_TESTCASE(int8_t, EQ, LEVEL2, true);
VEC_CMP_TESTCASE(float, EQ, LEVEL2, true);
VEC_CMP_TESTCASE(half, EQ, LEVEL2, true);
VEC_CMP_TESTCASE(int16_t, EQ, LEVEL2, true);
VEC_CMP_TESTCASE(bfloat16_t, EQ, LEVEL2, true);
VEC_CMP_TESTCASE(int32_t, EQ, LEVEL2, true);
VEC_CMP_TESTCASE(uint32_t, EQ, LEVEL2, true);
VEC_CMP_TESTCASE(uint16_t, EQ, LEVEL2, true);
VEC_CMP_TESTCASE(int64_t, EQ, LEVEL2, true);
VEC_CMP_TESTCASE(uint64_t, EQ, LEVEL2, true);
VEC_CMP_TESTCASE(uint8_t, LE, LEVEL2, true);
VEC_CMP_TESTCASE(int8_t, LE, LEVEL2, true);
VEC_CMP_TESTCASE(float, LE, LEVEL2, true);
VEC_CMP_TESTCASE(half, LE, LEVEL2, true);
VEC_CMP_TESTCASE(int16_t, LE, LEVEL2, true);
VEC_CMP_TESTCASE(bfloat16_t, LE, LEVEL2, true);
VEC_CMP_TESTCASE(int32_t, LE, LEVEL2, true);
VEC_CMP_TESTCASE(uint32_t, LE, LEVEL2, true);
VEC_CMP_TESTCASE(uint16_t, LE, LEVEL2, true);
VEC_CMP_TESTCASE(int64_t, LE, LEVEL2, true);
VEC_CMP_TESTCASE(uint64_t, LE, LEVEL2, true);
VEC_CMP_TESTCASE(uint8_t, GE, LEVEL2, true);
VEC_CMP_TESTCASE(int8_t, GE, LEVEL2, true);
VEC_CMP_TESTCASE(float, GE, LEVEL2, true);
VEC_CMP_TESTCASE(half, GE, LEVEL2, true);
VEC_CMP_TESTCASE(int16_t, GE, LEVEL2, true);
VEC_CMP_TESTCASE(int32_t, GE, LEVEL2, true);
VEC_CMP_TESTCASE(uint32_t, GE, LEVEL2, true);
VEC_CMP_TESTCASE(uint16_t, GE, LEVEL2, true);
VEC_CMP_TESTCASE(bfloat16_t, GE, LEVEL2, true);
VEC_CMP_TESTCASE(int64_t, GE, LEVEL2, true);
VEC_CMP_TESTCASE(uint64_t, GE, LEVEL2, true);
VEC_CMP_TESTCASE(uint8_t, NE, LEVEL2, true);
VEC_CMP_TESTCASE(int8_t, NE, LEVEL2, true);
VEC_CMP_TESTCASE(float, NE, LEVEL2, true);
VEC_CMP_TESTCASE(half, NE, LEVEL2, true);
VEC_CMP_TESTCASE(int16_t, NE, LEVEL2, true);
VEC_CMP_TESTCASE(int32_t, NE, LEVEL2, true);
VEC_CMP_TESTCASE(uint32_t, NE, LEVEL2, true);
VEC_CMP_TESTCASE(uint16_t, NE, LEVEL2, true);
VEC_CMP_TESTCASE(bfloat16_t, NE, LEVEL2, true);
VEC_CMP_TESTCASE(int64_t, NE, LEVEL2, true);
VEC_CMP_TESTCASE(uint64_t, NE, LEVEL2, true);
VEC_CMP_TESTCASE(float, LT, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(half, LT, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(int16_t, LT, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(bfloat16_t, LT, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(int32_t, LT, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(uint32_t, LT, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(uint16_t, LT, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(float, GT, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(half, GT, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(int16_t, GT, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(bfloat16_t, GT, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(int32_t, GT, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(uint32_t, GT, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(uint16_t, GT, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(float, EQ, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(half, EQ, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(int16_t, EQ, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(bfloat16_t, EQ, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(int32_t, EQ, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(uint32_t, EQ, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(uint16_t, EQ, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(float, LE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(half, LE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(int16_t, LE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(bfloat16_t, LE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(int32_t, LE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(uint32_t, LE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(uint16_t, LE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(float, GE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(half, GE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(int16_t, GE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(bfloat16_t, GE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(int32_t, GE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(uint32_t, GE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(uint16_t, GE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(float, NE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(half, NE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(int16_t, NE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(bfloat16_t, NE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(int32_t, NE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(uint32_t, NE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(uint16_t, NE, LEVEL0_BIT_MODE, true);
VEC_CMP_TESTCASE(float, LT, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(half, LT, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(int16_t, LT, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(bfloat16_t, LT, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(int32_t, LT, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(uint32_t, LT, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(uint16_t, LT, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(float, GT, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(half, GT, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(int16_t, GT, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(bfloat16_t, GT, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(int32_t, GT, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(uint32_t, GT, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(uint16_t, GT, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(float, EQ, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(half, EQ, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(int16_t, EQ, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(bfloat16_t, EQ, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(int32_t, EQ, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(uint32_t, EQ, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(uint16_t, EQ, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(float, LE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(half, LE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(int16_t, LE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(bfloat16_t, LE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(int32_t, LE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(uint32_t, LE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(uint16_t, LE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(float, GE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(half, GE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(int16_t, GE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(bfloat16_t, GE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(int32_t, GE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(uint32_t, GE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(uint16_t, GE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(float, NE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(half, NE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(int16_t, NE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(bfloat16_t, NE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(int32_t, NE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(uint32_t, NE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(uint16_t, NE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(uint8_t, LT, LEVEL3, true);
VEC_CMP_TESTCASE(int8_t, LT, LEVEL3, true);
VEC_CMP_TESTCASE(float, LT, LEVEL3, true);
VEC_CMP_TESTCASE(half, LT, LEVEL3, true);
VEC_CMP_TESTCASE(int16_t, LT, LEVEL3, true);
VEC_CMP_TESTCASE(bfloat16_t, LT, LEVEL3, true);
VEC_CMP_TESTCASE(int32_t, LT, LEVEL3, true);
VEC_CMP_TESTCASE(uint32_t, LT, LEVEL3, true);
VEC_CMP_TESTCASE(uint16_t, LT, LEVEL3, true);
VEC_CMP_TESTCASE(uint8_t, GT, LEVEL3, true);
VEC_CMP_TESTCASE(int8_t, GT, LEVEL3, true);
VEC_CMP_TESTCASE(float, GT, LEVEL3, true);
VEC_CMP_TESTCASE(half, GT, LEVEL3, true);
VEC_CMP_TESTCASE(int16_t, GT, LEVEL3, true);
VEC_CMP_TESTCASE(bfloat16_t, GT, LEVEL3, true);
VEC_CMP_TESTCASE(int32_t, GT, LEVEL3, true);
VEC_CMP_TESTCASE(uint32_t, GT, LEVEL3, true);
VEC_CMP_TESTCASE(uint16_t, GT, LEVEL3, true);
VEC_CMP_TESTCASE(uint8_t, EQ, LEVEL3, true);
VEC_CMP_TESTCASE(int8_t, EQ, LEVEL3, true);
VEC_CMP_TESTCASE(float, EQ, LEVEL3, true);
VEC_CMP_TESTCASE(half, EQ, LEVEL3, true);
VEC_CMP_TESTCASE(int16_t, EQ, LEVEL3, true);
VEC_CMP_TESTCASE(bfloat16_t, EQ, LEVEL3, true);
VEC_CMP_TESTCASE(int32_t, EQ, LEVEL3, true);
VEC_CMP_TESTCASE(uint32_t, EQ, LEVEL3, true);
VEC_CMP_TESTCASE(uint16_t, EQ, LEVEL3, true);
VEC_CMP_TESTCASE(uint8_t, LE, LEVEL3, true);
VEC_CMP_TESTCASE(int8_t, LE, LEVEL3, true);
VEC_CMP_TESTCASE(float, LE, LEVEL3, true);
VEC_CMP_TESTCASE(half, LE, LEVEL3, true);
VEC_CMP_TESTCASE(int16_t, LE, LEVEL3, true);
VEC_CMP_TESTCASE(bfloat16_t, LE, LEVEL3, true);
VEC_CMP_TESTCASE(int32_t, LE, LEVEL3, true);
VEC_CMP_TESTCASE(uint32_t, LE, LEVEL3, true);
VEC_CMP_TESTCASE(uint16_t, LE, LEVEL3, true);
VEC_CMP_TESTCASE(uint8_t, GE, LEVEL3, true);
VEC_CMP_TESTCASE(int8_t, GE, LEVEL3, true);
VEC_CMP_TESTCASE(float, GE, LEVEL3, true);
VEC_CMP_TESTCASE(half, GE, LEVEL3, true);
VEC_CMP_TESTCASE(int16_t, GE, LEVEL3, true);
VEC_CMP_TESTCASE(bfloat16_t, GE, LEVEL3, true);
VEC_CMP_TESTCASE(int32_t, GE, LEVEL3, true);
VEC_CMP_TESTCASE(uint32_t, GE, LEVEL3, true);
VEC_CMP_TESTCASE(uint16_t, GE, LEVEL3, true);
VEC_CMP_TESTCASE(uint8_t, NE, LEVEL3, true);
VEC_CMP_TESTCASE(int8_t, NE, LEVEL3, true);
VEC_CMP_TESTCASE(float, NE, LEVEL3, true);
VEC_CMP_TESTCASE(half, NE, LEVEL3, true);
VEC_CMP_TESTCASE(int16_t, NE, LEVEL3, true);
VEC_CMP_TESTCASE(bfloat16_t, NE, LEVEL3, true);
VEC_CMP_TESTCASE(int32_t, NE, LEVEL3, true);
VEC_CMP_TESTCASE(uint32_t, NE, LEVEL3, true);
VEC_CMP_TESTCASE(uint16_t, NE, LEVEL3, true);

VEC_CMP_TESTCASE(float, LT, LEVEL0_BIT_MODE, false);
VEC_CMP_TESTCASE(half, LT, LEVEL0_BIT_MODE, false);
VEC_CMP_TESTCASE(float, GT, LEVEL0_BIT_MODE, false);
VEC_CMP_TESTCASE(half, GT, LEVEL0_BIT_MODE, false);
VEC_CMP_TESTCASE(float, EQ, LEVEL0_BIT_MODE, false);
VEC_CMP_TESTCASE(half, EQ, LEVEL0_BIT_MODE, false);
VEC_CMP_TESTCASE(float, LE, LEVEL0_BIT_MODE, false);
VEC_CMP_TESTCASE(half, LE, LEVEL0_BIT_MODE, false);
VEC_CMP_TESTCASE(float, GE, LEVEL0_BIT_MODE, false);
VEC_CMP_TESTCASE(half, GE, LEVEL0_BIT_MODE, false);
VEC_CMP_TESTCASE(float, NE, LEVEL0_BIT_MODE, false);
VEC_CMP_TESTCASE(half, NE, LEVEL0_BIT_MODE, false);

VEC_CMP_TESTCASE(TTI8, LT, LEVEL2, true);
VEC_CMP_TESTCASE(TTU32, GE, LEVEL3, true);
VEC_CMP_TESTCASE(TTU16, NE, LEVEL0_COUNT_MODE, true);
VEC_CMP_TESTCASE(TTFloat, EQ, LEVEL0_BIT_MODE, false);

class TEST_COMPARE_SCALAR : public testing::Test {
protected:
    void SetUp()
    {
        SetGCoreType(2);
    }
    void TearDown()
    {
        SetGCoreType(0);
    }
};

template <typename T, TestMode TestMode, bool USE_COMPARES>
void MainVecCompareScalarDemo(__gm__ uint8_t* __restrict__ selMaskGm, __gm__ uint8_t* __restrict__ src0Gm,
    __gm__ uint8_t* __restrict__ src1Gm, CMPMODE cmpMode, uint32_t dataSize, uint32_t selMaskSize)
{
    TPipe tpipe;
    GlobalTensor<T> input0Global;

    GlobalTensor<uint8_t> selMaskGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(src0Gm), dataSize);
    selMaskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(selMaskGm), dataSize);

    LocalTensor<T> input0Local;
    TBuffAddr tbuf;
    tbuf.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    input0Local.SetAddr(tbuf);
    input0Local.InitBuffer(0, dataSize);

    PrimT<T> scalar = 1;
    LocalTensor<uint8_t> selMaskLocal;
    TBuffAddr tbuf2;
    tbuf2.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    selMaskLocal.SetAddr(tbuf2);
    selMaskLocal.InitBuffer(input0Local.GetSize() * sizeof(PrimT<T>), selMaskSize);

    DataCopy(input0Local, input0Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if constexpr (USE_COMPARES){
        if constexpr (TestMode == LEVEL2) {
            Compares(selMaskLocal, input0Local, scalar, cmpMode, dataSize);
            Compares(selMaskLocal, scalar, input0Local, cmpMode, dataSize);
        } else if constexpr (TestMode == LEVEL0_BIT_MODE) {
            uint8_t repeatTime = dataSize * sizeof(PrimT<T>) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask[2];
            uint64_t counterMask[2] = {144, 0};
            if (sizeof(PrimT<T>) == 2) {
                mask[0] = UINT64_MAX;
                mask[1] = UINT64_MAX;
            } else if (sizeof(PrimT<T>) == 4) {
                mask[0] = UINT64_MAX;
                mask[1] = 0;
            }
            Compares(selMaskLocal, input0Local, scalar, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, static_cast<PrimT<T>>(0), input0Local, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, input0Local, static_cast<PrimT<T>>(0), cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, input0Local, input0Local[0], cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, input0Local[0], input0Local, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<PrimT<T>, MaskMode::COUNTER>(0, 144);
            Compares<T, uint8_t ,false>(selMaskLocal, input0Local, scalar, cmpMode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(selMaskLocal, scalar, input0Local, cmpMode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(selMaskLocal, static_cast<PrimT<T>>(0), input0Local, cmpMode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(selMaskLocal, input0Local, static_cast<PrimT<T>>(0), cmpMode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(selMaskLocal, input0Local, input0Local[0], cmpMode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(selMaskLocal, input0Local[0], input0Local, cmpMode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            AscendC::ResetMask();
            Compares(selMaskLocal, input0Local, scalar, cmpMode, counterMask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, scalar, input0Local, cmpMode, counterMask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, static_cast<PrimT<T>>(0), input0Local, cmpMode, counterMask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, input0Local, static_cast<PrimT<T>>(0), cmpMode, counterMask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, input0Local, input0Local[0], cmpMode, counterMask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, input0Local[0], input0Local, cmpMode, counterMask, repeatTime, { 0, 1, 0, 8 });
            AscendC::SetMaskNorm();
        } else if constexpr (TestMode == LEVEL0_COUNT_MODE) {
            uint8_t repeatTime = dataSize * sizeof(PrimT<T>) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask = 0;
            if (sizeof(PrimT<T>) == 2) {
                mask = 128;
            } else if (sizeof(PrimT<T>) == 4) {
                mask = 64;
            }
            Compares(selMaskLocal, input0Local, scalar, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, scalar, input0Local, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, static_cast<PrimT<T>>(0), input0Local, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, input0Local, static_cast<PrimT<T>>(0), cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, input0Local, input0Local[0], cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, input0Local[0], input0Local, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });

            constexpr static BinaryConfig config = {1};
            Compares<T, uint8_t, false, config>(selMaskLocal, input0Local, scalar, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            Select<T, uint8_t, false, config>(input0Local, selMaskLocal, input0Local, scalar, SELMODE::VSEL_TENSOR_SCALAR_MODE, mask, 1, { 1, 1, 1, 8, 8, 8});

            AscendC::SetMaskCount();
            AscendC::SetVectorMask<T, MaskMode::COUNTER>(0, 144);
            Compares<T, uint8_t ,false>(selMaskLocal, input0Local, scalar, cmpMode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(selMaskLocal, scalar, input0Local, cmpMode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(selMaskLocal, static_cast<PrimT<T>>(0), input0Local, cmpMode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(selMaskLocal, input0Local, static_cast<PrimT<T>>(0), cmpMode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(selMaskLocal, input0Local, input0Local[0], cmpMode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(selMaskLocal, input0Local[0], input0Local, cmpMode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            AscendC::ResetMask();
            Compares(selMaskLocal, input0Local, scalar, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, scalar, input0Local, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, static_cast<PrimT<T>>(0), input0Local, cmpMode,  mask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, input0Local, static_cast<PrimT<T>>(0), cmpMode,  mask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, input0Local, input0Local[0], cmpMode,  mask, repeatTime, { 0, 1, 0, 8 });
            Compares(selMaskLocal, input0Local[0], input0Local, cmpMode,  mask, repeatTime, { 0, 1, 0, 8 });
            AscendC::SetMaskNorm();
        }
    }else{
        if constexpr (TestMode == LEVEL2) {
            CompareScalar(selMaskLocal, input0Local, scalar, cmpMode, dataSize);
            CompareScalar(selMaskLocal, scalar, input0Local, cmpMode, dataSize);
        } else if constexpr (TestMode == LEVEL0_BIT_MODE) {
            uint8_t repeatTime = dataSize * sizeof(PrimT<T>) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask[2];
            uint64_t counterMask[2] = {144, 0};
            if (sizeof(PrimT<T>) == 2) {
                mask[0] = UINT64_MAX;
                mask[1] = UINT64_MAX;
            } else if (sizeof(PrimT<T>) == 4) {
                mask[0] = UINT64_MAX;
                mask[1] = 0;
            }
            CompareScalar(selMaskLocal, input0Local, scalar, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, static_cast<PrimT<T>>(0), input0Local, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, input0Local, static_cast<PrimT<T>>(0), cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, input0Local, input0Local[0], cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, input0Local[0], input0Local, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<PrimT<T>, MaskMode::COUNTER>(0, 144);
            CompareScalar<T, uint8_t ,false>(selMaskLocal, input0Local, scalar, cmpMode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(selMaskLocal, scalar, input0Local, cmpMode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(selMaskLocal, static_cast<PrimT<T>>(0), input0Local, cmpMode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(selMaskLocal, input0Local, static_cast<PrimT<T>>(0), cmpMode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(selMaskLocal, input0Local, input0Local[0], cmpMode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(selMaskLocal, input0Local[0], input0Local, cmpMode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            AscendC::ResetMask();
            CompareScalar(selMaskLocal, input0Local, scalar, cmpMode, counterMask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, scalar, input0Local, cmpMode, counterMask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, static_cast<PrimT<T>>(0), input0Local, cmpMode, counterMask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, input0Local, static_cast<PrimT<T>>(0), cmpMode, counterMask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, input0Local, input0Local[0], cmpMode, counterMask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, input0Local[0], input0Local, cmpMode, counterMask, repeatTime, { 0, 1, 0, 8 });
            AscendC::SetMaskNorm();
        } else if constexpr (TestMode == LEVEL0_COUNT_MODE) {
            uint8_t repeatTime = dataSize * sizeof(PrimT<T>) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask = 0;
            if (sizeof(PrimT<T>) == 2) {
                mask = 128;
            } else if (sizeof(PrimT<T>) == 4) {
                mask = 64;
            }
            CompareScalar(selMaskLocal, input0Local, scalar, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, scalar, input0Local, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, static_cast<PrimT<T>>(0), input0Local, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, input0Local, static_cast<PrimT<T>>(0), cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, input0Local, input0Local[0], cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, input0Local[0], input0Local, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });

            constexpr static BinaryConfig config = {1};
            CompareScalar<T, uint8_t, false, config>(selMaskLocal, input0Local, scalar, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            Select<T, uint8_t, false, config>(input0Local, selMaskLocal, input0Local, scalar, SELMODE::VSEL_TENSOR_SCALAR_MODE, mask, 1, { 1, 1, 1, 8, 8, 8});

            AscendC::SetMaskCount();
            AscendC::SetVectorMask<T, MaskMode::COUNTER>(0, 144);
            CompareScalar<T, uint8_t ,false>(selMaskLocal, input0Local, scalar, cmpMode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(selMaskLocal, scalar, input0Local, cmpMode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(selMaskLocal, static_cast<PrimT<T>>(0), input0Local, cmpMode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(selMaskLocal, input0Local, static_cast<PrimT<T>>(0), cmpMode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(selMaskLocal, input0Local, input0Local[0], cmpMode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(selMaskLocal, input0Local[0], input0Local, cmpMode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            AscendC::ResetMask();
            CompareScalar(selMaskLocal, input0Local, scalar, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, scalar, input0Local, cmpMode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, static_cast<PrimT<T>>(0), input0Local, cmpMode,  mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, input0Local, static_cast<PrimT<T>>(0), cmpMode,  mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, input0Local, input0Local[0], cmpMode,  mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(selMaskLocal, input0Local[0], input0Local, cmpMode,  mask, repeatTime, { 0, 1, 0, 8 });
            AscendC::SetMaskNorm();
        }
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(selMaskGlobal, selMaskLocal, selMaskSize);

    pipe_barrier(PIPE_ALL);
}

#define VEC_CMPS_TESTCASE(dataType, relationOp, testMode)                                               \
    TEST_F(TEST_COMPARE, Compares##dataType##relationOp##testMode##Case)                    \
    {                                                                                                      \
        uint32_t dataSize = 256;                                                                          \
        uint32_t selMaskSize = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));                    \
        uint8_t input0Gm[dataSize * sizeof(PrimT<dataType>)];                                                  \
        uint8_t input1Gm[dataSize * sizeof(PrimT<dataType>)];                                                  \
        uint8_t outputGm[dataSize];                                                                      \
                                                                                                           \
        MainVecCompareScalarDemo<dataType, testMode, true>(outputGm, input0Gm, input1Gm, CMPMODE::relationOp,     \
            dataSize, selMaskSize);                                                          \
                                                                                                           \
        for (uint32_t i = 0; i < selMaskSize; i++) {                                                     \
            EXPECT_EQ(outputGm[i], 0x00);                                                                 \
        }                                                                                                  \
    }                                                                                                      \
    TEST_F(TEST_COMPARE_SCALAR, CompareScalar##dataType##relationOp##testMode##Case)                    \
    {                                                                                                      \
        uint32_t dataSize = 256;                                                                          \
        uint32_t selMaskSize = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));                    \
        uint8_t input0Gm[dataSize * sizeof(PrimT<dataType>)];                                                  \
        uint8_t input1Gm[dataSize * sizeof(PrimT<dataType>)];                                                  \
        uint8_t outputGm[dataSize];                                                                      \
                                                                                                           \
        MainVecCompareScalarDemo<dataType, testMode, false>(outputGm, input0Gm, input1Gm, CMPMODE::relationOp,     \
            dataSize, selMaskSize);                                                          \
                                                                                                           \
        for (uint32_t i = 0; i < selMaskSize; i++) {                                                     \
            EXPECT_EQ(outputGm[i], 0x00);                                                                 \
        }                                                                                                  \
    }
// Counter mode tests
VEC_CMPS_TESTCASE(uint8_t, LT, LEVEL2);
VEC_CMPS_TESTCASE(int8_t, LT, LEVEL2);
VEC_CMPS_TESTCASE(uint16_t, LT, LEVEL2);
VEC_CMPS_TESTCASE(int16_t, LT, LEVEL2);
VEC_CMPS_TESTCASE(bfloat16_t, LT, LEVEL2);
VEC_CMPS_TESTCASE(uint32_t, LT, LEVEL2);
VEC_CMPS_TESTCASE(int32_t, LT, LEVEL2);
VEC_CMPS_TESTCASE(float, LT, LEVEL2);
VEC_CMPS_TESTCASE(half, LT, LEVEL2);
VEC_CMPS_TESTCASE(uint64_t, LT, LEVEL2);
VEC_CMPS_TESTCASE(int64_t, LT, LEVEL2);

VEC_CMPS_TESTCASE(uint8_t, GT, LEVEL2);
VEC_CMPS_TESTCASE(int8_t, GT, LEVEL2);
VEC_CMPS_TESTCASE(uint16_t, GT, LEVEL2);
VEC_CMPS_TESTCASE(int16_t, GT, LEVEL2);
VEC_CMPS_TESTCASE(bfloat16_t, GT, LEVEL2);
VEC_CMPS_TESTCASE(uint32_t, GT, LEVEL2);
VEC_CMPS_TESTCASE(int32_t, GT, LEVEL2);
VEC_CMPS_TESTCASE(float, GT, LEVEL2);
VEC_CMPS_TESTCASE(half, GT, LEVEL2);
VEC_CMPS_TESTCASE(uint64_t, GT, LEVEL2);
VEC_CMPS_TESTCASE(int64_t, GT, LEVEL2);

VEC_CMPS_TESTCASE(uint8_t, EQ, LEVEL2);
VEC_CMPS_TESTCASE(int8_t, EQ, LEVEL2);
VEC_CMPS_TESTCASE(uint16_t, EQ, LEVEL2);
VEC_CMPS_TESTCASE(int16_t, EQ, LEVEL2);
VEC_CMPS_TESTCASE(bfloat16_t, EQ, LEVEL2);
VEC_CMPS_TESTCASE(uint32_t, EQ, LEVEL2);
VEC_CMPS_TESTCASE(int32_t, EQ, LEVEL2);
VEC_CMPS_TESTCASE(float, EQ, LEVEL2);
VEC_CMPS_TESTCASE(half, EQ, LEVEL2);
VEC_CMPS_TESTCASE(uint64_t, EQ, LEVEL2);
VEC_CMPS_TESTCASE(int64_t, EQ, LEVEL2);

VEC_CMPS_TESTCASE(uint8_t, LE, LEVEL2);
VEC_CMPS_TESTCASE(int8_t, LE, LEVEL2);
VEC_CMPS_TESTCASE(uint16_t, LE, LEVEL2);
VEC_CMPS_TESTCASE(int16_t, LE, LEVEL2);
VEC_CMPS_TESTCASE(bfloat16_t, LE, LEVEL2);
VEC_CMPS_TESTCASE(uint32_t, LE, LEVEL2);
VEC_CMPS_TESTCASE(int32_t, LE, LEVEL2);
VEC_CMPS_TESTCASE(float, LE, LEVEL2);
VEC_CMPS_TESTCASE(half, LE, LEVEL2);
VEC_CMPS_TESTCASE(uint64_t, LE, LEVEL2);
VEC_CMPS_TESTCASE(int64_t, LE, LEVEL2);

VEC_CMPS_TESTCASE(uint8_t, GE, LEVEL2);
VEC_CMPS_TESTCASE(int8_t, GE, LEVEL2);
VEC_CMPS_TESTCASE(uint16_t, GE, LEVEL2);
VEC_CMPS_TESTCASE(int16_t, GE, LEVEL2);
VEC_CMPS_TESTCASE(bfloat16_t, GE, LEVEL2);
VEC_CMPS_TESTCASE(uint32_t, GE, LEVEL2);
VEC_CMPS_TESTCASE(int32_t, GE, LEVEL2);
VEC_CMPS_TESTCASE(float, GE, LEVEL2);
VEC_CMPS_TESTCASE(half, GE, LEVEL2);
VEC_CMPS_TESTCASE(uint64_t, GE, LEVEL2);
VEC_CMPS_TESTCASE(int64_t, GE, LEVEL2);

VEC_CMPS_TESTCASE(uint8_t, NE, LEVEL2);
VEC_CMPS_TESTCASE(int8_t, NE, LEVEL2);
VEC_CMPS_TESTCASE(uint16_t, NE, LEVEL2);
VEC_CMPS_TESTCASE(int16_t, NE, LEVEL2);
VEC_CMPS_TESTCASE(bfloat16_t, NE, LEVEL2);
VEC_CMPS_TESTCASE(uint32_t, NE, LEVEL2);
VEC_CMPS_TESTCASE(int32_t, NE, LEVEL2);
VEC_CMPS_TESTCASE(float, NE, LEVEL2);
VEC_CMPS_TESTCASE(half, NE, LEVEL2);
VEC_CMPS_TESTCASE(uint64_t, NE, LEVEL2);
VEC_CMPS_TESTCASE(int64_t, NE, LEVEL2);

// Bit mode tests
VEC_CMPS_TESTCASE(uint16_t, LT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(int16_t, LT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(bfloat16_t, LT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(uint32_t, LT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(int32_t, LT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(float, LT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(half, LT, LEVEL0_BIT_MODE);

VEC_CMPS_TESTCASE(uint16_t, GT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(int16_t, GT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(bfloat16_t, GT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(uint32_t, GT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(int32_t, GT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(float, GT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(half, GT, LEVEL0_BIT_MODE);

VEC_CMPS_TESTCASE(uint16_t, EQ, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(int16_t, EQ, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(bfloat16_t, EQ, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(uint32_t, EQ, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(int32_t, EQ, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(float, EQ, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(half, EQ, LEVEL0_BIT_MODE);

VEC_CMPS_TESTCASE(uint16_t, LE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(int16_t, LE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(bfloat16_t, LE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(uint32_t, LE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(int32_t, LE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(float, LE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(half, LE, LEVEL0_BIT_MODE);

VEC_CMPS_TESTCASE(uint16_t, GE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(int16_t, GE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(bfloat16_t, GE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(uint32_t, GE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(int32_t, GE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(float, GE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(half, GE, LEVEL0_BIT_MODE);

VEC_CMPS_TESTCASE(uint16_t, NE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(int16_t, NE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(bfloat16_t, NE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(uint32_t, NE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(int32_t, NE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(float, NE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(half, NE, LEVEL0_BIT_MODE);

// Continuous mode tests
VEC_CMPS_TESTCASE(uint16_t, LT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(int16_t, LT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(bfloat16_t, LT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(uint32_t, LT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(int32_t, LT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(float, LT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(half, LT, LEVEL0_COUNT_MODE);

VEC_CMPS_TESTCASE(uint16_t, GT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(int16_t, GT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(bfloat16_t, GT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(uint32_t, GT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(int32_t, GT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(float, GT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(half, GT, LEVEL0_COUNT_MODE);

VEC_CMPS_TESTCASE(uint16_t, EQ, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(int16_t, EQ, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(bfloat16_t, EQ, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(uint32_t, EQ, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(int32_t, EQ, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(float, EQ, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(half, EQ, LEVEL0_COUNT_MODE);

VEC_CMPS_TESTCASE(uint16_t, LE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(int16_t, LE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(bfloat16_t, LE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(uint32_t, LE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(int32_t, LE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(float, LE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(half, LE, LEVEL0_COUNT_MODE);

VEC_CMPS_TESTCASE(uint16_t, GE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(int16_t, GE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(bfloat16_t, GE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(uint32_t, GE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(int32_t, GE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(float, GE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(half, GE, LEVEL0_COUNT_MODE);

VEC_CMPS_TESTCASE(uint16_t, NE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(int16_t, NE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(bfloat16_t, NE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(uint32_t, NE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(int32_t, NE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(float, NE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(half, NE, LEVEL0_COUNT_MODE);

// TensorTrait Case

// Counter mode tests
VEC_CMPS_TESTCASE(TTU8, LT, LEVEL2);
VEC_CMPS_TESTCASE(TTI8, LT, LEVEL2);
VEC_CMPS_TESTCASE(TTU16, LT, LEVEL2);
VEC_CMPS_TESTCASE(TTI16, LT, LEVEL2);
VEC_CMPS_TESTCASE(TTBF16, LT, LEVEL2);
VEC_CMPS_TESTCASE(TTU32, LT, LEVEL2);
VEC_CMPS_TESTCASE(TTI32, LT, LEVEL2);
VEC_CMPS_TESTCASE(TTFloat, LT, LEVEL2);
VEC_CMPS_TESTCASE(TTHalf, LT, LEVEL2);
VEC_CMPS_TESTCASE(TTU64, LT, LEVEL2);
VEC_CMPS_TESTCASE(TTI64, LT, LEVEL2);

VEC_CMPS_TESTCASE(TTU8, GT, LEVEL2);
VEC_CMPS_TESTCASE(TTI8, GT, LEVEL2);
VEC_CMPS_TESTCASE(TTU16, GT, LEVEL2);
VEC_CMPS_TESTCASE(TTI16, GT, LEVEL2);
VEC_CMPS_TESTCASE(TTBF16, GT, LEVEL2);
VEC_CMPS_TESTCASE(TTU32, GT, LEVEL2);
VEC_CMPS_TESTCASE(TTI32, GT, LEVEL2);
VEC_CMPS_TESTCASE(TTFloat, GT, LEVEL2);
VEC_CMPS_TESTCASE(TTHalf, GT, LEVEL2);
VEC_CMPS_TESTCASE(TTU64, GT, LEVEL2);
VEC_CMPS_TESTCASE(TTI64, GT, LEVEL2);

VEC_CMPS_TESTCASE(TTU8, EQ, LEVEL2);
VEC_CMPS_TESTCASE(TTI8, EQ, LEVEL2);
VEC_CMPS_TESTCASE(TTU16, EQ, LEVEL2);
VEC_CMPS_TESTCASE(TTI16, EQ, LEVEL2);
VEC_CMPS_TESTCASE(TTBF16, EQ, LEVEL2);
VEC_CMPS_TESTCASE(TTU32, EQ, LEVEL2);
VEC_CMPS_TESTCASE(TTI32, EQ, LEVEL2);
VEC_CMPS_TESTCASE(TTFloat, EQ, LEVEL2);
VEC_CMPS_TESTCASE(TTHalf, EQ, LEVEL2);
VEC_CMPS_TESTCASE(TTU64, EQ, LEVEL2);
VEC_CMPS_TESTCASE(TTI64, EQ, LEVEL2);

VEC_CMPS_TESTCASE(TTU8, LE, LEVEL2);
VEC_CMPS_TESTCASE(TTI8, LE, LEVEL2);
VEC_CMPS_TESTCASE(TTU16, LE, LEVEL2);
VEC_CMPS_TESTCASE(TTI16, LE, LEVEL2);
VEC_CMPS_TESTCASE(TTBF16, LE, LEVEL2);
VEC_CMPS_TESTCASE(TTU32, LE, LEVEL2);
VEC_CMPS_TESTCASE(TTI32, LE, LEVEL2);
VEC_CMPS_TESTCASE(TTFloat, LE, LEVEL2);
VEC_CMPS_TESTCASE(TTHalf, LE, LEVEL2);
VEC_CMPS_TESTCASE(TTU64, LE, LEVEL2);
VEC_CMPS_TESTCASE(TTI64, LE, LEVEL2);

VEC_CMPS_TESTCASE(TTU8, GE, LEVEL2);
VEC_CMPS_TESTCASE(TTI8, GE, LEVEL2);
VEC_CMPS_TESTCASE(TTU16, GE, LEVEL2);
VEC_CMPS_TESTCASE(TTI16, GE, LEVEL2);
VEC_CMPS_TESTCASE(TTBF16, GE, LEVEL2);
VEC_CMPS_TESTCASE(TTU32, GE, LEVEL2);
VEC_CMPS_TESTCASE(TTI32, GE, LEVEL2);
VEC_CMPS_TESTCASE(TTFloat, GE, LEVEL2);
VEC_CMPS_TESTCASE(TTHalf, GE, LEVEL2);
VEC_CMPS_TESTCASE(TTU64, GE, LEVEL2);
VEC_CMPS_TESTCASE(TTI64, GE, LEVEL2);

VEC_CMPS_TESTCASE(TTU8, NE, LEVEL2);
VEC_CMPS_TESTCASE(TTI8, NE, LEVEL2);
VEC_CMPS_TESTCASE(TTU16, NE, LEVEL2);
VEC_CMPS_TESTCASE(TTI16, NE, LEVEL2);
VEC_CMPS_TESTCASE(TTBF16, NE, LEVEL2);
VEC_CMPS_TESTCASE(TTU32, NE, LEVEL2);
VEC_CMPS_TESTCASE(TTI32, NE, LEVEL2);
VEC_CMPS_TESTCASE(TTFloat, NE, LEVEL2);
VEC_CMPS_TESTCASE(TTHalf, NE, LEVEL2);
VEC_CMPS_TESTCASE(TTU64, NE, LEVEL2);
VEC_CMPS_TESTCASE(TTI64, NE, LEVEL2);

// Bit mode tests
VEC_CMPS_TESTCASE(TTU16, LT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTI16, LT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTBF16, LT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTU32, LT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTI32, LT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTFloat, LT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTHalf, LT, LEVEL0_BIT_MODE);

VEC_CMPS_TESTCASE(TTU16, GT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTI16, GT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTBF16, GT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTU32, GT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTI32, GT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTFloat, GT, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTHalf, GT, LEVEL0_BIT_MODE);

VEC_CMPS_TESTCASE(TTU16, EQ, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTI16, EQ, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTBF16, EQ, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTU32, EQ, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTI32, EQ, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTFloat, EQ, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTHalf, EQ, LEVEL0_BIT_MODE);

VEC_CMPS_TESTCASE(TTU16, LE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTI16, LE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTBF16, LE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTU32, LE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTI32, LE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTFloat, LE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTHalf, LE, LEVEL0_BIT_MODE);

VEC_CMPS_TESTCASE(TTU16, GE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTI16, GE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTBF16, GE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTU32, GE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTI32, GE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTFloat, GE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTHalf, GE, LEVEL0_BIT_MODE);

VEC_CMPS_TESTCASE(TTU16, NE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTI16, NE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTBF16, NE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTU32, NE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTI32, NE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTFloat, NE, LEVEL0_BIT_MODE);
VEC_CMPS_TESTCASE(TTHalf, NE, LEVEL0_BIT_MODE);

// Continuous mode tests
VEC_CMPS_TESTCASE(TTU16, LT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTI16, LT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTBF16, LT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTU32, LT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTI32, LT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTFloat, LT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTHalf, LT, LEVEL0_COUNT_MODE);

VEC_CMPS_TESTCASE(TTU16, GT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTI16, GT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTBF16, GT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTU32, GT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTI32, GT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTFloat, GT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTHalf, GT, LEVEL0_COUNT_MODE);

VEC_CMPS_TESTCASE(TTU16, EQ, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTI16, EQ, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTBF16, EQ, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTU32, EQ, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTI32, EQ, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTFloat, EQ, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTHalf, EQ, LEVEL0_COUNT_MODE);

VEC_CMPS_TESTCASE(TTU16, LE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTI16, LE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTBF16, LE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTU32, LE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTI32, LE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTFloat, LE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTHalf, LE, LEVEL0_COUNT_MODE);

VEC_CMPS_TESTCASE(TTU16, GE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTI16, GE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTBF16, GE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTU32, GE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTI32, GE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTFloat, GE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTHalf, GE, LEVEL0_COUNT_MODE);

VEC_CMPS_TESTCASE(TTU16, NE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTI16, NE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTBF16, NE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTU32, NE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTI32, NE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTFloat, NE, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE(TTHalf, NE, LEVEL0_COUNT_MODE);
