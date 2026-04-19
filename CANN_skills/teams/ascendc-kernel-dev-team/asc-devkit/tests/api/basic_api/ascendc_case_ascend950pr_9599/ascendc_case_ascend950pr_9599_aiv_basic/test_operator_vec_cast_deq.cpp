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
    LEVEL2,
    LEVEL0_BIT_MODE,
    LEVEL0_COUNT_MODE,
};

class TEST_CAST_DEQ : public testing::Test {
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

template <typename T, typename U, TestMode TestMode, bool isSetMask, bool isCounterMode, bool isVecDeq, bool halfBlock, bool USE_CAST_DEQ>
void MainVecCastDeqDemo(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm,
    uint32_t dataSize)
{
    uint32_t LowMask = 0x0000000f;
    uint32_t HighMask = 0x000fff0f;
    uint32_t dstSize = dataSize * 2;
    int16_t Offset = 3;
    float Scale = 1.0;
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<U> dstGlobal;

    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<U>*>(dstGm), dstSize);

    LocalTensor<T> inputLocal;
    TBuffAddr tbuf;
    tbuf.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    inputLocal.SetAddr(tbuf);
    inputLocal.InitBuffer(0, dataSize);

    LocalTensor<U> dstLocal;
    TBuffAddr tbuf2;
    tbuf2.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    dstLocal.SetAddr(tbuf2);
    dstLocal.InitBuffer(0, dstSize);
    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if constexpr (TestMode == LEVEL0_BIT_MODE) {
        uint8_t repeatTimes = 2;
        UnaryRepeatParams repeatParams{8, 8, 1, 1};
        LocalTensor<uint64_t> vecDeqLocal;
        for (uint16_t i = 0; i < dstSize; i++) {
            dstLocal.SetValue(i, 0);
        }
        PopStackBuffer<uint64_t, TPosition::LCM>(vecDeqLocal);
        float vdeqScale[VDEQ_TENSOR_SIZE] = {0};
        int16_t vdeqOffset[VDEQ_TENSOR_SIZE] = {0};
        bool vdeqSignMode[VDEQ_TENSOR_SIZE] = {0};
        uint64_t mask[2] = {LowMask, HighMask};
        if constexpr (isCounterMode) {
            SetMaskCount();
        }
        if constexpr (!isSetMask) {
            if constexpr (isCounterMode) {
                SetVectorMask<PrimT<T>, AscendC::MaskMode::COUNTER>(HighMask, LowMask);
            } else {
                SetVectorMask<PrimT<T>, AscendC::MaskMode::NORMAL>(HighMask, LowMask);
            }
        }
        if constexpr (std::is_same<PrimT<U>, int8_t>::value) {
            if constexpr (isVecDeq) {
                for (uint16_t i = 0; i < 16; i++) {
                    vdeqScale[i] = Scale * i;
                    vdeqOffset[i] = i;
                    vdeqSignMode[i] = i > 9;
                }
                VdeqInfo vdeqInfo(vdeqScale, vdeqOffset, vdeqSignMode);
                SetDeqScale(vecDeqLocal, vdeqInfo);
            } else {
                SetDeqScale(Scale, Offset, 1);
            }
        } else {
            if constexpr (isVecDeq) {
                for (uint16_t i = 0; i < 16; i++) {
                    vdeqScale[i] = Scale * i;
                    vdeqOffset[i] = i;
                    vdeqSignMode[i] = i > 9;
                }
                VdeqInfo vdeqInfo(vdeqScale, vdeqOffset, vdeqSignMode);
                SetDeqScale(vecDeqLocal, vdeqInfo);
            } else {
                SetDeqScale(Scale, Offset, 0);
            }
        }
        if constexpr (USE_CAST_DEQ)
            CastDeq<U, T, isSetMask, isVecDeq, halfBlock>(dstLocal, inputLocal, mask, repeatTimes, repeatParams);
        else
            CastDequant<U, T, isSetMask, isVecDeq, halfBlock>(dstLocal, inputLocal, mask, repeatTimes, repeatParams);
        ResetMask();
        SetMaskNorm();
    } else if constexpr (TestMode == LEVEL0_COUNT_MODE) {
        uint8_t repeatTimes = 2;
        UnaryRepeatParams repeatParams{8, 8, 1, 1};
        LocalTensor<uint64_t> vecDeqLocal;
        for (uint16_t i = 0; i < dstSize; i++) {
            dstLocal.SetValue(i, 0);
        }
        PopStackBuffer<uint64_t, TPosition::LCM>(vecDeqLocal);
        float vdeqScale[VDEQ_TENSOR_SIZE] = {0};
        int16_t vdeqOffset[VDEQ_TENSOR_SIZE] = {0};
        bool vdeqSignMode[VDEQ_TENSOR_SIZE] = {0};
        uint64_t mask = LowMask;
        if constexpr (isCounterMode) {
            SetMaskCount();
        }
        if constexpr (!isSetMask) {
            if constexpr (isCounterMode) {
                SetVectorMask<PrimT<T>, AscendC::MaskMode::COUNTER>(mask);
            } else {
                SetVectorMask<PrimT<T>, AscendC::MaskMode::NORMAL>(mask);
            }
        }
        if constexpr (std::is_same<PrimT<U>, int8_t>::value) {
            if constexpr (isVecDeq) {
                for (uint16_t i = 0; i < 16; i++) {
                    vdeqScale[i] = Scale * i;
                    vdeqOffset[i] = i;
                    vdeqSignMode[i] = i > 9;
                }
                VdeqInfo vdeqInfo(vdeqScale, vdeqOffset, vdeqSignMode);
                SetDeqScale(vecDeqLocal, vdeqInfo);
            } else {
                SetDeqScale(Scale, Offset, 1);
            }
        } else {
            if constexpr (isVecDeq) {
                for (uint16_t i = 0; i < 16; i++) {
                    vdeqScale[i] = Scale * i;
                    vdeqOffset[i] = i;
                    vdeqSignMode[i] = i > 9;
                }
                VdeqInfo vdeqInfo(vdeqScale, vdeqOffset, vdeqSignMode);
                SetDeqScale(vecDeqLocal, vdeqInfo);
            } else {
                SetDeqScale(Scale, Offset, 0);
            }
        }
        if constexpr (USE_CAST_DEQ)
            CastDeq<U, T, isSetMask, isVecDeq, halfBlock>(dstLocal, inputLocal, mask, repeatTimes, repeatParams);
        else
            CastDequant<U, T, isSetMask, isVecDeq, halfBlock>(dstLocal, inputLocal, mask, repeatTimes, repeatParams);
        ResetMask();
        SetMaskNorm();
    } else {
        LocalTensor<uint64_t> vecDeqLocal;
        PopStackBuffer<uint64_t, TPosition::LCM>(vecDeqLocal);
        float vdeqScale[VDEQ_TENSOR_SIZE] = {0};
        int16_t vdeqOffset[VDEQ_TENSOR_SIZE] = {0};
        bool vdeqSignMode[VDEQ_TENSOR_SIZE] = {0};
        for (uint16_t i = 0; i < dstSize; i++) {
            dstLocal.SetValue(i, 0);
        }
        if constexpr (std::is_same<PrimT<U>, int8_t>::value) {
            if constexpr (isVecDeq) {
                for (uint16_t i = 0; i < 16; i++) {
                    vdeqScale[i] = Scale * i;
                    vdeqOffset[i] = i;
                    vdeqSignMode[i] = i > 9;
                }
                VdeqInfo vdeqInfo(vdeqScale, vdeqOffset, vdeqSignMode);
                SetDeqScale(vecDeqLocal, vdeqInfo);
            } else {
                SetDeqScale(Scale, Offset, 1);
            }
        } else {
            if constexpr (isVecDeq) {
                for (uint16_t i = 0; i < 16; i++) {
                    vdeqScale[i] = Scale * i;
                    vdeqOffset[i] = i;
                    vdeqSignMode[i] = i > 9;
                }
                VdeqInfo vdeqInfo(vdeqScale, vdeqOffset, vdeqSignMode);
                SetDeqScale(vecDeqLocal, vdeqInfo);
            } else {
                SetDeqScale(Scale, Offset, 0);
            }
        }
        if constexpr (USE_CAST_DEQ)
            CastDeq<U, T, isVecDeq, halfBlock>(dstLocal, inputLocal, LowMask);
        else
            CastDequant<U, T, isVecDeq, halfBlock>(dstLocal, inputLocal, LowMask);
    }
        DataCopy(dstGlobal, dstLocal, dstSize);

    
}

#define VEC_CAST_EQ_TESTCASE(t, u, testMode, isSetMask, isCounterMode, isVecDeq, halfBlock)                                                \
    TEST_F(TEST_CAST_DEQ, CastDeq##t##u##RELATION_OP##testMode##isSetMask##isCounterMode##isVecDeq##halfBlock##Case)                                 \
    {                                                                                                      \
        uint32_t dataSize = 2048;                                                                          \
        uint32_t selMaskSize = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));                    \
        uint8_t inputGm[dataSize * sizeof(PrimT<t>)];                                                  \
        uint8_t outputGm[dataSize * 2 * sizeof(PrimT<u>)];                                                                      \
        MainVecCastDeqDemo<t, u, testMode, isSetMask, isCounterMode, isVecDeq, halfBlock, true>(outputGm, inputGm, dataSize);   \
        for (uint32_t i = 0; i < selMaskSize; i++) {                                                     \
            EXPECT_EQ(outputGm[i], 0x00);                                                                 \
        }                                                                                                  \
    }                                                                                                      \
    TEST_F(TEST_CAST_DEQ, CastDequant##t##u##RELATION_OP##testMode##isSetMask##isCounterMode##isVecDeq##halfBlock##Case)                                 \
    {                                                                                                      \
        uint32_t dataSize = 2048;                                                                          \
        uint32_t selMaskSize = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));                    \
        uint8_t inputGm[dataSize * sizeof(PrimT<t>)];                                                  \
        uint8_t outputGm[dataSize * 2 * sizeof(PrimT<u>)];                                                                      \
        MainVecCastDeqDemo<t, u, testMode, isSetMask, isCounterMode, isVecDeq, halfBlock, false>(outputGm, inputGm, dataSize);  \
        for (uint32_t i = 0; i < selMaskSize; i++) {                                                     \
            EXPECT_EQ(outputGm[i], 0x00);                                                                 \
        }                                                                                                  \
    }

VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, true, true, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, true, true, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, true, true, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, true, false, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, false, true, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, true, true, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, true, false, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, false, true, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, true, false, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, false, true, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, false, false, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, false, false, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, false, false, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, false, true, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, true, false, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL2, false, false, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, true, true, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, true, true, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, true, true, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, true, false, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, false, true, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, true, true, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, true, false, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, false, true, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, true, false, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, false, true, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, false, false, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, false, false, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, false, false, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, false, true, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, true, false, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL2, false, false, false, false);

VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, true, true, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, true, true, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, true, true, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, true, false, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, false, true, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, true, true, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, true, false, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, false, true, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, true, false, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, false, true, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, false, false, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, false, false, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, false, false, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, false, true, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, true, false, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_BIT_MODE, false, false, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, true, true, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, true, true, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, true, true, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, true, false, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, false, true, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, true, true, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, true, false, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, false, true, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, true, false, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, false, true, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, false, false, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, false, false, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, false, false, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, false, true, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, true, false, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_BIT_MODE, false, false, false, false);

VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, true, true, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, true, true, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, true, true, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, true, false, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, false, true, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, true, true, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, true, false, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, false, true, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, true, false, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, false, true, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, false, false, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, false, false, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, false, false, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, false, true, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, true, false, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, uint8_t, LEVEL0_COUNT_MODE, false, false, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, true, true, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, true, true, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, true, true, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, true, false, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, false, true, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, true, true, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, true, false, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, false, true, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, true, false, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, false, true, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, false, false, true, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, false, false, false, true);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, false, false, true, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, false, true, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, true, false, false, false);
VEC_CAST_EQ_TESTCASE(int16_t, int8_t, LEVEL0_COUNT_MODE, false, false, false, false);

// TensorTrait
using TraitTensorInt16 = TensorTrait<int16_t>;
using TraitTensorUint8 = TensorTrait<uint8_t>;
VEC_CAST_EQ_TESTCASE(TraitTensorInt16, TraitTensorUint8, LEVEL2, true, true, true, true);
VEC_CAST_EQ_TESTCASE(TraitTensorInt16, TraitTensorUint8, LEVEL0_BIT_MODE, true, true, true, true);
VEC_CAST_EQ_TESTCASE(TraitTensorInt16, TraitTensorUint8, LEVEL0_COUNT_MODE, true, true, true, true);