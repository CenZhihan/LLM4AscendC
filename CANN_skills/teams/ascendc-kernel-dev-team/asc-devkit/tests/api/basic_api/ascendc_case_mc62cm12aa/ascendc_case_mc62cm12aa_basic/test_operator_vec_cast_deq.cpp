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

template <typename T, typename U, TestMode TEST_MODE, bool IS_SET_MASK, bool IS_COUNTER_MODE, bool IS_VEC_DEQ, bool HALF_BLOCK, bool USE_CAST_DEQ>
void main_vec_cast_deq_demo(__gm__ uint8_t* __restrict__ dst_gm, __gm__ uint8_t* __restrict__ src_gm,
    uint32_t data_size)
{
    uint32_t LowMask = 0x0000000f;
    uint32_t HighMask = 0x000fff0f;
    uint32_t dstSize = data_size * 2;
    int16_t Offset = 3;
    float Scale = 1.0;
    TPipe tpipe;
    GlobalTensor<T> input_global;
    GlobalTensor<U> dst_global;

    input_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src_gm), data_size);
    dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(dst_gm), dstSize);

    LocalTensor<T> input_local;
    TBuffAddr tbuf;
    tbuf.logicPos = (uint8_t)TPosition::VECCALC;
    input_local.SetAddr(tbuf);
    input_local.InitBuffer(0, data_size);

    LocalTensor<U> dst_local;
    TBuffAddr tbuf2;
    tbuf2.logicPos = (uint8_t)TPosition::VECCALC;
    dst_local.SetAddr(tbuf2);
    dst_local.InitBuffer(0, dstSize);
    DataCopy(input_local, input_global, data_size);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if constexpr (TEST_MODE == LEVEL0_BIT_MODE) {
        uint8_t repeatTimes = 2;
        UnaryRepeatParams repeatParams{8, 8, 1, 1};
        LocalTensor<uint64_t> vecDeqLocal;
        for (uint16_t i = 0; i < dstSize; i++) {
            dst_local.SetValue(i, 0);
        }
        PopStackBuffer<uint64_t, TPosition::LCM>(vecDeqLocal);
        float vdeqScale[VDEQ_TENSOR_SIZE] = {0};
        int16_t vdeqOffset[VDEQ_TENSOR_SIZE] = {0};
        bool vdeqSignMode[VDEQ_TENSOR_SIZE] = {0};
        uint64_t mask[2] = {LowMask, HighMask};
        if constexpr (IS_COUNTER_MODE) {
            SetMaskCount();
        }
        if constexpr (!IS_SET_MASK) {
            if constexpr (IS_COUNTER_MODE) {
                SetVectorMask<T, AscendC::MaskMode::COUNTER>(HighMask, LowMask);
            } else {
                SetVectorMask<T, AscendC::MaskMode::NORMAL>(HighMask, LowMask);
            }
        }
        if constexpr (std::is_same<U, int8_t>::value) {
            if constexpr (IS_VEC_DEQ) {
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
            if constexpr (IS_VEC_DEQ) {
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
            CastDeq<U, T, IS_SET_MASK, IS_VEC_DEQ, HALF_BLOCK>(dst_local, input_local, mask, repeatTimes, repeatParams);
        else
            CastDequant<U, T, IS_SET_MASK, IS_VEC_DEQ, HALF_BLOCK>(dst_local, input_local, mask, repeatTimes, repeatParams);
        ResetMask();
        SetMaskNorm();
    } else if constexpr (TEST_MODE == LEVEL0_COUNT_MODE) {
        uint8_t repeatTimes = 2;
        UnaryRepeatParams repeatParams{8, 8, 1, 1};
        LocalTensor<uint64_t> vecDeqLocal;
        for (uint16_t i = 0; i < dstSize; i++) {
            dst_local.SetValue(i, 0);
        }
        PopStackBuffer<uint64_t, TPosition::LCM>(vecDeqLocal);
        float vdeqScale[VDEQ_TENSOR_SIZE] = {0};
        int16_t vdeqOffset[VDEQ_TENSOR_SIZE] = {0};
        bool vdeqSignMode[VDEQ_TENSOR_SIZE] = {0};
        uint64_t mask = LowMask;
        if constexpr (IS_COUNTER_MODE) {
            SetMaskCount();
        }
        if constexpr (!IS_SET_MASK) {
            if constexpr (IS_COUNTER_MODE) {
                SetVectorMask<T, AscendC::MaskMode::COUNTER>(mask);
            } else {
                SetVectorMask<T, AscendC::MaskMode::NORMAL>(mask);
            }
        }
        if constexpr (std::is_same<U, int8_t>::value) {
            if constexpr (IS_VEC_DEQ) {
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
            if constexpr (IS_VEC_DEQ) {
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
            CastDeq<U, T, IS_SET_MASK, IS_VEC_DEQ, HALF_BLOCK>(dst_local, input_local, mask, repeatTimes, repeatParams);
        else
            CastDequant<U, T, IS_SET_MASK, IS_VEC_DEQ, HALF_BLOCK>(dst_local, input_local, mask, repeatTimes, repeatParams);
        ResetMask();
        SetMaskNorm();
    } else {
        LocalTensor<uint64_t> vecDeqLocal;
        PopStackBuffer<uint64_t, TPosition::LCM>(vecDeqLocal);
        float vdeqScale[VDEQ_TENSOR_SIZE] = {0};
        int16_t vdeqOffset[VDEQ_TENSOR_SIZE] = {0};
        bool vdeqSignMode[VDEQ_TENSOR_SIZE] = {0};
        for (uint16_t i = 0; i < dstSize; i++) {
            dst_local.SetValue(i, 0);
        }
        if constexpr (std::is_same<U, int8_t>::value) {
            if constexpr (IS_VEC_DEQ) {
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
            if constexpr (IS_VEC_DEQ) {
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
            CastDeq<U, T, IS_VEC_DEQ, HALF_BLOCK>(dst_local, input_local, LowMask);
        else
            CastDequant<U, T, IS_VEC_DEQ, HALF_BLOCK>(dst_local, input_local, LowMask);
    }
        DataCopy(dst_global, dst_local, dstSize);

    
}

#define VEC_CAST_EQ_TESTCASE(T , U, TEST_MODE, IS_SET_MASK, IS_COUNTER_MODE, IS_VEC_DEQ, HALF_BLOCK)                                                \
    TEST_F(TEST_CAST_DEQ, CastDeq##T##U##RELATION_OP##TEST_MODE##IS_SET_MASK##IS_COUNTER_MODE##IS_VEC_DEQ##HALF_BLOCK##Case)                                 \
    {                                                                                                      \
        uint32_t data_size = 2048;                                                                         \
        uint32_t sel_mask_size = data_size / AscendCUtils::GetBitSize(sizeof(uint8_t));                    \
        uint8_t input_gm[data_size * sizeof(T)];                                                           \
        uint8_t output_gm[data_size * 2 * sizeof(U)];                                                      \
        main_vec_cast_deq_demo<T, U, TEST_MODE, IS_SET_MASK, IS_COUNTER_MODE, IS_VEC_DEQ, HALF_BLOCK, true>(output_gm, input_gm, data_size);                                                                     \
        for (uint32_t i = 0; i < sel_mask_size; i++) {                                                     \
            EXPECT_EQ(output_gm[i], 0x00);                                                                 \
        }                                                                                                  \
    }                                                                                                      \
    TEST_F(TEST_CAST_DEQ, CastDequant##T##U##RELATION_OP##TEST_MODE##IS_SET_MASK##IS_COUNTER_MODE##IS_VEC_DEQ##HALF_BLOCK##Case)                                 \
    {                                                                                                      \
        uint32_t data_size = 2048;                                                                         \
        uint32_t sel_mask_size = data_size / AscendCUtils::GetBitSize(sizeof(uint8_t));                    \
        uint8_t input_gm[data_size * sizeof(T)];                                                           \
        uint8_t output_gm[data_size * 2 * sizeof(U)];                                                      \
        main_vec_cast_deq_demo<T, U, TEST_MODE, IS_SET_MASK, IS_COUNTER_MODE, IS_VEC_DEQ, HALF_BLOCK, false>(output_gm, input_gm, data_size);                                                                     \
        for (uint32_t i = 0; i < sel_mask_size; i++) {                                                     \
            EXPECT_EQ(output_gm[i], 0x00);                                                                 \
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