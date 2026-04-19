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

template <typename T, TestMode TEST_MODE, bool WITH_DST>
void main_vec_compare_demo(__gm__ uint8_t* __restrict__ sel_mask_gm, __gm__ uint8_t* __restrict__ src0_gm,
    __gm__ uint8_t* __restrict__ src1_gm, CMPMODE cmp_mode, uint32_t data_size, uint32_t sel_mask_size)
{
    TPipe tpipe;
    GlobalTensor<T> input0_global;
    GlobalTensor<T> input1_global;
    GlobalTensor<uint8_t> sel_mask_global;
    input0_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0_gm), data_size);
    input1_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1_gm), data_size);
    sel_mask_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(sel_mask_gm), data_size);

    LocalTensor<T> input0_local;
    TBuffAddr tbuf;
    tbuf.logicPos = (uint8_t)TPosition::VECCALC;
    input0_local.SetAddr(tbuf);
    input0_local.InitBuffer(0, data_size);

    LocalTensor<T> input1_local;
    TBuffAddr tbuf1;
    tbuf1.logicPos = (uint8_t)TPosition::VECCALC;
    input1_local.SetAddr(tbuf1);
    input1_local.InitBuffer(input0_local.GetSize() * sizeof(T), data_size);

    LocalTensor<uint8_t> sel_mask_local;
    TBuffAddr tbuf2;
    tbuf2.logicPos = (uint8_t)TPosition::VECCALC;
    sel_mask_local.SetAddr(tbuf2);
    sel_mask_local.InitBuffer(input0_local.GetSize() * sizeof(T) + input1_local.GetSize() * sizeof(T), sel_mask_size);

    DataCopy(input0_local, input0_global, data_size);
    DataCopy(input1_local, input1_global, data_size);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if constexpr(WITH_DST) {
        if constexpr (TEST_MODE == LEVEL2) {
            Compare(sel_mask_local, input0_local, input1_local, cmp_mode, data_size);
        } else if constexpr (TEST_MODE == LEVEL0_BIT_MODE) {
            uint8_t repeatTime = data_size * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask[2];
            uint64_t counterMask[2] = {144, 0};
            if (sizeof(T) == 2) {
                mask[0] = UINT64_MAX;
                mask[1] = UINT64_MAX;
            } else if (sizeof(T) == 4) {
                mask[0] = UINT64_MAX;
                mask[1] = 0;
            }
            Compare(sel_mask_local, input0_local, input1_local, cmp_mode, mask, repeatTime, { 0, 1, 1, 0, 8, 8 });
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<T, MaskMode::COUNTER>(0, 144);
            Compare<T, uint8_t ,false>(sel_mask_local, input0_local, input1_local, cmp_mode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 1, 0, 8, 8 });
            AscendC::ResetMask();
            Compare(sel_mask_local, input0_local, input1_local, cmp_mode, counterMask, repeatTime, { 0, 1, 1, 0, 8, 8 });
            AscendC::SetMaskNorm();
        } else if constexpr (TEST_MODE == LEVEL0_COUNT_MODE) {
            uint8_t repeatTime = data_size * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask = 0;
            if (sizeof(T) == 2) {
                mask = 128;
            } else if (sizeof(T) == 4) {
                mask = 64;
            }
            Compare(sel_mask_local, input0_local, input1_local, cmp_mode, mask, repeatTime, { 0, 1, 1, 0, 8, 8 });
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<T, MaskMode::COUNTER>(0, 144);
            Compare<T, uint8_t ,false>(sel_mask_local, input0_local, input1_local, cmp_mode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 1, 0, 8, 8 });
            AscendC::ResetMask();
            Compare(sel_mask_local, input0_local, input1_local, cmp_mode, mask, repeatTime, { 0, 1, 1, 0, 8, 8 });
            AscendC::SetMaskNorm();
        } else if constexpr (TEST_MODE == LEVEL3) {
            if (cmp_mode == CMPMODE::EQ) {
                sel_mask_local = input0_local == input1_local;
            } else if (cmp_mode == CMPMODE::GE) {
                sel_mask_local = input0_local >= input1_local;
            }else if (cmp_mode == CMPMODE::GT) {
                sel_mask_local = input0_local > input1_local;
            }else if (cmp_mode == CMPMODE::LE) {
                sel_mask_local = input0_local <= input1_local;
            }else if (cmp_mode == CMPMODE::LT) {
                sel_mask_local = input0_local < input1_local;
            }else if (cmp_mode == CMPMODE::NE) {
                sel_mask_local = input0_local != input1_local;
            }
        }
    } else {
        uint8_t repeatTime = data_size * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
        uint64_t mask[2];
        uint64_t counterMask[2] = {144, 0};
        if (sizeof(T) == 2) {
            mask[0] = UINT64_MAX;
            mask[1] = UINT64_MAX;
        } else if (sizeof(T) == 4) {
            mask[0] = UINT64_MAX;
            mask[1] = 0;
        }
        Compare(input0_local, input1_local, cmp_mode, mask, { 0, 1, 1, 0, 8, 8 });
        GetCmpMask(sel_mask_local);
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<T, MaskMode::COUNTER>(0, 144);
        Compare<T, false>(input0_local, input1_local, cmp_mode, AscendC::MASK_PLACEHOLDER_LIST, { 0, 1, 1, 0, 8, 8 });
        GetCmpMask(sel_mask_local);
        AscendC::ResetMask();
        Compare(input0_local, input1_local, cmp_mode, counterMask, { 0, 1, 1, 0, 8, 8 });
        GetCmpMask(sel_mask_local);
        AscendC::SetMaskNorm();

        Compare(input0_local, input1_local, cmp_mode, counterMask[0], { 0, 1, 1, 0, 8, 8 });
        GetCmpMask(sel_mask_local);
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<T, MaskMode::COUNTER>(0, 144);
        Compare<T, false>(input0_local, input1_local, cmp_mode, AscendC::MASK_PLACEHOLDER, { 0, 1, 1, 0, 8, 8 });
        GetCmpMask(sel_mask_local);
        AscendC::ResetMask();
        Compare(input0_local, input1_local, cmp_mode, counterMask[0], { 0, 1, 1, 0, 8, 8 });
        GetCmpMask(sel_mask_local);
        AscendC::SetMaskNorm();
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(sel_mask_global, sel_mask_local, sel_mask_size);

    pipe_barrier(PIPE_ALL);
}

#define VEC_CMP_TESTCASE(DATA_TYPE, RELATION_OP, TEST_MODE, WITH_DST)                                                \
    TEST_F(TEST_COMPARE, Compare##DATA_TYPE##RELATION_OP##TEST_MODE##WITH_DST##Case)                                 \
    {                                                                                                      \
        uint32_t data_size = 256;                                                                          \
        uint32_t sel_mask_size = data_size / AscendCUtils::GetBitSize(sizeof(uint8_t));                    \
        uint8_t input0_gm[data_size * sizeof(DATA_TYPE)];                                                  \
        uint8_t input1_gm[data_size * sizeof(DATA_TYPE)];                                                  \
        uint8_t output_gm[data_size];                                                                      \
                                                                                                           \
        main_vec_compare_demo<DATA_TYPE, TEST_MODE, WITH_DST>(output_gm, input0_gm, input1_gm, CMPMODE::RELATION_OP, data_size, \
            sel_mask_size);                                                                     \
                                                                                                           \
        for (uint32_t i = 0; i < sel_mask_size; i++) {                                                     \
            EXPECT_EQ(output_gm[i], 0x00);                                                                 \
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

class TEST_COMPARES : public testing::Test {
protected:
    void SetUp()
    {
        SetGCoreType(2);
    }
    void TearDown()
    {
        GlobalMockObject::reset();
        SetGCoreType(0);
    }
};

template <typename T, TestMode TEST_MODE>
void main_vec_compare_scalar_demo(__gm__ uint8_t* __restrict__ sel_mask_gm, __gm__ uint8_t* __restrict__ src0_gm,
    __gm__ uint8_t* __restrict__ src1_gm, CMPMODE cmp_mode, uint32_t data_size, uint32_t sel_mask_size, bool isCompares)
{
    TPipe tpipe;
    GlobalTensor<T> input0_global;

    GlobalTensor<uint8_t> sel_mask_global;
    input0_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0_gm), data_size);
    sel_mask_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(sel_mask_gm), data_size);

    LocalTensor<T> input0_local;
    TBuffAddr tbuf;
    tbuf.logicPos = (uint8_t)TPosition::VECCALC;
    input0_local.SetAddr(tbuf);
    input0_local.InitBuffer(0, data_size);

    T scalar = 1;
    LocalTensor<uint8_t> sel_mask_local;
    TBuffAddr tbuf2;
    tbuf2.logicPos = (uint8_t)TPosition::VECCALC;
    sel_mask_local.SetAddr(tbuf2);
    sel_mask_local.InitBuffer(input0_local.GetSize() * sizeof(T), sel_mask_size);

    DataCopy(input0_local, input0_global, data_size);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if (isCompares){
        if constexpr (TEST_MODE == LEVEL2) {
            Compares(sel_mask_local, input0_local, scalar, cmp_mode, data_size);
            Compares(sel_mask_local, scalar, input0_local, cmp_mode, data_size);
        } else if constexpr (TEST_MODE == LEVEL0_BIT_MODE) {
            uint8_t repeatTime = data_size * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask[2];
            uint64_t counterMask[2] = {144, 0};
            if (sizeof(T) == 2) {
                mask[0] = UINT64_MAX;
                mask[1] = UINT64_MAX;
            } else if (sizeof(T) == 4) {
                mask[0] = UINT64_MAX;
                mask[1] = 0;
            }
            Compares(sel_mask_local, input0_local, scalar, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, scalar, input0_local, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, (T)0, input0_local, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, input0_local, (T)0, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, input0_local, input0_local[0], cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, input0_local[0], input0_local, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<T, MaskMode::COUNTER>(0, 144);
            Compares<T, uint8_t ,false>(sel_mask_local, input0_local, scalar, cmp_mode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(sel_mask_local, scalar, input0_local, cmp_mode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(sel_mask_local, (T)0, input0_local, cmp_mode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(sel_mask_local, input0_local, (T)0, cmp_mode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(sel_mask_local, input0_local, input0_local[0], cmp_mode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(sel_mask_local, input0_local[0], input0_local, cmp_mode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            AscendC::ResetMask();
            Compares(sel_mask_local, input0_local, scalar, cmp_mode, counterMask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, scalar, input0_local, cmp_mode, counterMask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, (T)0, input0_local, cmp_mode, counterMask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, input0_local, (T)0, cmp_mode, counterMask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, input0_local, input0_local[0], cmp_mode, counterMask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, input0_local[0], input0_local, cmp_mode, counterMask, repeatTime, { 0, 1, 0, 8 });
            AscendC::SetMaskNorm();
        } else if constexpr (TEST_MODE == LEVEL0_COUNT_MODE) {
            uint8_t repeatTime = data_size * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask = 0;
            if (sizeof(T) == 2) {
                mask = 128;
            } else if (sizeof(T) == 4) {
                mask = 64;
            }
            Compares(sel_mask_local, input0_local, scalar, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, scalar, input0_local, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, (T)0, input0_local, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, input0_local, (T)0, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, input0_local, input0_local[0], cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, input0_local[0], input0_local, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });

            constexpr static BinaryConfig config = {1};
            Compares<T, uint8_t, false, config>(sel_mask_local, input0_local, scalar, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            Select<T, uint8_t, false, config>(input0_local, sel_mask_local, input0_local, scalar, SELMODE::VSEL_TENSOR_SCALAR_MODE, mask, 1, { 1, 1, 1, 8, 8, 8});

            AscendC::SetMaskCount();
            AscendC::SetVectorMask<T, MaskMode::COUNTER>(0, 144);
            Compares<T, uint8_t ,false>(sel_mask_local, input0_local, scalar, cmp_mode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(sel_mask_local, scalar, input0_local, cmp_mode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(sel_mask_local, (T)0, input0_local, cmp_mode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(sel_mask_local, input0_local, (T)0, cmp_mode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(sel_mask_local, input0_local, input0_local[0], cmp_mode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            Compares<T, uint8_t ,false>(sel_mask_local, input0_local[0], input0_local, cmp_mode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            AscendC::ResetMask();
            Compares(sel_mask_local, input0_local, scalar, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, scalar, input0_local, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, (T)0, input0_local, cmp_mode,  mask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, input0_local, (T)0, cmp_mode,  mask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, input0_local, input0_local[0], cmp_mode,  mask, repeatTime, { 0, 1, 0, 8 });
            Compares(sel_mask_local, input0_local[0], input0_local, cmp_mode,  mask, repeatTime, { 0, 1, 0, 8 });
            AscendC::SetMaskNorm();
        }
    }else{
        if constexpr (TEST_MODE == LEVEL2) {
            CompareScalar(sel_mask_local, input0_local, scalar, cmp_mode, data_size);
            CompareScalar(sel_mask_local, scalar, input0_local, cmp_mode, data_size);
        } else if constexpr (TEST_MODE == LEVEL0_BIT_MODE) {
            uint8_t repeatTime = data_size * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask[2];
            uint64_t counterMask[2] = {144, 0};
            if (sizeof(T) == 2) {
                mask[0] = UINT64_MAX;
                mask[1] = UINT64_MAX;
            } else if (sizeof(T) == 4) {
                mask[0] = UINT64_MAX;
                mask[1] = 0;
            }
            CompareScalar(sel_mask_local, input0_local, scalar, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, scalar, input0_local, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, (T)0, input0_local, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, input0_local, (T)0, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, input0_local, input0_local[0], cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, input0_local[0], input0_local, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<T, MaskMode::COUNTER>(0, 144);
            CompareScalar<T, uint8_t ,false>(sel_mask_local, input0_local, scalar, cmp_mode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(sel_mask_local, scalar, input0_local, cmp_mode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(sel_mask_local, (T)0, input0_local, cmp_mode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(sel_mask_local, input0_local, (T)0, cmp_mode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(sel_mask_local, input0_local, input0_local[0], cmp_mode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(sel_mask_local, input0_local[0], input0_local, cmp_mode, AscendC::MASK_PLACEHOLDER_LIST, repeatTime, { 0, 1, 0, 8 });
            AscendC::ResetMask();
            CompareScalar(sel_mask_local, input0_local, scalar, cmp_mode, counterMask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, scalar, input0_local, cmp_mode, counterMask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, (T)0, input0_local, cmp_mode, counterMask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, input0_local, (T)0, cmp_mode, counterMask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, input0_local, input0_local[0], cmp_mode, counterMask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, input0_local[0], input0_local, cmp_mode, counterMask, repeatTime, { 0, 1, 0, 8 });
            AscendC::SetMaskNorm();
        } else if constexpr (TEST_MODE == LEVEL0_COUNT_MODE) {
            uint8_t repeatTime = data_size * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask = 0;
            if (sizeof(T) == 2) {
                mask = 128;
            } else if (sizeof(T) == 4) {
                mask = 64;
            }
            CompareScalar(sel_mask_local, input0_local, scalar, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, scalar, input0_local, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, (T)0, input0_local, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, input0_local, (T)0, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, input0_local, input0_local[0], cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, input0_local[0], input0_local, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });

            constexpr static BinaryConfig config = {1};
            CompareScalar<T, uint8_t, false, config>(sel_mask_local, input0_local, scalar, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            Select<T, uint8_t, false, config>(input0_local, sel_mask_local, input0_local, scalar, SELMODE::VSEL_TENSOR_SCALAR_MODE, mask, 1, { 1, 1, 1, 8, 8, 8});

            AscendC::SetMaskCount();
            AscendC::SetVectorMask<T, MaskMode::COUNTER>(0, 144);
            CompareScalar<T, uint8_t ,false>(sel_mask_local, input0_local, scalar, cmp_mode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(sel_mask_local, scalar, input0_local, cmp_mode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(sel_mask_local, (T)0, input0_local, cmp_mode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(sel_mask_local, input0_local, (T)0, cmp_mode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(sel_mask_local, input0_local, input0_local[0], cmp_mode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            CompareScalar<T, uint8_t ,false>(sel_mask_local, input0_local[0], input0_local, cmp_mode, AscendC::MASK_PLACEHOLDER, repeatTime, { 0, 1, 0, 8 });
            AscendC::ResetMask();
            CompareScalar(sel_mask_local, input0_local, scalar, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, scalar, input0_local, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, (T)0, input0_local, cmp_mode,  mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, input0_local, (T)0, cmp_mode,  mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, input0_local, input0_local[0], cmp_mode,  mask, repeatTime, { 0, 1, 0, 8 });
            CompareScalar(sel_mask_local, input0_local[0], input0_local, cmp_mode,  mask, repeatTime, { 0, 1, 0, 8 });
            AscendC::SetMaskNorm();
        }
    }


    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(sel_mask_global, sel_mask_local, sel_mask_size);

    pipe_barrier(PIPE_ALL);
}


#define VEC_CMPS_TESTCASE(DATA_TYPE, RELATION_OP, TEST_MODE)                                               \
    TEST_F(TEST_COMPARE_SCALAR, CompareScalar##DATA_TYPE##RELATION_OP##TEST_MODE##Case)                    \
    {                                                                                                      \
        uint32_t data_size = 256;                                                                          \
        uint32_t sel_mask_size = data_size / AscendCUtils::GetBitSize(sizeof(uint8_t));                    \
        uint8_t input0_gm[data_size * sizeof(DATA_TYPE)];                                                  \
        uint8_t input1_gm[data_size * sizeof(DATA_TYPE)];                                                  \
        uint8_t output_gm[data_size];                                                                      \
                                                                                                           \
        main_vec_compare_scalar_demo<DATA_TYPE, TEST_MODE>(output_gm, input0_gm, input1_gm, CMPMODE::RELATION_OP,     \
            data_size, sel_mask_size, false);                                                              \
                                                                                                           \
        for (uint32_t i = 0; i < sel_mask_size; i++) {                                                     \
            EXPECT_EQ(output_gm[i], 0x00);                                                                 \
        }                                                                                                  \
    }                                                                                                      \
    TEST_F(TEST_COMPARES, Compares##DATA_TYPE##RELATION_OP##TEST_MODE##Case)                               \
    {                                                                                                      \
        uint32_t data_size = 256;                                                                          \
        uint32_t sel_mask_size = data_size / AscendCUtils::GetBitSize(sizeof(uint8_t));                    \
        uint8_t input0_gm[data_size * sizeof(DATA_TYPE)];                                                  \
        uint8_t input1_gm[data_size * sizeof(DATA_TYPE)];                                                  \
        uint8_t output_gm[data_size];                                                                      \
                                                                                                           \
        main_vec_compare_scalar_demo<DATA_TYPE, TEST_MODE>(output_gm, input0_gm, input1_gm, CMPMODE::RELATION_OP,     \
            data_size, sel_mask_size, true);                                                               \
                                                                                                           \
        for (uint32_t i = 0; i < sel_mask_size; i++) {                                                     \
            EXPECT_EQ(output_gm[i], 0x00);                                                                 \
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



template <typename T, TestMode TEST_MODE>
void main_vec_compare_scalar_demo_wrong(__gm__ uint8_t* __restrict__ sel_mask_gm, __gm__ uint8_t* __restrict__ src0_gm,
    __gm__ uint8_t* __restrict__ src1_gm, CMPMODE cmp_mode, uint32_t data_size, uint32_t sel_mask_size, bool isCompares)
{
    TPipe tpipe;
    GlobalTensor<T> input0_global;

    GlobalTensor<uint8_t> sel_mask_global;
    input0_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0_gm), data_size);
    sel_mask_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(sel_mask_gm), data_size);

    LocalTensor<T> input0_local;
    TBuffAddr tbuf;
    tbuf.logicPos = (uint8_t)TPosition::VECCALC;
    input0_local.SetAddr(tbuf);
    input0_local.InitBuffer(0, data_size);

    T scalar = 1;
    LocalTensor<uint8_t> sel_mask_local;
    TBuffAddr tbuf2;
    tbuf2.logicPos = (uint8_t)TPosition::VECCALC;
    sel_mask_local.SetAddr(tbuf2);
    sel_mask_local.InitBuffer(input0_local.GetSize() * sizeof(T), sel_mask_size);

    DataCopy(input0_local, input0_global, data_size);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if (isCompares) {
        if constexpr (TEST_MODE == LEVEL2) {
            // 覆盖重载 : Compares<T, U>(dst, src0, scalar, cmpMode, count)
            Compares(sel_mask_local, input0_local, scalar, cmp_mode, data_size);
            
        } else if constexpr (TEST_MODE == LEVEL0_BIT_MODE) {
            uint8_t repeatTime = data_size * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask[2];
            if (sizeof(T) == 2) {
                mask[0] = UINT64_MAX;
                mask[1] = UINT64_MAX;
            } else if (sizeof(T) == 4) {
                mask[0] = UINT64_MAX;
                mask[1] = 0;
            }
            // 覆盖重载 : Compares<T, U, isSetMask>(dst, src0, scalar, cmpMode, mask[], repeatTime, params)
            Compares(sel_mask_local, input0_local, scalar, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            
        } else if constexpr (TEST_MODE == LEVEL0_COUNT_MODE) {
            uint8_t repeatTime = data_size * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask = 0;
            if (sizeof(T) == 2) {
                mask = 128;
            } else if (sizeof(T) == 4) {
                mask = 64;
            }
            // 覆盖重载 : Compares<T, U, isSetMask>(dst, src0, scalar, cmpMode, mask, repeatTime, params)
            Compares(sel_mask_local, input0_local, scalar, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
        }
    }else{
        if constexpr (TEST_MODE == LEVEL2) {
            // 覆盖重载 : CompareScalar<T, U>(dst, src0, scalar, cmpMode, count)
            CompareScalar(sel_mask_local, input0_local, scalar, cmp_mode, data_size);
            
        } else if constexpr (TEST_MODE == LEVEL0_BIT_MODE) {
            uint8_t repeatTime = data_size * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask[2];
            if (sizeof(T) == 2) {
                mask[0] = UINT64_MAX;
                mask[1] = UINT64_MAX;
            } else if (sizeof(T) == 4) {
                mask[0] = UINT64_MAX;
                mask[1] = 0;
            }
            // 覆盖重载 : CompareScalar<T, U, isSetMask>(dst, src0, scalar, cmpMode, mask[], repeatTime, params)
            CompareScalar(sel_mask_local, input0_local, scalar, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
            
        } else if constexpr (TEST_MODE == LEVEL0_COUNT_MODE) {
            uint8_t repeatTime = data_size * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask = 0;
            if (sizeof(T) == 2) {
                mask = 128;
            } else if (sizeof(T) == 4) {
                mask = 64;
            }
            // 覆盖重载 : CompareScalar<T, U, isSetMask>(dst, src0, scalar, cmpMode, mask, repeatTime, params)
            CompareScalar(sel_mask_local, input0_local, scalar, cmp_mode, mask, repeatTime, { 0, 1, 0, 8 });
        }
    }
    
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(sel_mask_global, sel_mask_local, sel_mask_size);

    pipe_barrier(PIPE_ALL);
}

#define VEC_CMPS_TESTCASE_WRONG(DATA_TYPE, RELATION_OP, TEST_MODE)                                         \
    TEST_F(TEST_COMPARES, ComparesWrong##DATA_TYPE##RELATION_OP##TEST_MODE##Case)                          \
    {                                                                                                      \
        uint32_t data_size = 256;                                                                          \
        uint32_t sel_mask_size = data_size / AscendCUtils::GetBitSize(sizeof(uint8_t));                    \
        uint8_t input0_gm[data_size * sizeof(DATA_TYPE)];                                                  \
        uint8_t input1_gm[data_size * sizeof(DATA_TYPE)];                                                  \
        uint8_t output_gm[data_size];                                                                      \
        MOCKER(CheckFuncVecBinaryScalarCmp, bool(*)(const LocalTensor<uint8_t>&, const LocalTensor<DATA_TYPE>&, const DATA_TYPE&, const uint64_t, const uint8_t, const UnaryRepeatParams&, const char*)) \
            .stubs()                                                                                      \
            .will(returnValue(false));                                                                     \
        MOCKER(CheckFuncVecBinaryScalarCmp, bool(*)(const LocalTensor<uint8_t>&, const LocalTensor<DATA_TYPE>&, DATA_TYPE, const int32_t&, const char*)) \
            .stubs()                                                                                      \
            .will(returnValue(false));                                                                     \
        MOCKER(raise, int(*)(int)).times(6).will(returnValue(0));                                          \
        main_vec_compare_scalar_demo_wrong<DATA_TYPE, TEST_MODE>(output_gm, input0_gm, input1_gm, CMPMODE::RELATION_OP,     \
            data_size, sel_mask_size, true);                                                               \
        for (uint32_t i = 0; i < sel_mask_size; i++) {                                                     \
            EXPECT_EQ(output_gm[i], 0x00);                                                                 \
        }                                                                                                  \
    }

VEC_CMPS_TESTCASE_WRONG(float, LT, LEVEL2);
VEC_CMPS_TESTCASE_WRONG(float, LT, LEVEL0_COUNT_MODE);
VEC_CMPS_TESTCASE_WRONG(float, LT, LEVEL0_BIT_MODE);
