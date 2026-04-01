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

class TEST_SELECT : public testing::Test {
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

template <typename T, TestMode TEST_MODE, bool WITH_MASK>
void main_vec_select_level2_demo(__gm__ uint8_t* __restrict__ dst_gm, __gm__ uint8_t* __restrict__ sel_mask_gm,
    __gm__ uint8_t* __restrict__ src0_gm, __gm__ uint8_t* __restrict__ src1_gm, SELMODE sel_mode, uint32_t data_size,
    uint32_t sel_mask_size)
{
    TPipe tpipe;
    GlobalTensor<T> input0_global;
    GlobalTensor<T> input1_global;
    GlobalTensor<uint8_t> sel_mask_global;
    GlobalTensor<T> output_global;
    input0_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0_gm), data_size);
    input1_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1_gm), data_size);
    sel_mask_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(sel_mask_gm), sel_mask_size);
    output_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst_gm), data_size);

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

    LocalTensor<T> output_local;
    TBuffAddr tbuf3;
    tbuf3.logicPos = (uint8_t)TPosition::VECCALC;
    output_local.SetAddr(tbuf3);
    output_local.InitBuffer(input0_local.GetSize() * sizeof(T) + input1_local.GetSize() * sizeof(T) +
        sel_mask_local.GetSize() * sizeof(uint8_t),
        data_size);

    DataCopy(input0_local, input0_global, data_size);
    DataCopy(input1_local, input1_global, data_size);
    DataCopy(sel_mask_local, sel_mask_global, sel_mask_size);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if constexpr (WITH_MASK) {
        if constexpr (TEST_MODE == LEVEL2) {
            Select(output_local, sel_mask_local, input0_local, input1_local, sel_mode, data_size);
        } else if constexpr (TEST_MODE == LEVEL0_BIT_MODE) {
            uint8_t repeatTime = data_size * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask[2];
            uint64_t counterMask[2] = {140, 0};
            if (sizeof(T) == 2) {
                mask[0] = UINT64_MAX;
                mask[1] = UINT64_MAX;
            } else if (sizeof(T) == 4) {
                mask[0] = UINT64_MAX;
                mask[1] = 0;
            }
            if (sel_mode == SELMODE::VSEL_TENSOR_SCALAR_MODE) {
                Select(output_local, sel_mask_local, input0_local, static_cast<T>(0), sel_mode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(output_local, sel_mask_local, static_cast<T>(0), input0_local, sel_mode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(output_local, sel_mask_local, input0_local, input0_local[0], sel_mode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(output_local, sel_mask_local, input0_local[0], input0_local, sel_mode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskCount();
                Select(output_local, sel_mask_local, input0_local, static_cast<T>(0), sel_mode, counterMask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(output_local, sel_mask_local, static_cast<T>(0), input0_local, sel_mode, counterMask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(output_local, sel_mask_local, input0_local, input0_local[0], sel_mode, counterMask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(output_local, sel_mask_local, input0_local[0], input0_local, sel_mode, counterMask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskNorm();
            } else {
                Select(output_local, sel_mask_local, input0_local, input1_local, sel_mode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskCount();
                Select(output_local, sel_mask_local, input0_local, input1_local, sel_mode, counterMask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskNorm();
            }
        } else if constexpr (TEST_MODE == LEVEL0_COUNT_MODE) {
            uint8_t repeatTime = data_size * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
            uint64_t mask = 0;
            if (sizeof(T) == 2) {
                mask = 128;
            } else if (sizeof(T) == 4) {
                mask = 64;
            }
            if (sel_mode == SELMODE::VSEL_TENSOR_SCALAR_MODE) {
                Select(output_local, sel_mask_local, input0_local, static_cast<T>(0), sel_mode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(output_local, sel_mask_local, static_cast<T>(0), input0_local, sel_mode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(output_local, sel_mask_local, input0_local, input0_local[0], sel_mode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(output_local, sel_mask_local, input0_local[0], input0_local, sel_mode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskCount();
                Select(output_local, sel_mask_local, input0_local, static_cast<T>(0), sel_mode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(output_local, sel_mask_local, static_cast<T>(0), input0_local, sel_mode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(output_local, sel_mask_local, input0_local, input0_local[0], sel_mode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                Select(output_local, sel_mask_local, input0_local[0], input0_local, sel_mode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskNorm();
            } else {
                Select(output_local, sel_mask_local, input0_local, input1_local, sel_mode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskCount();
                Select(output_local, sel_mask_local, input0_local, input1_local, sel_mode, mask, repeatTime,
                    { 1, 1, 1, 8, 8, 8 });
                AscendC::SetMaskNorm();
            }
        }
    } else {
        uint8_t repeatTime = data_size * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
        LocalTensor<uint32_t> tmpLocal;
        PopStackBuffer<uint32_t, TPosition::LCM>(tmpLocal);
        AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>((uint64_t)122);
        AscendC::SetMaskCount();
        if (sel_mode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            uint32_t selAddr = static_cast<uint32_t>(
                reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t *>(sel_mask_local.GetPhyAddr())));
            Duplicate(tmpLocal, selAddr, 32);
            SetCmpMask(tmpLocal);
            Select<T, SELMODE::VSEL_TENSOR_TENSOR_MODE>(output_local, input0_local, input1_local, repeatTime, { 1, 1, 1, 8, 8, 8 });
        } else if (sel_mode == SELMODE::VSEL_TENSOR_SCALAR_MODE) {
            SetCmpMask(input1_local);
            Select(output_local, sel_mask_local, input0_local, repeatTime, { 1, 1, 1, 8, 8, 8 });
        } else if (sel_mode == SELMODE::VSEL_CMPMASK_SPR) {
            SetCmpMask(sel_mask_local);
            Select<T, SELMODE::VSEL_CMPMASK_SPR>(output_local, input0_local, input1_local, repeatTime, { 1, 1, 1, 8, 8, 8 });
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
        AscendC::SetVectorMask<T, AscendC::MaskMode::NORMAL>((uint64_t)0x00000000, (uint64_t)0x0000ffff);
        if (sel_mode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
            auto selAddr = reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t *>(sel_mask_local.GetPhyAddr()));
            auto tmpLocalPtr = (int64_t*)tmpLocal.GetPhyAddr();
            *tmpLocalPtr = selAddr;
            SetCmpMask(tmpLocal);
            Select<T, SELMODE::VSEL_TENSOR_TENSOR_MODE>(output_local, input0_local, input1_local, repeatTime, { 1, 1, 1, 8, 8, 8 });
        } else if (sel_mode == SELMODE::VSEL_TENSOR_SCALAR_MODE) {
            SetCmpMask(input1_local);
            Select(output_local, sel_mask_local, input0_local, repeatTime, { 1, 1, 1, 8, 8, 8 });
        } else if (sel_mode == SELMODE::VSEL_CMPMASK_SPR) {
            SetCmpMask(sel_mask_local);
            Select<T, SELMODE::VSEL_CMPMASK_SPR>(output_local, input0_local, input1_local, repeatTime, { 1, 1, 1, 8, 8, 8 });
        }
        
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(output_global, output_local, data_size);

    pipe_barrier(PIPE_ALL);
}

#define VEC_SELECT_LEVEL2_TESTCASE(DATA_TYPE, SEL_MODE, TEST_MODE, WITH_MASK)                                              \
    TEST_F(TEST_SELECT, Select##DATA_TYPE##SEL_MODE##TEST_MODE##WITH_MASK##Case)                                           \
    {                                                                                                           \
        uint32_t data_size = 256;                                                                               \
        uint32_t sel_mask_size = data_size / AscendCUtils::GetBitSize(sizeof(uint8_t));                            \
        uint8_t input0_gm[data_size * sizeof(DATA_TYPE)];                                                       \
        uint8_t input1_gm[data_size * sizeof(DATA_TYPE)];                                                       \
        uint8_t sel_mask_gm[data_size];                                                                         \
        uint8_t output_gm[data_size * sizeof(DATA_TYPE)];                                                       \
                                                                                                                \
        main_vec_select_level2_demo<DATA_TYPE, TEST_MODE, WITH_MASK>(output_gm, sel_mask_gm, input0_gm, input1_gm, SELMODE::SEL_MODE,\
            data_size, sel_mask_size);                                                               \
                                                                                                                \
    }

VEC_SELECT_LEVEL2_TESTCASE(uint64_t, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int64_t, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_CMPMASK_SPR, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, true);

VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, true);

VEC_SELECT_LEVEL2_TESTCASE(uint64_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int64_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int8_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint8_t, VSEL_TENSOR_SCALAR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int64_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint64_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(int8_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(uint8_t, VSEL_TENSOR_TENSOR_MODE, LEVEL2, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, true);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, true);

VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_SCALAR_MODE, LEVEL0_COUNT_MODE, false);

VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_TENSOR_TENSOR_MODE, LEVEL0_COUNT_MODE, false);

VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_CMPMASK_SPR, LEVEL0_BIT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(float, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int32_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint32_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(half, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(int16_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(bfloat16_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, false);
VEC_SELECT_LEVEL2_TESTCASE(uint16_t, VSEL_CMPMASK_SPR, LEVEL0_COUNT_MODE, false);
