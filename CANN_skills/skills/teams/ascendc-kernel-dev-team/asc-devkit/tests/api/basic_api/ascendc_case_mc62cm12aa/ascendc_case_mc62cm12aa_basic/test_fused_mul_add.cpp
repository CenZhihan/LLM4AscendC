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

class TEST_FUSED_MUL_ADD : public testing::Test {
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

template <typename T, bool USE_FUSED = true>
void main_vec_fused_mul_add_level2_demo(__gm__ uint8_t* __restrict__ dst_gm,
    __gm__ uint8_t* __restrict__ src0_gm, uint32_t data_size)
{
    BinaryRepeatParams repeatParams;
    TPipe tpipe;
    GlobalTensor<T> input0_global;
    GlobalTensor<T> output_global;
    input0_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0_gm), data_size);
    output_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst_gm), data_size);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, data_size * sizeof(T));
    LocalTensor<T> input0_local = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, data_size * sizeof(T));
    LocalTensor<T> output_local = tbuf1.Get<T>();

    DataCopy(input0_local, input0_global, data_size);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    uint64_t maskCount = 32;
    uint64_t mask[2] = {0xff, 0};
    FusedMulAdd(output_local, input0_local, input0_local, data_size);
    FusedMulAdd(output_local, input0_local, input0_local, maskCount, sizeof(T), repeatParams);
    FusedMulAdd(output_local, input0_local, input0_local, mask, sizeof(T), repeatParams);
    if constexpr (USE_FUSED) {
        FusedMulAddRelu(output_local, input0_local, input0_local, data_size);
        FusedMulAddRelu(output_local, input0_local, input0_local, maskCount, sizeof(T), repeatParams);
        FusedMulAddRelu(output_local, input0_local, input0_local, mask, sizeof(T), repeatParams);
    } else {
        MulAddRelu(output_local, input0_local, input0_local, data_size);
        MulAddRelu(output_local, input0_local, input0_local, maskCount, sizeof(T), repeatParams);
        MulAddRelu(output_local, input0_local, input0_local, mask, sizeof(T), repeatParams);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(output_global, output_local, data_size);

    pipe_barrier(PIPE_ALL);
}

#define VEC_FUSED_MUL_ADD_LEVEL2_TESTCASE(DATA_TYPE)                                              \
    TEST_F(TEST_FUSED_MUL_ADD, FusedMulAddRelu##DATA_TYPE##Case)                                  \
    {                                                                                              \
        uint32_t data_size = 256;                                                                  \
        uint32_t sel_mask_size = data_size / AscendCUtils::GetBitSize(sizeof(uint8_t));          \
        uint8_t input0_gm[data_size * sizeof(DATA_TYPE)];                                        \
        uint8_t output_gm[data_size * sizeof(DATA_TYPE)];                                        \
                                                                                                   \
        main_vec_fused_mul_add_level2_demo<DATA_TYPE, true>(output_gm, input0_gm, data_size);   \
                                                                                                   \
        for (uint32_t i = 0; i < data_size; i++) {                                               \
            EXPECT_EQ(output_gm[i], 0x00);                                                        \
        }                                                                                          \
    }                                                                                              \
    TEST_F(TEST_FUSED_MUL_ADD, MulAddRelu##DATA_TYPE##Case)                                      \
    {                                                                                              \
        uint32_t data_size = 256;                                                                  \
        uint32_t sel_mask_size = data_size / AscendCUtils::GetBitSize(sizeof(uint8_t));          \
        uint8_t input0_gm[data_size * sizeof(DATA_TYPE)];                                        \
        uint8_t output_gm[data_size * sizeof(DATA_TYPE)];                                        \
                                                                                                   \
        main_vec_fused_mul_add_level2_demo<DATA_TYPE, false>(output_gm, input0_gm, data_size);  \
                                                                                                   \
        for (uint32_t i = 0; i < data_size; i++) {                                               \
            EXPECT_EQ(output_gm[i], 0x00);                                                        \
        }                                                                                          \
    }


VEC_FUSED_MUL_ADD_LEVEL2_TESTCASE(float);
VEC_FUSED_MUL_ADD_LEVEL2_TESTCASE(half);

