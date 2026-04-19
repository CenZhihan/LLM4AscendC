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
    MODE_WITH_COUNT,
    MODE_WITH_TMPBUFFER,
};

class TEST_REGLU : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(2);
    }
    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};

template <typename T>
void main_vec_reglu_demo(__gm__ uint8_t* __restrict__ dst_gm, __gm__ uint8_t* __restrict__ src0_gm,
    __gm__ uint8_t* __restrict__ src1_gm, uint32_t data_size, TestMode test_mode)
{
    TPipe tpipe;
    GlobalTensor<T> input0_global;
    GlobalTensor<T> input1_global;
    GlobalTensor<T> output_global;
    input0_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0_gm), data_size);
    input1_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1_gm), data_size);
    output_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst_gm), data_size);

    TBuf<TPosition::VECCALC> tbuf_input0;
    tpipe.InitBuffer(tbuf_input0, data_size * sizeof(T));
    LocalTensor<T> input0_local = tbuf_input0.Get<T>();

    TBuf<TPosition::VECCALC> tbuf_input1;
    tpipe.InitBuffer(tbuf_input1, data_size * sizeof(T));
    LocalTensor<T> input1_local = tbuf_input1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf_output;
    tpipe.InitBuffer(tbuf_output, data_size * sizeof(T));
    LocalTensor<T> output_local = tbuf_output.Get<T>();

    DataCopy(input0_local, input0_global, data_size);
    DataCopy(input1_local, input1_global, data_size);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    if (test_mode == MODE_WITH_COUNT) {
        ReGlu<T, false>(output_local, input0_local, input1_local, data_size);
    } else if (test_mode == MODE_WITH_TMPBUFFER && sizeof(T) == sizeof(half)) {
        TBuf<TPosition::VECCALC> tbuf1;
        tpipe.InitBuffer(tbuf1, data_size * 6 * sizeof(half));
        LocalTensor<uint8_t> tmp_local = tbuf1.Get<uint8_t>();

        ReGlu<T, false>(output_local, input0_local, input1_local, tmp_local, data_size);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(output_global, output_local, data_size);

    PipeBarrier<PIPE_ALL>();
}
#define VEC_REGLU_TESTCASE(DATA_TYPE, TEST_MODE)                                                                \
    TEST_F(TEST_REGLU, ReGlu##DATA_TYPE##TEST_MODE##Case)                                                       \
    {                                                                                                           \
        uint32_t data_size = 256;                                                                               \
        uint32_t sel_mask_size = data_size / AscendCUtils::GetBitSize(sizeof(uint8_t));                         \
        uint8_t input0_gm[data_size * sizeof(DATA_TYPE)];                                                       \
        uint8_t input1_gm[data_size * sizeof(DATA_TYPE)];                                                       \
        uint8_t output_gm[data_size * sizeof(DATA_TYPE)];                                                       \
        main_vec_reglu_demo<DATA_TYPE>(output_gm, input0_gm, input1_gm, data_size, TEST_MODE);                  \
                                                                                                                \
        for (uint32_t i = 0; i < data_size; i++) {                                                              \
            EXPECT_EQ(output_gm[i], 0x00);                                                                      \
        }                                                                                                       \
    }

VEC_REGLU_TESTCASE(float, MODE_WITH_COUNT);
VEC_REGLU_TESTCASE(float, MODE_WITH_TMPBUFFER);
VEC_REGLU_TESTCASE(half, MODE_WITH_COUNT);
VEC_REGLU_TESTCASE(half, MODE_WITH_TMPBUFFER);