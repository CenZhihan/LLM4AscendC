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
    NORMAL_MODE,
    CAL_MODE,
    BUF_MODE,
    BUF_CAL_MODE
};

class TEST_XOR : public testing::Test {
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
void main_vec_xor_level2_demo(__gm__ uint8_t* __restrict__ dstGm,
    __gm__ uint8_t* __restrict__ src0Gm, __gm__ uint8_t* __restrict__ src1Gm, uint32_t dataSize,TestMode testMode)
{
    TPipe tpipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<T> input1Global;
    GlobalTensor<T> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    input1Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src1Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf0;
    tpipe.InitBuffer(tbuf0, dataSize * sizeof(T));
    LocalTensor<T> input0_local = tbuf0.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> input1_local = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T) * 2);
    LocalTensor<T> output_local = tbuf2.Get<T>();

    for (int32_t i = 0; i < dataSize; ++i) {
        input1_local.SetValue(i, 1);
    }

    SetVectorMask<uint8_t, MaskMode::NORMAL>(256);

    DataCopy(input0_local, input0Global, dataSize);
    DataCopy(input1_local, input1Global, dataSize);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetVectorMask<uint8_t, MaskMode::NORMAL>(128);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    if (testMode == NORMAL_MODE) {
        Xor(output_local, input0_local, input1_local);
    } else if (testMode == CAL_MODE) {
        Xor(output_local, input0_local, input1_local, dataSize);
    } else {
        TBuf<TPosition::VECCALC> tbuf3;
        tpipe.InitBuffer(tbuf3, dataSize * sizeof(uint8_t) * 3);
        LocalTensor<uint8_t> tmp_local = tbuf3.Get<uint8_t>();

        if (testMode == BUF_MODE) {
            Xor(output_local, input0_local, input1_local, tmp_local);
        }
        else if (testMode == BUF_CAL_MODE){
            Xor(output_local, input0_local, input1_local, tmp_local, dataSize);
        }
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outputGlobal, output_local, dataSize);

    PipeBarrier<PIPE_ALL>();
}
#define VEC_XOR_LEVEL2_TESTCASE(DATA_TYPE, TEST_MODE)                                                           \
    TEST_F(TEST_XOR, Xor##DATA_TYPE##TEST_MODE##Case)                                                           \
    {                                                                                                           \
        uint32_t dataSize = 256;                                                                               \
        uint32_t sel_mask_size = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));                         \
        uint8_t input0Gm[dataSize * sizeof(DATA_TYPE)];                                                       \
        uint8_t input1Gm[dataSize * sizeof(DATA_TYPE)];                                                       \
        uint8_t outputGm[dataSize * sizeof(DATA_TYPE)];                                                       \
                                                                                                                \
        main_vec_xor_level2_demo<DATA_TYPE>(outputGm, input0Gm, input1Gm, dataSize, TEST_MODE);             \
                                                                                                                \
        for (uint32_t i = 0; i < dataSize; i++) {                                                              \
            EXPECT_EQ(outputGm[i], 0x00);                                                                      \
        }                                                                                                       \
    }
VEC_XOR_LEVEL2_TESTCASE(int16_t, NORMAL_MODE);
VEC_XOR_LEVEL2_TESTCASE(int16_t, CAL_MODE);
VEC_XOR_LEVEL2_TESTCASE(int16_t, BUF_MODE);
VEC_XOR_LEVEL2_TESTCASE(int16_t, BUF_CAL_MODE);
VEC_XOR_LEVEL2_TESTCASE(uint16_t, NORMAL_MODE);
VEC_XOR_LEVEL2_TESTCASE(uint16_t, CAL_MODE);
VEC_XOR_LEVEL2_TESTCASE(uint16_t, BUF_MODE);
VEC_XOR_LEVEL2_TESTCASE(uint16_t, BUF_CAL_MODE);