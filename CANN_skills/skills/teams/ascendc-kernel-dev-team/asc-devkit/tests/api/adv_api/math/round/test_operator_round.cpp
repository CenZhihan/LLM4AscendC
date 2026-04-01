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
    CAL_MODE,
    BUF_CAL_MODE
};

class TEST_ROUND : public testing::Test {
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
void main_vec_round_level2_demo(__gm__ uint8_t* __restrict__ dstGm,
    __gm__ uint8_t* __restrict__ src0Gm, uint32_t dataSize,TestMode testMode)
{
    TPipe tPipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<T> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tPipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> input0Local = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tPipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    DataCopy(input0Local, input0Global, dataSize);

    auto eventIdMTE2ToV = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    if (testMode == CAL_MODE) {
        Round<T>(outputLocal, input0Local, dataSize);
    } else {
        TBuf<TPosition::VECCALC> tbuf2;
        if (sizeof(T) == sizeof(float)) {
            tPipe.InitBuffer(tbuf2, 128 * sizeof(float) * sizeof(uint8_t));
        } else {
            tPipe.InitBuffer(tbuf2, 128 * sizeof(half) * sizeof(uint8_t));
        }
        LocalTensor<uint8_t> tmpLocal = tbuf2.Get<uint8_t>();
        Round<T>(outputLocal, input0Local, tmpLocal, dataSize);
    }
    auto eventIdVToMTE3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    DataCopy(outputGlobal, outputLocal, dataSize);

    PipeBarrier<PIPE_ALL>();
}

#define VEC_ROUND_LEVEL2_TESTCASE(DATA_TYPE, TEST_MODE)                                              \
    TEST_F(TEST_ROUND, Round##DATA_TYPE##TEST_MODE##Case)                                           \
    {                                                                                                           \
        uint32_t dataSize = 1024;                                                                               \
        uint32_t sel_mask_size = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));                         \
        uint8_t input0Gm[dataSize * sizeof(DATA_TYPE)];                                                       \
        uint8_t outputGm[dataSize * sizeof(DATA_TYPE)];                                                       \
                                                                                                                \
        main_vec_round_level2_demo<DATA_TYPE>(outputGm, input0Gm, dataSize, TEST_MODE);                     \
                                                                                                                \
        for (uint32_t i = 0; i < dataSize; i++) {                                                              \
            EXPECT_EQ(outputGm[i], 0x00);                                                                      \
        }                                                                                                       \
    }
VEC_ROUND_LEVEL2_TESTCASE(half, CAL_MODE);
VEC_ROUND_LEVEL2_TESTCASE(half, BUF_CAL_MODE);
VEC_ROUND_LEVEL2_TESTCASE(float, CAL_MODE);
VEC_ROUND_LEVEL2_TESTCASE(float, BUF_CAL_MODE);