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
    POWER_TENSOR_TENSOR,
    POWER_TENSOR_SCALAR,
    POWER_SCALAR_TENSOR
};

enum MemMode {
    NORMAL_MODE,
    BASIC_BLK_MODE,
};

class TEST_POWER : public testing::Test {
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
void MainVecPowerLevel2Demo(__gm__ uint8_t* __restrict__ dstGm,
    __gm__ uint8_t* __restrict__ src0Gm, uint32_t dataSize, MemMode memMode)
{
    TPipe tpipe;
    constexpr uint32_t defaultTmpBuffer = 10240;
    GlobalTensor<T> input0Global;
    GlobalTensor<T> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> input0Local = tbuf1.Get<T>();

    LocalTensor<uint8_t> tmpLocal;

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf2.Get<T>();

    DataCopy(input0Local, input0Global, dataSize);

    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    if (memMode == NORMAL_MODE) {
        Power(outputLocal, input0Local, input0Local);
    } else if (memMode == BASIC_BLK_MODE) {
        TBuf<TPosition::VECCALC> tbuf3;
        tpipe.InitBuffer(tbuf3, defaultTmpBuffer);
        tmpLocal = tbuf3.Get<uint8_t>();
        Power(outputLocal, input0Local, input0Local, tmpLocal);
    }

    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    DataCopy(outputGlobal, outputLocal, dataSize);

    PipeBarrier<PIPE_ALL>();
}
#define VEC_POWER_LEVEL2_TESTCASE(DATA_TYPE, MEM_MODE)                                              \
    TEST_F(TEST_POWER, Power##DATA_TYPE##MEM_MODE##Case)                                           \
    {                                                                                                           \
        uint32_t dataSize = 256;                                                                               \
        uint8_t input0Gm[dataSize * sizeof(DATA_TYPE)];                                                       \
        uint8_t output0Gm[dataSize * sizeof(DATA_TYPE)];                                                       \
                                                                                                                \
        MainVecPowerLevel2Demo<DATA_TYPE>(output0Gm, input0Gm, dataSize, MEM_MODE);                     \
                                                                                                                \
        for (uint32_t i = 0; i < dataSize; i++) {                                                              \
            EXPECT_EQ(output0Gm[i], 0x00);                                                                      \
        }                                                                                                       \
    }
VEC_POWER_LEVEL2_TESTCASE(float, NORMAL_MODE);
VEC_POWER_LEVEL2_TESTCASE(float, BASIC_BLK_MODE);
VEC_POWER_LEVEL2_TESTCASE(half, NORMAL_MODE);
VEC_POWER_LEVEL2_TESTCASE(half, BASIC_BLK_MODE);
VEC_POWER_LEVEL2_TESTCASE(int32_t, NORMAL_MODE);
VEC_POWER_LEVEL2_TESTCASE(int32_t, BASIC_BLK_MODE);