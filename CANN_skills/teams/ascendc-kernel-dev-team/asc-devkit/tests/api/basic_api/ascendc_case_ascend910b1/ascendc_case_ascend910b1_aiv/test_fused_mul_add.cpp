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

class TestFusedMulAdd : public testing::Test {
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

template <typename T, bool UseMulAddRelu = false>
void MainVecFusedMulAddLevel2Demo(__gm__ uint8_t* __restrict__ dstGm,
    __gm__ uint8_t* __restrict__ src0Gm, uint32_t dataSize)
{
    BinaryRepeatParams repeatParams;
    TPipe tpipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<T> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> input0Local = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    DataCopy(input0Local, input0Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    uint64_t maskCount = 32;
    uint64_t mask[2] = {0xff, 0};
    FusedMulAdd(outputLocal, input0Local, input0Local, dataSize);
    FusedMulAdd(outputLocal, input0Local, input0Local, maskCount, sizeof(T), repeatParams);
    FusedMulAdd(outputLocal, input0Local, input0Local, mask, sizeof(T), repeatParams);
    if constexpr (UseMulAddRelu) {
        MulAddRelu(outputLocal, input0Local, input0Local, dataSize);
        MulAddRelu(outputLocal, input0Local, input0Local, maskCount, sizeof(T), repeatParams);
        MulAddRelu(outputLocal, input0Local, input0Local, mask, sizeof(T), repeatParams);
    }else{
        FusedMulAddRelu(outputLocal, input0Local, input0Local, dataSize);
        FusedMulAddRelu(outputLocal, input0Local, input0Local, maskCount, sizeof(T), repeatParams);
        FusedMulAddRelu(outputLocal, input0Local, input0Local, mask, sizeof(T), repeatParams);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    pipe_barrier(PIPE_ALL);
}

#define VEC_FUSED_MUL_ADD_LEVEL2_TESTCASE(dataType)                                              \
    TEST_F(TestFusedMulAdd, FusedMulAddRelu##dataType##Case)                                           \
    {                                                                                                           \
        uint32_t dataSize = 256;                                                                               \
        uint32_t selMaskSize = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));                         \
        uint8_t input0Gm[dataSize * sizeof(dataType)];                                                       \
        uint8_t outputGm[dataSize * sizeof(dataType)];                                                       \
                                                                                                                \
        MainVecFusedMulAddLevel2Demo<dataType, false>(outputGm, input0Gm, dataSize);                     \
                                                                                                                \
        for (uint32_t i = 0; i < dataSize; i++) {                                                              \
            EXPECT_EQ(outputGm[i], 0x00);                                                                      \
        }                                                                                                       \
    }                                                                                                           \
    TEST_F(TestFusedMulAdd, MulAddRelu##dataType##Case)                                           \
    {                                                                                                           \
        uint32_t dataSize = 256;                                                                               \
        uint32_t selMaskSize = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));                         \
        uint8_t input0Gm[dataSize * sizeof(dataType)];                                                       \
        uint8_t outputGm[dataSize * sizeof(dataType)];                                                       \
                                                                                                                \
        MainVecFusedMulAddLevel2Demo<dataType, true>(outputGm, input0Gm, dataSize);                     \
                                                                                                                \
        for (uint32_t i = 0; i < dataSize; i++) {                                                              \
            EXPECT_EQ(outputGm[i], 0x00);                                                                      \
        }                                                                                                       \
    }

VEC_FUSED_MUL_ADD_LEVEL2_TESTCASE(float);
VEC_FUSED_MUL_ADD_LEVEL2_TESTCASE(half);

