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
    COMPARE_TENSOR_TENSOR,
    COMPARE_TENSOR_SCALAR,
    COMPARE_SCALAR_TENSOR
};

enum MemMode {
    NORMAL_MODE,
    BASIC_BLK_MODE,
};

class TEST_COMPARE : public testing::Test {
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

template <typename T, bool USE_COMPARES>
void MainVecCompareLevel2Demo(__gm__ uint8_t* __restrict__ dstGm,
    __gm__ uint8_t* __restrict__ srcGm, uint32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> input0Global;
    GlobalTensor<uint8_t> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> input0Local = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(uint8_t));
    LocalTensor<uint8_t> outputLocal = tbuf1.Get<uint8_t>();

    DataCopy(input0Local, input0Global, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if constexpr (USE_COMPARES){
        Compares(outputLocal, input0Local, static_cast<T>(0), CMPMODE::EQ, dataSize);
        Compares(outputLocal, input0Local, static_cast<T>(0), CMPMODE::NE, dataSize);
        Compares(outputLocal, input0Local, static_cast<T>(0), CMPMODE::GT, dataSize);
        Compares(outputLocal, input0Local, static_cast<T>(0), CMPMODE::GE, dataSize);
        Compares(outputLocal, input0Local, static_cast<T>(0), CMPMODE::LT, dataSize);
        Compares(outputLocal, input0Local, static_cast<T>(0), CMPMODE::LE, dataSize);
    }else{
        CompareScalar(outputLocal, input0Local, static_cast<T>(0), CMPMODE::EQ, dataSize);
        CompareScalar(outputLocal, input0Local, static_cast<T>(0), CMPMODE::NE, dataSize);
        CompareScalar(outputLocal, input0Local, static_cast<T>(0), CMPMODE::GT, dataSize);
        CompareScalar(outputLocal, input0Local, static_cast<T>(0), CMPMODE::GE, dataSize);
        CompareScalar(outputLocal, input0Local, static_cast<T>(0), CMPMODE::LT, dataSize);
        CompareScalar(outputLocal, input0Local, static_cast<T>(0), CMPMODE::LE, dataSize);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    pipe_barrier(PIPE_ALL);

    SetCmpMask(input0Local);
    GetCmpMask(outputLocal);
    pipe_barrier(PIPE_ALL);
}

#define VEC_COMPARE_LEVEL2_TESTCASE(dataType)                                              \
    TEST_F(TEST_COMPARE, Compares##dataType##Case)                                          \
    {                                                                                                 \
        uint32_t dataSize = 256;                                                                     \
        uint32_t selMaskSize = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));               \
        uint8_t input0Gm[dataSize * sizeof(dataType)] = {0};                                       \
        uint8_t outputGm[dataSize * sizeof(uint8_t)] = {0};                                       \
                                                                                                      \
        MainVecCompareLevel2Demo<dataType, true>(outputGm, input0Gm, dataSize);           \
                                                                                                      \
        for (uint32_t i = 0; i < dataSize; i++) {                                                    \
            EXPECT_EQ(outputGm[i], 0x00);                                                            \
        }                                                                                             \
    }                                                                                             \
    TEST_F(TEST_COMPARE, CompareScalar##dataType##Case)                                          \
    {                                                                                                 \
        uint32_t dataSize = 256;                                                                     \
        uint32_t selMaskSize = dataSize / AscendCUtils::GetBitSize(sizeof(uint8_t));               \
        uint8_t input0Gm[dataSize * sizeof(dataType)] = {0};                                       \
        uint8_t outputGm[dataSize * sizeof(uint8_t)] = {0};                                       \
                                                                                                      \
        MainVecCompareLevel2Demo<dataType, false>(outputGm, input0Gm, dataSize);           \
                                                                                                      \
        for (uint32_t i = 0; i < dataSize; i++) {                                                    \
            EXPECT_EQ(outputGm[i], 0x00);                                                            \
        }                                                                                             \
    }

VEC_COMPARE_LEVEL2_TESTCASE(float);
VEC_COMPARE_LEVEL2_TESTCASE(half);
