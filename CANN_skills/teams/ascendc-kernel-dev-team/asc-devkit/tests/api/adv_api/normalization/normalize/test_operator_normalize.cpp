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
#define private public
#define protected public
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;

template <typename T, typename U>
__aicore__ inline void MainNormalizeTest(GM_ADDR inputXGm, GM_ADDR inputMeanGm, GM_ADDR inputVarGm, GM_ADDR gammGm,
    GM_ADDR betaGm, GM_ADDR outputGm, GM_ADDR outputRstdGm, uint32_t aLength, uint32_t rLength, uint32_t rWithPad)
{
    float epsilon = 0.001;
    GlobalTensor<T> inputXGlobal;
    GlobalTensor<float> inputMeanGlobal;
    GlobalTensor<float> inputVarGlobal;
    GlobalTensor<U> betaGlobal;
    GlobalTensor<U> gammaGlobal;
    GlobalTensor<T> outputGlobal;
    GlobalTensor<float> outputRstdGlobal;

    uint32_t arLength = aLength * rWithPad;

    inputXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputXGm), arLength);
    inputMeanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputMeanGm), aLength);
    inputVarGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputVarGm), aLength);
    gammaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gammGm), rWithPad);
    betaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(betaGm), rWithPad);

    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputGm), arLength);
    outputRstdGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputRstdGm), aLength);

    TPipe pipe;
    TQue<TPosition::VECIN, 1>  inQueueX;
    TQue<TPosition::VECIN, 1>  inQueueMean;
    TQue<TPosition::VECIN, 1>  inQueueVar;
    TQue<TPosition::VECIN, 1>  inQueueGamma;
    TQue<TPosition::VECIN, 1>  inQueueBeta;
    TQue<TPosition::VECOUT, 1> outQueue;
    TQue<TPosition::VECOUT, 1> outQueueRstd;

    pipe.InitBuffer(inQueueX,         1, sizeof(T) * arLength);
    pipe.InitBuffer(inQueueMean,      1, sizeof(float) * aLength);
    pipe.InitBuffer(inQueueVar,       1, sizeof(float) * aLength);
    pipe.InitBuffer(inQueueGamma,     1, sizeof(U) * rWithPad);
    pipe.InitBuffer(inQueueBeta,      1, sizeof(U) * rWithPad);
    pipe.InitBuffer(outQueue,         1, sizeof(T) * arLength);
    pipe.InitBuffer(outQueueRstd,     1, sizeof(float) * aLength);

    LocalTensor<T> inputXLocal = inQueueX.AllocTensor<T>();
    LocalTensor<float> inputMeanLocal = inQueueMean.AllocTensor<float>();
    LocalTensor<float> inputVarLocal = inQueueVar.AllocTensor<float>();
    LocalTensor<U> gammaLocal = inQueueGamma.AllocTensor<U>();
    LocalTensor<U> betaLocal = inQueueBeta.AllocTensor<U>();
    LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
    LocalTensor<float> outputRstdLocal = outQueueRstd.AllocTensor<float>();

    DataCopy(inputXLocal, inputXGlobal, arLength);
    DataCopy(inputMeanLocal, inputMeanGlobal, aLength);
    DataCopy(inputVarLocal, inputVarGlobal, aLength);
    DataCopy(gammaLocal, gammaGlobal, rWithPad);
    DataCopy(betaLocal, betaGlobal, rWithPad);
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> stackBuffer;
    bool ans = PopStackBuffer<float, TPosition::LCM>(stackBuffer);
    uint32_t stackBufferSize = stackBuffer.GetSize();

    NormalizePara para = {aLength, rLength, rWithPad};
    static constexpr NormalizeConfig config = GetNormalizeConfig(false, false);
    Normalize<U, T, false, config>(outputLocal, outputRstdLocal, inputMeanLocal, inputVarLocal, inputXLocal, gammaLocal,
        betaLocal, epsilon, para);
    PipeBarrier<PIPE_ALL>();

    DataCopy(outputGlobal, outputLocal, arLength);
    DataCopy(outputRstdGlobal, outputRstdLocal, aLength);

    inQueueX.FreeTensor(inputXLocal);
    inQueueMean.FreeTensor(inputMeanLocal);
    inQueueVar.FreeTensor(inputVarLocal);
    inQueueGamma.FreeTensor(gammaLocal);
    inQueueBeta.FreeTensor(betaLocal);
    outQueue.FreeTensor(outputLocal);
    outQueueRstd.FreeTensor(outputRstdLocal);
    PipeBarrier<PIPE_ALL>();
}

struct normalizeTestParams {
    uint32_t aLength;
    uint32_t rLength;
    uint32_t rWithPad;
    uint32_t typeSizeT;
    uint32_t typeSizeU;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t,
        uint32_t);
};

class normalizeTestSuite : public testing::Test, public testing::WithParamInterface<normalizeTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "normalizeTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "normalizeTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_normalize, normalizeTestSuite,
    ::testing::Values(
    // dtype combination
    normalizeTestParams { 8, 32, 32, sizeof(float), sizeof(float), MainNormalizeTest<float, float>},
    normalizeTestParams { 8, 32, 32, sizeof(half), sizeof(half), MainNormalizeTest<half, half>},
    normalizeTestParams { 8, 32, 32, sizeof(half), sizeof(float), MainNormalizeTest<half, float>},
    // R != RwithPad
    normalizeTestParams { 8, 27, 32, sizeof(float), sizeof(float), MainNormalizeTest<float, float>},
    normalizeTestParams { 8, 29, 32, sizeof(half), sizeof(half), MainNormalizeTest<half, half>},
    normalizeTestParams { 16, 31, 32, sizeof(half), sizeof(float), MainNormalizeTest<half, float>}
));

TEST_P(normalizeTestSuite, NormalizeTestCase)
{
    auto param = GetParam();
    uint32_t aLength = param.aLength;
    uint32_t rLength = param.rLength;
    uint32_t rLengthWithPadding = param.rWithPad;
    uint32_t typeSizeT = param.typeSizeT;
    uint32_t typeSizeU = param.typeSizeU;

    uint8_t inputXGm[aLength * rLengthWithPadding * typeSizeT] { 0x00 };  // [A, R]
    uint8_t inputMeanGm[aLength * sizeof(float)] { 0x00 };                // [A]
    uint8_t inputVarGm[aLength * sizeof(float)] { 0x00 };                 // [A]
    uint8_t gammGm[rLengthWithPadding * typeSizeU] { 0x00 };              // [R]
    uint8_t betaGm[rLengthWithPadding * typeSizeU] { 0x00 };              // [R]

    uint8_t outputGm[aLength * rLengthWithPadding * typeSizeT] {0x00};    // [A, R]
    uint8_t outputRstdGm[aLength * sizeof(float)] {0x00};                 // [A]

    param.cal_func(inputXGm, inputMeanGm, inputVarGm, gammGm, betaGm, outputGm, outputRstdGm, aLength, rLength,
        rLengthWithPadding);

    for (int32_t i = 0; i < aLength * rLengthWithPadding * typeSizeT; i++) {
        EXPECT_EQ(outputGm[i], 0x00);
    }
}