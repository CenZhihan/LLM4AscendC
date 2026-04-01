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

__aicore__ inline void GetLayerNormNDTilingInfo(const ShapeInfo& inputShapeInfo, const uint32_t stackBufferSize,
    const uint32_t typeSize, const bool isReuseSource, LayerNormSeparateTiling& tiling)
{
    uint32_t aLength = inputShapeInfo.shape[0];
    uint32_t rLength = inputShapeInfo.shape[1];

    if ((tiling.aLength == aLength) && (tiling.rLength == rLength)) {
        return;
    }

    tiling.aLength = aLength;
    tiling.rLength = rLength;
    int32_t typeAignSize = 32 / typeSize;
    uint32_t rLengthWithPadding = (rLength + typeAignSize - 1) / typeAignSize * typeAignSize;

    tiling.inputXSize = tiling.aLength * rLengthWithPadding;
    tiling.meanVarSize = tiling.aLength;

    tiling.numberOfTmpBuf = THREE_OF_STACK_BUFFER;
    if ((isReuseSource == true) && (typeSize == B32_BYTE_SIZE)) {
        tiling.numberOfTmpBuf = TWO_OF_STACK_BUFFER;
    }

    uint32_t oneBlockNum = ONE_BLK_SIZE / sizeof(float);
    tiling.varianceTmpTensorPos = 0;
    tiling.varianceTmpTensorSize = (tiling.meanVarSize + oneBlockNum - 1) / oneBlockNum * oneBlockNum;

    tiling.tmpBufSize = stackBufferSize / sizeof(float);

    tiling.oneTmpSize = (tiling.tmpBufSize - tiling.varianceTmpTensorSize) / tiling.numberOfTmpBuf;
    tiling.oneTmpSize = tiling.oneTmpSize / rLengthWithPadding * rLengthWithPadding;

    if (tiling.oneTmpSize > tiling.inputXSize) {
        tiling.oneTmpSize = tiling.inputXSize;
    }

    if (tiling.oneTmpSize == 0) {
        return;
    }

    tiling.firstTmpStartPos = tiling.varianceTmpTensorSize;
    tiling.secondTmpStartPos = tiling.firstTmpStartPos + tiling.oneTmpSize;
    tiling.thirdTmpStartPos = tiling.secondTmpStartPos + tiling.oneTmpSize;

    tiling.loopRound = tiling.inputXSize / tiling.oneTmpSize;

    tiling.inputRoundSize = tiling.oneTmpSize;
    tiling.inputTailSize = tiling.inputXSize % tiling.oneTmpSize;

    tiling.inputTailPos = tiling.inputXSize - tiling.inputTailSize;

    tiling.meanVarRoundSize = tiling.inputRoundSize / rLengthWithPadding;
    tiling.meanVarTailSize = tiling.inputTailSize / rLengthWithPadding;

    tiling.meanVarTailPos = tiling.meanVarSize - tiling.meanVarTailSize;

    tiling.arCurLength = tiling.inputRoundSize;
    tiling.aCurLength = tiling.meanVarRoundSize;

    float lastDimValueBack = 1.0;
    tiling.rValueBack = lastDimValueBack / (float)tiling.rLength;
}

template <typename U, typename T, bool isReuseSource = false>
__aicore__ inline void MainLayernormrstdTest(GM_ADDR inputXGm, GM_ADDR gammGm, GM_ADDR betaGm, GM_ADDR outputGm,
    GM_ADDR outputMeanGm, GM_ADDR outputRstdGm, uint32_t aLength, uint32_t rLength, uint32_t rLengthWithPadding)
{
    const float epsilon = 0.001;
    DataFormat dataFormat = DataFormat::ND;

    GlobalTensor<T> inputXGlobal;
    GlobalTensor<U> gammGlobal;
    GlobalTensor<U> betaGlobal;
    GlobalTensor<T> outputGlobal;
    GlobalTensor<float> outputMeanGlobal;
    GlobalTensor<float> outputRstdGlobal;

    uint32_t arLength = aLength * rLengthWithPadding;

    inputXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputXGm), arLength);
    gammGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gammGm), rLengthWithPadding);
    betaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(betaGm), rLengthWithPadding);

    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputGm), arLength);
    outputMeanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputMeanGm), aLength);
    outputRstdGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputRstdGm), aLength);

    TPipe pipe;
    TQue<TPosition::VECIN, 1>  inQueueX;
    TQue<TPosition::VECIN, 1>  inQueueGamma;
    TQue<TPosition::VECIN, 1>  inQueueBeta;
    TQue<TPosition::VECOUT, 1> outQueue;
    TQue<TPosition::VECOUT, 1> outQueueMean;
    TQue<TPosition::VECOUT, 1> outQueueRstd;

    pipe.InitBuffer(inQueueX,           1, sizeof(T) * arLength);
    pipe.InitBuffer(inQueueGamma,       1, sizeof(U) * rLengthWithPadding);
    pipe.InitBuffer(inQueueBeta,        1, sizeof(U) * rLengthWithPadding);
    pipe.InitBuffer(outQueue,           1, sizeof(T) * arLength);
    pipe.InitBuffer(outQueueMean,       1, sizeof(float) * aLength);
    pipe.InitBuffer(outQueueRstd,   1, sizeof(float) * aLength);

    LocalTensor<T> inputXLocal   = inQueueX.AllocTensor<T>();
    LocalTensor<U> gammaLocal    = inQueueGamma.AllocTensor<U>();
    LocalTensor<U> betaLocal     = inQueueBeta.AllocTensor<U>();
    LocalTensor<T> outputLocal   = outQueue.AllocTensor<T>();
    LocalTensor<float> meanLocal     = outQueueMean.AllocTensor<float>();
    LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();

    DataCopy(inputXLocal, inputXGlobal, arLength);
    DataCopy(gammaLocal, gammGlobal, rLengthWithPadding);
    DataCopy(betaLocal, betaGlobal, rLengthWithPadding);
    PipeBarrier<PIPE_ALL>();

    uint32_t stackBufferSize = 0;
    {
        LocalTensor<float> stackBuffer;
        bool ans = PopStackBuffer<float, TPosition::LCM>(stackBuffer);
        stackBufferSize = stackBuffer.GetSize();
    }

    LayerNormPara para;
    para.aLength = aLength;
    para.rLength = rLength;
    para.rLengthWithPadding = rLengthWithPadding;

    static constexpr LayerNormConfig config = GetLayerNormNormalConfig();

    LayerNormSeparateTiling tiling;
    uint32_t inputShape[2] = {aLength, rLength};
    ShapeInfo shapeInfo{ (uint8_t)3, inputShape, (uint8_t)3, inputShape, dataFormat };

    GetLayerNormNDTilingInfo(shapeInfo, stackBufferSize, sizeof(T), isReuseSource, tiling);

    LayerNorm<U, T, isReuseSource, config>(outputLocal, meanLocal, rstdLocal, inputXLocal, gammaLocal, betaLocal,
        epsilon, para, tiling);
    PipeBarrier<PIPE_ALL>();

    DataCopy(outputGlobal, outputLocal, arLength);
    DataCopy(outputMeanGlobal, meanLocal, aLength);
    DataCopy(outputRstdGlobal, rstdLocal, aLength);

    inQueueX.FreeTensor(inputXLocal);
    inQueueGamma.FreeTensor(gammaLocal);
    inQueueBeta.FreeTensor(betaLocal);
    outQueue.FreeTensor(outputLocal);
    outQueueMean.FreeTensor(meanLocal);
    outQueueRstd.FreeTensor(rstdLocal);
    PipeBarrier<PIPE_ALL>();
}

struct layernormrstdTestParams {
    uint32_t aLength;
    uint32_t rLength;
    uint32_t rLengthWithPadding;
    uint32_t typeSizeU;
    uint32_t typeSizeT;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t);
};

class layernormrstdTestSuite : public testing::Test, public testing::WithParamInterface<layernormrstdTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "layernormrstdTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "layernormrstdTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_layernormrstd, layernormrstdTestSuite,
    ::testing::Values(
    // dtype combination
    layernormrstdTestParams { 64, 16, 16, sizeof(half),  sizeof(half), MainLayernormrstdTest<half, half>  },
    layernormrstdTestParams { 64, 16, 16, sizeof(float),  sizeof(half), MainLayernormrstdTest<float, half> },
    layernormrstdTestParams { 64, 16, 16, sizeof(float), sizeof(float), MainLayernormrstdTest<float, float> },

    // rLength != rWithPad
    layernormrstdTestParams { 8, 27, 32, sizeof(half), sizeof(half), MainLayernormrstdTest<half, half> },
    layernormrstdTestParams { 8, 29, 32, sizeof(float), sizeof(half), MainLayernormrstdTest<float, half>},
    layernormrstdTestParams { 8, 27, 32, sizeof(float), sizeof(float),MainLayernormrstdTest<float, float> }
));

TEST_P(layernormrstdTestSuite, LayernormrstdTestCase)
{
    auto param = GetParam();

    uint32_t aLength = param.aLength;
    uint32_t rLength = param.rLength;
    uint32_t rLengthWithPadding = param.rLengthWithPadding;
    uint32_t typeSizeU = param.typeSizeU;
    uint32_t typeSizeT = param.typeSizeT;

    uint32_t arLength = aLength * rLength;

    uint8_t inputXGm[aLength * rLengthWithPadding * typeSizeT] { 0x00 };
    uint8_t gammGm[rLengthWithPadding * sizeof(float)] { 0x00 };
    uint8_t betaGm[rLengthWithPadding * sizeof(float)] { 0x00 };

    uint8_t outputGm[aLength * rLengthWithPadding * typeSizeT] {0x00};
    uint8_t outputMeanGm[aLength * sizeof(float)] {0x00};
    uint8_t outputRstdeGm[aLength * sizeof(float)] {0x00};

    param.cal_func(inputXGm, gammGm, betaGm, outputGm, outputMeanGm, outputRstdeGm, aLength, rLength, rLengthWithPadding);

    for (int32_t i = 0; i < aLength * rLengthWithPadding * typeSizeT; i++) {
        EXPECT_EQ(outputGm[i], 0x00);
    }
}