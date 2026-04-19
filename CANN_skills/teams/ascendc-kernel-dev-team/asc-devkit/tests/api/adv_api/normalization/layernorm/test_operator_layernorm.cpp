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
#define protect public
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;

__aicore__ inline void GetLayerNormNDTillingInfo(const ShapeInfo& inputShapeInfo, const uint32_t stackBufferSize,
    const uint32_t typeSize, const bool isReuseSource, LayerNormTiling& tiling)
{
    uint32_t bLength = inputShapeInfo.shape[0];
    uint32_t sLength = inputShapeInfo.shape[1];
    uint32_t hLength = inputShapeInfo.shape[2];
    uint32_t originalHLength = inputShapeInfo.originalShape[2];

    if ((tiling.bLength == bLength) && (tiling.sLength == sLength) && (tiling.hLength == hLength) &&
        (tiling.originalHLength == originalHLength)) {
        return;
    }

    tiling.bLength = bLength;
    tiling.sLength = sLength;
    tiling.hLength = hLength;
    tiling.originalHLength = originalHLength;

    tiling.inputXSize = tiling.bLength * tiling.sLength * tiling.hLength;
    tiling.meanVarSize = tiling.bLength * tiling.sLength;

    tiling.numberOfTmpBuf = THREE_OF_STACK_BUFFER;
    if ((isReuseSource == true) && (typeSize == B32_BYTE_SIZE)) {
        tiling.numberOfTmpBuf = TWO_OF_STACK_BUFFER;
    }

    uint32_t oneBlockNum = ONE_BLK_SIZE / sizeof(float);
    tiling.meanTmpTensorPos = 0;
    tiling.meanTmpTensorSize = (tiling.meanVarSize + oneBlockNum - 1) / oneBlockNum * oneBlockNum;
    tiling.varianceTmpTensorPos = tiling.meanTmpTensorSize;
    tiling.varianceTmpTensorSize = tiling.meanTmpTensorSize;

    uint32_t meanVarTotalSize = tiling.meanTmpTensorSize + tiling.varianceTmpTensorSize;
    if (typeSize == B32_BYTE_SIZE) {
        meanVarTotalSize = 0;
    }

    tiling.tmpBufSize = stackBufferSize / sizeof(float);

    tiling.oneTmpSize = (tiling.tmpBufSize - meanVarTotalSize) / tiling.numberOfTmpBuf;
    tiling.oneTmpSize = tiling.oneTmpSize / tiling.hLength * tiling.hLength;

    if (tiling.oneTmpSize > tiling.inputXSize) {
        tiling.oneTmpSize = tiling.inputXSize;
    }

    if (tiling.oneTmpSize == 0) {
        return;
    }

    tiling.firstTmpStartPos = meanVarTotalSize;
    tiling.secondTmpStartPos = tiling.firstTmpStartPos + tiling.oneTmpSize;
    tiling.thirdTmpStartPos = tiling.secondTmpStartPos + tiling.oneTmpSize;

    tiling.loopRound = tiling.inputXSize / tiling.oneTmpSize;

    tiling.inputRoundSize = tiling.oneTmpSize;
    tiling.inputTailSize = tiling.inputXSize % tiling.oneTmpSize;

    tiling.inputTailPos = tiling.inputXSize - tiling.inputTailSize;

    tiling.meanVarRoundSize = tiling.inputRoundSize / tiling.hLength;
    tiling.meanVarTailSize = tiling.inputTailSize / tiling.hLength;

    tiling.meanVarTailPos = tiling.meanVarSize - tiling.meanVarTailSize;

    tiling.bshCurLength = tiling.inputRoundSize;
    tiling.bsCurLength = tiling.meanVarRoundSize;

    float lastDimValueBack = 1.0;
    tiling.lastDimValueBack = lastDimValueBack / tiling.originalHLength;
}

template <typename dataType, bool isReuseSource = false>
__aicore__ inline void MainLayernormTest(GM_ADDR inputXGm, GM_ADDR gammGm, GM_ADDR betaGm, GM_ADDR outputGm,
    GM_ADDR outputMeanGm, GM_ADDR outputVarianceGm, uint32_t bLength, uint32_t sLength, uint32_t hLength)
{
    dataType epsilon = 0.001;
    DataFormat dataFormat = DataFormat::ND;

    GlobalTensor<dataType> inputXGlobal;
    GlobalTensor<dataType> gammGlobal;
    GlobalTensor<dataType> betaGlobal;
    GlobalTensor<dataType> outputGlobal;
    GlobalTensor<dataType> outputMeanGlobal;
    GlobalTensor<dataType> outputVarianceGlobal;

    uint32_t bshLength = bLength * sLength * hLength;
    uint32_t bsLength = bLength * sLength;

    inputXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(inputXGm), bshLength);
    gammGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(gammGm), hLength);
    betaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(betaGm), hLength);

    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(outputGm), bshLength);
    outputMeanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(outputMeanGm), bsLength);
    outputVarianceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(outputVarianceGm), bsLength);

    TPipe pipe;
    TQue<TPosition::VECIN, 1>  inQueueX;
    TQue<TPosition::VECIN, 1>  inQueueGamma;
    TQue<TPosition::VECIN, 1>  inQueueBeta;
    TQue<TPosition::VECOUT, 1> outQueue;
    TQue<TPosition::VECOUT, 1> outQueueMean;
    TQue<TPosition::VECOUT, 1> outQueueVariance;

    pipe.InitBuffer(inQueueX,           1, sizeof(dataType) * bshLength);
    pipe.InitBuffer(inQueueGamma,       1, sizeof(dataType) * hLength);
    pipe.InitBuffer(inQueueBeta,        1, sizeof(dataType) * hLength);
    pipe.InitBuffer(outQueue,           1, sizeof(dataType) * bshLength);
    pipe.InitBuffer(outQueueMean,       1, sizeof(dataType) * bsLength);
    pipe.InitBuffer(outQueueVariance,   1, sizeof(dataType) * bsLength);

    LocalTensor<dataType> inputXLocal   = inQueueX.AllocTensor<dataType>();
    LocalTensor<dataType> gammaLocal    = inQueueGamma.AllocTensor<dataType>();
    LocalTensor<dataType> betaLocal     = inQueueBeta.AllocTensor<dataType>();
    LocalTensor<dataType> outputLocal   = outQueue.AllocTensor<dataType>();
    LocalTensor<dataType> meanLocal     = outQueueMean.AllocTensor<dataType>();
    LocalTensor<dataType> varianceLocal = outQueueVariance.AllocTensor<dataType>();

    DataCopy(inputXLocal, inputXGlobal, bshLength);
    DataCopy(gammaLocal, gammGlobal, hLength);
    DataCopy(betaLocal, betaGlobal, hLength);
    PipeBarrier<PIPE_ALL>();

    uint32_t stackBufferSize = 0;
    {
        LocalTensor<float> stackBuffer;
        bool ans = PopStackBuffer<float, TPosition::LCM>(stackBuffer);
        stackBufferSize = stackBuffer.GetSize();
    }

    LayerNormTiling layerNormTiling;
    uint32_t inputShape[3] = {bLength, sLength, hLength};
    ShapeInfo shapeInfo{ (uint8_t)3, inputShape, (uint8_t)3, inputShape, dataFormat };

    GetLayerNormNDTillingInfo(shapeInfo, stackBufferSize, sizeof(dataType), isReuseSource, layerNormTiling);

    LayerNorm<dataType, isReuseSource>(outputLocal, meanLocal, varianceLocal, inputXLocal, gammaLocal, betaLocal,
        (dataType)epsilon, layerNormTiling);
    PipeBarrier<PIPE_ALL>();

    DataCopy(outputGlobal, outputLocal, bshLength);
    DataCopy(outputMeanGlobal, meanLocal, bsLength);
    DataCopy(outputVarianceGlobal, varianceLocal, bsLength);

    inQueueX.FreeTensor(inputXLocal);
    inQueueGamma.FreeTensor(gammaLocal);
    inQueueBeta.FreeTensor(betaLocal);
    outQueue.FreeTensor(outputLocal);
    outQueueMean.FreeTensor(meanLocal);
    outQueueVariance.FreeTensor(varianceLocal);
    PipeBarrier<PIPE_ALL>();
}

struct layernormTestParams {
    uint32_t bLength;
    uint32_t sLength;
    uint32_t hLength;
    uint32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t);
};

class layernormTestSuite : public testing::Test, public testing::WithParamInterface<layernormTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "layernormTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "layernormTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_layernorm, layernormTestSuite,
   ::testing::Values(
   layernormTestParams { 2, 32, 16, sizeof(half),  MainLayernormTest<half,  true>  },
   layernormTestParams { 2, 32, 16, sizeof(half),  MainLayernormTest<half,  false> },
   layernormTestParams { 2, 32, 16, sizeof(float), MainLayernormTest<float, true>  },
   layernormTestParams { 2, 32, 16, sizeof(float), MainLayernormTest<float, false> },
   layernormTestParams { 2, 16, 32, sizeof(half),  MainLayernormTest<half,  true>  },
   layernormTestParams { 2, 16, 32, sizeof(half),  MainLayernormTest<half,  false> },
   layernormTestParams { 2, 16, 32, sizeof(float), MainLayernormTest<float, true>  },
   layernormTestParams { 2, 16, 32, sizeof(float), MainLayernormTest<float, false> }
   ));

TEST_P(layernormTestSuite, LayernormTestCase)
{
   auto param = GetParam();

   uint32_t bLength = param.bLength;
   uint32_t sLength = param.sLength;
   uint32_t hLength = param.hLength;
   uint32_t typeSize = param.typeSize;

   uint32_t bshLength = bLength * sLength * hLength;
   uint32_t bsLength = bLength * sLength;

   uint8_t inputXGm[bshLength * typeSize] { 0x00 };
   uint8_t gammGm[hLength * typeSize] { 0x00 };
   uint8_t betaGm[hLength * typeSize] { 0x00 };

   uint8_t outputGm[bshLength * typeSize] {0x00};
   uint8_t outputMeanGm[bsLength * typeSize] {0x00};
   uint8_t outputVarianceGm[bsLength * typeSize] {0x00};

   param.cal_func(inputXGm, gammGm, betaGm, outputGm, outputMeanGm, outputVarianceGm, bLength, sLength, hLength);

   for (int32_t i = 0; i < bshLength * typeSize; i++) {
       EXPECT_EQ(outputGm[i], 0x00);
   }
}