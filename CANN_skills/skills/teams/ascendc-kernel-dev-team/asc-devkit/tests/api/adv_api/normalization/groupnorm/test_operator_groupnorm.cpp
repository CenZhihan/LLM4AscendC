/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file test_operator_groupnorm.cpp
 * \brief
 */

#include <gtest/gtest.h>
#define private public
#define protected public
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;
constexpr uint32_t GROUPNORM_SIZEOF_HALF = 2;

constexpr uint32_t GROUPNORM_MAX_MASK_VAL = 64;
constexpr uint32_t GROUPNORM_STEP_MASK_VAL = 8;
constexpr uint32_t GROUPNORM_MAX_REPEAT_VAL = 255;
constexpr uint32_t GROUPNORM_MIN_BSCURLENGTH_IN_ITERATION = 8;
constexpr uint32_t GROUPNORM_REDUCESUM_MAX_FLOAT_NUM = 64;
constexpr uint32_t GROUPNORM_REDUCESUM_MAX_REPEAT_SMALLSHAPE = 8;

__aicore__ inline void GetGroupNormNDTillingInfo(const ShapeInfo& inputShapeInfo, const uint32_t stackBufferSize,
    const uint32_t typeSize, const bool isReuseSource, const uint32_t groupNum, GroupNormTiling& tiling)
{
    
    uint32_t n = inputShapeInfo.shape[0];
    uint32_t c = inputShapeInfo.shape[1];
    uint32_t h = inputShapeInfo.shape[2];
    uint32_t w = inputShapeInfo.shape[3];

    tiling.n = n;
    tiling.c = c;
    tiling.hw = h * w;
    tiling.g = groupNum;
    tiling.d = c / tiling.g;
    tiling.hwAlignSize = (typeSize * tiling.hw + ONE_BLK_SIZE - 1) / 
    ONE_BLK_SIZE * ONE_BLK_SIZE / typeSize;
    tiling.dhwAlignSize = tiling.d * tiling.hwAlignSize;

    tiling.inputXSize = n * c * tiling.hwAlignSize;
    tiling.meanVarSize = n * tiling.g;

    uint32_t oneBlockNum = ONE_BLK_SIZE / B32_BYTE_SIZE;
    tiling.meanTmpTensorPos = 0;
    tiling.meanTmpTensorSize = (tiling.meanVarSize + oneBlockNum - 1) / oneBlockNum * oneBlockNum;
    tiling.varianceTmpTensorPos = tiling.meanTmpTensorSize;
    tiling.varianceTmpTensorSize = tiling.meanTmpTensorSize;


    uint32_t meanVarTotalSize = tiling.meanTmpTensorSize + tiling.varianceTmpTensorSize;
    if (typeSize == B32_BYTE_SIZE) {
        meanVarTotalSize = 0;
    }

    tiling.numberOfTmpBuf = THREE_OF_STACK_BUFFER;
    if (isReuseSource && (typeSize == B32_BYTE_SIZE)) {
        tiling.numberOfTmpBuf = TWO_OF_STACK_BUFFER;
    }

    tiling.tmpBufSize = stackBufferSize / ONE_BLK_SIZE * ONE_BLK_SIZE / B32_BYTE_SIZE;
    tiling.oneTmpSize = (tiling.tmpBufSize - meanVarTotalSize) / tiling.numberOfTmpBuf;

    // to enable MeanVarTensor to directly use Add without need to use GetValue
    // it is necessary to ensure that each iteration has at least 8 integer multiples of groups
    tiling.bsCurLength = tiling.oneTmpSize / (GROUPNORM_MIN_BSCURLENGTH_IN_ITERATION * tiling.d * tiling.hwAlignSize) *
        GROUPNORM_MIN_BSCURLENGTH_IN_ITERATION;

    // determine whether the condition for smallShape is met
    uint32_t k = GROUPNORM_REDUCESUM_MAX_REPEAT_SMALLSHAPE;
    while ((tiling.dhwAlignSize / (ONE_BLK_SIZE / B32_BYTE_SIZE)) % k != 0) {
        k--;
    }
    tiling.smallShape = (tiling.hwAlignSize <= GROUPNORM_REDUCESUM_MAX_FLOAT_NUM) && 
    (tiling.hwAlignSize * tiling.d <= GROUPNORM_REDUCESUM_MAX_FLOAT_NUM * k);

    // the constraints instroduced by the ReduceSum0 interface
    // based one the DHW calculation of the mask/repeat for 2 ReduceSum operations,
    // as well as the value range of DHW/bsCurLength
    if (tiling.smallShape) {
        uint32_t mask1{GROUPNORM_MAX_MASK_VAL};
        if (tiling.dhwAlignSize > GROUPNORM_MAX_MASK_VAL) {
            while (tiling.dhwAlignSize % mask1 != 0) {
                mask1 -= GROUPNORM_STEP_MASK_VAL;
            }
        } else {
            mask1 = tiling.dhwAlignSize;
        }
        uint32_t max_bsCurLength = (GROUPNORM_MAX_REPEAT_VAL / (tiling.dhwAlignSize / mask1) / 
            GROUPNORM_MIN_BSCURLENGTH_IN_ITERATION) * GROUPNORM_MIN_BSCURLENGTH_IN_ITERATION;
        if (max_bsCurLength < tiling.bsCurLength) {
            tiling.bsCurLength = max_bsCurLength;
        }
    }

    if (typeSize == GROUPNORM_SIZEOF_HALF && tiling.bsCurLength * tiling.dhwAlignSize < c) {
        return;
    }

    tiling.oneTmpSize = tiling.bsCurLength * tiling.d * tiling.hwAlignSize;

    if (tiling.oneTmpSize > tiling.inputXSize) {
        tiling.bsCurLength = tiling.meanVarSize;
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

    tiling.meanVarRoundSize = tiling.inputRoundSize / tiling.dhwAlignSize;
    tiling.meanVarTailSize = tiling.inputTailSize / tiling.dhwAlignSize;
    tiling.meanVarTailPos = tiling.meanVarSize - tiling.meanVarTailSize;

    tiling.bshCurLength = tiling.inputRoundSize;

    tiling.factor = 1.0f / (tiling.d * tiling.hw);
    cout << tiling.n << ", " << tiling.c <<  ", " << tiling.hw <<  ", " << tiling.g <<  ", " << tiling.hwAlignSize << endl;
    cout << "inputXSize: " << tiling.inputXSize << endl;
    cout << "meanVarSize: " << tiling.meanVarSize << endl;
    cout << "numberOfTmpBuf: " << tiling.numberOfTmpBuf << endl;
    cout << "meanTmpTensorPos: " << tiling.meanTmpTensorPos << endl;
    cout << "varianceTmpTensorPos: " << tiling.varianceTmpTensorPos << endl;
    cout << "oneTmpSize: " << tiling.oneTmpSize << endl;
    cout << "firstTmpStartPos: " << tiling.firstTmpStartPos << endl;
    cout << "thirdTmpStartPos: " << tiling.thirdTmpStartPos << endl;
    cout << "bsCurLength: " << tiling.bsCurLength << endl;
    cout << "bshCurLength: " << tiling.bshCurLength << endl;
    cout << "factor: " << tiling.factor << endl;
    cout << "hwAlignSize: " << tiling.hwAlignSize << endl;
    cout << "smallShape: " << tiling.smallShape << endl;

}
// __aicore__ inline void MainGroupnormTest(GM_ADDR inputXGm, GM_ADDR gammGm, GM_ADDR betaGm, GM_ADDR outputGm,
//     GM_ADDR outputMeanGm, GM_ADDR outputVarianceGm, uint32_t n, uint32_t c, uint32_t h, uint32_t w, uint32_t g)
template <typename dataType, bool isReuseSource = false>
__aicore__ inline void MainGroupnormTest(GM_ADDR inputXGm, GM_ADDR gammGm, GM_ADDR betaGm, GM_ADDR outputGm,
    uint32_t n, uint32_t c, uint32_t h, uint32_t w, uint32_t g)
{
    dataType epsilon = 0.001;
    DataFormat dataFormat = DataFormat::ND;

    GlobalTensor<dataType> inputXGlobal;
    GlobalTensor<dataType> gammGlobal;
    GlobalTensor<dataType> betaGlobal;
    GlobalTensor<dataType> outputGlobal;
    // GlobalTensor<dataType> outputMeanGlobal;
    // GlobalTensor<dataType> outputVarianceGlobal;

    uint32_t bshLength = n*c*h*w;
    uint32_t bsLength = g*n;

    inputXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(inputXGm), bshLength);
    gammGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(gammGm), c);
    betaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(betaGm), c);

    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(outputGm), bshLength);
    // outputMeanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(outputMeanGm), bsLength);
    // outputVarianceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(outputVarianceGm), bsLength);

    TPipe pipe;
    TQue<TPosition::VECIN, 1>  inQueueX;
    TQue<TPosition::VECIN, 1>  inQueueGamma;
    TQue<TPosition::VECIN, 1>  inQueueBeta;
    TQue<TPosition::VECOUT, 1> outQueue;
    // TQue<TPosition::VECOUT, 1> outQueueMean;
    // TQue<TPosition::VECOUT, 1> outQueueVariance;
    TBuf<TPosition::VECCALC> meanBuffer, varBuffer;

    uint32_t hwAlignSize = (sizeof(dataType) * h * w + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE / sizeof(dataType);
    pipe.InitBuffer(inQueueX,           1, sizeof(dataType) * n * c * hwAlignSize);
    pipe.InitBuffer(inQueueGamma,       1, (sizeof(dataType) * c + 31) / 32 * 32);
    pipe.InitBuffer(inQueueBeta,        1, (sizeof(dataType) * c + 31) / 32 * 32);
    pipe.InitBuffer(outQueue,           1, sizeof(dataType) * n * c * hwAlignSize);
    // pipe.InitBuffer(outQueueMean,       1, (sizeof(dataType) * g * n + 31) / 32 * 32);
    // pipe.InitBuffer(outQueueVariance,   1, (sizeof(dataType) * g * n + 31) / 32 * 32);
    pipe.InitBuffer(meanBuffer, (sizeof(dataType) * g * n + 31) / 32 * 32);
    pipe.InitBuffer(varBuffer, (sizeof(dataType) * g * n + 31) / 32 * 32);

    LocalTensor<dataType> inputXLocal   = inQueueX.AllocTensor<dataType>();
    LocalTensor<dataType> gammaLocal    = inQueueGamma.AllocTensor<dataType>();
    LocalTensor<dataType> betaLocal     = inQueueBeta.AllocTensor<dataType>();
    LocalTensor<dataType> outputLocal   = outQueue.AllocTensor<dataType>();
    // LocalTensor<dataType> meanLocal     = outQueueMean.AllocTensor<dataType>();
    // LocalTensor<dataType> varianceLocal = outQueueVariance.AllocTensor<dataType>();
    LocalTensor<dataType> meanLocal     = meanBuffer.Get<dataType>();
    LocalTensor<dataType> varianceLocal = varBuffer.Get<dataType>();

    DataCopyParams copyParams{static_cast<uint16_t>(n*c), static_cast<uint16_t>(h*w*sizeof(dataType)), 0, 0};
    DataCopyPadParams padParams{true, 0, static_cast<uint8_t>(hwAlignSize - h * w), 0};
    DataCopyPad(inputXLocal, inputXGlobal, copyParams, padParams);
    // DataCopy(inputXLocal, inputXGlobal, bshLength);
    DataCopyParams copyParamsGamma{1, static_cast<uint16_t>(c*sizeof(dataType)), 0, 0};
    DataCopyPadParams padParamsGamma{false, 0, 0, 0};
    DataCopyPad(gammaLocal, gammGlobal, copyParamsGamma, padParamsGamma);
    DataCopyPad(betaLocal, betaGlobal, copyParamsGamma, padParamsGamma);

    // DataCopy(gammaLocal, gammGlobal, c);
    // DataCopy(betaLocal, betaGlobal, c);
    PipeBarrier<PIPE_ALL>();

    uint32_t stackBufferSize = 0;
    {
        LocalTensor<float> stackBuffer;
        bool ans = PopStackBuffer<float, TPosition::LCM>(stackBuffer);
        stackBufferSize = stackBuffer.GetSize();
    }

    GroupNormTiling groupNormTiling;
    uint32_t inputShape[4] = {n, c, h, w};
    ShapeInfo shapeInfo{ (uint8_t)4, inputShape, (uint8_t)4, inputShape, dataFormat };

    GetGroupNormNDTillingInfo(shapeInfo, stackBufferSize, sizeof(dataType), isReuseSource, g, groupNormTiling);

    GroupNorm<dataType, isReuseSource>(outputLocal, meanLocal, varianceLocal, inputXLocal, gammaLocal, betaLocal,
        (dataType)epsilon, groupNormTiling);
    PipeBarrier<PIPE_ALL>();

    // DataCopy(outputGlobal, outputLocal, bshLength);
    DataCopyPad(outputGlobal, outputLocal, copyParams);
    // DataCopy(outputMeanGlobal, meanLocal, bsLength);
    // DataCopy(outputVarianceGlobal, varianceLocal, bsLength);

    inQueueX.FreeTensor(inputXLocal);
    inQueueGamma.FreeTensor(gammaLocal);
    inQueueBeta.FreeTensor(betaLocal);
    outQueue.FreeTensor(outputLocal);
    // outQueueMean.FreeTensor(meanLocal);
    // outQueueVariance.FreeTensor(varianceLocal);
    PipeBarrier<PIPE_ALL>();
}

struct groupnormTestParams {
    uint32_t n;
    uint32_t c;
    uint32_t h;
    uint32_t w;
    uint32_t g;
    uint32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
};

class groupnormTestSuite : public testing::Test, public testing::WithParamInterface<groupnormTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "groupnormTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "groupnormTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_groupnorm, groupnormTestSuite,
    ::testing::Values(
    groupnormTestParams { 2, 16, 8, 8, 4, sizeof(float), MainGroupnormTest<float, false> },
    groupnormTestParams { 2, 16, 8, 8, 4, sizeof(half), MainGroupnormTest<half, false> },
    groupnormTestParams { 2, 16, 9, 9, 4, sizeof(float), MainGroupnormTest<float, false> },
    groupnormTestParams { 2, 16, 9, 9, 4, sizeof(half), MainGroupnormTest<half, false> },
    groupnormTestParams { 2, 16, 8, 8, 4, sizeof(float), MainGroupnormTest<float, true> },
    groupnormTestParams { 2, 16, 9, 9, 4, sizeof(float), MainGroupnormTest<float, true> }
    ));

TEST_P(groupnormTestSuite, GroupnormTestCase)
{
    auto param = GetParam();

    uint32_t n = param.n;
    uint32_t c = param.c;
    uint32_t h = param.h;
    uint32_t w = param.w;
    uint32_t g = param.g;
    uint32_t typeSize = param.typeSize;

    uint32_t bshLength = n * c * h * w;
    uint32_t bsLength = n * c / g;

    uint8_t inputXGm[bshLength * typeSize] { 0x00 };
    uint8_t gammGm[c * typeSize] { 0x00 };
    uint8_t betaGm[c * typeSize] { 0x00 };

    uint8_t outputGm[bshLength * typeSize] {0x00};
    // uint8_t outputMeanGm[bsLength * typeSize] {0x00};
    // uint8_t outputVarianceGm[bsLength * typeSize] {0x00};

    param.cal_func(inputXGm, gammGm, betaGm, outputGm, n, c, h, w, g);

    for (int32_t i = 0; i < bshLength * typeSize; i++) {
        EXPECT_EQ(outputGm[i], 0x00);
    }
}