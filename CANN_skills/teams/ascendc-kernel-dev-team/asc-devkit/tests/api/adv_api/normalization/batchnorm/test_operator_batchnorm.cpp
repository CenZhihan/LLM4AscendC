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

namespace test_batch_norm {
constexpr uint32_t BATCHNORM_SIZEOF_FLOAT = 4;
constexpr uint32_t BATCHNORM_SIZEOF_HALF = 2;
constexpr uint32_t FLOAT_BLOCK_NUMBER = 8;
constexpr uint32_t HALF_BLOCK_NUMBER = 16;
constexpr uint32_t BATCHNORM_HALF_ELE = 16;
constexpr uint32_t BATCHNORM_THREE_TIMES = 3;
constexpr uint32_t BATCHNORM_ONE_BLK_SIZE = 32;
constexpr uint32_t BATCHNORM_ZERO_NUMBER = 0;
constexpr uint32_t BASIC_FLOAT_BLK_SHLENGTH = 64;
constexpr int32_t MAX_REPEAT_TIMES = 255;
const uint8_t DEFAULT_REPEAT_STRIDE = 8;

__aicore__ inline uint32_t AlignToBlockSize(const uint32_t inputX, const uint32_t inputY) {
    return (inputX + inputY - 1) / inputY * inputY;
}

__aicore__ inline bool GetBatchNormNDTilingInfo(const ShapeInfo srcShape, const ShapeInfo originSrcShape,
    const uint32_t stackBufferByteSize, const uint32_t typeSize, const bool isReuseSource, BatchNormTiling &tilling,
    const bool isBasicBlock)
{
    const uint32_t bLength = srcShape.shape[0];
    const uint32_t sLength = srcShape.shape[1];
    const uint32_t hLength = srcShape.shape[2];
    const uint32_t originalBLength = originSrcShape.shape[0];
    const uint32_t originalInputXSize = originalBLength * sLength * hLength;
    const uint32_t meanVarSize = sLength * hLength;

    if ((meanVarSize * typeSize % BATCHNORM_ONE_BLK_SIZE) != 0) {
        return false;
    }

    uint32_t numberOfTmpBuf = BATCHNORM_THREE_TIMES;
    constexpr uint32_t meanTmpTensorPos = BATCHNORM_ZERO_NUMBER;
    const uint32_t meanTmpTensorSize = AlignToBlockSize(meanVarSize, FLOAT_BLOCK_NUMBER);
    const uint32_t varianceTmpTensorPos = meanTmpTensorSize;
    const uint32_t varianceTmpTensorSize = meanTmpTensorSize;

    uint32_t meanVarTotalSize = meanTmpTensorSize + varianceTmpTensorSize;
    if (typeSize == BATCHNORM_SIZEOF_FLOAT) {
        meanVarTotalSize = BATCHNORM_ZERO_NUMBER;
    }

    const uint32_t tmpBufSize = stackBufferByteSize / BATCHNORM_SIZEOF_FLOAT;

    uint32_t oneTmpSize = (tmpBufSize - meanVarTotalSize) / numberOfTmpBuf;
    if (typeSize != BATCHNORM_SIZEOF_FLOAT) {
        oneTmpSize = oneTmpSize / (originalBLength * BATCHNORM_HALF_ELE) * (originalBLength * BATCHNORM_HALF_ELE);
    } else {
        oneTmpSize = oneTmpSize / (originalBLength * FLOAT_BLOCK_NUMBER) * (originalBLength * FLOAT_BLOCK_NUMBER);
    }

    if (oneTmpSize > originalInputXSize) {
        oneTmpSize = originalInputXSize;
    }

    if (oneTmpSize == BATCHNORM_ZERO_NUMBER) {
        return false;
    }

    uint32_t shCurLength = oneTmpSize / originalBLength;
    const uint32_t shCurLengthAlign = shCurLength / BASIC_FLOAT_BLK_SHLENGTH * BASIC_FLOAT_BLK_SHLENGTH;
    if (isBasicBlock && (shCurLength % BASIC_FLOAT_BLK_SHLENGTH != 0)) {
        shCurLength = shCurLengthAlign;
        oneTmpSize = shCurLength * originalBLength;
    }

    const uint32_t firstTmpStartPos = meanVarTotalSize;
    const uint32_t secondTmpStartPos = firstTmpStartPos + oneTmpSize;
    const uint32_t thirdTmpStartPos = secondTmpStartPos + oneTmpSize;
    const uint32_t loopRound = originalInputXSize / oneTmpSize;
    const uint32_t inputTailSize = originalInputXSize % oneTmpSize;
    const uint32_t inputTailPos = (originalInputXSize - inputTailSize) / originalBLength;
    const uint32_t meanVarTailSize = inputTailSize / originalBLength;
    const uint32_t meanVarTailPos = meanVarSize - meanVarTailSize;
    const uint32_t bshCurLength = oneTmpSize;
    float firstDimValueBack = (float)1.0 / originalBLength;
    const uint32_t castHalfRepStride = DEFAULT_REPEAT_STRIDE / BATCHNORM_SIZEOF_HALF;
    const uint32_t shCurLengthBlockNum = shCurLength / FLOAT_BLOCK_NUMBER;
    const uint32_t castHalfOutRepStride = meanVarSize / HALF_BLOCK_NUMBER;

    tilling.originalBLength = originalBLength;
    tilling.meanVarSize = meanVarSize;
    tilling.meanTmpTensorPos = meanTmpTensorPos;
    tilling.varianceTmpTensorPos = varianceTmpTensorPos;
    tilling.tmpBufSize = tmpBufSize;
    tilling.oneTmpSize = oneTmpSize;
    tilling.firstTmpStartPos = firstTmpStartPos;
    tilling.secondTmpStartPos = secondTmpStartPos;
    tilling.thirdTmpStartPos = thirdTmpStartPos;
    tilling.loopRound = loopRound;
    tilling.inputTailSize = inputTailSize;
    tilling.inputTailPos = inputTailPos;
    tilling.meanVarTailSize = meanVarTailSize;
    tilling.meanVarTailPos = meanVarTailPos;
    tilling.bshCurLength = bshCurLength;
    tilling.shCurLength = shCurLength;
    tilling.firstDimValueBack = firstDimValueBack;
    tilling.castHalfRepStride = castHalfRepStride;
    tilling.shCurLengthBlockNum = shCurLengthBlockNum;
    tilling.castHalfOutRepStride = castHalfOutRepStride;
    return true;
};
}// namespace test_batch_norm

template <typename dataType, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void main_batchnorm_test(GM_ADDR inputX_gm, GM_ADDR gamm_gm, GM_ADDR beta_gm, GM_ADDR output_gm,
    GM_ADDR outputMean_gm, GM_ADDR outputVariance_gm, uint32_t bLength, uint32_t sLength, uint32_t hLength,
    uint32_t originbLength)
{
    TPipe tpipe;
    dataType epsilon = 0.001;
    DataFormat dataFormat = DataFormat::ND;

    GlobalTensor<dataType> inputX_global;
    GlobalTensor<dataType> gamm_global;
    GlobalTensor<dataType> beta_global;
    GlobalTensor<dataType> output_global;
    GlobalTensor<dataType> outputMean_global;
    GlobalTensor<dataType> outputVariance_global;

    uint32_t bshLength = originbLength * sLength * hLength;
    uint32_t shLength = sLength * hLength;

    inputX_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(inputX_gm), bshLength);
    gamm_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(gamm_gm), bLength);
    beta_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(beta_gm), bLength);

    output_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(output_gm), bshLength);
    outputMean_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(outputMean_gm), shLength);
    outputVariance_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(outputVariance_gm), shLength);

    TPipe pipe;
    TQue<TPosition::VECIN, 1>  inQueueX;
    TQue<TPosition::VECIN, 1>  inQueueGamma;
    TQue<TPosition::VECIN, 1>  inQueueBeta;
    TQue<TPosition::VECOUT, 1> outQueue;
    TQue<TPosition::VECOUT, 1> outQueueMean;
    TQue<TPosition::VECOUT, 1> outQueueVariance;

    pipe.InitBuffer(inQueueX, 1, sizeof(dataType) * bshLength);
    pipe.InitBuffer(inQueueGamma, 1, sizeof(dataType) * bLength);
    pipe.InitBuffer(inQueueBeta, 1, sizeof(dataType) * bLength);
    pipe.InitBuffer(outQueue, 1, sizeof(dataType) * bshLength);
    pipe.InitBuffer(outQueueMean, 1, sizeof(dataType) * shLength);
    pipe.InitBuffer(outQueueVariance, 1, sizeof(dataType) * shLength);

    LocalTensor<dataType> inputXLocal = inQueueX.AllocTensor<dataType>();
    LocalTensor<dataType> gammaLocal = inQueueGamma.AllocTensor<dataType>();
    LocalTensor<dataType> betaLocal = inQueueBeta.AllocTensor<dataType>();
    LocalTensor<dataType> outputLocal = outQueue.AllocTensor<dataType>();
    LocalTensor<dataType> meanLocal = outQueueMean.AllocTensor<dataType>();
    LocalTensor<dataType> varianceLocal = outQueueVariance.AllocTensor<dataType>();

    DataCopy(inputXLocal, inputX_global, bshLength);
    DataCopy(gammaLocal, gamm_global, bLength);
    DataCopy(betaLocal, beta_global, bLength);
    PipeBarrier<PIPE_ALL>();

    uint32_t stackBufferSize = 0;
    {
        LocalTensor<float> stackBuffer;
        bool ans = PopStackBuffer<float, TPosition::LCM>(stackBuffer);
        stackBufferSize = stackBuffer.GetSize();
    }

    BatchNormTiling batchNormTiling;
    uint32_t inputShape[3] = {bLength, sLength, hLength};
    uint32_t originInputShape[3] = {originbLength, sLength, hLength};
    ShapeInfo shapeInfo{ 3, inputShape, 3, inputShape, dataFormat };
    ShapeInfo oriShapeInfo{ 3, originInputShape, 3, originInputShape, dataFormat };

    bool tilingRes = test_batch_norm::GetBatchNormNDTilingInfo(shapeInfo, oriShapeInfo, stackBufferSize, sizeof(dataType), isReuseSource,
        batchNormTiling, isBasicBlock);

    // if the data exceeds the UB size, tiling is 0, change tilingData
    if (!tilingRes) {
        batchNormTiling.originalBLength = 32;
        batchNormTiling.meanVarSize = 64;
        batchNormTiling.meanTmpTensorPos = 0;
        batchNormTiling.varianceTmpTensorPos = 64;
        batchNormTiling.tmpBufSize = 6272;
        batchNormTiling.oneTmpSize = 2048;
        batchNormTiling.firstTmpStartPos = 128;
        batchNormTiling.secondTmpStartPos = 2176;
        batchNormTiling.thirdTmpStartPos = 4224;
        batchNormTiling.loopRound = 1;
        batchNormTiling.inputTailSize = 0;
        batchNormTiling.inputTailPos = 64;
        batchNormTiling.meanVarTailSize = 0;
        batchNormTiling.meanVarTailPos = 64;
        batchNormTiling.bshCurLength = 2048;
        batchNormTiling.shCurLength = 64;
        batchNormTiling.firstDimValueBack = 0.03125;
        batchNormTiling.castHalfRepStride = 4;
        batchNormTiling.shCurLengthBlockNum = 8;
        batchNormTiling.castHalfOutRepStride = 4;
    }

    BatchNorm<dataType, isReuseSource, isBasicBlock>(outputLocal, meanLocal, varianceLocal, inputXLocal, gammaLocal,
        betaLocal, (dataType)epsilon, batchNormTiling);
    PipeBarrier<PIPE_ALL>();

    DataCopy(output_global, outputLocal, bshLength);
    DataCopy(outputMean_global, meanLocal, shLength);
    DataCopy(outputVariance_global, varianceLocal, shLength);

    inQueueX.FreeTensor(inputXLocal);
    inQueueGamma.FreeTensor(gammaLocal);
    inQueueBeta.FreeTensor(betaLocal);
    outQueue.FreeTensor(outputLocal);
    outQueueMean.FreeTensor(meanLocal);
    outQueueVariance.FreeTensor(varianceLocal);
    PipeBarrier<PIPE_ALL>();
}

struct batchnormTestParams {
    uint32_t bLength;
    uint32_t sLength;
    uint32_t hLength;
    uint32_t originbLength;
    uint32_t typeSize;
    void (*cal_func)(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint32_t, uint32_t, uint32_t,
        uint32_t);
};

class batchnormTestSuite : public testing::Test, public testing::WithParamInterface<batchnormTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "batchnormTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "batchnormTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_batchnorm, batchnormTestSuite,
    ::testing::Values(
    batchnormTestParams { 16, 16, 16, 16, sizeof(half),  main_batchnorm_test<half,  false, true>  },
    batchnormTestParams { 16, 16, 16, 16, sizeof(half),  main_batchnorm_test<half,  false, false> },
    batchnormTestParams { 32, 4, 16, 1, sizeof(half),  main_batchnorm_test<half,  false, false>  },
    batchnormTestParams { 256, 4, 16, 256, sizeof(half), main_batchnorm_test<half, false, true> },
    batchnormTestParams { 272, 4, 16, 272, sizeof(half), main_batchnorm_test<half, false, true> },
    batchnormTestParams { 272, 4, 16, 272, sizeof(half), main_batchnorm_test<half, false, false> },
    batchnormTestParams { 8, 16, 16, 8, sizeof(float), main_batchnorm_test<float, true, true>  },
    batchnormTestParams { 8, 16, 16, 8, sizeof(float), main_batchnorm_test<float, false, false> },
    batchnormTestParams { 256, 1, 16, 256, sizeof(float), main_batchnorm_test<float, true, false>  }
    ));

TEST_P(batchnormTestSuite, batchnormTestCase)
{
    auto param = GetParam();

    uint32_t bLength = param.bLength;
    uint32_t sLength = param.sLength;
    uint32_t hLength = param.hLength;
    uint32_t originbLength = param.originbLength;
    uint32_t typeSize = param.typeSize;

    uint32_t bshLength = originbLength * sLength * hLength;
    uint32_t shLength = sLength * hLength;

    uint8_t inputX_gm[bshLength * typeSize] { 0x00 };
    uint8_t gamm_gm[bLength * typeSize] { 0x00 };
    uint8_t beta_gm[bLength * typeSize] { 0x00 };

    uint8_t output_gm[bshLength * typeSize] {0x00};
    uint8_t outputMean_gm[shLength * typeSize] {0x00};
    uint8_t outputVariance_gm[shLength * typeSize] {0x00};

    param.cal_func(inputX_gm, gamm_gm, beta_gm, output_gm, outputMean_gm, outputVariance_gm, bLength, sLength, hLength,
        originbLength);

    for (int32_t i = 0; i < bshLength * typeSize; i++) {
        EXPECT_EQ(output_gm[i], 0x00);
    }
}