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

namespace AscendC {

   __aicore__ inline void GetDeepNormNDTilingInfo(const ShapeInfo& inputShapeInfo, const ShapeInfo& originShapeInfo,
    const uint32_t stackBufferSize, const uint32_t typeSize, const bool isReuseSource, const bool isBasicBlk,
    DeepNormTiling& tiling)
{
    uint32_t bLength = inputShapeInfo.shape[0];
    uint32_t sLength = inputShapeInfo.shape[1];
    uint32_t hLength = inputShapeInfo.shape[2];                    // H after alignment

    uint32_t originalBLength = originShapeInfo.shape[0];
    uint32_t originalSLength = originShapeInfo.shape[1];
    uint32_t originalHLength = originShapeInfo.shape[2];           // H before alignment

    // do not need to update tiling
    if ((tiling.bLength == bLength) && (tiling.sLength == sLength) && (tiling.hLength == hLength) &&
        (tiling.originalHLength == originalHLength)) {
        return;
    }

    tiling.bLength = bLength;
    tiling.sLength = sLength;
    tiling.hLength = hLength;
    tiling.originalHLength = originalHLength;
    tiling.inputXSize = tiling.bLength * tiling.sLength * tiling.hLength;   // B * S * H
    tiling.meanVarSize = tiling.bLength * tiling.sLength;                   // B * S

    tiling.numberOfTmpBuf = (isReuseSource && (typeSize == 4))? TWO_OF_STACK_BUFFER: THREE_OF_STACK_BUFFER;

    uint32_t oneBlockNum = ONE_BLK_SIZE / sizeof(float);
    tiling.meanTmpTensorPos = 0;
    tiling.meanTmpTensorSize = (tiling.meanVarSize + oneBlockNum - 1) / oneBlockNum * oneBlockNum;
    tiling.varianceTmpTensorPos = tiling.meanTmpTensorSize;
    tiling.varianceTmpTensorSize = tiling.meanTmpTensorSize;

    uint32_t meanVarTotalSize = (typeSize == B32_BYTE_SIZE) ? 0 : tiling.meanTmpTensorSize +
        tiling.varianceTmpTensorSize;

    tiling.tmpBufSize = stackBufferSize / sizeof(float);
    tiling.oneTmpSize = (tiling.tmpBufSize - meanVarTotalSize) / tiling.numberOfTmpBuf;
    tiling.oneTmpSize = tiling.oneTmpSize / tiling.hLength * tiling.hLength;
    tiling.oneTmpSize = (tiling.oneTmpSize > tiling.inputXSize) ? tiling.inputXSize : tiling.oneTmpSize;

    tiling.firstTmpStartPos = meanVarTotalSize;
    tiling.secondTmpStartPos = tiling.firstTmpStartPos + tiling.oneTmpSize;
    tiling.thirdTmpStartPos = tiling.secondTmpStartPos + tiling.oneTmpSize;

    tiling.loopRound = tiling.inputXSize / tiling.oneTmpSize;
    tiling.inputTailSize = tiling.inputXSize % tiling.oneTmpSize;
    tiling.inputTailPos = tiling.inputXSize - tiling.inputTailSize;
    tiling.inputRoundSize = tiling.oneTmpSize;
    tiling.meanVarRoundSize = tiling.inputRoundSize / tiling.hLength;
    tiling.meanVarTailSize = tiling.inputTailSize / tiling.hLength;
    tiling.meanVarTailPos = tiling.meanVarSize - tiling.meanVarTailSize;

    tiling.bshCurLength = tiling.inputRoundSize;
    tiling.bsCurLength = tiling.meanVarRoundSize;

    float lastDimValueBack = 1.0;
    lastDimValueBack = lastDimValueBack / tiling.originalHLength;
    tiling.lastDimValueBack = lastDimValueBack;
}

template <typename dataType, bool isReuseSrc = false, bool isBasicBlock = false>
class KernelDeepNorm {
public:
    __aicore__ inline KernelDeepNorm() {}

    __aicore__ inline void Init(GM_ADDR inputXGm, GM_ADDR inputGxGm, GM_ADDR betaGm, GM_ADDR gammGm,
        GM_ADDR outputGm, GM_ADDR outputMeanGm, GM_ADDR outputVarianceGm, uint32_t bLength, uint32_t sLength,
        uint32_t hLength)
    {
        this->bLength = bLength;
        this->sLength = sLength;
        this->hLength = hLength;
        this->alpha = 1.35;
        this->epsilon = 0.001;
        this->dataFormat = DataFormat::ND;

        uint32_t oneBlockNum = 32 / sizeof(dataType);
        bshLength = bLength * sLength * hLength;
        bsLength = bLength * sLength;
        originalBsLength = bsLength;
        bsLength = (bsLength + oneBlockNum - 1) / oneBlockNum * oneBlockNum;

        inputX_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(inputXGm), bshLength);
        inputGx_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(inputGxGm), bshLength);
        beta_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(betaGm), hLength);
        gamm_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(gammGm), hLength);

        output_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(outputGm), bshLength);
        outputMean_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(outputMeanGm), bsLength);
        outputVariance_global.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(outputVarianceGm), bsLength);

        pipe.InitBuffer(inQueueX, 1, sizeof(dataType) * bshLength);
        pipe.InitBuffer(inQueueGx, 1, sizeof(dataType) * bshLength);
        pipe.InitBuffer(inQueueBeta, 1, sizeof(dataType) * hLength);
        pipe.InitBuffer(inQueueGamma, 1, sizeof(dataType) * hLength);
        pipe.InitBuffer(outQueue, 1, sizeof(dataType) * bshLength);
        pipe.InitBuffer(outQueueMean, 1, sizeof(dataType) * bsLength);
        pipe.InitBuffer(outQueueVariance, 1, sizeof(dataType) * bsLength);
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<dataType> inputXLocal = inQueueX.AllocTensor<dataType>();
        LocalTensor<dataType> inputGxLocal = inQueueGx.AllocTensor<dataType>();
        LocalTensor<dataType> gammaLocal = inQueueGamma.AllocTensor<dataType>();
        LocalTensor<dataType> betaLocal = inQueueBeta.AllocTensor<dataType>();

        DataCopy(inputXLocal, inputX_global, bshLength);
        DataCopy(inputGxLocal, inputGx_global, bshLength);
        DataCopy(gammaLocal, gamm_global, hLength);
        DataCopy(betaLocal, beta_global, hLength);

        inQueueX.EnQue(inputXLocal);
        inQueueGx.EnQue(inputGxLocal);
        inQueueBeta.EnQue(betaLocal);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<dataType> inputXLocal = inQueueX.DeQue<dataType>();
        LocalTensor<dataType> inputGxLocal = inQueueGx.DeQue<dataType>();
        LocalTensor<dataType> betaLocal = inQueueBeta.DeQue<dataType>();
        LocalTensor<dataType> gammaLocal = inQueueGamma.DeQue<dataType>();
        LocalTensor<dataType> outputLocal = outQueue.AllocTensor<dataType>();
        LocalTensor<dataType> meanLocal = outQueueMean.AllocTensor<dataType>();
        LocalTensor<dataType> varianceLocal = outQueueVariance.AllocTensor<dataType>();

        DeepNormTiling tiling;

        uint32_t inputShape[3] = {bLength, sLength, hLength};
        ShapeInfo shapeInfo{3, inputShape, 3, inputShape, dataFormat};
        ShapeInfo orishapeInfo{3, inputShape, 3, inputShape, dataFormat};

        LocalTensor<uint8_t> stackBuffer;
        bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackBuffer);
        stackBufferSize = stackBuffer.GetSize();

        GetDeepNormNDTilingInfo(shapeInfo, orishapeInfo, stackBufferSize, sizeof(dataType), isReuseSrc, isBasicBlock, tiling);
        DeepNorm<dataType, isReuseSrc, isBasicBlock>(outputLocal, meanLocal, varianceLocal, inputXLocal,
            inputGxLocal, betaLocal, gammaLocal, (dataType)alpha, (dataType)epsilon, tiling);

        outQueue.EnQue<dataType>(outputLocal);
        outQueueMean.EnQue<dataType>(meanLocal);
        outQueueVariance.EnQue<dataType>(varianceLocal);

        inQueueX.FreeTensor(inputXLocal);
        inQueueGx.FreeTensor(inputGxLocal);
        inQueueBeta.FreeTensor(betaLocal);
        inQueueGamma.FreeTensor(gammaLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<dataType> outputLocal = outQueue.DeQue<dataType>();
        LocalTensor<dataType> meanLocal = outQueueMean.DeQue<dataType>();
        LocalTensor<dataType> varianceLocal = outQueueVariance.DeQue<dataType>();

        DataCopyParams copyParams;
        copyParams.blockLen = originalBsLength * sizeof(dataType);

        DataCopy(output_global, outputLocal, bshLength);
        DataCopyPad(outputMean_global, meanLocal, copyParams);
        DataCopyPad(outputVariance_global, varianceLocal, copyParams);

        outQueue.FreeTensor(outputLocal);
        outQueueMean.FreeTensor(meanLocal);
        outQueueVariance.FreeTensor(varianceLocal);
    }

private:
    GlobalTensor<dataType> inputX_global;
    GlobalTensor<dataType> inputGx_global;
    GlobalTensor<dataType> beta_global;
    GlobalTensor<dataType> gamm_global;
    GlobalTensor<dataType> output_global;
    GlobalTensor<dataType> outputMean_global;
    GlobalTensor<dataType> outputVariance_global;

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECIN, 1> inQueueGx;
    TQue<TPosition::VECIN, 1> inQueueBeta;
    TQue<TPosition::VECIN, 1> inQueueGamma;
    TQue<TPosition::VECOUT, 1> outQueue;
    TQue<TPosition::VECOUT, 1> outQueueMean;
    TQue<TPosition::VECOUT, 1> outQueueVariance;

    uint32_t bLength;
    uint32_t sLength;
    uint32_t hLength;
    dataType alpha;
    dataType epsilon;
    DataFormat dataFormat;

    uint32_t bshLength;
    uint32_t bsLength;
    uint32_t originalBsLength;
    uint32_t stackBufferSize = 0;
};
} // namespace AscendC

template <typename dataType, bool isReuseSrc = false, bool isBasicBlock = false>
__global__ __aicore__ void kernel_deepnorm_operator(GM_ADDR inputXGm, GM_ADDR inputGxGm, GM_ADDR betaGm,
    GM_ADDR gammGm, GM_ADDR outputGm, GM_ADDR outputMeanGm, GM_ADDR outputVarianceGm, uint32_t bLength,
    uint32_t sLength, uint32_t hLength)
{
    AscendC::KernelDeepNorm<dataType, isReuseSrc, isBasicBlock> op;
    op.Init(inputXGm, inputGxGm, betaGm, gammGm, outputGm, outputMeanGm, outputVarianceGm,
        bLength, sLength, hLength);
    op.Process();
}

struct DeepNormTestParams {
    uint32_t bLength;
    uint32_t sLength;
    uint32_t hLength;
    uint32_t typeSize;
    void (*calFunc) (uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t,
        uint32_t);
};

class DeepNormTestsuite : public testing::Test, public testing::WithParamInterface<DeepNormTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DEEPNORM, DeepNormTestsuite,
    ::testing::Values(
    DeepNormTestParams {2, 32, 16, sizeof(half),  kernel_deepnorm_operator<half,  true, false>  },
    DeepNormTestParams {2, 32, 16, sizeof(half),  kernel_deepnorm_operator<half,  false, false> },
    DeepNormTestParams {2, 16, 32, sizeof(half),  kernel_deepnorm_operator<half,  true, false>  },
    DeepNormTestParams {2, 16, 32, sizeof(half),  kernel_deepnorm_operator<half,  false, false> },
    DeepNormTestParams {2, 32, 16, sizeof(float), kernel_deepnorm_operator<float, true, false>  },
    DeepNormTestParams {2, 32, 16, sizeof(float), kernel_deepnorm_operator<float, false, false> },
    DeepNormTestParams {2, 16, 32, sizeof(float), kernel_deepnorm_operator<float, true, false>  },
    DeepNormTestParams {2, 16, 32, sizeof(float), kernel_deepnorm_operator<float, false, false> },
    // basic block
    DeepNormTestParams {1, 16, 128, sizeof(half),  kernel_deepnorm_operator<half, true, true>  },
    DeepNormTestParams {1, 16, 128, sizeof(half),  kernel_deepnorm_operator<half, false, true> },
    DeepNormTestParams {1, 16, 128, sizeof(float), kernel_deepnorm_operator<float, true, true> },
    DeepNormTestParams {1, 16, 128, sizeof(float), kernel_deepnorm_operator<float, false, true>}
));

TEST_P(DeepNormTestsuite, DeepNormOpTestCase)
{
    auto param = GetParam();

    uint32_t bLength = param.bLength;
    uint32_t sLength = param.sLength;
    uint32_t hLength = param.hLength;
    uint32_t typeSize = param.typeSize;
    uint32_t oneBlockNum = 32 / typeSize;
    uint32_t bshLength = bLength * sLength * hLength;
    uint32_t bsLength = bLength * sLength;
    bsLength = (bsLength + oneBlockNum - 1) / oneBlockNum * oneBlockNum;

    uint8_t inputXGm[bshLength * typeSize] { 0x00 };
    uint8_t inputGxGm[bshLength * typeSize] { 0x00 };
    uint8_t gammGm[hLength * typeSize] { 0x00 };
    uint8_t betaGm[hLength * typeSize] { 0x00 };

    uint8_t outputGm[bshLength * typeSize] {0x00};
    uint8_t outputMeanGm[bsLength * typeSize] {0x00};
    uint8_t outputVarianceGm[bsLength * typeSize] {0x00};

    param.calFunc(inputXGm, inputGxGm, gammGm, betaGm, outputGm, outputMeanGm, outputVarianceGm, bLength,
        sLength, hLength);
}