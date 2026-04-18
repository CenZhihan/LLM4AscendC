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

__aicore__ inline void GetLayerNormNDTilingInfo(const ShapeInfo &inputShapeInfo, const uint32_t stackBufferSize,
                                                const uint32_t typeSize, const bool isReuseSource, LayerNormSeparateTiling &tiling)
{
    constexpr uint32_t LAYERNORM_FOLD_NUM = 2;
    uint32_t aLength = inputShapeInfo.shape[0];
    uint32_t rLength = inputShapeInfo.shape[1];

    int32_t typeAignSize = 32 / typeSize;
    uint32_t rLengthWithPadding = (rLength + typeAignSize - 1) / typeAignSize * typeAignSize;

    uint32_t rHeadLength = 64;
    uint32_t k = 6;
    for (uint32_t i = 0; i < rLengthWithPadding; i++) {
        if (rHeadLength * LAYERNORM_FOLD_NUM > rLength) {
            k += i;
            break;
        }
        rHeadLength *= LAYERNORM_FOLD_NUM;
    }
    tiling.rLength = rLength;
    tiling.oneTmpSize = k;
    tiling.rHeadLength = rHeadLength;
    uint32_t kOverflow = k;
    if (pow(LAYERNORM_FOLD_NUM, kOverflow) < rLength) {
        kOverflow += 1;
    }
    uint32_t rLengthOverflow = static_cast<uint32_t>(pow(LAYERNORM_FOLD_NUM, kOverflow));
    float k2Rec = static_cast<float>(1) / static_cast<float>(rLengthOverflow);
    float k2RRec = static_cast<float>(rLengthOverflow) / static_cast<float>(rLength);
    tiling.k2Rec = k2Rec;
    tiling.k2RRec = k2RRec;
}

template <typename U, typename T, bool isNoBeta = false, bool isNoGamma = false, bool isReuseSource = false>
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

    inputXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(inputXGm), arLength);
    gammGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(gammGm), rLengthWithPadding);
    betaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(betaGm), rLengthWithPadding);

    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(outputGm), arLength);
    outputMeanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputMeanGm), aLength);
    outputRstdGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputRstdGm), aLength);

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECIN, 1> inQueueGamma;
    TQue<TPosition::VECIN, 1> inQueueBeta;
    TQue<TPosition::VECOUT, 1> outQueue;
    TQue<TPosition::VECOUT, 1> outQueueMean;
    TQue<TPosition::VECOUT, 1> outQueueRstd;
    LocalTensor<U> gammaLocal;
    LocalTensor<U> betaLocal;

    pipe.InitBuffer(inQueueX, 1, sizeof(T) * arLength);
    if constexpr (!isNoGamma) {
        pipe.InitBuffer(inQueueGamma, 1, sizeof(U) * rLengthWithPadding);
        gammaLocal = inQueueGamma.AllocTensor<U>();
    }
    if constexpr (!isNoBeta) {
        pipe.InitBuffer(inQueueBeta, 1, sizeof(U) * rLengthWithPadding);
        betaLocal = inQueueBeta.AllocTensor<U>();
    }
    pipe.InitBuffer(outQueue, 1, sizeof(T) * arLength);
    pipe.InitBuffer(outQueueMean, 1, sizeof(float) * aLength);
    pipe.InitBuffer(outQueueRstd, 1, sizeof(float) * aLength);

    LocalTensor<T> inputXLocal = inQueueX.AllocTensor<T>();
    LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
    LocalTensor<float> meanLocal = outQueueMean.AllocTensor<float>();
    LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();

    DataCopyPadParams dataCopyPadParams;
    for (int32_t i = 0; i < aLength; i++) {
        DataCopyPad(inputXLocal[i * rLengthWithPadding], inputXGlobal[i * rLengthWithPadding], {1, static_cast<uint16_t>(rLengthWithPadding * sizeof(T)), 0, 0}, dataCopyPadParams);
    }
    if constexpr (!isNoGamma) {
        DataCopy(gammaLocal, gammGlobal, rLengthWithPadding);
    }
    if constexpr (!isNoBeta) {
        DataCopy(betaLocal, betaGlobal, rLengthWithPadding);
    }

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

    LayerNormSeparateTiling tiling;
    uint32_t inputShape[2] = {aLength, rLength};
    ShapeInfo shapeInfo{(uint8_t)3, inputShape, (uint8_t)3, inputShape, dataFormat};

    GetLayerNormNDTilingInfo(shapeInfo, stackBufferSize, sizeof(T), isReuseSource, tiling);
    static constexpr LayerNormConfig config = {isNoBeta, isNoGamma, false};
    if constexpr (!isNoGamma && !isNoBeta) {
        LayerNorm<U, T, isReuseSource, config>(outputLocal, meanLocal, rstdLocal, inputXLocal, gammaLocal, betaLocal,
                                               epsilon, para, tiling);
    } else if constexpr (!isNoGamma) {
        LayerNorm<U, T, isReuseSource, config>(outputLocal, meanLocal, rstdLocal, inputXLocal, gammaLocal, gammaLocal,
                                               epsilon, para, tiling);
    } else if constexpr (!isNoBeta) {
        LayerNorm<U, T, isReuseSource, config>(outputLocal, meanLocal, rstdLocal, inputXLocal, betaLocal, betaLocal,
                                               epsilon, para, tiling);
    } else {
        LayerNorm<U, T, isReuseSource, config>(outputLocal, meanLocal, rstdLocal, inputXLocal, inputXLocal, inputXLocal,
                                               epsilon, para, tiling);
    }
    for (int32_t i = 0; i < aLength; i++) {
        DataCopyPad(outputGlobal[i * rLengthWithPadding], outputLocal[i * rLengthWithPadding], {1, static_cast<uint16_t>(rLengthWithPadding * sizeof(T)), 0, 0});
    }

    DataCopyPad(outputMeanGlobal, meanLocal, {1, static_cast<uint16_t>(aLength * sizeof(T)), 0, 0});
    DataCopyPad(outputRstdGlobal, rstdLocal, {1, static_cast<uint16_t>(aLength * sizeof(T)), 0, 0});

    inQueueX.FreeTensor(inputXLocal);
    if constexpr (!isNoGamma) {
        inQueueGamma.FreeTensor(gammaLocal);
    }
    if constexpr (!isNoBeta) {
        inQueueBeta.FreeTensor(betaLocal);
    }
    outQueue.FreeTensor(outputLocal);
    outQueueMean.FreeTensor(meanLocal);
    outQueueRstd.FreeTensor(rstdLocal);
}

struct layernormrstdTestParams
{
    uint32_t aLength;
    uint32_t rLength;
    uint32_t rLengthWithPadding;
    uint32_t typeSizeU;
    uint32_t typeSizeT;
    void (*cal_func)(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint32_t, uint32_t, uint32_t);
};

class layernormrstdTestSuite : public testing::Test, public testing::WithParamInterface<layernormrstdTestParams>
{
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
                            layernormrstdTestParams{64, 16, 16, sizeof(half), sizeof(half), MainLayernormrstdTest<half, half, false, false>},
                            layernormrstdTestParams{64, 16, 16, sizeof(float), sizeof(half), MainLayernormrstdTest<float, half, false, false>},
                            layernormrstdTestParams{64, 16, 16, sizeof(float), sizeof(float), MainLayernormrstdTest<float, float, false, false>},
                            // dtype combination
                            layernormrstdTestParams{8, 128, 128, sizeof(half), sizeof(half), MainLayernormrstdTest<half, half, false, false>},
                            layernormrstdTestParams{8, 128, 128, sizeof(float), sizeof(half), MainLayernormrstdTest<float, half, false, false>},
                            layernormrstdTestParams{8, 128, 128, sizeof(float), sizeof(float), MainLayernormrstdTest<float, float, false, false>},
                            // dtype combination
                            layernormrstdTestParams{1, 4080, 4080, sizeof(float), sizeof(float), MainLayernormrstdTest<float, float, false, false>},
                            layernormrstdTestParams{1, 4096, 4096, sizeof(float), sizeof(float), MainLayernormrstdTest<float, float, false, false>},
                            layernormrstdTestParams{1, 4112, 4112, sizeof(float), sizeof(float), MainLayernormrstdTest<float, float, false, false>},
                            // dtype combination
                            layernormrstdTestParams{1, 16256, 16256, sizeof(float), sizeof(half), MainLayernormrstdTest<float, half, false, false>},
                            layernormrstdTestParams{1, 16384, 16384, sizeof(float), sizeof(half), MainLayernormrstdTest<float, half, false, false>},
                            layernormrstdTestParams{1, 16400, 16400, sizeof(float), sizeof(half), MainLayernormrstdTest<float, half, false, false>},
                            layernormrstdTestParams{1, 16512, 16512, sizeof(float), sizeof(half), MainLayernormrstdTest<float, half, false, false>},
                            // dtype combinationtrue
                            layernormrstdTestParams{1, 32768, 32768, sizeof(half), sizeof(half), MainLayernormrstdTest<half, half, true, true>},
                            layernormrstdTestParams{1, 32784, 32784, sizeof(half), sizeof(half), MainLayernormrstdTest<half, half, true, true>},
                            layernormrstdTestParams{1, 32896, 32896, sizeof(half), sizeof(half), MainLayernormrstdTest<half, half, true, true>},
                            // rLength != rWithPad
                            layernormrstdTestParams{8, 27, 32, sizeof(half), sizeof(half), MainLayernormrstdTest<half, half, false, false>},
                            layernormrstdTestParams{8, 29, 32, sizeof(float), sizeof(half), MainLayernormrstdTest<float, half, false, false>},
                            layernormrstdTestParams{8, 27, 32, sizeof(float), sizeof(float), MainLayernormrstdTest<float, float, false, false>}));

TEST_P(layernormrstdTestSuite, LayernormrstdTestCase)
{
    auto param = GetParam();

    uint32_t aLength = param.aLength;
    uint32_t rLength = param.rLength;
    uint32_t rLengthWithPadding = param.rLengthWithPadding;
    uint32_t typeSizeU = param.typeSizeU;
    uint32_t typeSizeT = param.typeSizeT;

    uint32_t arLength = aLength * rLength;

    uint8_t inputXGm[aLength * rLengthWithPadding * typeSizeT]{0x00};
    uint8_t gammGm[rLengthWithPadding * sizeof(float)]{0x00};
    uint8_t betaGm[rLengthWithPadding * sizeof(float)]{0x00};

    uint8_t outputGm[aLength * rLengthWithPadding * typeSizeT]{0x00};
    uint8_t outputMeanGm[aLength * sizeof(float)]{0x00};
    uint8_t outputRstdeGm[aLength * sizeof(float)]{0x00};

    param.cal_func(inputXGm, gammGm, betaGm, outputGm, outputMeanGm, outputRstdeGm, aLength, rLength, rLengthWithPadding);

    for (int32_t i = 0; i < aLength * rLengthWithPadding * typeSizeT; i++) {
        EXPECT_EQ(outputGm[i], 0x00);
    }
}