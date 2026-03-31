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

class TEST_layernormgradbeta : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TEST_layernormgradbeta SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "TEST_layernormgradbeta TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

__aicore__ inline void CheckLayerNormGradBetaNDTilingInfo(const ShapeInfo info, const uint32_t stackBufferSize,
    const uint32_t typeSize)
{
    const uint32_t bLength = info.shape[0];
    const uint32_t sLength = info.shape[1];
    const uint32_t hLength = info.shape[2];
    const uint32_t originalHLength = info.originalShape[2];

    ASCENDC_ASSERT((bLength > 0),
                   { KERNEL_LOG(KERNEL_ERROR, "GetLayerNormGradBetaNDTilingInfo ShapeInfo bLength must > 0!"); });
    ASCENDC_ASSERT((sLength > 0),
                   { KERNEL_LOG(KERNEL_ERROR, "GetLayerNormGradBetaNDTilingInfo ShapeInfo sLength must > 0!"); });
    ASCENDC_ASSERT((hLength > 0),
                   { KERNEL_LOG(KERNEL_ERROR, "GetLayerNormGradBetaNDTilingInfo ShapeInfo hLength must > 0!"); });
    ASCENDC_ASSERT((originalHLength > 0),
                   { KERNEL_LOG(KERNEL_ERROR, "GetLayerNormGradBetaNDTilingInfo originalHLength must > 0!"); });
    ASCENDC_ASSERT(((stackBufferSize / sizeof(float)) >= (hLength + hLength)), {
        KERNEL_LOG(KERNEL_ERROR, "tiling.stackBufferSize (%d) <= 2 * tiling.hLength (%d)!", stackBufferSize, hLength);
    });
    ASCENDC_ASSERT(((typeSize == sizeof(half)) || (typeSize == sizeof(float))),
                   { KERNEL_LOG(KERNEL_ERROR, "GetLayerNormGradBetaNDTilingInfo ShapeInfo bLength must > 0!"); });
}

__aicore__ inline void GetLayerNormGradBetaNDTilingInfo(const ShapeInfo info, const uint32_t stackBufferSize,
    const uint32_t typeSize, const bool isReuseSource, LayerNormGradBetaTiling &tiling)
{
    CheckLayerNormGradBetaNDTilingInfo(info, stackBufferSize, typeSize);

    uint32_t bLength = info.shape[0];
    uint32_t sLength = info.shape[1];
    uint32_t hLength = info.shape[2];
    uint32_t originalHLength = info.originalShape[2];

    if ((tiling.bLength == bLength) && (tiling.sLength == sLength) && (tiling.hLength == hLength) &&
        (tiling.originalHLength == originalHLength)) {
        return;
    }

    tiling.bLength = bLength;
    tiling.sLength = sLength;
    tiling.hLength = hLength;
    tiling.originalHLength = originalHLength;

    tiling.bshLength = tiling.bLength * tiling.sLength * tiling.hLength;
    tiling.bsLength = tiling.bLength * tiling.sLength;

    tiling.stackBufferSize = stackBufferSize / sizeof(float);

    tiling.oneCalSize = tiling.stackBufferSize;
    if (typeSize == B16_BYTE_SIZE) {
        tiling.numberOfTmpBuf = TWO_OF_STACK_BUFFER;
        tiling.oneCalSize = tiling.oneCalSize - tiling.hLength - tiling.hLength;
        tiling.oneCalSize = tiling.oneCalSize / tiling.numberOfTmpBuf;
    }

    tiling.oneCalSize = tiling.oneCalSize / tiling.hLength * tiling.hLength;
    ASCENDC_ASSERT((tiling.oneCalSize != 0),
                   { KERNEL_LOG(KERNEL_ERROR, "GetLayerNormGradBetaNDTilingInfo tiling.oneCalSize == 0!"); });

    if (tiling.oneCalSize == 0) {
        return;
    }

    if (tiling.oneCalSize >= tiling.bshLength) {
        tiling.oneCalSize = tiling.bshLength;
    }

    tiling.loopRound = tiling.bshLength / tiling.oneCalSize;
    tiling.inputTailSize = tiling.bshLength % tiling.oneCalSize;

    tiling.inputTailPos = tiling.bshLength - tiling.inputTailSize;
    tiling.bsTailSize = tiling.inputTailSize / tiling.hLength;

    tiling.bshCurLength = tiling.oneCalSize;
    tiling.bsCurLength = tiling.oneCalSize / tiling.hLength;

    tiling.resForGammaTmpTensorPos = 0;

    if (typeSize == B16_BYTE_SIZE) {
        tiling.gammaTempTensorPos = 0;
        tiling.betaTempTensorPos = tiling.gammaTempTensorPos + tiling.hLength;
        tiling.inputDyTmpTensorPos = tiling.betaTempTensorPos + tiling.hLength;
        tiling.resForGammaTmpTensorPos = tiling.inputDyTmpTensorPos + tiling.oneCalSize;
    }
}

template <typename T, bool isReuseSource = false> class KernelLayernormGradBeta {
public:
    __aicore__ inline KernelLayernormGradBeta() {}
    __aicore__ inline void Init(__gm__ uint8_t* resForGammaGm, __gm__ uint8_t* inputDyGm,
        __gm__ uint8_t* outputPdGammaGm, __gm__ uint8_t* outputPdBetaGm, uint32_t bLength, uint32_t sLength,
        uint32_t hLength, DataFormat dataFormat)
    {
        this->bLength = bLength;
        this->sLength = sLength;
        this->hLength = hLength;
        this->dataFormat = dataFormat;

        bshLength = bLength * sLength * hLength;
        bsLength = bLength * sLength;

        resForGammaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(resForGammaGm), bshLength);
        inputDyGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputDyGm), bshLength);

        outputPdGammaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputPdGammaGm), hLength);
        outputPdBetaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputPdBetaGm), hLength);


        pipe.InitBuffer(inQueueResForGamma, 1, sizeof(T) * bshLength);
        pipe.InitBuffer(inQueueDy, 1, sizeof(T) * bshLength);

        pipe.InitBuffer(outQueuePdGamma, 1, sizeof(T) * hLength);
        pipe.InitBuffer(outQueuePdBeta, 1, sizeof(T) * hLength);

        LocalTensor<uint8_t> stackBuffer;
        bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackBuffer);
        stackBufferSize = stackBuffer.GetSize();
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
        LocalTensor<T> resForGammaLocal = inQueueResForGamma.AllocTensor<T>();
        LocalTensor<T> inputDyLocal = inQueueDy.AllocTensor<T>();

        DataCopy(resForGammaLocal, resForGammaGlobal, bshLength);
        DataCopy(inputDyLocal, inputDyGlobal, bshLength);

        inQueueResForGamma.EnQue(resForGammaLocal);
        inQueueDy.EnQue(inputDyLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> resForGammaLocal = inQueueResForGamma.DeQue<T>();
        LocalTensor<T> inputDyLocal = inQueueDy.DeQue<T>();
        LocalTensor<T> outputPdGammaLocal = outQueuePdGamma.AllocTensor<T>();
        LocalTensor<T> outputPdBetaLocal = outQueuePdBeta.AllocTensor<T>();

        uint32_t inputShape3[3] = {bLength, sLength, hLength};
        ShapeInfo shapeInfo3{ 3, inputShape3, 3, inputShape3, dataFormat};

        LayerNormGradBetaTiling layerNormGradBetaTiling;
        GetLayerNormGradBetaNDTilingInfo(shapeInfo3, stackBufferSize, sizeof(T), isReuseSource,
            layerNormGradBetaTiling);

        LayerNormGradBeta<T, isReuseSource>(outputPdGammaLocal, outputPdBetaLocal, resForGammaLocal, inputDyLocal,
            layerNormGradBetaTiling);

        outQueuePdGamma.EnQue<T>(outputPdGammaLocal);
        outQueuePdBeta.EnQue<T>(outputPdBetaLocal);

        inQueueResForGamma.FreeTensor(resForGammaLocal);
        inQueueDy.FreeTensor(inputDyLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> outputPdGammaLocal = outQueuePdGamma.DeQue<T>();
        LocalTensor<T> outputPdBetaLocal = outQueuePdBeta.DeQue<T>();

        DataCopy(outputPdGammaGlobal, outputPdGammaLocal, hLength);
        DataCopy(outputPdBetaGlobal, outputPdBetaLocal, hLength);

        outQueuePdGamma.FreeTensor(outputPdGammaLocal);
        outQueuePdBeta.FreeTensor(outputPdBetaLocal);
    }

private:
    TPipe pipe;

    TQue<TPosition::VECIN, 1> inQueueResForGamma, inQueueDy;
    TQue<TPosition::VECOUT, 1> outQueuePdGamma, outQueuePdBeta;

    GlobalTensor<T> resForGammaGlobal;
    GlobalTensor<T> inputDyGlobal;
    GlobalTensor<T> outputPdGammaGlobal;
    GlobalTensor<T> outputPdBetaGlobal;

    uint32_t bLength;
    uint32_t sLength;
    uint32_t hLength;
    DataFormat dataFormat;

    uint32_t bshLength;
    uint32_t bsLength;

    uint32_t stackBufferSize = 0;
};

template <typename dataType, bool isReuseSource = false>
__aicore__ inline void mainLayernormgradbetaTest(GM_ADDR outputPdGammaGm, GM_ADDR outputPdBetaGm,
    GM_ADDR resForGammaGm, GM_ADDR inputDyGm, uint32_t bLength, uint32_t sLength, uint32_t hLength)
{
    KernelLayernormGradBeta<dataType, isReuseSource> op;
    op.Init(resForGammaGm, inputDyGm, outputPdGammaGm, outputPdBetaGm, bLength, sLength, hLength, DataFormat::ND);
    op.Process();
}

struct layernormgradbetaTestParams {
    uint32_t bLength;
    uint32_t sLength;
    uint32_t hLength;
    uint32_t typeSize;
    void (*cal_func)(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint32_t, uint32_t, uint32_t);
};

class layernormgradbetaTestSuite : public testing::Test,
    public testing::WithParamInterface<layernormgradbetaTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "layernormgradbetaTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "layernormgradbetaTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_layernormgradbeta, layernormgradbetaTestSuite,
    ::testing::Values(layernormgradbetaTestParams { 2, 32, 16, sizeof(half), mainLayernormgradbetaTest<half, true> },
    layernormgradbetaTestParams { 2, 32, 16, sizeof(half), mainLayernormgradbetaTest<half, false> },
    layernormgradbetaTestParams { 2, 32, 16, sizeof(float), mainLayernormgradbetaTest<float, true> },
    layernormgradbetaTestParams { 2, 32, 16, sizeof(float), mainLayernormgradbetaTest<float, false> },
    layernormgradbetaTestParams { 2, 16, 32, sizeof(half), mainLayernormgradbetaTest<half, true> },
    layernormgradbetaTestParams { 2, 16, 32, sizeof(half), mainLayernormgradbetaTest<half, false> },
    layernormgradbetaTestParams { 2, 16, 32, sizeof(float), mainLayernormgradbetaTest<float, true> },
    layernormgradbetaTestParams { 2, 16, 32, sizeof(float), mainLayernormgradbetaTest<float, false> }));

TEST_P(layernormgradbetaTestSuite, layernormgradbetaTestCase)
{
    auto param = GetParam();

    uint32_t bLength = param.bLength;
    uint32_t sLength = param.sLength;
    uint32_t hLength = param.hLength;
    uint32_t typeSize = param.typeSize;

    uint32_t bshLength = bLength * sLength * hLength;
    uint32_t bsLength = bLength * sLength;

    uint8_t outputPdGammaGm[hLength * typeSize] { 0x00 };
    uint8_t outputPdBetaGm[hLength * typeSize] { 0x00 };

    uint8_t resForGammaGm[bshLength * typeSize] { 0x00 };
    uint8_t inputDyGm[bshLength * typeSize] { 0x00 };
    param.cal_func(outputPdGammaGm, outputPdBetaGm, resForGammaGm, inputDyGm, bLength, sLength, hLength);

    for (int32_t i = 0; i < hLength * typeSize; i++) {
        EXPECT_EQ(outputPdGammaGm[i], 0x00);
        EXPECT_EQ(outputPdBetaGm[i], 0x00);
    }
}