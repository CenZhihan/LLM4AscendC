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

class TEST_layernormgrad : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TEST_layernormgrad SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "TEST_layernormgrad TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

template <typename T>
void GetLayerNormGradNDLoopInfo(const LocalTensor<T> &input, const uint32_t stackBufferSize, const uint32_t typeSize,
    const bool isReuseSource, LayerNormGradTiling &tiling)
{
    ShapeInfo info = input.GetShapeInfo();
    tiling.bLength = info.shape[0];
    tiling.sLength = info.shape[1];
    tiling.hLength = info.shape[2];
    tiling.originalHLength = info.shape[2];

    uint32_t inputXSize = tiling.bLength * tiling.sLength * tiling.hLength;
    uint32_t meanVarSize = tiling.bLength * tiling.sLength;

    uint32_t needBufferBlock;
    if (typeSize == B16_BYTE_SIZE) {
        needBufferBlock = 9;
    } else if (isReuseSource == true) {
        needBufferBlock = 4;
    } else {
        needBufferBlock = 6;
    }

    tiling.stackBufferSize = stackBufferSize;
    tiling.oneCalSize = stackBufferSize * sizeof(uint8_t) / sizeof(float) / needBufferBlock;
    tiling.oneCalSize = tiling.oneCalSize / tiling.hLength * tiling.hLength;
    tiling.nohCalSize = tiling.oneCalSize / tiling.hLength;
    ASSERT(tiling.oneCalSize > 0);
    tiling.loopNum = inputXSize / tiling.oneCalSize;

    tiling.tailSize = inputXSize % tiling.oneCalSize;
    tiling.nohTailSize = tiling.tailSize / tiling.hLength;
    ASSERT(tiling.tailSize % tiling.hLength == 0);

    tiling.tmpTensorBSHPos = 0;
    tiling.tmpTensorBSHSize = tiling.oneCalSize;

    tiling.pdVarTensorPos = tiling.tmpTensorBSHPos + tiling.tmpTensorBSHSize;
    tiling.pdVarTensorSize = tiling.oneCalSize;

    tiling.pdMeanTensorPos = tiling.pdVarTensorPos + tiling.pdVarTensorSize;
    tiling.pdMeanTensorSize = tiling.oneCalSize;

    tiling.x1TensorPos = 0;
    tiling.x1TensorSize = tiling.oneCalSize;

    tiling.x2TensorPos = 0;
    tiling.x2TensorSize = tiling.oneCalSize;

    tiling.x3TensorPos = tiling.pdMeanTensorPos + tiling.pdMeanTensorSize;
    tiling.x3TensorSize = tiling.oneCalSize;

    if (!(isReuseSource == true && typeSize == B32_BYTE_SIZE)) {
        tiling.x1TensorPos = tiling.pdMeanTensorPos + tiling.pdMeanTensorSize;
        tiling.x1TensorSize = tiling.oneCalSize;
        tiling.x2TensorPos = tiling.x1TensorPos + tiling.x1TensorSize;
        tiling.x2TensorSize = tiling.oneCalSize;
        tiling.x3TensorPos = tiling.x2TensorPos + tiling.x2TensorSize;
        tiling.x3TensorSize = tiling.oneCalSize;
    }

    tiling.tmpTensorPos = 0;
    tiling.tmpTensorSize = tiling.oneCalSize;
    tiling.tmpTensor1Pos = 0;
    tiling.tmpTensor1Size = tiling.oneCalSize;
    tiling.tmpTensor2Pos = 0;
    tiling.tmpTensor2Size = tiling.oneCalSize;

    if (typeSize == B16_BYTE_SIZE) {
        tiling.tmpTensorPos = tiling.x3TensorPos + tiling.x3TensorSize;
        tiling.tmpTensorSize = tiling.oneCalSize;
        tiling.tmpTensor1Pos = tiling.tmpTensorPos + tiling.tmpTensorSize;
        tiling.tmpTensor1Size = tiling.oneCalSize;
        tiling.tmpTensor2Pos = tiling.tmpTensor1Pos + tiling.tmpTensor1Size;
        tiling.tmpTensor2Size = tiling.oneCalSize;
    }

    float lastDimValueBack = 1.0;
    lastDimValueBack = lastDimValueBack / static_cast<float>(tiling.originalHLength);
    tiling.lastDimValueBack = *(uint32_t *)(&lastDimValueBack);

    float lastDimValueBackMulTwo = 2.0;
    lastDimValueBackMulTwo = lastDimValueBackMulTwo / static_cast<float>(tiling.originalHLength);
    tiling.lastDimValueBackMulTwo = *(uint32_t *)(&lastDimValueBackMulTwo);
}

template <typename dataType, bool isReuseSource = false>
__aicore__ inline void mainLayernormgradTest(GM_ADDR outputPdXGm, GM_ADDR resForGammaGm, GM_ADDR inputDyGm,
    GM_ADDR inputXGm, GM_ADDR inputVarianceGm, GM_ADDR inputMeanGm, GM_ADDR inputGammaGm, uint32_t bLength,
    uint32_t sLength, uint32_t hLength)
{
    TPipe tPipe;
    dataType epsilon = 0.001;
    DataFormat dataFormat = DataFormat::ND;

    GlobalTensor<dataType> outputPdXGlobal;
    GlobalTensor<dataType> resForGammaGlobal;
    GlobalTensor<dataType> inputDyGlobal;
    GlobalTensor<dataType> inputXGlobal;
    GlobalTensor<dataType> inputVarianceGlobal;
    GlobalTensor<dataType> inputMeanGlobal;
    GlobalTensor<dataType> inputGammaGlobal;

    uint32_t bshLength = bLength * sLength * hLength;
    uint32_t bsLength = bLength * sLength;

    outputPdXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(outputPdXGm), bshLength);
    resForGammaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(resForGammaGm), bshLength);

    inputDyGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputDyGm), bshLength);
    inputXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputXGm), bshLength);
    inputVarianceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputVarianceGm), bsLength);
    inputMeanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputMeanGm), bsLength);
    inputGammaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputGammaGm), hLength);


    LOCAL_TENSOR_REGISTER(outputPdXLocal, dataType, VECCALC, 0, bshLength);
    LOCAL_TENSOR_REGISTER(resForGammaLocal, dataType, VECCALC, bshLength * sizeof(dataType), bshLength);

    LOCAL_TENSOR_REGISTER(inputDyLocal, dataType, VECCALC, 2 * bshLength * sizeof(dataType), bshLength);
    LOCAL_TENSOR_REGISTER(inputXLocal, dataType, VECCALC, 3 * bshLength * sizeof(dataType), bshLength);
    LOCAL_TENSOR_REGISTER(inputVarianceLocal, dataType, VECCALC, 4 * bshLength * sizeof(dataType), bsLength);
    LOCAL_TENSOR_REGISTER(inputMeanLocal, dataType, VECCALC, 5 * bshLength * sizeof(dataType), bsLength);
    LOCAL_TENSOR_REGISTER(inputGammaLocal, dataType, VECCALC, 6 * bshLength * sizeof(dataType), hLength);


    uint32_t inputShape1[1] = {static_cast<int>(hLength)};
    uint32_t inputShape2[2] = {static_cast<int>(bLength), static_cast<int>(sLength)};
    uint32_t inputShape3[3] = {static_cast<int>(bLength), static_cast<int>(sLength), static_cast<int>(hLength)};

    ShapeInfo shapeInfo1 { 1, inputShape1, dataFormat };
    ShapeInfo shapeInfo2 { 2, inputShape2, dataFormat };
    ShapeInfo shapeInfo3 { 3, inputShape3, dataFormat };

    inputDyLocal.SetShapeInfo(shapeInfo3);
    inputXLocal.SetShapeInfo(shapeInfo3);
    inputVarianceLocal.SetShapeInfo(shapeInfo2);
    inputMeanLocal.SetShapeInfo(shapeInfo2);
    inputGammaLocal.SetShapeInfo(shapeInfo1);

    DataCopy(inputDyLocal, inputDyGlobal, bshLength);
    DataCopy(inputXLocal, inputXGlobal, bshLength);
    DataCopy(inputVarianceLocal, inputVarianceGlobal, bsLength);
    DataCopy(inputMeanLocal, inputMeanGlobal, bsLength);
    DataCopy(inputGammaLocal, inputGammaGlobal, hLength);

    PipeBarrier<PIPE_ALL>();

    uint32_t stackBufferSize = 0;
    {
        LocalTensor<uint8_t> stackBuffer;
        bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackBuffer);
        stackBufferSize = stackBuffer.GetSize();
    }

    LayerNormGradTiling layerNormGradTiling;
    GetLayerNormGradNDLoopInfo(inputDyLocal, stackBufferSize, sizeof(dataType), isReuseSource, layerNormGradTiling);
    LayerNormGrad<dataType, isReuseSource>(outputPdXLocal, resForGammaLocal, inputDyLocal, inputXLocal,
        inputVarianceLocal, inputMeanLocal, inputGammaLocal, (dataType)epsilon, layerNormGradTiling,
        { DataFormat::ND });
    PipeBarrier<PIPE_ALL>();

    DataCopy(outputPdXGlobal, outputPdXLocal, bshLength);
    DataCopy(resForGammaGlobal, resForGammaLocal, bshLength);
    PipeBarrier<PIPE_ALL>();
}

struct layernormgradTestParams {
    uint32_t bLength;
    uint32_t sLength;
    uint32_t hLength;
    uint32_t typeSize;
    void (*cal_func)(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint32_t, uint32_t,
        uint32_t);
};

class layernormgradTestSuite : public testing::Test, public testing::WithParamInterface<layernormgradTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "layernormgradTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "layernormgradTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_layernormgrad, layernormgradTestSuite,
    ::testing::Values(layernormgradTestParams { 2, 32, 16, sizeof(half), mainLayernormgradTest<half, true> },
    layernormgradTestParams { 2, 32, 16, sizeof(half), mainLayernormgradTest<half, false> },
    layernormgradTestParams { 2, 32, 16, sizeof(float), mainLayernormgradTest<float, true> },
    layernormgradTestParams { 2, 32, 16, sizeof(float), mainLayernormgradTest<float, false> },
    layernormgradTestParams { 2, 16, 32, sizeof(half), mainLayernormgradTest<half, true> },
    layernormgradTestParams { 2, 16, 32, sizeof(half), mainLayernormgradTest<half, false> },
    layernormgradTestParams { 2, 16, 32, sizeof(float), mainLayernormgradTest<float, true> },
    layernormgradTestParams { 2, 16, 32, sizeof(float), mainLayernormgradTest<float, false> }));

TEST_P(layernormgradTestSuite, layernormgradTestCase)
{
    auto param = GetParam();

    uint32_t bLength = param.bLength;
    uint32_t sLength = param.sLength;
    uint32_t hLength = param.hLength;
    uint32_t typeSize = param.typeSize;

    uint32_t bshLength = bLength * sLength * hLength;
    uint32_t bsLength = bLength * sLength;

    uint8_t outputPdXGm[bshLength * typeSize] { 0x00 };
    uint8_t resForGammaGm[bshLength * typeSize] { 0x00 };

    uint8_t inputDyGm[bshLength * typeSize] { 0x00 };
    uint8_t inputXGm[bshLength * typeSize] { 0x00 };
    uint8_t inputVarianceGm[bsLength * typeSize] { 0x00 };
    uint8_t inputMeanGm[bsLength * typeSize] { 0x00 };
    uint8_t inputGammaGm[hLength * typeSize] { 0x00 };
    param.cal_func(outputPdXGm, resForGammaGm, inputDyGm, inputXGm, inputVarianceGm, inputMeanGm, inputGammaGm,
        bLength, sLength, hLength);

    for (int32_t i = 0; i < bshLength * typeSize; i++) {
        EXPECT_EQ(outputPdXGm[i], 0x00);
        EXPECT_EQ(resForGammaGm[i], 0x00);
    }
}