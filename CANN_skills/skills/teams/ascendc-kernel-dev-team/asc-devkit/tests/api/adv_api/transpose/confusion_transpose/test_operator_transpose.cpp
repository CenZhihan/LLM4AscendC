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
#include "kernel_utils.h"
#include <math.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace AscendC;

// scene 1
namespace AscendC {
// scene1,2: srcShape[B, A1, A2, A3]
__aicore__ inline void GetConfusionTranspose0213TilingInfo(const ShapeInfo srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, ConfusionTransposeTiling& tiling)
{
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t shapeB = srcShape.originalShape[0];
    uint32_t shapeA1 = srcShape.originalShape[1];
    uint32_t alignA2 = (srcShape.originalShape[2] + BLOCK_CUBE - 1) / BLOCK_CUBE * BLOCK_CUBE;
    uint32_t widthTiling = (srcShape.originalShape[3] + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t alignA3 = widthTiling * BLOCK_CUBE;

    // stackBuffer is aligned to [16,16]
    uint32_t newPopSize = (stackBufferSize / CUBE_MAX_SIZE) * CUBE_MAX_SIZE; // element
    uint32_t newPopH = newPopSize / BLOCK_CUBE;
    uint32_t needSize = alignA2 * BLOCK_CUBE;
    uint32_t mainBlocks = needSize / newPopSize;
    uint32_t tailSize = needSize % newPopSize;
    uint32_t alignA2MulAlignA3 = alignA2 * alignA3;
    uint32_t batchOffset = shapeA1 * alignA2MulAlignA3;
    uint32_t alignA3MulA1 = shapeA1 * alignA3;
    uint32_t shapeA1BlockCube = shapeA1 * BLOCK_CUBE;
    uint32_t mainOffset = newPopH * mainBlocks * alignA3MulA1;

    tiling.param0 = blockSize;
    tiling.param1 = shapeB;
    tiling.param2 = shapeA1;
    tiling.param3 = alignA3;
    tiling.param4 = alignA2;
    tiling.param5 = widthTiling;
    tiling.param6 = newPopSize;
    tiling.param7 = newPopH;
    tiling.param8 = needSize;
    tiling.param9 = mainBlocks;
    tiling.param10 = tailSize;
    tiling.param11 = alignA2MulAlignA3;
    tiling.param12 = batchOffset;
    tiling.param13 = alignA3MulA1;
    tiling.param14 = shapeA1BlockCube;
    tiling.param15 = mainOffset;
}

// scene 3: srcShape[B, N, S, H/N]
__aicore__ inline void GetConfusionTranspose2NZ012NTilingInfo(const ShapeInfo srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, ConfusionTransposeTiling& tiling)
{
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t shapeB = srcShape.originalShape[0];
    uint32_t shapeN = srcShape.originalShape[1];
    uint32_t hnDiv = srcShape.originalShape[3];
    uint32_t blockNum = BLOCK_CUBE / blockSize;
    uint32_t shapeH = hnDiv * shapeN;
    uint32_t hBlockNum = (shapeH + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t sBlockNum = (srcShape.originalShape[2] + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t alignH = hBlockNum * BLOCK_CUBE;
    uint32_t alignS = sBlockNum * BLOCK_CUBE;
    uint32_t hnDivBlockNum = (hnDiv + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t alignHnDiv = hnDivBlockNum * BLOCK_CUBE;
    uint32_t gap = alignHnDiv - hnDiv;
    uint32_t alignsBlockCube = alignS * BLOCK_CUBE;
    uint32_t prehBlockNum = shapeN * hnDivBlockNum;
    uint32_t dstBatchOffset = shapeN * alignHnDiv * alignS;
    uint32_t srcBatchOffset = alignH * alignS;

    tiling.param0 = blockSize;
    tiling.param1 = shapeB;
    tiling.param2 = shapeN;
    tiling.param3 = hnDiv;
    tiling.param4 = blockNum;
    tiling.param5 = shapeH;
    tiling.param6 = hBlockNum;
    tiling.param7 = sBlockNum;
    tiling.param8 = alignH;
    tiling.param9 = alignS;
    tiling.param10 = hnDivBlockNum;
    tiling.param11 = alignHnDiv;
    tiling.param12 = gap;
    tiling.param13 = alignsBlockCube;
    tiling.param14 = prehBlockNum;
    tiling.param15 = dstBatchOffset;
    tiling.param16 = srcBatchOffset;
}

// scene4: srcShape[B, N, S, H/N]
__aicore__ inline void GetConfusionTranspose2ND012NTilingInfo(const ShapeInfo srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, ConfusionTransposeTiling& tiling)
{
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t shapeB = srcShape.originalShape[0];
    uint32_t shapeN = srcShape.originalShape[1];
    uint32_t hnDiv = srcShape.originalShape[3];
    uint32_t shapeH = shapeN * hnDiv;
    uint32_t hBlockNum = (shapeH + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t sBlockNum = (srcShape.originalShape[2] + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t hnDivBlockNum = (hnDiv + BLOCK_CUBE - 1) / 16;
    uint32_t alignHnDiv = hnDivBlockNum * BLOCK_CUBE;
    uint32_t gap = alignHnDiv - hnDiv;
    uint32_t alignsCube = sBlockNum * CUBE_MAX_SIZE;
    uint32_t prehBlockNum = shapeN * hnDivBlockNum;
    uint32_t alignsMulAlignHnDiv = sBlockNum * BLOCK_CUBE * alignHnDiv;
    uint32_t alignHnDivCube = alignHnDiv * BLOCK_CUBE;
    uint32_t alignHnDivBlockSize = alignHnDiv * blockSize;
    uint32_t dstBatchOffset = shapeN * alignHnDiv * sBlockNum * BLOCK_CUBE;
    uint32_t srcBatchOffset = hBlockNum * sBlockNum * CUBE_MAX_SIZE;
    uint32_t blockNum = BLOCK_CUBE / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = shapeB;
    tiling.param2 = shapeN;
    tiling.param3 = hnDiv;
    tiling.param4 = shapeH;
    tiling.param5 = hBlockNum;
    tiling.param6 = sBlockNum;
    tiling.param7 = hnDivBlockNum;
    tiling.param8 = alignHnDiv;
    tiling.param9 = gap;
    tiling.param10 = alignsCube;
    tiling.param11 = prehBlockNum;
    tiling.param12 = alignsMulAlignHnDiv;
    tiling.param13 = alignHnDivCube;
    tiling.param14 = alignHnDivBlockSize;
    tiling.param15 = dstBatchOffset;
    tiling.param16 = srcBatchOffset;
    tiling.param17 = blockNum;
}

// scene5,6: srcShape[B, N, S, H/N]
__aicore__ inline void GetConfusionTranspose012TilingInfo(const ShapeInfo srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, ConfusionTransposeTiling& tiling)
{
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t shapeB = srcShape.originalShape[0];
    uint32_t shapeN = srcShape.originalShape[1];
    uint32_t hnDiv = srcShape.originalShape[3];
    uint32_t shapeH = shapeN * hnDiv;
    uint32_t hBlockNum = (shapeH + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t sBlockNum = (srcShape.originalShape[2] + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t hnDivBlockNum = (hnDiv + BLOCK_CUBE - 1) / BLOCK_CUBE;
    uint32_t alignH = hBlockNum * BLOCK_CUBE;
    uint32_t alignsCube = sBlockNum * CUBE_MAX_SIZE;
    uint32_t alignhBlockCube = alignH * BLOCK_CUBE;
    uint32_t blockSizeMulAlignH = blockSize * alignH;
    uint32_t srcBatchOffset = shapeN * hnDivBlockNum * BLOCK_CUBE * sBlockNum * BLOCK_CUBE;
    uint32_t dstBatchOffset = alignH * sBlockNum * BLOCK_CUBE;
    uint32_t blockNum = BLOCK_CUBE / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = shapeB;
    tiling.param2 = shapeN;
    tiling.param3 = hnDiv;
    tiling.param4 = shapeH;
    tiling.param5 = hBlockNum;
    tiling.param6 = sBlockNum;
    tiling.param7 = hnDivBlockNum;
    tiling.param8 = alignH;
    tiling.param9 = alignsCube;
    tiling.param10 = alignhBlockCube;
    tiling.param11 = blockSizeMulAlignH;
    tiling.param12 = srcBatchOffset;
    tiling.param13 = dstBatchOffset;
    tiling.param14 = blockNum;
}

// scene7: srcShape[height, width]
__aicore__ inline void GetTransposeOnlyTilingInfo(const ShapeInfo srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, ConfusionTransposeTiling& tiling)
{
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t height = srcShape.originalShape[0];
    uint32_t width = srcShape.originalShape[1];
    uint32_t highBlock = height / BLOCK_CUBE;
    uint32_t stride = height * blockSize * typeSize / ONE_BLK_SIZE;
    uint32_t repeat = width / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;
}

__aicore__ inline void GetTransposeTilingInfo(const ShapeInfo srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, const uint32_t transposeTypeIn, ConfusionTransposeTiling& tiling)
{
    if (transposeTypeIn == 1 || transposeTypeIn == 2) {
        GetConfusionTranspose0213TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    } else if (transposeTypeIn == 3) {
        GetConfusionTranspose2NZ012NTilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    } else if (transposeTypeIn == 4) {
        GetConfusionTranspose2ND012NTilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    } else if (transposeTypeIn == 5 || transposeTypeIn == 6) {
        GetConfusionTranspose012TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    } else if (transposeTypeIn == 7) {
        GetTransposeOnlyTilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    }
}

template <typename T>
class KernelTransposeFirst {
public:
    __aicore__ inline KernelTransposeFirst() { }

    __aicore__ inline void Init(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint32_t A1In, uint32_t A2In, uint32_t A3In,
        TransposeType transposetypeIn)
    {
        A1 = A1In;
        A2 = A2In;
        A3 = A3In;
        TransposeType transposetype = transposetypeIn;
        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrcVecIn, 1, A1 * A2 * A3 * sizeof(T));
        pipe.InitBuffer(inQueueSrcVecOut, 1, A1 * A2 * A3 * sizeof(T));
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
        LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal, A1 * A2 * A3);
        inQueueSrcVecIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> srcLocal = inQueueSrcVecIn.DeQue<T>();
        LocalTensor<T> dstLocal = inQueueSrcVecOut.AllocTensor<T>();
        uint32_t srcOriShape[] = {1, A1, A2, A3};
        uint32_t srcShape[] = { 1, A1, A3 / 16, A2 / 16, 16, 16 };
        ShapeInfo srcShapeInfo(6, srcShape, 4, srcOriShape, DataFormat::NZ);

        uint32_t dstOriShape[] = {1, A2, A1, A3};
        uint32_t dstShape[] = {1, A2, A1, A3};
        ShapeInfo dstShapeInfo(4, dstShape, 4, dstOriShape, DataFormat::ND);

        LocalTensor<T> stackBuffer;
        PopStackBuffer<T, TPosition::LCM>(stackBuffer);
        uint32_t popSize = stackBuffer.GetSize();
        ConfusionTransposeTiling tiling;
        GetTransposeTilingInfo(srcShapeInfo, popSize, sizeof(T), 1, tiling);
        AscendC::Transpose(dstLocal, srcLocal, TransposeType::TRANSPOSE_NZ2ND_0213, tiling);

        inQueueSrcVecOut.EnQue<T>(dstLocal);
        inQueueSrcVecIn.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = inQueueSrcVecOut.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, A1 * A2 * A3);
        inQueueSrcVecOut.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrcVecIn;
    TQue<TPosition::VECOUT, 1> inQueueSrcVecOut;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    int32_t A1, A2, A3;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void TransposeFirst(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint32_t A1In, uint32_t A2In,
    uint32_t A3In, TransposeType transposetypeIn)
{
    AscendC::KernelTransposeFirst<T> op;
    op.Init(dstGm, srcGm, A1In, A2In, A3In, transposetypeIn);
    op.Process();
}

struct TransposeFirstTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, TransposeType);
    uint32_t A1In;
    uint32_t A2In;
    uint32_t A3In;
    TransposeType transposetypeIn;
};

class TransposeFirstTestsuite : public testing::Test,
    public testing::WithParamInterface<TransposeFirstTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_CONFUSION_TRANSPOSE, TransposeFirstTestsuite,
    ::testing::Values(TransposeFirstTestParams { 2, TransposeFirst<half>, 2, 64, 32, TransposeType::TRANSPOSE_NZ2ND_0213 },
    TransposeFirstTestParams { 4, TransposeFirst<float>, 2, 64, 32, TransposeType::TRANSPOSE_NZ2ND_0213 }));

TEST_P(TransposeFirstTestsuite, TransposeFirstTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.A1In * param.A2In * param.A3In * param.typeSize] = {0};
    uint8_t dstGm[param.A1In * param.A2In * param.A3In * param.typeSize] = {0};

    param.cal_func(dstGm, srcGm, param.A1In, param.A2In, param.A3In, param.transposetypeIn);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

// scene 2
namespace AscendC {
template <typename T>
class KernelTransposeSecond {
public:
    __aicore__ inline KernelTransposeSecond() { }

    __aicore__ inline void Init(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint32_t A1In, uint32_t A2In, uint32_t A3In,
        TransposeType transposetypeIn)
    {
        A1 = A1In;
        A2 = A2In;
        A3 = A3In;
        TransposeType transposetype = transposetypeIn;
        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrcVecIn, 1, A1 * A2 * A3 * sizeof(T));
        pipe.InitBuffer(inQueueSrcVecOut, 1, A1 * A2 * A3 * sizeof(T));
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
        LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal, A1 * A2 * A3);
        inQueueSrcVecIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> srcLocal = inQueueSrcVecIn.DeQue<T>();
        LocalTensor<T> dstLocal = inQueueSrcVecOut.AllocTensor<T>();
        uint32_t srcOriShape[] = {1, A1, A2, A3};
        uint32_t srcShape[] = { 1, A1, A3 / 16, A2 / 16, 16, 16 };
        ShapeInfo srcShapeInfo(6, srcShape, 4, srcOriShape, DataFormat::NZ);

        uint32_t dstOriShape[] = {1, A2, A1, A3};
        uint32_t dstShape[] = {1, A2, A1, A3};
        ShapeInfo dstShapeInfo(4, dstShape, 4, dstOriShape, DataFormat::ND);

        LocalTensor<T> stackBuffer;
        PopStackBuffer<T, TPosition::LCM>(stackBuffer);
        uint32_t popSize = stackBuffer.GetSize();
        ConfusionTransposeTiling tiling;
        GetTransposeTilingInfo(srcShapeInfo, popSize, sizeof(T), 2, tiling);
        AscendC::Transpose(dstLocal, srcLocal, TransposeType::TRANSPOSE_NZ2NZ_0213, tiling);

        inQueueSrcVecOut.EnQue<T>(dstLocal);
        inQueueSrcVecIn.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = inQueueSrcVecOut.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, A1 * A2 * A3);
        inQueueSrcVecOut.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrcVecIn;
    TQue<TPosition::VECOUT, 1> inQueueSrcVecOut;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    int32_t A1, A2, A3;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void TransposeSecond(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint32_t A1In, uint32_t A2In,
    uint32_t A3In, TransposeType transposetypeIn)
{
    AscendC::KernelTransposeSecond<T> op;
    op.Init(dstGm, srcGm, A1In, A2In, A3In, transposetypeIn);
    op.Process();
}

struct TransposeSecondTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, TransposeType);
    uint32_t A1In;
    uint32_t A2In;
    uint32_t A3In;
    TransposeType transposetypeIn;
};

class TransposeSecondTestsuite : public testing::Test,
    public testing::WithParamInterface<TransposeSecondTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_CONFUSION_TRANSPOSE, TransposeSecondTestsuite,
    ::testing::Values(TransposeSecondTestParams { 2, TransposeSecond<half>, 16, 80, 32, TransposeType::TRANSPOSE_NZ2NZ_0213 },
    TransposeSecondTestParams { 2, TransposeSecond<half>, 32, 48, 16, TransposeType::TRANSPOSE_NZ2NZ_0213 },
    TransposeSecondTestParams { 2, TransposeSecond<half>, 16, 32, 32, TransposeType::TRANSPOSE_NZ2NZ_0213 },
    TransposeSecondTestParams { 4, TransposeSecond<float>, 32, 32, 16, TransposeType::TRANSPOSE_NZ2NZ_0213 },
    TransposeSecondTestParams { 4, TransposeSecond<float>, 16, 32, 32, TransposeType::TRANSPOSE_NZ2NZ_0213 }));

TEST_P(TransposeSecondTestsuite, TransposeSecondTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.A1In * param.A2In * param.A3In * param.typeSize] = {0};
    uint8_t dstGm[param.A1In * param.A2In * param.A3In * param.typeSize] = {0};

    param.cal_func(dstGm, srcGm, param.A1In, param.A2In, param.A3In, param.transposetypeIn);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

// scene 3
namespace AscendC {
template <typename T>
class KernelTransposeThird {
public:
    __aicore__ inline KernelTransposeThird() { }

    __aicore__ inline void Init(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint32_t NIn, uint32_t SIn, uint32_t HIn,
        uint32_t hnDivBlockNumIn, uint32_t hBlockNumIn, uint32_t sBlockNumIn, TransposeType transposetypeIn)
    {
        N = NIn;
        S = SIn;
        H = HIn;
        hnDivBlockNum = hnDivBlockNumIn;
        hBlockNum = hBlockNumIn;
        sBlockNum = sBlockNumIn;
        TransposeType transposetype = transposetypeIn;

        hAlign = hBlockNum * 16;
        hnDivAlign = hnDivBlockNum * 16;
        gap = hnDivAlign - H/N;

        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);

        pipe.InitBuffer(inQueueSrcVecIn, 1, hBlockNum * sBlockNum * 16 * 16 * sizeof(T));
        pipe.InitBuffer(inQueueSrcVecOut, 1, N * sBlockNum * hnDivBlockNum * 16 * 16 * sizeof(T));
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
        LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal, hBlockNum * sBlockNum * 16 * 16);
        inQueueSrcVecIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> srcLocal = inQueueSrcVecIn.DeQue<T>();
        LocalTensor<T> dstLocal = inQueueSrcVecOut.AllocTensor<T>();
        uint32_t srcOriShape[] = {1, S, H};
        uint32_t srcShape[] = { 1, (H + 16-1)/16, S/16, 16, 16 };
        ShapeInfo srcShapeInfo(5, srcShape, 3, srcOriShape, DataFormat::NZ);

        uint32_t dstOriShape[] = { 1, N, S, H / N };
        uint32_t dstShape[] = { 1, N, (H / N + 16-1)/16, S/16, 16, 16 };
        ShapeInfo dstShapeInfo(6, dstShape, 4, dstOriShape, DataFormat::NZ);

        LocalTensor<T> stackBuffer;
        PopStackBuffer<T, TPosition::LCM>(stackBuffer);
        uint32_t popSize = stackBuffer.GetSize();
        ConfusionTransposeTiling tiling;
        GetTransposeTilingInfo(dstShapeInfo, popSize, sizeof(T), 3, tiling);
        AscendC::Transpose(dstLocal, srcLocal, TransposeType::TRANSPOSE_NZ2NZ_012_WITH_N, tiling);

        uint64_t mask[2];
        mask[1] = 0;
        mask[0] = ((1 << (gap)) - 1) << (16-gap);
        uint32_t tensorOffset = sBlockNum * 16 * (hnDivBlockNum - 1) * 16;
        uint32_t dstRepeatStride = 16 * sizeof(T) / 32;
        for (uint16_t n = 0; n < N; n++) {
            AscendC::Duplicate(dstLocal[tensorOffset + n * S * hnDivAlign], static_cast<T>(0), mask, S, 1, dstRepeatStride);
        }
        inQueueSrcVecOut.EnQue<T>(dstLocal);
        inQueueSrcVecIn.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = inQueueSrcVecOut.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, N * hnDivBlockNum * sBlockNum * 16 * 16);
        inQueueSrcVecOut.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrcVecIn;
    TQue<TPosition::VECOUT, 1> inQueueSrcVecOut;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    int32_t N, S, H;
    int32_t hnDivBlockNum, hBlockNum, sBlockNum;
    uint32_t gap, hnDivAlign, hAlign;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void TransposeThird(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint32_t NIn, uint32_t SIn,
    uint32_t HIn, uint32_t hnDivBlockNumIn, uint32_t hBlockNumIn, uint32_t sBlockNumIn, TransposeType transposetypeIn)
{
    AscendC::KernelTransposeThird<T> op;
    op.Init(dstGm, srcGm, NIn, SIn, HIn, hnDivBlockNumIn, hBlockNumIn, sBlockNumIn, transposetypeIn);
    op.Process();
}

struct TransposeThirdTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, TransposeType);
    uint32_t NIn;
    uint32_t SIn;
    uint32_t HIn;
    uint32_t hnDivBlockNumIn;
    uint32_t hBlockNumIn;
    uint32_t sBlockNumIn;
    TransposeType transposetypeIn;
};

class TransposeThirdTestsuite : public testing::Test,
    public testing::WithParamInterface<TransposeThirdTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_CONFUSION_TRANSPOSE, TransposeThirdTestsuite,
    ::testing::Values(TransposeThirdTestParams { 2, TransposeThird<half>, 3, 16, 18, 1, 2, 1, TransposeType::TRANSPOSE_NZ2NZ_012_WITH_N },
    TransposeThirdTestParams { 2, TransposeThird<half>, 6, 32, 18, 1, 2, 2, TransposeType::TRANSPOSE_NZ2NZ_012_WITH_N },
    TransposeThirdTestParams { 2, TransposeThird<half>, 6, 16, 18, 1, 2, 1, TransposeType::TRANSPOSE_NZ2NZ_012_WITH_N },
    TransposeThirdTestParams { 2, TransposeThird<half>, 2, 16, 40, 2, 3, 1, TransposeType::TRANSPOSE_NZ2NZ_012_WITH_N },
    TransposeThirdTestParams { 4, TransposeThird<float>, 6, 32, 18, 1, 2, 2, TransposeType::TRANSPOSE_NZ2NZ_012_WITH_N },
    TransposeThirdTestParams { 4, TransposeThird<float>, 6, 16, 18, 1, 2, 1, TransposeType::TRANSPOSE_NZ2NZ_012_WITH_N },
    TransposeThirdTestParams { 4, TransposeThird<float>, 2, 16, 40, 2, 3, 1, TransposeType::TRANSPOSE_NZ2NZ_012_WITH_N }));

TEST_P(TransposeThirdTestsuite, TransposeThirdTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.hBlockNumIn * param.sBlockNumIn * 16 * 16 * param.typeSize] = {0};
    uint8_t dstGm[param.NIn * param.sBlockNumIn * param.hnDivBlockNumIn * 16 * 16 * param.typeSize] = {0};

    param.cal_func(dstGm, srcGm, param.NIn, param.SIn, param.HIn, param.hnDivBlockNumIn, param.hBlockNumIn, param.sBlockNumIn, param.transposetypeIn);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

// scene 4
namespace AscendC {
template <typename T>
class KernelTransposeFourth {
public:
    __aicore__ inline KernelTransposeFourth() { }

    __aicore__ inline void Init(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint32_t NIn, uint32_t SIn, uint32_t HIn,
        uint32_t hnDivBlockNumIn, uint32_t hBlockNumIn, uint32_t sBlockNumIn, TransposeType transposetypeIn)
    {
        N = NIn;
        S = SIn;
        H = HIn;
        hnDivBlockNum = hnDivBlockNumIn;
        hBlockNum = hBlockNumIn;
        sBlockNum = sBlockNumIn;
        TransposeType transposetype = transposetypeIn;

        hAlign = hBlockNum * 16;
        hnDivAlign = hnDivBlockNum * 16;
        gap = hnDivAlign - H/N;

        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);

        pipe.InitBuffer(inQueueSrcVecIn, 1, hBlockNum * sBlockNum * 16 * 16 * sizeof(T));
        pipe.InitBuffer(inQueueSrcVecOut, 1, N * sBlockNum * hnDivBlockNum * 16 * 16 * sizeof(T));
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
        LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal, hBlockNum * sBlockNum * 16 * 16);
        inQueueSrcVecIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> srcLocal = inQueueSrcVecIn.DeQue<T>();
        LocalTensor<T> dstLocal = inQueueSrcVecOut.AllocTensor<T>();
        uint32_t srcOriShape[] = {1, S, H};
        uint32_t srcShape[] = { 1, (H + 16-1)/16, S/16, 16, 16 };
        ShapeInfo srcShapeInfo(5, srcShape, 3, srcOriShape, DataFormat::NZ);

        uint32_t dstOriShape[] = { 1, N, S, H / N };
        uint32_t dstShape[] = { 1, N, (H / N + 16-1)/16, S/16, 16, 16 };
        ShapeInfo dstShapeInfo(6, dstShape, 4, dstOriShape, DataFormat::NZ);

        LocalTensor<T> stackBuffer;
        PopStackBuffer<T, TPosition::LCM>(stackBuffer);
        uint32_t popSize = stackBuffer.GetSize();
        ConfusionTransposeTiling tiling;
        GetTransposeTilingInfo(dstShapeInfo, popSize, sizeof(T), 4, tiling);
        AscendC::Transpose(dstLocal, srcLocal, TransposeType::TRANSPOSE_NZ2ND_012_WITH_N, tiling);

        uint64_t mask[2];
        mask[1] = 0;
        mask[0] = ((1 << (gap)) - 1) << (16-gap);
        uint32_t tensorOffset = sBlockNum * 16 * (hnDivBlockNum - 1) * 16;
        uint32_t dstRepeatStride = 16 * sizeof(T) / 32;
        for (uint16_t n = 0; n < N; n++) {
            AscendC::Duplicate(dstLocal[tensorOffset + n * S * hnDivAlign], static_cast<T>(0), mask, S, 1, dstRepeatStride);
        }
        inQueueSrcVecOut.EnQue<T>(dstLocal);
        inQueueSrcVecIn.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = inQueueSrcVecOut.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, N * hnDivBlockNum * sBlockNum * 16 * 16);
        inQueueSrcVecOut.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrcVecIn;
    TQue<TPosition::VECOUT, 1> inQueueSrcVecOut;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    int32_t N, S, H;
    int32_t hnDivBlockNum, hBlockNum, sBlockNum;
    uint32_t gap, hnDivAlign, hAlign;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void TransposeFourth(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint32_t NIn, uint32_t SIn,
    uint32_t HIn, uint32_t hnDivBlockNumIn, uint32_t hBlockNumIn, uint32_t sBlockNumIn, TransposeType transposetypeIn)
{
    AscendC::KernelTransposeFourth<T> op;
    op.Init(dstGm, srcGm, NIn, SIn, HIn, hnDivBlockNumIn, hBlockNumIn, sBlockNumIn, transposetypeIn);
    op.Process();
}

struct TransposeFourthTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, TransposeType);
    uint32_t NIn;
    uint32_t SIn;
    uint32_t HIn;
    uint32_t hnDivBlockNumIn;
    uint32_t hBlockNumIn;
    uint32_t sBlockNumIn;
    TransposeType transposetypeIn;
};

class TransposeFourthTestsuite : public testing::Test,
    public testing::WithParamInterface<TransposeFourthTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_CONFUSION_TRANSPOSE, TransposeFourthTestsuite,
    ::testing::Values(TransposeFourthTestParams { 2, TransposeFourth<half>, 3, 16, 18, 1, 2, 1, TransposeType::TRANSPOSE_NZ2ND_012_WITH_N },
    TransposeFourthTestParams { 2, TransposeFourth<half>, 6, 32, 18, 1, 2, 2, TransposeType::TRANSPOSE_NZ2ND_012_WITH_N },
    TransposeFourthTestParams { 2, TransposeFourth<half>, 6, 16, 18, 1, 2, 1, TransposeType::TRANSPOSE_NZ2ND_012_WITH_N },
    TransposeFourthTestParams { 2, TransposeFourth<half>, 2, 16, 40, 2, 3, 1, TransposeType::TRANSPOSE_NZ2ND_012_WITH_N },
    TransposeFourthTestParams { 4, TransposeFourth<float>, 6, 32, 18, 1, 2, 2, TransposeType::TRANSPOSE_NZ2ND_012_WITH_N },
    TransposeFourthTestParams { 4, TransposeFourth<float>, 6, 16, 18, 1, 2, 1, TransposeType::TRANSPOSE_NZ2ND_012_WITH_N },
    TransposeFourthTestParams { 4, TransposeFourth<float>, 2, 16, 40, 2, 3, 1, TransposeType::TRANSPOSE_NZ2ND_012_WITH_N }));

TEST_P(TransposeFourthTestsuite, TransposeFourthTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.hBlockNumIn * param.sBlockNumIn * 16 * 16 * param.typeSize] = {0};
    uint8_t dstGm[param.NIn * param.sBlockNumIn * param.hnDivBlockNumIn * 16 * 16 * param.typeSize] = {0};

    param.cal_func(dstGm, srcGm, param.NIn, param.SIn, param.HIn, param.hnDivBlockNumIn, param.hBlockNumIn, param.sBlockNumIn, param.transposetypeIn);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}


// scene 5
namespace AscendC {
template <typename T>
class KernelTransposeFifth {
public:
    __aicore__ inline KernelTransposeFifth() { }

    __aicore__ inline void Init(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint32_t NIn, uint32_t SIn, uint32_t HIn,
        uint32_t hnDivBlockNumIn, uint32_t hBlockNumIn, uint32_t sBlockNumIn, TransposeType transposetypeIn)
    {
        N = NIn;
        S = SIn;
        H = HIn;
        hnDivBlockNum = hnDivBlockNumIn;
        hBlockNum = hBlockNumIn;
        sBlockNum = sBlockNumIn;
        TransposeType transposetype = transposetypeIn;

        hAlign = hBlockNum * 16;
        gap = hAlign - H;

        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);

        pipe.InitBuffer(inQueueSrcVecIn, 1, N * hnDivBlockNum * sBlockNum * 16 * 16 * sizeof(T));
        pipe.InitBuffer(inQueueSrcVecOut, 1, sBlockNum * hBlockNum * 16 * 16 * sizeof(T));
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
        LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal, N * hnDivBlockNum * sBlockNum * 16 * 16);
        inQueueSrcVecIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> srcLocal = inQueueSrcVecIn.DeQue<T>();
        LocalTensor<T> dstLocal = inQueueSrcVecOut.AllocTensor<T>();

        uint32_t srcOriShape[] = { 1, N, S, H / N };
        uint32_t srcShape[] = { 1, N, static_cast<int32_t>(hnDivBlockNum), S / 16, 16, 16 };
        ShapeInfo srcShapeInfo(6, srcShape, 4, srcOriShape, DataFormat::NZ);

        uint32_t dstOriShape[] = {1, S, H};
        uint32_t dstShape[] = {1, S, H};
        ShapeInfo dstShapeInfo(3, dstShape, 3, dstOriShape, DataFormat::ND);

        LocalTensor<T> stackBuffer;
        PopStackBuffer<T, TPosition::LCM>(stackBuffer);
        uint32_t popSize = stackBuffer.GetSize();
        ConfusionTransposeTiling tiling;
        GetTransposeTilingInfo(srcShapeInfo, popSize, sizeof(T), 5, tiling);
        AscendC::Transpose(dstLocal, srcLocal, TransposeType::TRANSPOSE_NZ2ND_012_WITHOUT_N, tiling);

        uint64_t mask[2];
        mask[1] = 0;
        mask[0] = ((1 << (gap)) - 1) << (16-gap);
        uint32_t tensorOffset = (hBlockNum - 1) * 16;
        uint32_t dstRepeatStride = (hBlockNum * 16) * sizeof(T) / 32;
        AscendC::Duplicate(dstLocal[tensorOffset], static_cast<T>(0), mask, S, 1, dstRepeatStride);

        inQueueSrcVecOut.EnQue<T>(dstLocal);
        inQueueSrcVecIn.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = inQueueSrcVecOut.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, sBlockNum * hBlockNum * 16 * 16);
        inQueueSrcVecOut.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrcVecIn;
    TQue<TPosition::VECOUT, 1> inQueueSrcVecOut;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    int32_t N, S, H;
    int32_t hnDivBlockNum, hBlockNum, sBlockNum;
    uint32_t gap, hAlign;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void TransposeFifth(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint32_t NIn, uint32_t SIn,
    uint32_t HIn, uint32_t hnDivBlockNumIn, uint32_t hBlockNumIn, uint32_t sBlockNumIn, TransposeType transposetypeIn)
{
    AscendC::KernelTransposeFifth<T> op;
    op.Init(dstGm, srcGm, NIn, SIn, HIn, hnDivBlockNumIn, hBlockNumIn, sBlockNumIn, transposetypeIn);
    op.Process();
}

struct TransposeFifthTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, TransposeType);
    uint32_t NIn;
    uint32_t SIn;
    uint32_t HIn;
    uint32_t hnDivBlockNumIn;
    uint32_t hBlockNumIn;
    uint32_t sBlockNumIn;
    TransposeType transposetypeIn;
};

class TransposeFifthTestsuite : public testing::Test,
    public testing::WithParamInterface<TransposeFifthTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_CONFUSION_TRANSPOSE, TransposeFifthTestsuite,
    ::testing::Values(TransposeFifthTestParams { 2, TransposeFifth<half>, 3, 16, 18, 1, 2, 1, TransposeType::TRANSPOSE_NZ2ND_012_WITH_N },
    TransposeFifthTestParams { 2, TransposeFifth<half>, 6, 32, 18, 1, 2, 2, TransposeType::TRANSPOSE_NZ2ND_012_WITH_N },
    TransposeFifthTestParams { 2, TransposeFifth<half>, 6, 16, 18, 1, 2, 1, TransposeType::TRANSPOSE_NZ2ND_012_WITH_N },
    TransposeFifthTestParams { 2, TransposeFifth<half>, 2, 16, 40, 2, 3, 1, TransposeType::TRANSPOSE_NZ2ND_012_WITH_N },
    TransposeFifthTestParams { 4, TransposeFifth<float>, 6, 32, 18, 1, 2, 2, TransposeType::TRANSPOSE_NZ2ND_012_WITH_N },
    TransposeFifthTestParams { 4, TransposeFifth<float>, 6, 16, 18, 1, 2, 1, TransposeType::TRANSPOSE_NZ2ND_012_WITH_N },
    TransposeFifthTestParams { 4, TransposeFifth<float>, 2, 16, 40, 2, 3, 1, TransposeType::TRANSPOSE_NZ2ND_012_WITH_N }));

TEST_P(TransposeFifthTestsuite, TransposeFifthTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.NIn * param.sBlockNumIn * param.hnDivBlockNumIn * 16 * 16 * param.typeSize] = {0};
    uint8_t dstGm[param.hBlockNumIn * param.sBlockNumIn * 16 * 16 * param.typeSize] = {0};

    param.cal_func(dstGm, srcGm, param.NIn, param.SIn, param.HIn, param.hnDivBlockNumIn, param.hBlockNumIn, param.sBlockNumIn, param.transposetypeIn);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}


// scene 6
namespace AscendC {
template <typename T>
class KernelTransposeSixth {
public:
    __aicore__ inline KernelTransposeSixth() { }

    __aicore__ inline void Init(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint32_t NIn, uint32_t SIn, uint32_t HIn,
        uint32_t hnDivBlockNumIn, uint32_t hBlockNumIn, uint32_t sBlockNumIn, TransposeType transposetypeIn)
    {
        N = NIn;
        S = SIn;
        H = HIn;
        hnDivBlockNum = hnDivBlockNumIn;
        hBlockNum = hBlockNumIn;
        sBlockNum = sBlockNumIn;
        TransposeType transposetype = transposetypeIn;

        hAlign = hBlockNum * 16;
        gap = hAlign - H;

        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);

        pipe.InitBuffer(inQueueSrcVecIn, 1, N * hnDivBlockNum * sBlockNum * 16 * 16 * sizeof(T));
        pipe.InitBuffer(inQueueSrcVecOut, 1, sBlockNum * hBlockNum * 16 * 16 * sizeof(T));
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
        LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal, N * hnDivBlockNum * sBlockNum * 16 * 16);
        inQueueSrcVecIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> srcLocal = inQueueSrcVecIn.DeQue<T>();
        LocalTensor<T> dstLocal = inQueueSrcVecOut.AllocTensor<T>();

        uint32_t srcOriShape[] = { 1, N, S, H / N };
        uint32_t srcShape[] = { 1, N, static_cast<int32_t>(hnDivBlockNum), S / 16, 16, 16 };
        ShapeInfo srcShapeInfo(6, srcShape, 4, srcOriShape, DataFormat::NZ);

        uint32_t dstOriShape[] = {1, S, H};
        uint32_t dstShape[] = { 1, H / 16, S / 16, 16, 16 };
        ShapeInfo dstShapeInfo(5, dstShape, 3, dstOriShape, DataFormat::NZ);

        LocalTensor<T> stackBuffer;
        PopStackBuffer<T, TPosition::LCM>(stackBuffer);
        uint32_t popSize = stackBuffer.GetSize();
        ConfusionTransposeTiling tiling;
        GetTransposeTilingInfo(srcShapeInfo, popSize, sizeof(T), 6, tiling);
        AscendC::Transpose(dstLocal, srcLocal, TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N, tiling);

        uint64_t mask[2];
        mask[1] = 0;
        mask[0] = ((1 << (gap)) - 1) << (16-gap);
        uint32_t tensorOffset = sBlockNum * 16 * (hBlockNum - 1) * 16;
        uint32_t dstRepeatStride = 16 * sizeof(T) / 32;
        AscendC::Duplicate(dstLocal[tensorOffset], static_cast<T>(0), mask, S, 1, dstRepeatStride);

        inQueueSrcVecOut.EnQue<T>(dstLocal);
        inQueueSrcVecIn.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = inQueueSrcVecOut.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, sBlockNum * hBlockNum * 16 * 16);
        inQueueSrcVecOut.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrcVecIn;
    TQue<TPosition::VECOUT, 1> inQueueSrcVecOut;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    int32_t N, S, H;
    int32_t hnDivBlockNum, hBlockNum, sBlockNum;
    uint32_t gap, hAlign;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void TransposeSixth(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint32_t NIn, uint32_t SIn,
    uint32_t HIn, uint32_t hnDivBlockNumIn, uint32_t hBlockNumIn, uint32_t sBlockNumIn, TransposeType transposetypeIn)
{
    AscendC::KernelTransposeSixth<T> op;
    op.Init(dstGm, srcGm, NIn, SIn, HIn, hnDivBlockNumIn, hBlockNumIn, sBlockNumIn, transposetypeIn);
    op.Process();
}

struct TransposeSixthTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, TransposeType);
    uint32_t NIn;
    uint32_t SIn;
    uint32_t HIn;
    uint32_t hnDivBlockNumIn;
    uint32_t hBlockNumIn;
    uint32_t sBlockNumIn;
    TransposeType transposetypeIn;
};

class TransposeSixthTestsuite : public testing::Test,
    public testing::WithParamInterface<TransposeSixthTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_CONFUSION_TRANSPOSE, TransposeSixthTestsuite,
    ::testing::Values(TransposeSixthTestParams { 2, TransposeSixth<half>, 3, 16, 18, 1, 2, 1, TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N },
    TransposeSixthTestParams { 2, TransposeSixth<half>, 6, 32, 18, 1, 2, 2, TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N },
    TransposeSixthTestParams { 2, TransposeSixth<half>, 6, 16, 18, 1, 2, 1, TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N },
    TransposeSixthTestParams { 2, TransposeSixth<half>, 2, 16, 40, 2, 3, 1, TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N },
    TransposeSixthTestParams { 4, TransposeSixth<float>, 6, 32, 18, 1, 2, 2, TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N },
    TransposeSixthTestParams { 4, TransposeSixth<float>, 6, 16, 18, 1, 2, 1, TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N },
    TransposeSixthTestParams { 4, TransposeSixth<float>, 2, 16, 40, 2, 3, 1, TransposeType::TRANSPOSE_NZ2NZ_012_WITHOUT_N }));

TEST_P(TransposeSixthTestsuite, TransposeSixthTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.NIn * param.sBlockNumIn * param.hnDivBlockNumIn * 16 * 16 * param.typeSize] = {0};
    uint8_t dstGm[param.hBlockNumIn * param.sBlockNumIn * 16 * 16 * param.typeSize] = {0};

    param.cal_func(dstGm, srcGm, param.NIn, param.SIn, param.HIn, param.hnDivBlockNumIn, param.hBlockNumIn, param.sBlockNumIn, param.transposetypeIn);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

// scene 7
namespace AscendC {
template <typename T>
class KernelTransposeSeventh {
public:
    __aicore__ inline KernelTransposeSeventh() { }

    __aicore__ inline void Init(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint32_t widthIn, uint32_t heightIn, TransposeType transposetypeIn)
    {
        width = widthIn;
        height = heightIn;
        TransposeType transposetype = transposetypeIn;

        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);

        pipe.InitBuffer(inQueueSrcVecIn, 1, width * height * sizeof(T));
        pipe.InitBuffer(inQueueSrcVecOut, 1, width * height * sizeof(T));
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
        LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal, height * width);
        inQueueSrcVecIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> srcLocal = inQueueSrcVecIn.DeQue<T>();
        LocalTensor<T> dstLocal = inQueueSrcVecOut.AllocTensor<T>();

        uint32_t srcOriShape[] = {height, static_cast<int32_t>(width)};
        uint32_t srcShape[] = {static_cast<int32_t>(height), static_cast<int32_t>(width)};
        ShapeInfo srcShapeInfo(2, srcShape, 2, srcOriShape, DataFormat::NZ);

        uint32_t dstOriShape[] = {static_cast<int32_t>(width), static_cast<int32_t>(height)};
        uint32_t dstShape[] = {static_cast<int32_t>(width), static_cast<int32_t>(height)};
        ShapeInfo dstShapeInfo(2, dstShape, 2, dstOriShape, DataFormat::NZ);

        LocalTensor<T> stackBuffer;
        PopStackBuffer<T, TPosition::LCM>(stackBuffer);
        uint32_t popSize = stackBuffer.GetSize();
        ConfusionTransposeTiling tiling;
        GetTransposeTilingInfo(srcShapeInfo, popSize, sizeof(T), 7, tiling);
        AscendC::Transpose(dstLocal, srcLocal, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling);

        inQueueSrcVecOut.EnQue<T>(dstLocal);
        inQueueSrcVecIn.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = inQueueSrcVecOut.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, height * width);
        inQueueSrcVecOut.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrcVecIn;
    TQue<TPosition::VECOUT, 1> inQueueSrcVecOut;

    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    uint32_t width;
    uint32_t height;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void TransposeSeventh(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint32_t widthIn, uint32_t heightIn, TransposeType transposetypeIn)
{
    AscendC::KernelTransposeSeventh<T> op;
    op.Init(dstGm, srcGm, widthIn, heightIn, transposetypeIn);
    op.Process();
}

struct TransposeSeventhTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t, uint32_t, TransposeType);
    uint32_t widthIn;
    uint32_t heightIn;
    TransposeType transposetypeIn;
};

class TransposeSeventhTestsuite : public testing::Test,
    public testing::WithParamInterface<TransposeSeventhTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_CONFUSION_TRANSPOSE, TransposeSeventhTestsuite,
    ::testing::Values(TransposeSeventhTestParams { 2, TransposeSeventh<half>, 32, 32, TransposeType::TRANSPOSE_ND2ND_ONLY },
    TransposeSeventhTestParams { 2, TransposeSeventh<half>, 16, 16, TransposeType::TRANSPOSE_ND2ND_ONLY },
    TransposeSeventhTestParams { 2, TransposeSeventh<half>, 16, 32, TransposeType::TRANSPOSE_ND2ND_ONLY },
    TransposeSeventhTestParams { 4, TransposeSeventh<float>, 16, 32, TransposeType::TRANSPOSE_ND2ND_ONLY },
    TransposeSeventhTestParams { 4, TransposeSeventh<float>, 16, 8, TransposeType::TRANSPOSE_ND2ND_ONLY },
    TransposeSeventhTestParams { 4, TransposeSeventh<float>, 32, 32, TransposeType::TRANSPOSE_ND2ND_ONLY },
    TransposeSeventhTestParams { 4, TransposeSeventh<float>, 16, 16, TransposeType::TRANSPOSE_ND2ND_ONLY }));

TEST_P(TransposeSeventhTestsuite, TransposeSeventhTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.widthIn * param.heightIn * param.typeSize] = {0};
    uint8_t dstGm[param.widthIn * param.heightIn * param.typeSize] = {0};

    param.cal_func(dstGm, srcGm, param.widthIn, param.heightIn, param.transposetypeIn);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}