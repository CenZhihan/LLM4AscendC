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

namespace AscendC {
__aicore__ inline void UnPadTilingFunc(
    const ShapeInfo srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, UnPadTiling& tiling)
{
    if (typeSize == 0) {
        return;
    }

    uint32_t srcHeight=srcShape.shape[0];
    uint32_t srcWidth=srcShape.shape[1];

    tiling.srcHeight = srcHeight;
    tiling.srcWidth = srcWidth;

    uint32_t baseBlockLen = 16 * ONE_BLK_SIZE;          // Byte; x multiply 32B, for brcb
    uint32_t baseBlockSize = baseBlockLen / typeSize;  // ele
    uint32_t tmpBuffer1BlockNum = stackBufferSize * sizeof(uint8_t) / typeSize / baseBlockSize / 2;
    uint32_t tmpBuffer2Offset = tmpBuffer1BlockNum * baseBlockSize;

    tiling.tmpBuffer1BlockNum=tmpBuffer1BlockNum;
    tiling.tmpBuffer1RowNum=16*tmpBuffer1BlockNum;
    tiling.tmpBuffer2Offset=tmpBuffer2Offset;

    uint32_t widthTiling = 16 * tmpBuffer1BlockNum;  // elements
    uint32_t widthFractal = srcWidth / widthTiling;
    uint32_t widthFractalTail = srcWidth % widthTiling;
    if (widthFractalTail) {
        widthFractal += 1;
    } else {
        widthFractalTail = widthTiling;
    }

    tiling.widthTiling=widthTiling;
    tiling.widthFractal=widthFractal;
    tiling.widthFractalTail=widthFractalTail;
}

template <typename T>
class KernelUnPad {
public:
    __aicore__ inline KernelUnPad()
    {}
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR srcGm, uint16_t heightIn, uint16_t widthIn, uint16_t oriWidthIn,UnPadParams& unPadParamsIn)
    {
        height=heightIn;
        width=widthIn;
        oriWidth=oriWidthIn;
        unPadParams=unPadParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T *)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T *)dstGm);
        pipe.InitBuffer(inQueueSrcVecIn, 1, height * width * sizeof(T));
        pipe.InitBuffer(inQueueSrcVecOut, 1, height * width * sizeof(T));
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
        LocalTensor<T> dstLocal = inQueueSrcVecIn.DeQue<T>();

        uint32_t shape[] = {height, width};
        uint32_t oriShape[] = {height, oriWidth};
        ShapeInfo shapeInfo(2, shape, 2, oriShape, DataFormat::ND);

        LocalTensor<T> srcOutLocal = inQueueSrcVecOut.AllocTensor<T>();
        uint32_t shape1[] = {height, static_cast<uint32_t>(width - unPadParams.leftPad - unPadParams.rightPad)};
        uint32_t oriShape1[] = {height, oriWidth};
        ShapeInfo shapeInfo1(2, shape1, 2, oriShape1, DataFormat::ND);

        LocalTensor<uint8_t> stackBuffer;
        PopStackBuffer<uint8_t, TPosition::LCM>(stackBuffer);
        uint32_t stackBufferSize = stackBuffer.GetSize();

        UnPadTiling tiling;
        UnPadTilingFunc(shapeInfo, stackBufferSize, sizeof(T), tiling);

        AscendC::UnPad(srcOutLocal, dstLocal, unPadParams, tiling);
        // std::cout<<"after unpad srcOutLocal is:"<<std::endl;
        // srcOutLocal.Print();
        inQueueSrcVecOut.EnQue(srcOutLocal);
        inQueueSrcVecIn.FreeTensor(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> srcOutLocalDe = inQueueSrcVecOut.DeQue<T>();
        DataCopy(dstGlobal, srcOutLocalDe, height * width);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrcVecIn;
    TQue<TPosition::VECOUT, 1> inQueueSrcVecOut;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    uint16_t height;
    uint16_t width;
    uint16_t oriWidth;
    UnPadParams unPadParams;

};
}  // namespace AscendC

template <typename T>
__global__ __aicore__ void MainUnPad(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint16_t heightIn, uint16_t widthIn,
    uint16_t oriWidthIn, UnPadParams &unPadParamsIn)
{
    AscendC::KernelUnPad<T> op;
    op.Init(dstGm, srcGm, heightIn, widthIn, oriWidthIn, unPadParamsIn);
    op.Process();
}

struct UnPadTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint16_t, uint16_t, uint16_t, UnPadParams&);
    uint16_t heightIn;
    uint16_t widthIn;
    uint16_t oriWidthIn;
    UnPadParams unPadParamsIn;
};

class UnPadTestsuite : public testing::Test,
    public testing::WithParamInterface<UnPadTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_UNPAD, UnPadTestsuite,
    ::testing::Values(
        UnPadTestParams { 2, MainUnPad<half>, 32, 32, 32, {0, 2} },
        UnPadTestParams { 4, MainUnPad<float>, 32, 32, 32, {1, 1} },
        UnPadTestParams { 4, MainUnPad<float>, 1, 256, 256, {1, 1} },
        UnPadTestParams { 4, MainUnPad<float>, 1, 256, 256, {0, 2} }
    ));

TEST_P(UnPadTestsuite, UnPadTestsuiteOpTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.heightIn * param.widthIn * param.typeSize] = {0};
    uint8_t dstGm[param.heightIn * (param.widthIn - param.unPadParamsIn.leftPad - param.unPadParamsIn.rightPad) *
                  param.typeSize] = {0};

    param.cal_func(dstGm, srcGm, param.heightIn, param.widthIn, param.oriWidthIn, param.unPadParamsIn);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}


namespace AscendC {
__aicore__ inline void PadTilingFunc(
    const ShapeInfo srcShape, const uint32_t stackBufferSize, const uint32_t typeSize, PadTiling& tiling)
{
    if (typeSize == 0) {
        return;
    }

    // common
    uint32_t srcHeight=srcShape.shape[0];
    uint32_t srcWidth=srcShape.shape[1];
    uint32_t srcOriWidth=srcShape.originalShape[1];
    tiling.srcHeight = srcHeight;
    tiling.srcWidth = srcWidth;
    tiling.srcOriWidth = srcOriWidth;

    // width 32B aligned
    uint32_t widthWithoutLastBlock=srcWidth - ONE_BLK_SIZE / typeSize;
    tiling.widthWithoutLastBlock = widthWithoutLastBlock;

    uint32_t blocksPerRow=srcWidth * typeSize / ONE_BLK_SIZE;
    tiling.blocksPerRow = blocksPerRow;

    uint32_t heightTiling = MAX_REPEAT_TIMES;
    uint32_t heightFractal = srcHeight / heightTiling;
    uint32_t heightFractalTail = srcHeight % heightTiling;
    tiling.heightTiling = heightTiling;
    tiling.heightFractal = heightFractal;
    tiling.heightFractalTail = heightFractalTail;

    uint32_t mainLoopOffset=heightTiling * srcWidth;
    uint32_t tailBlockOffset=heightFractal * heightTiling * srcWidth+widthWithoutLastBlock;
    tiling.mainLoopOffset = mainLoopOffset ;
    tiling.tailBlockOffset = tailBlockOffset;

    // width 32B unaligned
    uint32_t baseBlockLen = 16 * ONE_BLK_SIZE;          // Byte; x multiply 32B, for brcb
    uint32_t baseBlockSize = baseBlockLen / typeSize;  // ele
    uint32_t tmpBuffer1BlockNum = stackBufferSize * sizeof(uint8_t) / typeSize / baseBlockSize / 2;
    uint32_t tmpBuffer2Offset = tmpBuffer1BlockNum * baseBlockSize;
    tiling.tmpBuffer1BlockNum=tmpBuffer1BlockNum;
    tiling.tmpBuffer1RowNum=16*tmpBuffer1BlockNum;
    tiling.tmpBuffer2Offset=tmpBuffer2Offset;

    uint32_t widthTiling = 16 * tmpBuffer1BlockNum;  // elements
    uint32_t widthFractal = srcWidth / widthTiling;
    uint32_t widthFractalTail = srcWidth % widthTiling;
    // aligned to 8 or 16
    uint32_t widthFractalTailAlingned =
        ((widthFractalTail - 1) / (ONE_BLK_SIZE / typeSize) + 1) * (ONE_BLK_SIZE / typeSize);

    tiling.widthTiling=widthTiling;
    tiling.widthFractal=widthFractal;
    tiling.widthFractalTail=widthFractalTail;
    tiling.widthFractalTailAlingned=widthFractalTailAlingned;

    uint32_t brcbTiling = 16 * tmpBuffer1BlockNum;  // elements
    uint32_t brcbFractal = srcHeight * srcWidth / brcbTiling;
    uint32_t brcbFractalTail = srcHeight * srcWidth % brcbTiling;
    tiling.brcbTiling=brcbTiling;
    tiling.brcbFractal=brcbFractal;
    tiling.brcbFractalTail=brcbFractalTail;

    uint32_t maxRepeatTimes = 254; // 255*8 not 32B aligned, so 254
    uint32_t brcbTilingRepeatTimes = brcbTiling / 8 / maxRepeatTimes;
    uint32_t brcbTilingRepeatTimesTail = brcbTiling / 8 % maxRepeatTimes;
    uint32_t brcbFractalTailRepeatTimes = brcbFractalTail / 8 / maxRepeatTimes;
    uint32_t brcbFractalTailRepeatTimesTail = brcbFractalTail / 8 % maxRepeatTimes;
    tiling.maxRepeatTimes=maxRepeatTimes;
    tiling.brcbTilingRepeatTimes=brcbTilingRepeatTimes;
    tiling.brcbTilingRepeatTimesTail=brcbTilingRepeatTimesTail;
    tiling.brcbFractalTailRepeatTimes=brcbFractalTailRepeatTimes;
    tiling.brcbFractalTailRepeatTimesTail=brcbFractalTailRepeatTimesTail;
}

template <typename T>
class KernelPad {
public:
    __aicore__ inline KernelPad()
    {}
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR srcGm, uint32_t heightIn, uint32_t widthIn, uint32_t oriWidthIn,PadParams& padParamsIn)
    {
        height=heightIn;
        width=widthIn;
        oriWidth=oriWidthIn;
        padParams=padParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T *)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T *)dstGm);
        alignedWidth= ((width * sizeof(T)-1)/32+1)*32/sizeof(T);
        pipe.InitBuffer(inQueueSrcVecIn, 1, height * alignedWidth * sizeof(T));
        pipe.InitBuffer(inQueueSrcVecOut, 1, height * alignedWidth * sizeof(T));
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
        DataCopy(srcLocal, srcGlobal, height * alignedWidth);
        inQueueSrcVecIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> dstLocal = inQueueSrcVecIn.DeQue<T>();

        uint32_t shape[] = {height, width};
        uint32_t oriShape[] = {height, oriWidth};
        ShapeInfo shapeInfo(2, shape, 2, oriShape, DataFormat::ND);

        LocalTensor<T> srcOutLocal = inQueueSrcVecOut.AllocTensor<T>();

        LocalTensor<uint8_t> stackBuffer;
        PopStackBuffer<uint8_t, TPosition::LCM>(stackBuffer);
        uint32_t stackBufferSize = stackBuffer.GetSize();

        PadTiling tiling;
        PadTilingFunc(shapeInfo, stackBufferSize, sizeof(T), tiling);

        AscendC::Pad(srcOutLocal, dstLocal, padParams, tiling);
        inQueueSrcVecOut.EnQue(srcOutLocal);
        inQueueSrcVecIn.FreeTensor(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> srcOutLocalDe = inQueueSrcVecOut.DeQue<T>();
        DataCopy(dstGlobal, srcOutLocalDe, height * alignedWidth);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrcVecIn;
    TQue<TPosition::VECOUT, 1> inQueueSrcVecOut;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    uint32_t height;
    uint32_t width;
    uint32_t oriWidth;
    uint32_t alignedWidth;
    PadParams padParams;

};
}  // namespace AscendC

template <typename T>
__global__ __aicore__ void MainPad(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, uint16_t heightIn, uint16_t widthIn,
    uint16_t oriWidthIn, PadParams &padParamsIn)
{
    AscendC::KernelPad<T> op;
    op.Init(dstGm, srcGm, heightIn, widthIn, oriWidthIn, padParamsIn);
    op.Process();
}

struct PadTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint16_t, uint16_t, uint16_t, PadParams&);
    uint16_t heightIn;
    uint16_t widthIn;
    uint16_t oriWidthIn;
    PadParams padParamsIn;
};

class PadTestsuite : public testing::Test,
    public testing::WithParamInterface<PadTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_PAD, PadTestsuite,
    ::testing::Values(
        PadTestParams{2, MainPad<half>, 34, 32, 31, {0, 1, 321}},
        PadTestParams{4, MainPad<float>, 34, 32, 31, {0, 1, 321}},
        PadTestParams{2, MainPad<half>, 16, 30, 30, {1, 1, 321}},
        PadTestParams{4, MainPad<float>, 16, 30, 30, {1, 1, 321}},
        PadTestParams{4, MainPad<float>, 1, 256, 254, {0, 2, 321}}
    ));

TEST_P(PadTestsuite, PadTestsuiteOpTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.heightIn * param.widthIn * param.typeSize] = {0};
    uint8_t dstGm[param.heightIn * (((param.widthIn * param.typeSize - 1) / 32 + 1) * 32 / param.typeSize) *
                  param.typeSize] = {0};

    param.cal_func(dstGm, srcGm, param.heightIn, param.widthIn, param.oriWidthIn, param.padParamsIn);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
