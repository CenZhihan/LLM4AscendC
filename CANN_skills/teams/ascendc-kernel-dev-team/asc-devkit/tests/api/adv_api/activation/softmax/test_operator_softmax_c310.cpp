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
constexpr SoftmaxConfig config = {true, 16, 16};
namespace AscendC {
template <typename T1, typename T2, bool isBasicBlock = false, bool isDataFormatNZ = false>
class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm,
        uint32_t inheight, uint32_t inwidth)
    {
        height = inheight;
        width = inwidth;
        int elementNumPerBlk = 32 / sizeof(T2); // half=16 float=8
        src1Global.SetGlobalBuffer((__gm__ T1*)src0Gm);
        src2Global.SetGlobalBuffer((__gm__ T1*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ T1*)dstGm);
        pipe.InitBuffer(inQueueSrc1, 1, height * width * sizeof(T1));
        pipe.InitBuffer(inQueueSrc2, 1, height * width * sizeof(T1));
        pipe.InitBuffer(outQueueDst, 1, height * width * sizeof(T1));
        pipe.InitBuffer(inMeanQueue, 1, height * elementNumPerBlk * sizeof(T2));
        pipe.InitBuffer(inMaxQueue, 1, height * elementNumPerBlk * sizeof(T2));
        pipe.InitBuffer(inSumQueue, 1, height * elementNumPerBlk * sizeof(T2));
        elementNumPerBlk = 32 / sizeof(T1); // half=16 float=8
        pipe.InitBuffer(expMaxQueue, 1, height * elementNumPerBlk * sizeof(T1));
        height = inheight;
        width = inwidth;
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
        LocalTensor<T1> srcLocal1 = inQueueSrc1.AllocTensor<T1>();
        LocalTensor<T1> srcLocal2 = inQueueSrc2.AllocTensor<T1>();
        DataCopy(srcLocal1, src1Global, height * width);
        DataCopy(srcLocal2, src2Global, height * width);
        inQueueSrc1.EnQue(srcLocal1);
        inQueueSrc2.EnQue(srcLocal2);
    }
    __aicore__ inline void Compute()
    {
        uint32_t elementNumPerBlk = 32 / sizeof(T2); // half=16 float=8
        LocalTensor<T1> srcLocal1 = inQueueSrc1.DeQue<T1>();
        LocalTensor<T1> srcLocal2 = inQueueSrc2.DeQue<T1>();

        LocalTensor<T2> inmeanLocal = inMeanQueue.AllocTensor<T2>();
        LocalTensor<T2> inmaxLocal = inMaxQueue.AllocTensor<T2>();
        LocalTensor<T2> insumLocal = inSumQueue.AllocTensor<T2>();
        LocalTensor<T1> expMaxTensor = expMaxQueue.AllocTensor<T1>();

        LocalTensor<T1> dstLocal = outQueueDst.AllocTensor<T1>();

        const uint32_t shapeDim = 2;
        DataFormat dataFormat{ DataFormat::ND };
        if (isDataFormatNZ) {
            dataFormat = DataFormat::NZ;
        }
        uint32_t array[2] = {height, width};
        uint32_t arrayOrigin[2] = {height, width};
        ShapeInfo srcShape = ShapeInfo(shapeDim, array, shapeDim, arrayOrigin, dataFormat);
        srcLocal1.SetShapeInfo(srcShape);
        srcLocal2.SetShapeInfo(srcShape);
        dstLocal.SetShapeInfo(srcShape);

        SoftMaxShapeInfo alignSrcShape = { height, width, height, width };
        SoftMaxShapeInfo unalignedSrcShape = { height, width, height, width - 8 };
        SoftMaxTiling tiling;

        SoftMax<T1, false, isBasicBlock, isDataFormatNZ>(srcLocal1, insumLocal, inmaxLocal, srcLocal1, tiling, alignSrcShape);
        SoftMax<T1, false, isBasicBlock, isDataFormatNZ>(srcLocal1, insumLocal, inmaxLocal, srcLocal1, tiling, unalignedSrcShape);
        SoftMax<T1, false, isBasicBlock>(srcLocal1, srcLocal1, tiling, alignSrcShape);
        AdjustSoftMaxRes<T1, T2, isDataFormatNZ>(srcLocal1, inmaxLocal, 9, (T1)1.0, alignSrcShape);
        SoftMax<T1, false, isBasicBlock>(srcLocal1, srcLocal1, tiling, unalignedSrcShape);
        AdjustSoftMaxRes<T1, T2, isDataFormatNZ>(srcLocal1, inmaxLocal, 9, (T1)1.0, unalignedSrcShape);
        SimpleSoftMax<T1, false, isBasicBlock, isDataFormatNZ>(dstLocal, insumLocal, inmaxLocal, srcLocal1, tiling, alignSrcShape);
        SimpleSoftMax<T1, false, isBasicBlock, isDataFormatNZ>(dstLocal, insumLocal, inmaxLocal, srcLocal1, tiling, unalignedSrcShape);
        SimpleSoftMax<T1, false, isBasicBlock, isDataFormatNZ, config>(dstLocal, insumLocal, inmaxLocal, srcLocal1, tiling, alignSrcShape);
        SimpleSoftMax<T1, false, isBasicBlock, isDataFormatNZ, config>(dstLocal, insumLocal, inmaxLocal, srcLocal1, tiling, unalignedSrcShape);

        if constexpr (sizeof(T1) == sizeof(T2)) {
            AscendC::LocalTensor<uint8_t> sharedTmpBuffer;
            AscendC::PopStackBuffer<uint8_t, AscendC::TPosition::LCM>(sharedTmpBuffer);
            LogSoftMaxTiling logTiling;
            LogSoftMax<T1, false, isDataFormatNZ>(srcLocal1, insumLocal, inmaxLocal, srcLocal1, sharedTmpBuffer, logTiling, alignSrcShape);
            LogSoftMax<T1, false, isDataFormatNZ>(srcLocal1, insumLocal, inmaxLocal, srcLocal1, sharedTmpBuffer, logTiling, unalignedSrcShape);
        }

        SoftmaxFlash<T1, false, isBasicBlock>(dstLocal, insumLocal, inmaxLocal, srcLocal1, expMaxTensor, insumLocal,
            inmaxLocal, tiling, false, alignSrcShape);
        SoftmaxFlash<T1, false, isBasicBlock>(dstLocal, insumLocal, inmaxLocal, srcLocal1, expMaxTensor, insumLocal,
            inmaxLocal, tiling, false, unalignedSrcShape);
        SoftmaxFlash<T1, false, isBasicBlock>(dstLocal, insumLocal, inmaxLocal, srcLocal2, expMaxTensor, insumLocal,
            inmaxLocal, tiling, true, alignSrcShape);
        SoftmaxFlash<T1, false, isBasicBlock>(dstLocal, insumLocal, inmaxLocal, srcLocal2, expMaxTensor, insumLocal,
            inmaxLocal, tiling, true, unalignedSrcShape);

        SoftmaxFlash<T1, false, isBasicBlock>(dstLocal, insumLocal, inmaxLocal, srcLocal1, expMaxTensor, insumLocal,
            inmaxLocal, tiling, false);
        SoftmaxFlash<T1, false, isBasicBlock>(dstLocal, insumLocal, inmaxLocal, srcLocal2, expMaxTensor, insumLocal,
            inmaxLocal, tiling, true);

        SoftMaxTiling flashv2Tiling;
        SoftmaxFlashV2<T1, false, false, isBasicBlock, isDataFormatNZ>(dstLocal, insumLocal, inmaxLocal, srcLocal1,
            expMaxTensor, insumLocal, inmaxLocal, flashv2Tiling, alignSrcShape);
        SoftmaxFlashV2<T1, false, false, isBasicBlock, isDataFormatNZ>(dstLocal, insumLocal, inmaxLocal, srcLocal1,
            expMaxTensor, insumLocal, inmaxLocal, flashv2Tiling, unalignedSrcShape);
        SoftmaxFlashV2<T1, true, false, isBasicBlock, isDataFormatNZ>(dstLocal, insumLocal, inmaxLocal, srcLocal2,
            expMaxTensor, insumLocal, inmaxLocal, flashv2Tiling, alignSrcShape);
        SoftmaxFlashV2<T1, true, false, isBasicBlock, isDataFormatNZ>(dstLocal, insumLocal, inmaxLocal, srcLocal2,
            expMaxTensor, insumLocal, inmaxLocal, flashv2Tiling, unalignedSrcShape);
        
        if constexpr (Std::is_same<T1, half>::value && Std::is_same<T2, float>::value) {
            if (height == 8 && width == 512) {
                SoftMaxTiling flashv3Tiling;
                Duplicate(inmeanLocal, (float)9, height * 8);
                SoftmaxFlashV3<T1, T2, false>(dstLocal, inmeanLocal, insumLocal, inmaxLocal, srcLocal1,
                    expMaxTensor, inmeanLocal, insumLocal, inmaxLocal, flashv3Tiling, {8, 512, 8, 512, 2, 8, 0.9375});
                SoftmaxFlashV3<T1, T2, true>(dstLocal, inmeanLocal, insumLocal, inmaxLocal, srcLocal1,
                    expMaxTensor, inmeanLocal, insumLocal, inmaxLocal, flashv3Tiling, {8, 512, 8, 512, 2, 8, 0.9375});
            }
        }

        SoftMaxTiling gradTiling;
        SoftmaxGrad<T1, false, isDataFormatNZ>(expMaxTensor, srcLocal2, srcLocal1, gradTiling, true, alignSrcShape);
        SoftmaxGrad<T1, false, isDataFormatNZ>(expMaxTensor, srcLocal2, srcLocal1, gradTiling, true, unalignedSrcShape);
        SoftmaxGrad<T1, false, isDataFormatNZ>(dstLocal, srcLocal2, srcLocal1, gradTiling, false, alignSrcShape);
        SoftmaxGrad<T1, false, isDataFormatNZ>(dstLocal, srcLocal2, srcLocal1, gradTiling, false, unalignedSrcShape);

        outQueueDst.EnQue<T1>(dstLocal);
        inMeanQueue.FreeTensor(inmeanLocal);
        inMaxQueue.FreeTensor(inmaxLocal);
        inSumQueue.FreeTensor(insumLocal);
        inQueueSrc1.FreeTensor(srcLocal1);
        inQueueSrc2.FreeTensor(srcLocal2);
        expMaxQueue.FreeTensor(expMaxTensor);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T1> dstLocal = outQueueDst.DeQue<T1>();
        DataCopy(dstGlobal, dstLocal, height * width);
        outQueueDst.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrc1;
    TQue<TPosition::VECIN, 1> inQueueSrc2;
    TQue<TPosition::VECIN, 1> inMeanQueue;
    TQue<TPosition::VECIN, 1> inMaxQueue;
    TQue<TPosition::VECIN, 1> inSumQueue;
    TQue<TPosition::VECIN, 1> expMaxQueue;

    TQue<TPosition::VECOUT, 1> outQueueDst;
    GlobalTensor<T1> src1Global, dstGlobal;
    GlobalTensor<T1> src2Global;
    uint32_t height = 16;
    uint32_t width = 16;
};
} // namespace AscendC

template <typename T1, typename T2, bool isBasicBlock = false, bool isDataFormatNZ = false>
__global__ __aicore__ void MainSoftmax(__gm__ uint8_t* dstGm, __gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm,
    uint32_t height, uint32_t width)
{
    AscendC::KernelSoftmax<T1, T2, isBasicBlock, isDataFormatNZ> op;
    op.Init(src0Gm, src1Gm, dstGm, height, width);
    op.Process();
}

struct SoftMaxTestParams {
    uint32_t typeSize;
    uint32_t height;
    uint32_t width;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t);
};

class SoftMaxTestsuite : public testing::Test, public testing::WithParamInterface<SoftMaxTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_SOFTMAX, SoftMaxTestsuite,
    ::testing::Values(SoftMaxTestParams{ 2, 128, 144, MainSoftmax<half, half, false, true> },
    SoftMaxTestParams{ 2, 16, 16, MainSoftmax<half, half, false, false> },
    SoftMaxTestParams{ 2, 16, 32, MainSoftmax<half, half, false, false> },
    SoftMaxTestParams{ 2, 16, 16, MainSoftmax<half, half, false, true> },
    SoftMaxTestParams{ 2, 129, 256, MainSoftmax<half, half> },
    SoftMaxTestParams{ 2, 128, 256, MainSoftmax<half, half> },
    SoftMaxTestParams{ 2, 64, 128, MainSoftmax<half, float> },
    SoftMaxTestParams{ 2, 8, 512, MainSoftmax<half, float> },
    SoftMaxTestParams{ 2, 64, 128, MainSoftmax<half, float, false, true> },
    SoftMaxTestParams{ 4, 64, 128, MainSoftmax<float, float> },
    SoftMaxTestParams{ 4, 64, 128, MainSoftmax<float, float, false, true> }));

TEST_P(SoftMaxTestsuite, SoftMaxOpTestCase)
{
    auto param = GetParam();
    uint8_t src0Gm[param.height * param.width * param.typeSize];
    uint8_t src1Gm[param.height * param.width * param.typeSize];
    uint8_t dstGm[param.height * param.width * param.typeSize];
    param.cal_func(dstGm, src0Gm, src1Gm, param.height, param.width);
}
