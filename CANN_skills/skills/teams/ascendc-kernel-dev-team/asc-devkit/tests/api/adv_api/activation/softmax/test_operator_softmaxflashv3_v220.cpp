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
template <typename T, typename U, bool isUpdate = false> 
class KernelSoftmaxFlashV3 {
public:
    __aicore__ inline KernelSoftmaxFlashV3() {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* dstGm, uint32_t inheight, uint32_t inwidth)
    {
        height = inheight;
        width = inwidth;
        int elementNumPerBlk = 32 / sizeof(U); // half=16 float=8
        src1Global.SetGlobalBuffer((__gm__ T*)src0Gm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrc1, 1, height * width * sizeof(T));
        pipe.InitBuffer(outQueueDst, 1, height * width * sizeof(T));
        pipe.InitBuffer(inMaxQueue, 1, height * elementNumPerBlk * sizeof(U));
        pipe.InitBuffer(inMeanQueue, 1, height * elementNumPerBlk * sizeof(U));
        pipe.InitBuffer(inSumQueue, 1, height * elementNumPerBlk * sizeof(U));
        elementNumPerBlk = 32 / sizeof(T); // half=16 float=8
        pipe.InitBuffer(expMaxQueue, 1, height * elementNumPerBlk * sizeof(T));
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
        LocalTensor<T> srcLocal1 = inQueueSrc1.AllocTensor<T>();
        DataCopy(srcLocal1, src1Global, height * width);
        inQueueSrc1.EnQue(srcLocal1);
    }
    __aicore__ inline void Compute()
    {
        uint32_t elementNumPerBlk = 32 / sizeof(U); // half=16 float=8
        LocalTensor<T> srcLocal1 = inQueueSrc1.DeQue<T>();
        LocalTensor<U> inmaxLocal = inMaxQueue.AllocTensor<U>();
        LocalTensor<U> insumLocal = inSumQueue.AllocTensor<U>();
        LocalTensor<U> inmeanLocal = inMeanQueue.AllocTensor<U>();
        LocalTensor<T> expMaxTensor = expMaxQueue.AllocTensor<T>();

        LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
        const int shapeDim = 2;
        uint32_t array[2] = {height, width};

        srcLocal1.SetShapeInfo(ShapeInfo(shapeDim, array)); // default ND
        dstLocal.SetShapeInfo(ShapeInfo(shapeDim, array));
        SoftMaxParams params = { height, width, height, width, loopCnt, splitMeanCnt, alpha };
        SoftMaxTiling flashTiling = { height, width, height * width, height, 8, height * 8,
            height, width, height * width, height, 8, height * 8, 1, 0, 0, 0};
        SoftmaxFlashV3<T, U, isUpdate, false, false>(dstLocal, inmeanLocal, insumLocal, inmaxLocal, srcLocal1, expMaxTensor,
            inmeanLocal, insumLocal, inmaxLocal, flashTiling, params);
        outQueueDst.EnQue<T>(dstLocal);
        inMaxQueue.FreeTensor(inmaxLocal);
        inSumQueue.FreeTensor(insumLocal);
        inQueueSrc1.FreeTensor(srcLocal1);
        expMaxQueue.FreeTensor(expMaxTensor);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, height * width);
        outQueueDst.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrc1;
    TQue<TPosition::VECIN, 1> inMeanQueue;
    TQue<TPosition::VECIN, 1> inMaxQueue;
    TQue<TPosition::VECIN, 1> inSumQueue;
    TQue<TPosition::VECIN, 1> expMaxQueue;

    TQue<TPosition::VECOUT, 1> outQueueDst;
    GlobalTensor<T> src1Global, dstGlobal;
    uint32_t height = 16;
    uint32_t width = 16;
    uint32_t loopCnt = 2;
    uint32_t splitMeanCnt = 8;
    float alpha = 0.9375;
};
} // namespace AscendC

template <typename T, typename U, bool isUpdate = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__global__ __aicore__ void MainSoftmax(__gm__ uint8_t* dstGm, __gm__ uint8_t* src0Gm, uint32_t height, uint32_t width)
{
    AscendC::KernelSoftmaxFlashV3<T, U, isUpdate> op;
    op.Init(src0Gm, dstGm, height, width);
    op.Process();
}

struct SoftMaxV3TestParams {
    uint32_t typeSize;
    uint32_t height;
    uint32_t width;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t, uint32_t);
};

class SoftMaxFlashV3Testsuite : public testing::Test, public testing::WithParamInterface<SoftMaxV3TestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_SOFTMAXFLASHV3, SoftMaxFlashV3Testsuite,
    ::testing::Values(SoftMaxV3TestParams{ 2, 5, 1024, MainSoftmax<half, float, true> },
    SoftMaxV3TestParams{ 2, 5, 1024, MainSoftmax<half, float, false> },
    SoftMaxV3TestParams{ 2, 8, 1024, MainSoftmax<half, float, true> },
    SoftMaxV3TestParams{ 2, 8, 1024, MainSoftmax<half, float, false> }));

TEST_P(SoftMaxFlashV3Testsuite, SoftMaxFlashV3OpTestCase)
{
    auto param = GetParam();
    uint8_t src0Gm[param.height * param.width * param.typeSize];
    uint8_t dstGm[param.height * param.width * param.typeSize];
    param.cal_func(dstGm, src0Gm, param.height, param.width);
}
