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
template <typename T1, typename T2> class KernelSoftmax {
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
        pipe.InitBuffer(inMaxQueue, 1, height * elementNumPerBlk * sizeof(T2));
        pipe.InitBuffer(inSumQueue, 1, height * elementNumPerBlk * sizeof(T2));
        elementNumPerBlk = 32 / sizeof(T1); // half=16 float=8
        pipe.InitBuffer(expMaxQueue, 1, height * elementNumPerBlk * sizeof(T1));
        pipe.InitBuffer(logSumQueue, 1, height * 8 * sizeof(float));
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

        LocalTensor<T2> inmaxLocal = inMaxQueue.AllocTensor<T2>();
        LocalTensor<T2> insumLocal = inSumQueue.AllocTensor<T2>();
        LocalTensor<T1> expMaxTensor = expMaxQueue.AllocTensor<T1>();
        LocalTensor<float> logSumLocal = logSumQueue.AllocTensor<float>();

        LocalTensor<T1> dstLocal = outQueueDst.AllocTensor<T1>();
        const int shapeDim = 2;
        uint32_t array[2] = {height, width};

        srcLocal1.SetShapeInfo(ShapeInfo(shapeDim, array)); // default ND
        srcLocal2.SetShapeInfo(ShapeInfo(shapeDim, array)); // default ND
        dstLocal.SetShapeInfo(ShapeInfo(shapeDim, array));

        SoftMaxShapeInfo srcShape = { height, width, height, width };
        SoftMaxTiling tiling;
        SoftMax<T1, false>(srcLocal1, insumLocal, inmaxLocal, srcLocal1, tiling);
        SimpleSoftMax<T1, false>(srcLocal1, insumLocal, inmaxLocal, srcLocal1, tiling);
        outQueueDst.EnQue<T1>(dstLocal);
        inMaxQueue.FreeTensor(inmaxLocal);
        inSumQueue.FreeTensor(insumLocal);
        inQueueSrc1.FreeTensor(srcLocal1);
        inQueueSrc2.FreeTensor(srcLocal2);
        expMaxQueue.FreeTensor(expMaxTensor);
        logSumQueue.FreeTensor(logSumLocal);
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
    TQue<TPosition::VECIN, 1> inMaxQueue;
    TQue<TPosition::VECIN, 1> inSumQueue;
    TQue<TPosition::VECIN, 1> expMaxQueue;
    TQue<TPosition::VECIN, 1> logSumQueue;

    TQue<TPosition::VECOUT, 1> outQueueDst;
    GlobalTensor<T1> src1Global, dstGlobal;
    GlobalTensor<T1> src2Global;
    uint32_t height = 16;
    uint32_t width = 16;
};
} // namespace AscendC

template <typename T1, typename T2>
__global__ __aicore__ void MainSoftmax(__gm__ uint8_t* dstGm, __gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm,
    uint32_t height, uint32_t width)
{
    AscendC::KernelSoftmax<T1, T2> op;
    op.Init(src0Gm, src1Gm, dstGm, height, width);
    op.Process();
}

struct SoftMaxTestParams {
    uint32_t typeSize;
    uint32_t height;
    uint32_t width;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t);
};

class SoftMaxTestsuite : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SoftMaxTestsuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "SoftMaxTestsuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown()
    {
    }
};

template <typename srcType> bool AddImplCheckSrcValue(__ubuf__ srcType* src)
{
    return (src[0] == (srcType)0);
}
bool AddImplCheckCalCount(const int32_t& calCount)
{
    return (calCount == 1020);
}

TEST_F(SoftMaxTestsuite, SoftMaxTestsuite_half_8_1024)
{
    SoftMaxTestParams param{ sizeof(half), 8, 1024 };

    uint8_t src0Gm[param.height * param.width * param.typeSize]{ 0x00 };
    uint8_t src1Gm[param.height * param.width * param.typeSize]{ 0x00 };
    uint8_t dstGm[param.height * param.width * param.typeSize]{ 0x00 };

    MainSoftmax<half, half>(dstGm, src0Gm, src1Gm, param.height, param.width);
}
TEST_F(SoftMaxTestsuite, SoftMaxTestsuite_half_64_128)
{
    SoftMaxTestParams param{ sizeof(half), 64, 128 };

    uint8_t src0Gm[param.height * param.width * param.typeSize]{ 0x00 };
    uint8_t src1Gm[param.height * param.width * param.typeSize]{ 0x00 };
    uint8_t dstGm[param.height * param.width * param.typeSize]{ 0x00 };

    MainSoftmax<half, half>(dstGm, src0Gm, src1Gm, param.height, param.width);
}

TEST_F(SoftMaxTestsuite, SoftMaxTestsuite_half_60_144)
{
    SoftMaxTestParams param{ sizeof(half), 60, 144 };

    uint8_t src0Gm[param.height * param.width * param.typeSize]{ 0x00 };
    uint8_t src1Gm[param.height * param.width * param.typeSize]{ 0x00 };
    uint8_t dstGm[param.height * param.width * param.typeSize]{ 0x00 };

    MainSoftmax<half, half>(dstGm, src0Gm, src1Gm, param.height, param.width);
}
TEST_F(SoftMaxTestsuite, SoftMaxTestsuite_float_8_1024)
{
    SoftMaxTestParams param{ sizeof(float), 8, 1024 };

    uint8_t src0Gm[param.height * param.width * param.typeSize]{ 0x00 };
    uint8_t src1Gm[param.height * param.width * param.typeSize]{ 0x00 };
    uint8_t dstGm[param.height * param.width * param.typeSize]{ 0x00 };

    MainSoftmax<float, float>(dstGm, src0Gm, src1Gm, param.height, param.width);
}

TEST_F(SoftMaxTestsuite, SoftMaxTestsuite_float_64_128)
{
    SoftMaxTestParams param{ sizeof(float), 64, 128 };

    uint8_t src0Gm[param.height * param.width * param.typeSize]{ 0x00 };
    uint8_t src1Gm[param.height * param.width * param.typeSize]{ 0x00 };
    uint8_t dstGm[param.height * param.width * param.typeSize]{ 0x00 };

    MainSoftmax<float, float>(dstGm, src0Gm, src1Gm, param.height, param.width);
}

TEST_F(SoftMaxTestsuite, SoftMaxTestsuite_float_256_56)
{
    SoftMaxTestParams param{ sizeof(float), 256, 56 };

    uint8_t src0Gm[param.height * param.width * param.typeSize]{ 0x00 };
    uint8_t src1Gm[param.height * param.width * param.typeSize]{ 0x00 };
    uint8_t dstGm[param.height * param.width * param.typeSize]{ 0x00 };

    MainSoftmax<float, float>(dstGm, src0Gm, src1Gm, param.height, param.width);
}