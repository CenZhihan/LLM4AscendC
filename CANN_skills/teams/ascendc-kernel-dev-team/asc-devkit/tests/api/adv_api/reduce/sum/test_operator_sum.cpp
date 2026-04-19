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

template <typename T> class KernelSum {
public:
    __aicore__ inline KernelSum() {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* dstGm)
    {
        int elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);         // half=16 float=8
        int elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
        src1Global.SetGlobalBuffer((__gm__ T*)src0Gm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrc1, 1, 8 * 160 * sizeof(T));
        pipe.InitBuffer(outQueueDst, 1, ONE_BLK_SIZE); // align the 8 numbers as a whole
        int32_t repeatTimes = (160 + elementNumPerRep - 1) / elementNumPerRep; // workSize = ceil(repeatTimes)
        int32_t finalWorkSize = (repeatTimes + elementNumPerBlk - 1) / elementNumPerBlk * elementNumPerBlk * sizeof(T);
        pipe.InitBuffer(workQueue, 1, finalWorkSize); // round up
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
        DataCopy(srcLocal1, src1Global, 8*160);
        inQueueSrc1.EnQue(srcLocal1);
    }
    __aicore__ inline void Compute()
    {
        uint32_t elementNumPerBlk = 32 / sizeof(T);         // half=16 float=8
        LocalTensor<T> srcLocal1 = inQueueSrc1.DeQue<T>();

        LocalTensor<uint8_t> workLocal = workQueue.AllocTensor<uint8_t>();
        LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();

        SumParams params {8, 160, 152};
        Sum(dstLocal, srcLocal1, workLocal, params);

        outQueueDst.EnQue<T>(dstLocal);
        workQueue.FreeTensor(workLocal);
        inQueueSrc1.FreeTensor(srcLocal1);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, ONE_BLK_SIZE / sizeof(T));
        outQueueDst.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    // used for applying a temporary tensor
    TQue<TPosition::VECIN, 1> inQueueSrc1; 
    TQue<TPosition::VECIN, 1> workQueue;

    TQue<TPosition::VECOUT, 1> outQueueDst;
    // used for associating Gm
    GlobalTensor<T> src1Global;
    GlobalTensor<T> dstGlobal; 
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void MainSum(__gm__ uint8_t* dstGm, __gm__ uint8_t* src0Gm)
{
    AscendC::KernelSum<T> op;
    op.Init(src0Gm, dstGm);
    op.Process();
}

struct SumTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*);
};

class SumTestsuite : public testing::Test, public testing::WithParamInterface<SumTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_SUM, SumTestsuite,
    ::testing::Values(
        SumTestParams { 2, MainSum<half> },
        SumTestParams { 4, MainSum<float> }));

TEST_P(SumTestsuite, SumOpTestCase)
{
    auto param = GetParam();
    uint8_t src0Gm[8 * 160 * param.typeSize]; // external guarantee inner is 32B aligned
    uint32_t dstLen = (8 * param.typeSize + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    uint8_t dstGm[dstLen];
    param.cal_func(dstGm, src0Gm);
}
