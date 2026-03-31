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
#include "kernel_operator.h"
#include "impl/adv_api/detail/api_check/kernel_api_check.h"
#include "../reduce_case_common.h"

class ReduceXorSumAPICheck : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        AscendC::SetGCoreType(2);
        AscendC::KernelRaise::GetInstance().SetRaiseMode(false);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
        AscendC::KernelRaise::GetInstance().SetRaiseMode(true);
    }
};

TEST_F(ReduceXorSumAPICheck, ReduceXorSumAPICheckReduceXorSumSrc0CalCount)
{
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX2;
    AscendC::TQue<TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<TPosition::VECCALC> tmplocalBuf;
    uint32_t computeSize = 16;
    pipe.InitBuffer(inQueueX, 1, computeSize * sizeof(int16_t));
    pipe.InitBuffer(inQueueX2, 1, computeSize * sizeof(int16_t));
    pipe.InitBuffer(outQueueY, 1, 32);
    pipe.InitBuffer(tmplocalBuf, computeSize);
    AscendC::LocalTensor<int16_t> src0Tensor = inQueueX.AllocTensor<int16_t>();
    AscendC::LocalTensor<int16_t> src1Tensor = inQueueX2.AllocTensor<int16_t>();
    AscendC::LocalTensor<int16_t> dstTensor = outQueueY.AllocTensor<int16_t>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    uint32_t calCount = 32;
    constexpr bool isReuseSource = false;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncReduceXorSum<
        int16_t, isReuseSource>("ReduceXorSum",
        dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
    inQueueX.FreeTensor(src0Tensor);
    inQueueX2.FreeTensor(src1Tensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 2);
}

TEST_F(ReduceXorSumAPICheck, ReduceXorSumAPICheckReduceXorSumSrc1CalCount)
{
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX2;
    AscendC::TQue<TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<TPosition::VECCALC> tmplocalBuf;
    uint32_t computeSize = 16;
    pipe.InitBuffer(inQueueX, 1, computeSize * sizeof(int16_t));
    pipe.InitBuffer(inQueueX2, 1, computeSize * sizeof(int16_t));
    pipe.InitBuffer(outQueueY, 1, 32);
    pipe.InitBuffer(tmplocalBuf, computeSize);
    AscendC::LocalTensor<int16_t> src0Tensor = inQueueX.AllocTensor<int16_t>();
    AscendC::LocalTensor<int16_t> src1Tensor = inQueueX2.AllocTensor<int16_t>();
    AscendC::LocalTensor<int16_t> dstTensor = outQueueY.AllocTensor<int16_t>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    uint32_t calCount = 32;
    constexpr bool isReuseSource = false;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncReduceXorSum<
        int16_t, isReuseSource>("ReduceXorSum",
        dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
    inQueueX.FreeTensor(src0Tensor);
    inQueueX2.FreeTensor(src1Tensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 2);
}


TEST_F(ReduceXorSumAPICheck, ReduceXorSumAPICheckReduceXorSumSrc0Pos)
{
    
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::A1, 1> inQueueX;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX2;
    AscendC::TQue<TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<TPosition::VECCALC> tmplocalBuf;
    uint32_t computeSize = 16;
    pipe.InitBuffer(inQueueX, 1, computeSize * sizeof(int16_t));
    pipe.InitBuffer(inQueueX2, 1, computeSize * sizeof(int16_t));
    pipe.InitBuffer(outQueueY, 1, 32);
    pipe.InitBuffer(tmplocalBuf, computeSize);
    AscendC::LocalTensor<int16_t> src0Tensor = inQueueX.AllocTensor<int16_t>();
    AscendC::LocalTensor<int16_t> src1Tensor = inQueueX2.AllocTensor<int16_t>();
    AscendC::LocalTensor<int16_t> dstTensor = outQueueY.AllocTensor<int16_t>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    uint32_t calCount = 16;
    constexpr bool isReuseSource = false;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncReduceXorSum<
        int16_t, isReuseSource>("ReduceXorSum",
        dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
    inQueueX.FreeTensor(src0Tensor);
    inQueueX2.FreeTensor(src1Tensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(ReduceXorSumAPICheck, ReduceXorSumAPICheckReduceXorSumSrc1Pos)
{
    
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<TPosition::A1, 1> inQueueX2;
    AscendC::TQue<TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<TPosition::VECCALC> tmplocalBuf;
    uint32_t computeSize = 16;
    pipe.InitBuffer(inQueueX, 1, computeSize * sizeof(int16_t));
    pipe.InitBuffer(inQueueX2, 1, computeSize * sizeof(int16_t));
    pipe.InitBuffer(outQueueY, 1, 32);
    pipe.InitBuffer(tmplocalBuf, computeSize);
    AscendC::LocalTensor<int16_t> src0Tensor = inQueueX.AllocTensor<int16_t>();
    AscendC::LocalTensor<int16_t> src1Tensor = inQueueX2.AllocTensor<int16_t>();
    AscendC::LocalTensor<int16_t> dstTensor = outQueueY.AllocTensor<int16_t>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    uint32_t calCount = 16;
    constexpr bool isReuseSource = false;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncReduceXorSum<
        int16_t, isReuseSource>("ReduceXorSum",
        dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
    inQueueX.FreeTensor(src0Tensor);
    inQueueX2.FreeTensor(src1Tensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(ReduceXorSumAPICheck, ReduceXorSumAPICheckReduceXorSumDstPos)
{
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX2;
    AscendC::TQue<TPosition::A1, 1> outQueueY;
    AscendC::TBuf<TPosition::VECCALC> tmplocalBuf;
    uint32_t computeSize = 16;
    pipe.InitBuffer(inQueueX, 1, computeSize * sizeof(int16_t));
    pipe.InitBuffer(inQueueX2, 1, computeSize * sizeof(int16_t));
    pipe.InitBuffer(outQueueY, 1, 32);
    pipe.InitBuffer(tmplocalBuf, computeSize);
    AscendC::LocalTensor<int16_t> src0Tensor = inQueueX.AllocTensor<int16_t>();
    AscendC::LocalTensor<int16_t> src1Tensor = inQueueX2.AllocTensor<int16_t>();
    AscendC::LocalTensor<int16_t> dstTensor = outQueueY.AllocTensor<int16_t>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    uint32_t calCount = 16;
    constexpr bool isReuseSource = false;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncReduceXorSum<
        int16_t, isReuseSource>("ReduceXorSum",
        dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
    inQueueX.FreeTensor(src0Tensor);
    inQueueX2.FreeTensor(src1Tensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}


TEST_F(ReduceXorSumAPICheck, ReduceXorSumAPICheckTmpPos)
{
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX2;
    AscendC::TQue<TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<TPosition::A1> tmplocalBuf;
    uint32_t computeSize = 16;
    pipe.InitBuffer(inQueueX, 1, computeSize * sizeof(int16_t));
    pipe.InitBuffer(inQueueX2, 1, computeSize * sizeof(int16_t));
    pipe.InitBuffer(outQueueY, 1, 32);
    pipe.InitBuffer(tmplocalBuf, computeSize);
    AscendC::LocalTensor<int16_t> src0Tensor = inQueueX.AllocTensor<int16_t>();
    AscendC::LocalTensor<int16_t> src1Tensor = inQueueX2.AllocTensor<int16_t>();
    AscendC::LocalTensor<int16_t> dstTensor = outQueueY.AllocTensor<int16_t>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    uint32_t calCount = 16;
    constexpr bool isReuseSource = false;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncReduceXorSum<
        int16_t, isReuseSource>("ReduceXorSum",
        dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
    inQueueX.FreeTensor(src0Tensor);
    inQueueX2.FreeTensor(src1Tensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(ReduceXorSumAPICheck, ReduceXorSumAPICheckOverlap)
{
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<TPosition::VECIN, 1> inQueueX2;
    AscendC::TQue<TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<TPosition::VECCALC> tmplocalBuf;
    uint32_t computeSize = 16;
    pipe.InitBuffer(inQueueX, 1, computeSize * sizeof(int16_t));
    pipe.InitBuffer(inQueueX2, 1, computeSize * sizeof(int16_t));
    pipe.InitBuffer(outQueueY, 1, 32);
    pipe.InitBuffer(tmplocalBuf, computeSize);
    AscendC::LocalTensor<int16_t> src0Tensor = inQueueX.AllocTensor<int16_t>();
    AscendC::LocalTensor<int16_t> src1Tensor = inQueueX2.AllocTensor<int16_t>();
    AscendC::LocalTensor<int16_t> dstTensor = outQueueY.AllocTensor<int16_t>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    uint32_t calCount = 16;
    constexpr bool isReuseSource = false;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncReduceXorSum<
        int16_t, isReuseSource>("ReduceXorSum",
        src0Tensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
    inQueueX.FreeTensor(src0Tensor);
    inQueueX2.FreeTensor(src1Tensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 2);
}


