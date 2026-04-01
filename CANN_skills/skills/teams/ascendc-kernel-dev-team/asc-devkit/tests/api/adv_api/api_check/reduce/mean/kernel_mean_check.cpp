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

class MeanAPICheck : public testing::Test {
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


TEST_F(MeanAPICheck, MeanAPICheckTestDataType)
{
    MeanParams meanParams = {1, 32, 2};
    uint32_t finalWorkSize = ComputeTmpBufSize<uint8_t, MeanParams>(meanParams);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, meanParams.outter * meanParams.inner * sizeof(uint8_t));
    pipe.InitBuffer(outQueueY, 1, (meanParams.outter * sizeof(uint8_t) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
    pipe.InitBuffer(tmplocalBuf, finalWorkSize);
    AscendC::LocalTensor<uint8_t> srcTensor = inQueueX.AllocTensor<uint8_t>();
    AscendC::LocalTensor<uint8_t> dstTensor = outQueueY.AllocTensor<uint8_t>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    constexpr bool isReuseSource = false;
    constexpr bool isBasicBlock = false;
    constexpr int32_t reduceDim = -1;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncMean<
        uint8_t, float, isReuseSource, isBasicBlock, reduceDim>(
        "Mean", dstTensor, srcTensor, sharedTmpBuffer, meanParams, finalWorkSize);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);

}

TEST_F(MeanAPICheck, MeanAPICheckMeanOutter)
{
    MeanParams meanParams = { 1, 32, 2 };
    uint32_t finalWorkSize = ComputeTmpBufSize<float, MeanParams>(meanParams);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, meanParams.outter * meanParams.inner * sizeof(float));
    pipe.InitBuffer(outQueueY, 1, (meanParams.outter * sizeof(float) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
    pipe.InitBuffer(tmplocalBuf, finalWorkSize);
    AscendC::LocalTensor<float> srcTensor = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> dstTensor = outQueueY.AllocTensor<float>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    
    constexpr bool isReuseSource = false;
    constexpr bool isBasicBlock = false;
    constexpr int32_t reduceDim = -1;
    meanParams.outter = 0;

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncMean<
        float, float, isReuseSource, isBasicBlock, reduceDim>(
        "Mean", dstTensor, srcTensor, sharedTmpBuffer, meanParams, finalWorkSize);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(MeanAPICheck, MeanAPICheckMeanInnnerAlign)
{
    MeanParams meanParams = { 1, 32, 2 };
    uint32_t finalWorkSize = ComputeTmpBufSize<float, MeanParams>(meanParams);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, meanParams.outter * meanParams.inner * sizeof(float));
    pipe.InitBuffer(outQueueY, 1, (meanParams.outter * sizeof(float) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
    pipe.InitBuffer(tmplocalBuf, finalWorkSize);
    AscendC::LocalTensor<float> srcTensor = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> dstTensor = outQueueY.AllocTensor<float>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();


    
    constexpr bool isReuseSource = false;
    constexpr bool isBasicBlock = false;
    constexpr int32_t reduceDim = -1;
    meanParams.inner = 15;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncMean<
        float, float, isReuseSource, isBasicBlock, reduceDim>(
        "Mean", dstTensor, srcTensor, sharedTmpBuffer, meanParams, finalWorkSize);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(MeanAPICheck, MeanAPICheckMeanNSize)
{
    MeanParams meanParams = { 1, 32, 2 };
    uint32_t finalWorkSize = ComputeTmpBufSize<float, MeanParams>(meanParams);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, meanParams.outter * meanParams.inner * sizeof(float));
    pipe.InitBuffer(outQueueY, 1, (meanParams.outter * sizeof(float) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
    pipe.InitBuffer(tmplocalBuf, finalWorkSize);
    AscendC::LocalTensor<float> srcTensor = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> dstTensor = outQueueY.AllocTensor<float>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();


    
    constexpr bool isReuseSource = false;
    constexpr bool isBasicBlock = false;
    constexpr int32_t reduceDim = -1;
    meanParams.n = 0;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncMean<
        float, float, isReuseSource, isBasicBlock, reduceDim>(
        "Mean", dstTensor, srcTensor, sharedTmpBuffer, meanParams, finalWorkSize);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(MeanAPICheck, MeanAPICheckMeanSrcSize)
{
    MeanParams meanParams = { 2, 32, 2 };
    uint32_t finalWorkSize = ComputeTmpBufSize<float, MeanParams>(meanParams);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, (1 * meanParams.inner) * sizeof(float));
    pipe.InitBuffer(outQueueY, 1, (meanParams.outter * sizeof(float) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
    pipe.InitBuffer(tmplocalBuf, finalWorkSize);
    AscendC::LocalTensor<float> srcTensor = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> dstTensor = outQueueY.AllocTensor<float>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    
    constexpr bool isReuseSource = false;
    constexpr bool isBasicBlock = false;
    constexpr int32_t reduceDim = -1;

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncMean<
        float, float, isReuseSource, isBasicBlock, reduceDim>(
        "Mean", dstTensor, srcTensor, sharedTmpBuffer, meanParams, finalWorkSize);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(MeanAPICheck, MeanAPICheckMeanDstSize)
{
    MeanParams meanParams = { 9, 32, 2 };
    uint32_t finalWorkSize = ComputeTmpBufSize<float, MeanParams>(meanParams);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, meanParams.outter * meanParams.inner * sizeof(float));
    pipe.InitBuffer(outQueueY, 1, 8);
    pipe.InitBuffer(tmplocalBuf, finalWorkSize);
    AscendC::LocalTensor<float> srcTensor = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> dstTensor = outQueueY.AllocTensor<float>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    
    constexpr bool isReuseSource = false;
    constexpr bool isBasicBlock = false;
    constexpr int32_t reduceDim = -1;

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncMean<
        float, float, isReuseSource, isBasicBlock, reduceDim>(
        "Mean", dstTensor, srcTensor, sharedTmpBuffer, meanParams, finalWorkSize);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(MeanAPICheck, MeanAPICheckMeanTmpSize)
{
    MeanParams meanParams = { 1, 32, 2 };
    uint32_t finalWorkSize = ComputeTmpBufSize<float, MeanParams>(meanParams);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, meanParams.outter * meanParams.inner * sizeof(float));
    pipe.InitBuffer(outQueueY, 1, (meanParams.outter * sizeof(float) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
    pipe.InitBuffer(tmplocalBuf, 16);
    AscendC::LocalTensor<float> srcTensor = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> dstTensor = outQueueY.AllocTensor<float>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    
    constexpr bool isReuseSource = false;
    constexpr bool isBasicBlock = false;
    constexpr int32_t reduceDim = -1;

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncMean<
        float, float, isReuseSource, isBasicBlock, reduceDim>(
        "Mean", dstTensor, srcTensor, sharedTmpBuffer, meanParams, finalWorkSize);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(MeanAPICheck, MeanAPICheckMeanSrcPos)
{
    MeanParams meanParams = { 1, 32, 2 };
    uint32_t finalWorkSize = ComputeTmpBufSize<float, MeanParams>(meanParams);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, meanParams.outter * meanParams.inner * sizeof(float));
    pipe.InitBuffer(outQueueY, 1, (meanParams.outter * sizeof(float) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
    pipe.InitBuffer(tmplocalBuf, finalWorkSize);
    AscendC::LocalTensor<float> srcTensor = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> dstTensor = outQueueY.AllocTensor<float>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    
    constexpr bool isReuseSource = false;
    constexpr bool isBasicBlock = false;
    constexpr int32_t reduceDim = -1;

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncMean<
        float, float, isReuseSource, isBasicBlock, reduceDim>(
        "Mean", dstTensor, srcTensor, sharedTmpBuffer, meanParams, finalWorkSize);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(MeanAPICheck, MeanAPICheckMeanDstPos)
{
    MeanParams meanParams = { 1, 32, 2 };
    uint32_t finalWorkSize = ComputeTmpBufSize<float, MeanParams>(meanParams);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::A1, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, meanParams.outter * meanParams.inner * sizeof(float));
    pipe.InitBuffer(outQueueY, 1, (meanParams.outter * sizeof(float) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
    pipe.InitBuffer(tmplocalBuf, finalWorkSize);
    AscendC::LocalTensor<float> srcTensor = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> dstTensor = outQueueY.AllocTensor<float>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();


    
    constexpr bool isReuseSource = false;
    constexpr bool isBasicBlock = false;
    constexpr int32_t reduceDim = -1;

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncMean<
        float, float, isReuseSource, isBasicBlock, reduceDim>(
        "Mean", dstTensor, srcTensor, sharedTmpBuffer, meanParams, finalWorkSize);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(MeanAPICheck, MeanAPICheckMeanTmpPos)
{
    MeanParams meanParams = { 1, 32, 2 };
    uint32_t finalWorkSize = ComputeTmpBufSize<float, MeanParams>(meanParams);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::A1> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, meanParams.outter * meanParams.inner * sizeof(float));
    pipe.InitBuffer(outQueueY, 1, (meanParams.outter * sizeof(float) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
    pipe.InitBuffer(tmplocalBuf, finalWorkSize);
    AscendC::LocalTensor<float> srcTensor = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> dstTensor = outQueueY.AllocTensor<float>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    constexpr bool isReuseSource = false;
    constexpr bool isBasicBlock = false;
    constexpr int32_t reduceDim = -1;

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncMean<
        float, float, isReuseSource, isBasicBlock, reduceDim>(
        "Mean", dstTensor, srcTensor, sharedTmpBuffer, meanParams, finalWorkSize);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(MeanAPICheck, MeanAPICheckMeanOverlap)
{
    MeanParams meanParams = { 1, 32, 2 };
    uint32_t finalWorkSize = ComputeTmpBufSize<float, MeanParams>(meanParams);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, meanParams.outter * meanParams.inner * sizeof(float));
    pipe.InitBuffer(outQueueY, 1, (meanParams.outter * sizeof(float) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE);
    pipe.InitBuffer(tmplocalBuf, finalWorkSize);
    AscendC::LocalTensor<float> srcTensor = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> dstTensor = outQueueY.AllocTensor<float>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();

    constexpr bool isReuseSource = false;
    constexpr bool isBasicBlock = false;
    constexpr int32_t reduceDim = -1;

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    HighLevelApiCheck::CheckFuncMean<
        float, float, isReuseSource, isBasicBlock, reduceDim>(
        "Mean", srcTensor, srcTensor, sharedTmpBuffer, meanParams, finalWorkSize);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 2);
}

