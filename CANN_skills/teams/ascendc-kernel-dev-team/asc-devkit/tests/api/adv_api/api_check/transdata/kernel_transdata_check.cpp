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
using namespace AscendC;

class TransDataAPICheck : public testing::Test {
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


TEST_F(TransDataAPICheck, TransDataAPICheckTestConfig)
{
    int32_t n = 1;
    int32_t c = 1;
    int32_t d = 1;
    int32_t h = 1;
    int32_t w = 1;
    int32_t n0 = 16;
    int32_t c0 = 16;
    int32_t c1 = (c + c0 - 1) / c0;
    int32_t n1 = (n + n0 - 1) / n0;
    int32_t hw0 = 16;
    int32_t hw1 = (h * w + hw0 - 1) / hw0;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, n1 * n0 * c1 * c0 * d * h * w * sizeof(half));
    pipe.InitBuffer(outQueueY, 1, n1 * n0 * c1 * c0 * d * h * w * sizeof(half));
    pipe.InitBuffer(tmplocalBuf, c * d * hw0 * hw1 + n1 * n0 * d * c1 * c0 * hw0 * hw1);

    Layout ncdhwLayout = MakeLayout(MakeShape(n, c, d, h, w), MakeStride());
    Layout ndc1hwc0Layout = MakeLayout(MakeShape(n, d, c1, h, w, c0), MakeStride());
    Layout fractalzLayout = MakeLayout(MakeShape(d, c1, h, w, n1, n0, c0), MakeStride());

    AscendC::LocalTensor<half> srcTensor = inQueueX.AllocTensor<half>();
    AscendC::LocalTensor<half> dstTensor = outQueueY.AllocTensor<half>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    static constexpr TransDataConfig config = {DataFormat::FRACTAL_Z_3D, DataFormat::FRACTAL_Z_3D};
    TransDataParams<decltype(fractalzLayout), decltype(fractalzLayout)> params = {fractalzLayout, fractalzLayout};

    HighLevelApiCheck::CheckFuncTransData<config, half, decltype(fractalzLayout), decltype(fractalzLayout)>(
        "TransData", dstTensor, srcTensor, sharedTmpBuffer, params);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);

}

TEST_F(TransDataAPICheck, TransDataAPICheckTransDataSrcShapeDstShape)
{
    int32_t n = 1;
    int32_t c = 1;
    int32_t d = 1;
    int32_t h = 1;
    int32_t w = 1;
    int32_t n0 = 16;
    int32_t c0 = 16;
    int32_t c1 = (c + c0 - 1) / c0;
    int32_t n1 = (n + n0 - 1) / n0;
    int32_t hw0 = 16;
    int32_t hw1 = (h * w + hw0 - 1) / hw0;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, n * c * d * hw0 * hw1 * sizeof(half));
    pipe.InitBuffer(outQueueY, 1, n1 * n0 * c1 * c0 * d * h * w * sizeof(half));
    pipe.InitBuffer(tmplocalBuf, c * d * hw0 * hw1 + n1 * n0 * d * c1 * c0 * hw0 * hw1);

    Layout ncdhwLayout = MakeLayout(MakeShape(n, c, d, 2, w), MakeStride());
    Layout fractalzLayout = MakeLayout(MakeShape(d, c1, h, w, n1, n0, c0), MakeStride());

    AscendC::LocalTensor<half> srcTensor = inQueueX.AllocTensor<half>();
    AscendC::LocalTensor<half> dstTensor = outQueueY.AllocTensor<half>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    static constexpr TransDataConfig config = {DataFormat::NCDHW, DataFormat::FRACTAL_Z_3D};
    TransDataParams<decltype(ncdhwLayout), decltype(fractalzLayout)> params = {ncdhwLayout, fractalzLayout};

    HighLevelApiCheck::CheckFuncTransData<config, half, decltype(ncdhwLayout), decltype(fractalzLayout)>("TransData", dstTensor, srcTensor, sharedTmpBuffer, params);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(TransDataAPICheck, TransDataAPICheckTransDataSrcDstSize)
{
    int32_t n = 1;
    int32_t c = 1;
    int32_t d = 3;
    int32_t h = 3;
    int32_t w = 3;
    int32_t n0 = 16;
    int32_t c0 = 16;
    int32_t c1 = (c + c0 - 1) / c0;
    int32_t n1 = (n + n0 - 1) / n0;
    int32_t hw0 = 16;
    int32_t hw1 = (h * w + hw0 - 1) / hw0;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, 16 * sizeof(half));
    pipe.InitBuffer(outQueueY, 1, n1 * n0 * c1 * c0 * d * h * w * sizeof(half));
    pipe.InitBuffer(tmplocalBuf, c * d * hw0 * hw1 + n1 * n0 * d * c1 * c0 * hw0 * hw1);

    Layout ncdhwLayout = MakeLayout(MakeShape(n, c, d, h, w), MakeStride());
    Layout fractalzLayout = MakeLayout(MakeShape(d, c1, h, w, n1, n0, c0), MakeStride());

    AscendC::LocalTensor<half> srcTensor = inQueueX.AllocTensor<half>();
    AscendC::LocalTensor<half> dstTensor = outQueueY.AllocTensor<half>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    static constexpr TransDataConfig config = {DataFormat::NCDHW, DataFormat::FRACTAL_Z_3D};
    TransDataParams<decltype(ncdhwLayout), decltype(fractalzLayout)> params = {ncdhwLayout, fractalzLayout};

    HighLevelApiCheck::CheckFuncTransData<config, half, decltype(ncdhwLayout), decltype(fractalzLayout)>("TransData", dstTensor, srcTensor, sharedTmpBuffer, params);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}


TEST_F(TransDataAPICheck, TransDataAPICheckTransDataSrcPos)
{
    int32_t n = 1;
    int32_t c = 1;
    int32_t d = 1;
    int32_t h = 1;
    int32_t w = 1;
    int32_t n0 = 16;
    int32_t c0 = 16;
    int32_t c1 = (c + c0 - 1) / c0;
    int32_t n1 = (n + n0 - 1) / n0;
    int32_t hw0 = 16;
    int32_t hw1 = (h * w + hw0 - 1) / hw0;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, n * c * d * hw0 * hw1 * sizeof(half));
    pipe.InitBuffer(outQueueY, 1, n1 * n0 * c1 * c0 * d * h * w * sizeof(half));
    pipe.InitBuffer(tmplocalBuf, c * d * hw0 * hw1 + n1 * n0 * d * c1 * c0 * hw0 * hw1);

    Layout ncdhwLayout = MakeLayout(MakeShape(n, c, d, h, w), MakeStride());
    Layout fractalzLayout = MakeLayout(MakeShape(d, c1, h, w, n1, n0, c0), MakeStride());

    AscendC::LocalTensor<half> srcTensor = inQueueX.AllocTensor<half>();
    AscendC::LocalTensor<half> dstTensor = outQueueY.AllocTensor<half>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    static constexpr TransDataConfig config = {DataFormat::NCDHW, DataFormat::FRACTAL_Z_3D};
    TransDataParams<decltype(ncdhwLayout), decltype(fractalzLayout)> params = {ncdhwLayout, fractalzLayout};

    HighLevelApiCheck::CheckFuncTransData<config, half, decltype(ncdhwLayout), decltype(fractalzLayout)>("TransData", dstTensor, srcTensor, sharedTmpBuffer, params);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(TransDataAPICheck, TransDataAPICheckTransDataDstPos)
{
    int32_t n = 1;
    int32_t c = 1;
    int32_t d = 1;
    int32_t h = 1;
    int32_t w = 1;
    int32_t n0 = 16;
    int32_t c0 = 16;
    int32_t c1 = (c + c0 - 1) / c0;
    int32_t n1 = (n + n0 - 1) / n0;
    int32_t hw0 = 16;
    int32_t hw1 = (h * w + hw0 - 1) / hw0;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::A1, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, n * c * d * hw0 * hw1 * sizeof(half));
    pipe.InitBuffer(outQueueY, 1, n1 * n0 * c1 * c0 * d * h * w * sizeof(half));
    pipe.InitBuffer(tmplocalBuf, c * d * hw0 * hw1 + n1 * n0 * d * c1 * c0 * hw0 * hw1);

    Layout ncdhwLayout = MakeLayout(MakeShape(n, c, d, h, w), MakeStride());
    Layout fractalzLayout = MakeLayout(MakeShape(d, c1, h, w, n1, n0, c0), MakeStride());

    AscendC::LocalTensor<half> srcTensor = inQueueX.AllocTensor<half>();
    AscendC::LocalTensor<half> dstTensor = outQueueY.AllocTensor<half>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    static constexpr TransDataConfig config = {DataFormat::NCDHW, DataFormat::FRACTAL_Z_3D};
    TransDataParams<decltype(ncdhwLayout), decltype(fractalzLayout)> params = {ncdhwLayout, fractalzLayout};

    HighLevelApiCheck::CheckFuncTransData<config, half, decltype(ncdhwLayout), decltype(fractalzLayout)>("TransData", dstTensor, srcTensor, sharedTmpBuffer, params);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(TransDataAPICheck, TransDataAPICheckTransDataTmpPos)
{
    int32_t n = 1;
    int32_t c = 1;
    int32_t d = 1;
    int32_t h = 1;
    int32_t w = 1;
    int32_t n0 = 16;
    int32_t c0 = 16;
    int32_t c1 = (c + c0 - 1) / c0;
    int32_t n1 = (n + n0 - 1) / n0;
    int32_t hw0 = 16;
    int32_t hw1 = (h * w + hw0 - 1) / hw0;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::A1> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, n * c * d * hw0 * hw1 * sizeof(half));
    pipe.InitBuffer(outQueueY, 1, n1 * n0 * c1 * c0 * d * h * w * sizeof(half));
    pipe.InitBuffer(tmplocalBuf, c * d * hw0 * hw1 + n1 * n0 * d * c1 * c0 * hw0 * hw1);

    Layout ncdhwLayout = MakeLayout(MakeShape(n, c, d, h, w), MakeStride());
    Layout fractalzLayout = MakeLayout(MakeShape(d, c1, h, w, n1, n0, c0), MakeStride());

    AscendC::LocalTensor<half> srcTensor = inQueueX.AllocTensor<half>();
    AscendC::LocalTensor<half> dstTensor = outQueueY.AllocTensor<half>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    static constexpr TransDataConfig config = {DataFormat::NCDHW, DataFormat::FRACTAL_Z_3D};
    TransDataParams<decltype(ncdhwLayout), decltype(fractalzLayout)> params = {ncdhwLayout, fractalzLayout};

    HighLevelApiCheck::CheckFuncTransData<config, half, decltype(ncdhwLayout), decltype(fractalzLayout)>("TransData", dstTensor, srcTensor, sharedTmpBuffer, params);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 1);
}

TEST_F(TransDataAPICheck, TransDataAPICheckTransDataReuse)
{
    int32_t n = 1;
    int32_t c = 1;
    int32_t d = 1;
    int32_t h = 1;
    int32_t w = 1;
    int32_t n0 = 16;
    int32_t c0 = 16;
    int32_t c1 = (c + c0 - 1) / c0;
    int32_t n1 = (n + n0 - 1) / n0;
    int32_t hw0 = 16;
    int32_t hw1 = (h * w + hw0 - 1) / hw0;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;
    pipe.InitBuffer(inQueueX, 1, n * c * d * hw0 * hw1 * sizeof(half));
    pipe.InitBuffer(outQueueY, 1, n1 * n0 * c1 * c0 * d * h * w * sizeof(half));
    pipe.InitBuffer(tmplocalBuf, c * d * hw0 * hw1 + n1 * n0 * d * c1 * c0 * hw0 * hw1);

    Layout ncdhwLayout = MakeLayout(MakeShape(n, c, d, h, w), MakeStride());
    Layout fractalzLayout = MakeLayout(MakeShape(d, c1, h, w, n1, n0, c0), MakeStride());

    AscendC::LocalTensor<half> srcTensor = inQueueX.AllocTensor<half>();
    AscendC::LocalTensor<half> dstTensor = outQueueY.AllocTensor<half>();
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmplocalBuf.Get<uint8_t>();
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    static constexpr TransDataConfig config = {DataFormat::NCDHW, DataFormat::FRACTAL_Z_3D};
    TransDataParams<decltype(ncdhwLayout), decltype(fractalzLayout)> params = {ncdhwLayout, fractalzLayout};

    HighLevelApiCheck::CheckFuncTransData<config, half, decltype(ncdhwLayout), decltype(fractalzLayout)>("TransData", srcTensor, srcTensor, sharedTmpBuffer, params);
    inQueueX.FreeTensor(srcTensor);
    outQueueY.FreeTensor(dstTensor);
    tmplocalBuf.FreeTensor(sharedTmpBuffer);
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 3);
}

