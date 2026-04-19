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
#ifdef ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_check/activation/softmax/softmax/softmax_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class SoftMaxAPICheck : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        AscendC::KernelRaise::GetInstance().SetRaiseMode(false);
    }
    void TearDown() {
        AscendC::KernelRaise::GetInstance().SetRaiseMode(true);
    }
};

TEST_F(SoftMaxAPICheck, SoftMaxAPICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> dst = m_queDst.AllocTensor<float>();
    dst.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcsumTensor;
    m_pipe.InitBuffer(m_queSrcsumTensor, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> sumTensor = m_queSrcsumTensor.AllocTensor<float>();
    sumTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcmaxTensor;
    m_pipe.InitBuffer(m_queSrcmaxTensor, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> maxTensor = m_queSrcmaxTensor.AllocTensor<float>();
    maxTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> src = m_queSrc.AllocTensor<float>();
    src.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(float));
    AscendC::LocalTensor<float> sharedTmpBuffer = tmpBuf.Get<float>();
    sharedTmpBuffer.SetSize(32);

    static constexpr AscendC::SoftmaxConfig configSoftMax = {true, 0, 0};
    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {1, 8, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftMax, (float, float, false, false, false, configSoftMax),
        (dst, sumTensor, maxTensor, src, sharedTmpBuffer, tiling, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(SoftMaxAPICheck, SoftMaxAPICheckTestFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstTensor = m_queDst.AllocTensor<uint8_t>();
    dstTensor.SetSize(16);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(float));
    AscendC::LocalTensor<float> sharedTmpBuffer = tmpBuf.Get<float>();
    sharedTmpBuffer.SetSize(1);

    static constexpr AscendC::SoftmaxConfig configSoftMax1 = {false, 1, 1};
    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {12, 9, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftMax, (uint8_t, uint8_t, true, true, true, configSoftMax1),
        (dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, tiling, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 16);
}

TEST_F(SoftMaxAPICheck, SoftMaxAPICheckTestNoSumSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> dst = m_queDst.AllocTensor<float>();
    dst.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> src = m_queSrc.AllocTensor<float>();
    src.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(float));
    AscendC::LocalTensor<float> sharedTmpBuffer = tmpBuf.Get<float>();
    sharedTmpBuffer.SetSize(32);

    static constexpr AscendC::SoftmaxConfig configSoftMax2 = {true, 0, 0};
    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {1, 8, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftMax, (float, false, false, configSoftMax2), (
        dst, src, sharedTmpBuffer, tiling, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(SoftMaxAPICheck, SoftMaxAPICheckTestNoSumFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstTensor = m_queDst.AllocTensor<uint8_t>();
    dstTensor.SetSize(16);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(float));
    AscendC::LocalTensor<float> sharedTmpBuffer = tmpBuf.Get<float>();
    sharedTmpBuffer.SetSize(1);

    static constexpr AscendC::SoftmaxConfig configSoftMax3 = {false, 1, 1};
    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {12, 9, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftMax, (uint8_t, false, false, configSoftMax3), (dstTensor[1], dstTensor[1],
        sharedTmpBuffer, tiling, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 5);
}
