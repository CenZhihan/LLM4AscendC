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
#include "impl/adv_api/detail/api_check/kernel_check/activation/softmax/softmax_flashv3/softmax_flashv3_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class SoftMaxFlashV3APICheck : public testing::Test {
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

TEST_F(SoftMaxFlashV3APICheck, SoftMaxFlashV3APICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 1024 * sizeof(half));
    AscendC::LocalTensor<half> dstTensor = m_queDst.AllocTensor<half>();
    dstTensor.SetSize(1024);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queMean;
    m_pipe.InitBuffer(m_queMean, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> meanTensor = m_queMean.AllocTensor<float>();
    meanTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queExpSum;
    m_pipe.InitBuffer(m_queExpSum, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> expSumTensor = m_queExpSum.AllocTensor<float>();
    expSumTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queMax;
    m_pipe.InitBuffer(m_queMax, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> maxTensor = m_queMax.AllocTensor<float>();
    maxTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 1024 * sizeof(half));
    AscendC::LocalTensor<half> srcTensor = m_queSrc.AllocTensor<half>();
    srcTensor.SetSize(1024);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queExpMax;
    m_pipe.InitBuffer(m_queExpMax, 1, 32 * sizeof(half));
    AscendC::LocalTensor<half> expMaxTensor = m_queExpMax.AllocTensor<half>();
    expMaxTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queInMean;
    m_pipe.InitBuffer(m_queInMean, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inMeanTensor = m_queInMean.AllocTensor<float>();
    inMeanTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queInexpSum;
    m_pipe.InitBuffer(m_queInexpSum, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inexpSumTensor = m_queInexpSum.AllocTensor<float>();
    inexpSumTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queInMax;
    m_pipe.InitBuffer(m_queInMax, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inMaxTensor = m_queInMax.AllocTensor<float>();
    inMaxTensor.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(float));
    AscendC::LocalTensor<float> sharedTmpBuffer = tmpBuf.Get<float>();
    sharedTmpBuffer.SetSize(32);

    static constexpr AscendC::SoftmaxConfig configFlashV3 = AscendC::SOFTMAX_DEFAULT_CFG;
    SoftMaxTiling tiling;
    AscendC::SoftMaxParams params = {1, 512, 1, 512, 1, 8, 0.975};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlashV3, (half, float, false, false, false, false, configFlashV3), (
        dstTensor, meanTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
        inMeanTensor, inexpSumTensor, inMaxTensor, sharedTmpBuffer, tiling, params));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(SoftMaxFlashV3APICheck, SoftMaxFlashV3APICheckTestSubFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstTensor = m_queDst.AllocTensor<uint8_t>();
    dstTensor.SetSize(16);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(float));
    AscendC::LocalTensor<float> sharedTmpBuffer = tmpBuf.Get<float>();
    sharedTmpBuffer.SetSize(16);

    static constexpr AscendC::SoftmaxConfig configFlashV3_1 = AscendC::SOFTMAX_DEFAULT_CFG;
    SoftMaxTiling tiling;
    AscendC::SoftMaxParams params = {1, 8, 1, 8, 1, 8, 0.975};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlashV3, (uint8_t, uint8_t, false, false, false, false, configFlashV3_1), (
        dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1],
        dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, tiling, params));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 13);
}

TEST_F(SoftMaxFlashV3APICheck, SoftMaxFlashV3APICheckTestFailure)
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

    static constexpr AscendC::SoftmaxConfig configFlashV3_2 = AscendC::SOFTMAX_DEFAULT_CFG;
    SoftMaxTiling tiling;
    AscendC::SoftMaxParams params = {13, 9, 1, 8, 0, 18, 0.975};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlashV3, (uint8_t, uint8_t, true, true, true, true, configFlashV3_2), (
        dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1],
        dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, tiling, params));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 18);
}
