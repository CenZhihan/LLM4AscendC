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
#include "impl/adv_api/detail/api_check/kernel_check/normalization/welfordfinalize/welfordfinalize_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class WelfordFinalizeAPICheck : public testing::Test {
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

TEST_F(WelfordFinalizeAPICheck, WelfordFinalizeAPICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> outputMean = m_queDst.AllocTensor<float>();
    outputMean.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queOutVar;
    m_pipe.InitBuffer(m_queOutVar, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> outputVariance = m_queOutVar.AllocTensor<float>();
    outputVariance.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcMean;
    m_pipe.InitBuffer(m_queSrcMean, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inputMean = m_queSrcMean.AllocTensor<float>();
    inputMean.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcVar;
    m_pipe.InitBuffer(m_queSrcVar, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inputVariance = m_queSrcVar.AllocTensor<float>();
    inputVariance.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(32);

    AscendC::WelfordFinalizePara para = {1, 8, 1, 3, 1, 5, 0.125, 0.125};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(WelfordFinalize, (false), (outputMean, outputVariance,
        inputMean, inputVariance, sharedTmpBuffer, para));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(WelfordFinalizeAPICheck, WelfordFinalizeAPICheckTestFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> dstTensor = m_queDst.AllocTensor<float>();
    dstTensor.SetSize(16);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(0);

    AscendC::WelfordFinalizePara para = {35, 8, 1, 2, 1, 5, 0.1256, 0.1256};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(WelfordFinalize, (true), (dstTensor[1], dstTensor[1],
        dstTensor[1], dstTensor[1], sharedTmpBuffer, para));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 29);
}

TEST_F(WelfordFinalizeAPICheck, WelfordFinalizeAPICheckTestCountsSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> outputMean = m_queDst.AllocTensor<float>();
    outputMean.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queOutVar;
    m_pipe.InitBuffer(m_queOutVar, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> outputVariance = m_queOutVar.AllocTensor<float>();
    outputVariance.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcMean;
    m_pipe.InitBuffer(m_queSrcMean, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inputMean = m_queSrcMean.AllocTensor<float>();
    inputMean.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcVar;
    m_pipe.InitBuffer(m_queSrcVar, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inputVariance = m_queSrcVar.AllocTensor<float>();
    inputVariance.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(int32_t));
    AscendC::LocalTensor<int32_t> counts = m_queSrc.AllocTensor<int32_t>();
    counts.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(32);

    AscendC::WelfordFinalizePara para = {1, 8, 1, 3, 1, 5, 0.125, 0.125};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(WelfordFinalize, (false), (outputMean, outputVariance, inputMean,
        inputVariance, counts, sharedTmpBuffer, para));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(WelfordFinalizeAPICheck, WelfordFinalizeAPICheckTestCountsFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> dstTensor = m_queDst.AllocTensor<float>();
    dstTensor.SetSize(16);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(0);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(int32_t));
    AscendC::LocalTensor<int32_t> counts = m_queSrc.AllocTensor<int32_t>();
    counts.SetSize(32);

    AscendC::WelfordFinalizePara para = {35, 8, 1, 2, 1, 5, 0.1256, 0.1256};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(WelfordFinalize, (true), (dstTensor[1], dstTensor[1], dstTensor[1],
        dstTensor[1], counts, sharedTmpBuffer, para));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 40);
}
