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
#include "impl/adv_api/detail/api_check/kernel_check/normalization/welforupdate/welfordupdate_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class WelfordUpdateAPICheck : public testing::Test {
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

TEST_F(WelfordUpdateAPICheck, WelfordUpdateAPICheckTestSuccess)
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
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inputX = m_queSrc.AllocTensor<float>();
    inputX.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(32);

    AscendC::WelfordUpdateParam para;
    para.rnLength = 1;
    para.abLength = 8;
    para.abComputeLength = 8;
    para.nRec = 1;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(WelfordUpdate, (float, float, false, AscendC::WFUPDATE_DEFAULT_CFG), (
        outputMean, outputVariance, inputMean, inputVariance, inputX, sharedTmpBuffer, para));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(WelfordUpdateAPICheck, WelfordUpdateAPICheckTestSubFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstTensor = m_queDst.AllocTensor<uint8_t>();
    dstTensor.SetSize(16);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(16);

    AscendC::WelfordUpdateParam para;
    para.rnLength = 1;
    para.abLength = 8;
    para.abComputeLength = 8;
    para.nRec = 1;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(WelfordUpdate, (uint8_t, uint8_t, false, AscendC::WFUPDATE_DEFAULT_CFG), (
        dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, para));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 26);
}

TEST_F(WelfordUpdateAPICheck, WelfordUpdateAPICheckTestFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstTensor = m_queDst.AllocTensor<uint8_t>();
    dstTensor.SetSize(16);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(0);

    AscendC::WelfordUpdateParam para;
    para.rnLength = 9;
    para.abLength = 8;
    para.abComputeLength = 8;
    para.nRec = 1;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(WelfordUpdate, (uint8_t, uint8_t, true, AscendC::WFUPDATE_DEFAULT_CFG), (
        dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, para));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 52);
}
