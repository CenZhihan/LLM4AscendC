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
#include "impl/adv_api/detail/api_check/kernel_check/normalization/normalize/normalize_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class NormalizeAPICheck : public testing::Test {
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

TEST_F(NormalizeAPICheck, NormalizeAPICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> output = m_queDst.AllocTensor<float>();
    output.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queRstd;
    m_pipe.InitBuffer(m_queRstd, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> outputRstd = m_queRstd.AllocTensor<float>();
    outputRstd.SetSize(32);

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

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queGamma;
    m_pipe.InitBuffer(m_queGamma, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> gamma = m_queGamma.AllocTensor<float>();
    gamma.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queBeta;
    m_pipe.InitBuffer(m_queBeta, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> beta = m_queBeta.AllocTensor<float>();
    beta.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(32);

    static constexpr AscendC::NormalizeConfig configNormalize = {AscendC::ReducePattern::AR, -1, false};
    AscendC::NormalizePara para;
    para.aLength = 1;
    para.rLength = 8;
    para.rLengthWithPadding = 8;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(Normalize, (float, float, false, configNormalize), (output, outputRstd, inputMean,
        inputVariance, inputX, gamma, beta, sharedTmpBuffer, 0.7, para));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(NormalizeAPICheck, NormalizeAPICheckTestSubFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstTensor = m_queDst.AllocTensor<uint8_t>();
    dstTensor.SetSize(16);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDstFloat;
    m_pipe.InitBuffer(m_queDstFloat, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> output = m_queDstFloat.AllocTensor<float>();
    output.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(16);

    static constexpr AscendC::NormalizeConfig configNormal1 = {AscendC::ReducePattern::RA, 1, false};
    AscendC::NormalizePara para;
    para.aLength = 9;
    para.rLength = 8;
    para.rLengthWithPadding = 17;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(Normalize, (uint8_t, uint8_t, true, configNormal1), (dstTensor[1], output[1],
        output[1], output[1], dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, 0.7, para));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 23);
}

TEST_F(NormalizeAPICheck, NormalizeAPICheckTestFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstTensor = m_queDst.AllocTensor<uint8_t>();
    dstTensor.SetSize(16);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDstFloat;
    m_pipe.InitBuffer(m_queDstFloat, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> output = m_queDstFloat.AllocTensor<float>();
    output.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(0);

    static constexpr AscendC::NormalizeConfig configNormal2 = {AscendC::ReducePattern::RA, -2, true};
    AscendC::NormalizePara para;
    para.aLength = 9;
    para.rLength = 8;
    para.rLengthWithPadding = 17;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(Normalize, (uint8_t, uint8_t, true, configNormal2), (dstTensor[1], output[1],
        output[1], output[1], dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, 0.7, para));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 25);
}
