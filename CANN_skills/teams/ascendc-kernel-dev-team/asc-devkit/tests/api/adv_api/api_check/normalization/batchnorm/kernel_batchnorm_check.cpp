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
#include "impl/adv_api/detail/api_check/kernel_check/normalization/batchnorm/batchnorm_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class BatchNormAPICheck : public testing::Test {
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

TEST_F(BatchNormAPICheck, BatchNormAPICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> output = m_queDst.AllocTensor<float>();
    output.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDstMean;
    m_pipe.InitBuffer(m_queDstMean, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> outputMean = m_queDstMean.AllocTensor<float>();
    outputMean.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDstVar;
    m_pipe.InitBuffer(m_queDstVar, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> outputVariance = m_queDstVar.AllocTensor<float>();
    outputVariance.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inputX = m_queSrc.AllocTensor<float>();
    inputX.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queGamm;
    m_pipe.InitBuffer(m_queGamm, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> gamm = m_queGamm.AllocTensor<float>();
    gamm.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queBeta;
    m_pipe.InitBuffer(m_queBeta, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> beta = m_queBeta.AllocTensor<float>();
    beta.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(32);

    BatchNormTiling tiling;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(BatchNorm, (float, false, false),
        (output, outputMean, outputVariance, inputX, gamm, beta, sharedTmpBuffer, 1.1, tiling));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(BatchNormAPICheck, BatchNormAPICheckTestSubFailure)
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

    BatchNormTiling tiling;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(BatchNorm, (uint8_t, true, false),
        (dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, 1.1, tiling));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 15);
}

TEST_F(BatchNormAPICheck, BatchNormAPICheckTestFailure)
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

    BatchNormTiling tiling;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(BatchNorm, (uint8_t, true, true),
        (dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, 1.1, tiling));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 17);
}
