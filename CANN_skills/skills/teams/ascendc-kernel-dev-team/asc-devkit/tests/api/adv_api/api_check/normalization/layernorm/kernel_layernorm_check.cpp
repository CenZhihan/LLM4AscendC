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
#include "impl/adv_api/detail/api_check/kernel_check/normalization/layernorm/layernorm_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class LayerNormAPICheck : public testing::Test {
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

TEST_F(LayerNormAPICheck, LayerNormAPICheckTestRstdSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> output = m_queDst.AllocTensor<float>();
    output.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queOutMean;
    m_pipe.InitBuffer(m_queOutMean, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> outputMean = m_queOutMean.AllocTensor<float>();
    outputMean.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queRstd;
    m_pipe.InitBuffer(m_queRstd, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> outputRstd = m_queRstd.AllocTensor<float>();
    outputRstd.SetSize(32);

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

    static constexpr AscendC::LayerNormConfig configLayerNorm = {false, false, false};
    AscendC::LayerNormPara para;
    para.aLength = 1;
    para.rLength = 8;
    para.rLengthWithPadding = 8;
    LayerNormSeparateTiling tiling;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(LayerNorm, (float, float, false, configLayerNorm), (output, outputMean, outputRstd,
        inputX, gamma, beta, 0.4, sharedTmpBuffer, para, tiling));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(LayerNormAPICheck, LayerNormAPICheckTestRstdFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queOut;
    m_pipe.InitBuffer(m_queOut, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> output = m_queOut.AllocTensor<float>();
    output.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstTensor = m_queDst.AllocTensor<uint8_t>();
    dstTensor.SetSize(16);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(0);

    static constexpr AscendC::LayerNormConfig configLayerNorm1 = {false, false, true};
    AscendC::LayerNormPara para;
    para.aLength = 6;
    para.rLength = 15;
    para.rLengthWithPadding = 17;
    LayerNormSeparateTiling tiling;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(LayerNorm, (uint8_t, uint8_t, true, configLayerNorm1), (dstTensor[1], output[1], output[1],
        dstTensor[1], dstTensor[1], dstTensor[1], 0.4, sharedTmpBuffer, para, tiling));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 49);
}

TEST_F(LayerNormAPICheck, LayerNormAPICheckTestVarSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> output = m_queDst.AllocTensor<float>();
    output.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queOutMean;
    m_pipe.InitBuffer(m_queOutMean, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> outputMean = m_queOutMean.AllocTensor<float>();
    outputMean.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queVar;
    m_pipe.InitBuffer(m_queVar, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> outputVariance = m_queVar.AllocTensor<float>();
    outputVariance.SetSize(32);

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

    LayerNormTiling tiling;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(LayerNorm, (float, false), (output, outputMean, outputVariance, inputX, gamma, beta,
        sharedTmpBuffer, 0.8, tiling));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(LayerNormAPICheck, LayerNormAPICheckTestVarFailure)
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

    LayerNormTiling tiling;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(LayerNorm, (uint8_t, true), (dstTensor[1], dstTensor[1], dstTensor[1],
        dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, 0.8, tiling));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 63);
}
