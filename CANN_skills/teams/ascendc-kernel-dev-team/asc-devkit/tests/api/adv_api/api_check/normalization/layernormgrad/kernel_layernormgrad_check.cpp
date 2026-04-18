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
#include "impl/adv_api/detail/api_check/kernel_check/normalization/layernormgrad/layernormgrad_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class LayerNormGradAPICheck : public testing::Test {
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

TEST_F(LayerNormGradAPICheck, LayerNormGradAPICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> outputPdX = m_queDst.AllocTensor<float>();
    outputPdX.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queForGamma;
    m_pipe.InitBuffer(m_queForGamma, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> resForGamma = m_queForGamma.AllocTensor<float>();
    resForGamma.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcDy;
    m_pipe.InitBuffer(m_queSrcDy, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inputDy = m_queSrcDy.AllocTensor<float>();
    inputDy.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inputX = m_queSrc.AllocTensor<float>();
    inputX.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcVariance;
    m_pipe.InitBuffer(m_queSrcVariance, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inputVariance = m_queSrcVariance.AllocTensor<float>();
    inputVariance.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcMean;
    m_pipe.InitBuffer(m_queSrcMean, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inputMean = m_queSrcMean.AllocTensor<float>();
    inputMean.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcGamma;
    m_pipe.InitBuffer(m_queSrcGamma, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inputGamma = m_queSrcGamma.AllocTensor<float>();
    inputGamma.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(32);

    LayerNormGradTiling tiling;
    AscendC::LayerNormGradShapeInfo shapeInfo;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(LayerNormGrad, (float, false), (outputPdX, resForGamma, inputDy, inputX,
        inputVariance, inputMean, inputGamma, sharedTmpBuffer, 0.3, tiling, shapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(LayerNormGradAPICheck, LayerNormGradAPICheckTestSubFailure)
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

    LayerNormGradTiling tiling;
    AscendC::LayerNormGradShapeInfo shapeInfo;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(LayerNormGrad, (uint8_t, false), (dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1],
        dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, 0.3, tiling, shapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 11);
}

TEST_F(LayerNormGradAPICheck, LayerNormGradAPICheckTestFailure)
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

    LayerNormGradTiling tiling;
    AscendC::LayerNormGradShapeInfo shapeInfo;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(LayerNormGrad, (uint8_t, true), (dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1],
        dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, 0.3, tiling, shapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 13);
}
