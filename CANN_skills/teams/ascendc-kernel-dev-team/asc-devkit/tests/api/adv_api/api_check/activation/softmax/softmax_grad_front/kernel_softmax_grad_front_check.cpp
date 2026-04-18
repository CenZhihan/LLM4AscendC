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
#include "impl/adv_api/detail/api_check/kernel_check/activation/softmax/softmax_grad_front/softmax_grad_front_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class SoftMaxGradFrontAPICheck : public testing::Test {
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

TEST_F(SoftMaxGradFrontAPICheck, SoftMaxGradFrontAPICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> dstTensor = m_queDst.AllocTensor<float>();
    dstTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> srcTensor = m_queSrc.AllocTensor<float>();
    srcTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queGrad;
    m_pipe.InitBuffer(m_queGrad, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> gradTensor = m_queGrad.AllocTensor<float>();
    gradTensor.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(float));
    AscendC::LocalTensor<float> sharedTmpBuffer = tmpBuf.Get<float>();
    sharedTmpBuffer.SetSize(32);

    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {1, 8, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftmaxGradFront, (float, false, false), (
        dstTensor, gradTensor, srcTensor, sharedTmpBuffer, tiling, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(SoftMaxGradFrontAPICheck, SoftMaxGradFrontAPICheckTestSubFailure)
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

    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {1, 8, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftmaxGradFront, (uint8_t, false, false), (
        dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, tiling, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 5);
}

TEST_F(SoftMaxGradFrontAPICheck, SoftMaxGradFrontAPICheckTestFailure)
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

    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {11, 18, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftmaxGradFront, (uint8_t, true, true), (
        dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, tiling, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 6);
}
