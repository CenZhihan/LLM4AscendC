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
#include "impl/adv_api/detail/api_check/kernel_check/activation/softmax/simple_softmax/simple_softmax_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class SimpleSoftMaxAPICheck : public testing::Test {
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

TEST_F(SimpleSoftMaxAPICheck, SimpleSoftMaxAPICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> dst = m_queDst.AllocTensor<float>();
    dst.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcinSum;
    m_pipe.InitBuffer(m_queSrcinSum, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inSumTensor = m_queSrcinSum.AllocTensor<float>();
    inSumTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcMax;
    m_pipe.InitBuffer(m_queSrcMax, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inMaxTensor = m_queSrcMax.AllocTensor<float>();
    inMaxTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> src = m_queSrc.AllocTensor<float>();
    src.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(float));
    AscendC::LocalTensor<float> sharedTmpBuffer = tmpBuf.Get<float>();
    sharedTmpBuffer.SetSize(32);

    static constexpr AscendC::SoftmaxConfig configSimple = {true, 0, 0};
    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {1, 8, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SimpleSoftMax, (float, float, false, false, false, configSimple),
        (dst, inSumTensor, inMaxTensor, src, sharedTmpBuffer, tiling, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(SimpleSoftMaxAPICheck, SimpleSoftMaxAPICheckTestSubFailure)
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

    static constexpr AscendC::SoftmaxConfig configSimple1 = {true, 0, 0};
    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {1, 8, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SimpleSoftMax, (uint8_t, uint8_t, false, false, false, configSimple1),
        (dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, tiling, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 7);
}

TEST_F(SimpleSoftMaxAPICheck, SimpleSoftMaxAPICheckTestFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstTensor = m_queDst.AllocTensor<uint8_t>();
    dstTensor.SetSize(16);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(float));
    AscendC::LocalTensor<float> sharedTmpBuffer = tmpBuf.Get<float>();
    sharedTmpBuffer.SetSize(0);

    static constexpr AscendC::SoftmaxConfig configSimple2 = {false, 1, 1};
    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {9, 9, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SimpleSoftMax, (uint8_t, uint8_t, true, true, true, configSimple2),
        (dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, tiling, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 12);
}
