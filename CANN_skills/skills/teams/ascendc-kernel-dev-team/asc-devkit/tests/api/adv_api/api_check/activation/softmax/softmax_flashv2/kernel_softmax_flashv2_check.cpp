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
#include "impl/adv_api/detail/api_check/kernel_check/activation/softmax/softmax_flashv2/softmax_flashv2_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class SoftMaxFlashV2APICheck : public testing::Test {
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

TEST_F(SoftMaxFlashV2APICheck, SoftMaxFlashV2APICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> dstTensor = m_queDst.AllocTensor<float>();
    dstTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSum;
    m_pipe.InitBuffer(m_queSum, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> sumTensor = m_queSum.AllocTensor<float>();
    sumTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queMax;
    m_pipe.InitBuffer(m_queMax, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> maxTensor = m_queMax.AllocTensor<float>();
    maxTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> srcTensor = m_queSrc.AllocTensor<float>();
    srcTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queExpMax;
    m_pipe.InitBuffer(m_queExpMax, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> expMaxTensor = m_queExpMax.AllocTensor<float>();
    expMaxTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queInSum;
    m_pipe.InitBuffer(m_queInSum, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inSumTensor = m_queInSum.AllocTensor<float>();
    inSumTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queInMax;
    m_pipe.InitBuffer(m_queInMax, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inMaxTensor = m_queInMax.AllocTensor<float>();
    inMaxTensor.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(float));
    AscendC::LocalTensor<float> sharedTmpBuffer = tmpBuf.Get<float>();
    sharedTmpBuffer.SetSize(32);

    static constexpr AscendC::SoftmaxConfig configFlashV2 = {true, 0, 0, AscendC::SoftmaxMode::SOFTMAX_NORMAL};
    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {1, 8, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlashV2, (float, float, false, false, false, false, configFlashV2), (dstTensor,
        sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor, inMaxTensor, sharedTmpBuffer, tiling, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(SoftMaxFlashV2APICheck, SoftMaxFlashV2APICheckTestSubFailure)
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

    static constexpr AscendC::SoftmaxConfig configFlashV2_1 = {true, 0, 0, AscendC::SoftmaxMode::SOFTMAX_NORMAL};
    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {1, 8, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlashV2, (uint8_t, uint8_t, false, false, false, false, configFlashV2_1), (dstTensor[1],
        dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, tiling, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 10);
}

TEST_F(SoftMaxFlashV2APICheck, SoftMaxFlashV2APICheckTestFailure)
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

    static constexpr AscendC::SoftmaxConfig configFlashV2_2 = {false, 1, 1, AscendC::SoftmaxMode::SOFTMAX_NORMAL};
    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {23, 9, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlashV2, (uint8_t, uint8_t, true, true, true, true, configFlashV2_2), (dstTensor[1],
        dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, tiling, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 13);
}
