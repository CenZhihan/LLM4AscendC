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
#include "impl/adv_api/detail/api_check/kernel_check/activation/softmax/softmax_flash/softmax_flash_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class SoftMaxFlashAPICheck : public testing::Test {
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

TEST_F(SoftMaxFlashAPICheck, SoftMaxFlashAPICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> dstTensor = m_queDst.AllocTensor<float>();
    dstTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcSumTensor;
    m_pipe.InitBuffer(m_queSrcSumTensor, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> sumTensor = m_queSrcSumTensor.AllocTensor<float>();
    sumTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcMax;
    m_pipe.InitBuffer(m_queSrcMax, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> maxTensor = m_queSrcMax.AllocTensor<float>();
    maxTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> srcTensor = m_queSrc.AllocTensor<float>();
    srcTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queExpMax;
    m_pipe.InitBuffer(m_queExpMax, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> expMaxTensor = m_queExpMax.AllocTensor<float>();
    expMaxTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcInSum;
    m_pipe.InitBuffer(m_queSrcInSum, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inSumTensor = m_queSrcInSum.AllocTensor<float>();
    inSumTensor.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcInMax;
    m_pipe.InitBuffer(m_queSrcInMax, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> inMaxTensor = m_queSrcInMax.AllocTensor<float>();
    inMaxTensor.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(32);

    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {1, 8, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlash, (float, false, false), (dstTensor, sumTensor, maxTensor, srcTensor,
        expMaxTensor, inSumTensor, inMaxTensor, sharedTmpBuffer, tiling, false, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(SoftMaxFlashAPICheck, SoftMaxFlashAPICheckTestSubFailure)
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

    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {1, 8, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlash, (uint8_t, false, false), (dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1],
        dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, tiling, false, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 57);
}

TEST_F(SoftMaxFlashAPICheck, SoftMaxFlashAPICheckTestFailure)
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

    SoftMaxTiling tiling;
    AscendC::SoftMaxShapeInfo softmaxShapeInfo = {13, 13, 1, 8};
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlash, (uint8_t, true, true), (dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1],
        dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, tiling, false, softmaxShapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 94);
}
