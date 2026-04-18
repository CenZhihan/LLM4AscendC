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
#include "impl/adv_api/detail/api_check/kernel_check/sort/concat/concat_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class ConcatAPICheck : public testing::Test {
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

TEST_F(ConcatAPICheck, ConcatAPICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> dstValue = m_queDst.AllocTensor<float>();
    dstValue.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> src = m_queSrc.AllocTensor<float>();
    src.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(float));
    AscendC::LocalTensor<float> sharedTmpBuffer = tmpBuf.Get<float>();
    sharedTmpBuffer.SetSize(32);

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(Concat, (float), (
        dstValue, src, sharedTmpBuffer, 8));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(ConcatAPICheck, ConcatAPICheckTestSubFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstValue = m_queDst.AllocTensor<uint8_t>();
    dstValue.SetSize(16);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> src = m_queSrc.AllocTensor<uint8_t>();
    src.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(16);

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(Concat, (uint8_t), (
        dstValue[1], src[1], sharedTmpBuffer[1], 8));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 3);
}

TEST_F(ConcatAPICheck, ConcatAPICheckTestFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstValue = m_queDst.AllocTensor<uint8_t>();
    dstValue.SetSize(16);

    AscendC::TQue<AscendC::TPosition::A1, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> src = m_queSrc.AllocTensor<uint8_t>();
    src.SetSize(2);

    AscendC::TBuf<AscendC::TPosition::B1> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(0);

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(Concat, (uint8_t), (
        dstValue[1], src[1], sharedTmpBuffer, -1));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 7);
}
