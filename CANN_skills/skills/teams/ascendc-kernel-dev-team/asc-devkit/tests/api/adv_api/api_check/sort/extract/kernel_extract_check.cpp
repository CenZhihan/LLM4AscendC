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
#include "impl/adv_api/detail/api_check/kernel_check/sort/extract/extract_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class ExtractAPICheck : public testing::Test {
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

TEST_F(ExtractAPICheck, ExtractAPICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECOUT, 1> m_queDstValue;
    m_pipe.InitBuffer(m_queDstValue, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> dstValue = m_queDstValue.AllocTensor<float>();
    dstValue.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECOUT, 1> m_queDstIndex;
    m_pipe.InitBuffer(m_queDstIndex, 1, 32 * sizeof(uint32_t));
    AscendC::LocalTensor<uint32_t> dstIndex = m_queDstIndex.AllocTensor<uint32_t>();
    dstIndex.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECIN> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 32 * sizeof(float));
    AscendC::LocalTensor<float> sorted = m_queSrc.Get<float>();
    sorted.SetSize(32);

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(Extract, (float), (
        dstValue, dstIndex, sorted, 8));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(ExtractAPICheck, ExtractAPICheckTestSubFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECOUT, 1> m_queDstValue;
    m_pipe.InitBuffer(m_queDstValue, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstValue = m_queDstValue.AllocTensor<uint8_t>();
    dstValue.SetSize(16);

    AscendC::TQue<AscendC::TPosition::VECOUT, 1> m_queDstIndex;
    m_pipe.InitBuffer(m_queDstIndex, 1, 32 * sizeof(uint32_t));
    AscendC::LocalTensor<uint32_t> dstIndex = m_queDstIndex.AllocTensor<uint32_t>();
    dstIndex.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECIN> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sorted =  m_queSrc.Get<uint8_t>();
    sorted.SetSize(16);

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(Extract, (uint8_t), (
        dstValue[1], dstIndex[1], sorted[1], 8));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 4);
}

TEST_F(ExtractAPICheck, ExtractAPICheckTestFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::A1, 1> m_queDstValue;
    m_pipe.InitBuffer(m_queDstValue, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstValue = m_queDstValue.AllocTensor<uint8_t>();
    dstValue.SetSize(2);

    AscendC::LocalTensor<uint32_t> dstIndex = dstValue.template ReinterpretCast<uint32_t>();
    dstIndex.SetSize(2);

    AscendC::TQue<AscendC::TPosition::B1, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sorted = m_queSrc.AllocTensor<uint8_t>();
    sorted.SetSize(0);

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(Extract, (uint8_t), (
        dstValue[1], dstIndex[1], sorted, 512));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 9);
}
