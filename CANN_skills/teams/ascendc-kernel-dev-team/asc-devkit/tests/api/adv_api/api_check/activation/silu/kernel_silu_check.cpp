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
#include "impl/adv_api/detail/api_check/kernel_check/activation/silu/silu_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class SiluAPICheck : public testing::Test {
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

TEST_F(SiluAPICheck, SiluAPICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> dstLocal = m_queDst.AllocTensor<float>();
    dstLocal.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> srcLocal = m_queSrc.AllocTensor<float>();
    srcLocal.SetSize(32);

    uint8_t dataSize = 8;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(Silu, (float, false), (dstLocal, srcLocal, dataSize));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(SiluAPICheck, SiluAPICheckTestSubFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstTensor = m_queDst.AllocTensor<uint8_t>();
    dstTensor.SetSize(16);

    uint8_t dataSize = 32;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(Silu, (uint8_t, false), (dstTensor[1], dstTensor[1], dataSize));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 7);
}

TEST_F(SiluAPICheck, SiluAPICheckTestFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstTensor = m_queDst.AllocTensor<uint8_t>();
    dstTensor.SetSize(16);

    uint8_t calCount = 32;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(Silu, (uint8_t, true), (dstTensor[1], dstTensor[1], calCount));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 7);
}
