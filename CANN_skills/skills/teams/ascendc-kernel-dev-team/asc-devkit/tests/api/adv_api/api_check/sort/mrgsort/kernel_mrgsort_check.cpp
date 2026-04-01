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
#include "impl/adv_api/detail/api_check/kernel_check/sort/mrgsort/mrgsort_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class MrgSortAPICheck : public testing::Test {
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

TEST_F(MrgSortAPICheck, MrgSortAPICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 128 * 8);
    AscendC::LocalTensor<float> dst = m_queDst.AllocTensor<float>();
    dst.SetSize(128 * 8);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 128 * 8);
    AscendC::LocalTensor<float> src = m_queSrc.AllocTensor<float>();
    src.SetSize(128 * 8);

    uint16_t singleProposalElementCount = 32;
    uint16_t baseOffset = singleProposalElementCount * 8 / sizeof(float);

    AscendC::MrgSortSrcList sortList =
        AscendC::MrgSortSrcList(src[0], src[baseOffset], src[baseOffset * 2], src[baseOffset * 3]);

    uint32_t sortedNum[4];

    const uint16_t elementCountList[4] = {singleProposalElementCount, singleProposalElementCount,
                                          singleProposalElementCount, singleProposalElementCount};

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();

    AscendC::CHECK_FUNC_HIGHLEVEL_API(MrgSort, (float, false), (dst, sortList, elementCountList, sortedNum, 0b1111, 1));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(MrgSortAPICheck, MrgSortAPICheckTestSubFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 128 * 8);
    AscendC::LocalTensor<uint32_t> dst = m_queDst.AllocTensor<uint32_t>();
    dst.SetSize(128 * 8);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 128 * 8);
    AscendC::LocalTensor<uint32_t> src = m_queSrc.AllocTensor<uint32_t>();
    src.SetSize(128 * 8);

    uint16_t singleProposalElementCount = 32;
    uint16_t baseOffset = singleProposalElementCount * 8 / sizeof(uint32_t);

    AscendC::MrgSortSrcList sortList =
        AscendC::MrgSortSrcList(src[0], src[baseOffset], src[baseOffset * 2], src[baseOffset * 3]);

    uint32_t sortedNum[4];

    const uint16_t elementCountList[4] = {singleProposalElementCount, singleProposalElementCount,
                                          singleProposalElementCount, singleProposalElementCount};

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();

    AscendC::CHECK_FUNC_HIGHLEVEL_API(MrgSort, (uint32_t, false),
                                      (dst, sortList, elementCountList, sortedNum, 0b1111, 0));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 2);
}

TEST_F(MrgSortAPICheck, MrgSortAPICheckTestFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::A1, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 4096 * 4 * 8);
    AscendC::LocalTensor<uint32_t> dst = m_queDst.AllocTensor<uint32_t>();
    dst.SetSize(4096 * 4 * 8);

    AscendC::TQue<AscendC::TPosition::B1, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 4096 * 4 * 8);
    AscendC::LocalTensor<uint32_t> src = m_queSrc.AllocTensor<uint32_t>();
    src.SetSize(4096 * 4 * 8);

    uint16_t singleProposalElementCount = 4096;
    uint16_t baseOffset = singleProposalElementCount * 8 / sizeof(uint32_t);

    AscendC::MrgSortSrcList sortList =
        AscendC::MrgSortSrcList(src[1], src[baseOffset + 1], src[baseOffset * 2 + 1], src[baseOffset * 3 + 1]);

    uint32_t sortedNum[4];

    const uint16_t elementCountList[4] = {singleProposalElementCount, singleProposalElementCount,
                                          singleProposalElementCount, singleProposalElementCount};

    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();

    AscendC::CHECK_FUNC_HIGHLEVEL_API(MrgSort, (uint32_t, false),
                                      (dst[1], sortList, elementCountList, sortedNum, 0b1111, 0));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 6);
}
