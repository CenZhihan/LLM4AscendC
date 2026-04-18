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
#include "impl/adv_api/detail/api_check/kernel_check/sort/topk/topk_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class TopKAPICheck : public testing::Test {
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

TEST_F(TopKAPICheck, TopKAPICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> dstValueLocal = m_queDst.AllocTensor<float>();
    dstValueLocal.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDstIndex;
    m_pipe.InitBuffer(m_queDstIndex, 1, 32 * sizeof(int32_t));
    AscendC::LocalTensor<int32_t> dstIndexLocal = m_queDstIndex.AllocTensor<int32_t>();
    dstIndexLocal.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 32 * sizeof(float));
    AscendC::LocalTensor<float> srcLocal = m_queSrc.AllocTensor<float>();
    srcLocal.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcIndex;
    m_pipe.InitBuffer(m_queSrcIndex, 1, 32 * sizeof(int32_t));
    AscendC::LocalTensor<int32_t> srcIndexLocal = m_queSrcIndex.AllocTensor<int32_t>();
    srcIndexLocal.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queFinish;
    m_pipe.InitBuffer(m_queFinish, 1, 32 * sizeof(bool));
    AscendC::LocalTensor<bool> finishLocal = m_queFinish.AllocTensor<bool>();
    finishLocal.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(32);

    AscendC::TopKInfo topKInfo;
    topKInfo.outter = 1;
    topKInfo.inner = 32;
    topKInfo.n = 8;
    TopkTiling tilling;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(TopK, (float, true, true, false, AscendC::TopKMode::TOPK_NORMAL), (
        dstValueLocal, dstIndexLocal, srcLocal, srcIndexLocal, finishLocal, sharedTmpBuffer, 7, tilling, topKInfo, false));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(TopKAPICheck, TopKAPICheckTestSubFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstTensor = m_queDst.AllocTensor<uint8_t>();
    dstTensor.SetSize(16);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcIndex;
    m_pipe.InitBuffer(m_queSrcIndex, 1, 32 * sizeof(int32_t));
    AscendC::LocalTensor<int32_t> srcIndexLocal = m_queSrcIndex.AllocTensor<int32_t>();
    srcIndexLocal.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queFinish;
    m_pipe.InitBuffer(m_queFinish, 1, 32 * sizeof(bool));
    AscendC::LocalTensor<bool> finishLocal = m_queFinish.AllocTensor<bool>();
    finishLocal.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(16);

    AscendC::TopKInfo topKInfo;
    topKInfo.outter = 1;
    topKInfo.inner = 8;
    topKInfo.n = 8;
    TopkTiling tilling;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(TopK, (uint8_t, true, true, false, AscendC::TopKMode::TOPK_NORMAL), (
        dstTensor[1], srcIndexLocal[1], dstTensor[1], srcIndexLocal[1], finishLocal[1], sharedTmpBuffer,
        7, tilling, topKInfo, false));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 12);
}

TEST_F(TopKAPICheck, TopKAPICheckTestFailure)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> dstTensor = m_queDst.AllocTensor<uint8_t>();
    dstTensor.SetSize(16);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrcIndex;
    m_pipe.InitBuffer(m_queSrcIndex, 1, 32 * sizeof(int32_t));
    AscendC::LocalTensor<int32_t> srcIndexLocal = m_queSrcIndex.AllocTensor<int32_t>();
    srcIndexLocal.SetSize(32);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queFinish;
    m_pipe.InitBuffer(m_queFinish, 1, 32 * sizeof(bool));
    AscendC::LocalTensor<bool> finishLocal = m_queFinish.AllocTensor<bool>();
    finishLocal.SetSize(32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(0);

    AscendC::TopKInfo topKInfo;
    topKInfo.outter = 23;
    topKInfo.inner = 26;
    topKInfo.n = 4097;
    TopkTiling tilling;
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(TopK, (uint8_t, true, true, true, AscendC::TopKMode::TOPK_NORMAL), (
        dstTensor[1], srcIndexLocal[1], dstTensor[1], srcIndexLocal[1], finishLocal[1], sharedTmpBuffer, 7,
        tilling, topKInfo, false));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 17);
}
