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
#include "impl/adv_api/detail/api_check/kernel_check/quantization/antiquant/antiquant_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "impl/adv_api/detail/api_check/kernel_api_check.h"

class AntiQuantAPICheck : public testing::Test {
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

TEST_F(AntiQuantAPICheck, AntiQuantAPICheckTestSuccess)
{
    AscendC::TPipe m_pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queDst;
    m_pipe.InitBuffer(m_queDst, 1, 64 * sizeof(half));
    AscendC::LocalTensor<half> dst = m_queDst.AllocTensor<half>();
    dst.SetSize(64);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queSrc;
    m_pipe.InitBuffer(m_queSrc, 1, 64 * sizeof(int8_t));
    AscendC::LocalTensor<int8_t> src = m_queSrc.AllocTensor<int8_t>();
    src.SetSize(64);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queOffset;
    m_pipe.InitBuffer(m_queOffset, 1, 64 * sizeof(half));
    AscendC::LocalTensor<half> offset = m_queOffset.AllocTensor<half>();
    offset.SetSize(64);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> m_queScale;
    m_pipe.InitBuffer(m_queScale, 1, 64 * sizeof(half));
    AscendC::LocalTensor<half> scale = m_queScale.AllocTensor<half>();
    scale.SetSize(64);

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    m_pipe.InitBuffer(tmpBuf, 32 * sizeof(uint8_t));
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpBuf.Get<uint8_t>();
    sharedTmpBuffer.SetSize(32);

    AscendC::AntiQuantShapeInfo shapeInfo = { 1, 16, 1, 16 };
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(AscendAntiQuant, (int8_t, half, false),
        (dst, src, offset, scale, sharedTmpBuffer, 8, shapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 0);
}

TEST_F(AntiQuantAPICheck, AntiQuantAPICheckTestSubFailure)
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

    AscendC::AntiQuantShapeInfo shapeInfo = { 1, 16, 1, 16 };
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(AscendAntiQuant, (uint8_t, uint8_t, false),
        (dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, 8, shapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 14);
}

TEST_F(AntiQuantAPICheck, AntiQuantAPICheckTestFailure)
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

    AscendC::AntiQuantShapeInfo shapeInfo = { 7, 16, 7, 16 };
    uint64_t startCounts = AscendC::KernelRaise::GetInstance().GetRaiseCount();
    AscendC::CHECK_FUNC_HIGHLEVEL_API(AscendAntiQuant, (uint8_t, uint8_t, false),
        (dstTensor[1], dstTensor[1], dstTensor[1], dstTensor[1], sharedTmpBuffer, 8, shapeInfo));
    EXPECT_EQ(AscendC::KernelRaise::GetInstance().GetRaiseCount() - startCounts, 16);
}
