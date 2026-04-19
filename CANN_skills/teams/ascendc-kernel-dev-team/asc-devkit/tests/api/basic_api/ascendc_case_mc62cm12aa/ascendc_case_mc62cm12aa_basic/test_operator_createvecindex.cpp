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
#define private public
#define protect public
#include "kernel_operator.h"
#include "kernel_utils.h"

#include "kernel_operator.h"

namespace AscendC {
template <typename T>
class CreateVecIndexTest {
public:
    __aicore__ inline CreateVecIndexTest() {}
    __aicore__ inline void Init(GM_ADDR dstGm, uint64_t mask, uint8_t repeatTimes,
        uint16_t dstBlkStride, uint8_t dstRepStride)
    {
        m_mask = mask;
        m_repeatTimes = repeatTimes;
        m_dstBlkStride = dstBlkStride;
        m_dstRepStride = dstRepStride;
        m_elementCount = m_dstBlkStride * m_dstRepStride * 32 * m_repeatTimes / sizeof(T);
        m_dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        m_pipe.InitBuffer(m_queOut, 1, m_dstBlkStride * m_dstRepStride * 32 * m_repeatTimes);
        m_pipe.InitBuffer(m_queTmp, 1, 1024);
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        ;
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> dstLocal = m_queOut.AllocTensor<T>();
        LocalTensor<uint8_t> tmpLocal = m_queTmp.AllocTensor<uint8_t>();
        Duplicate(dstLocal, (T)0, m_elementCount);
        pipe_barrier(PIPE_ALL);
        uint64_t maskbit[2] = {0xFFFFFFFFFFFFFFFF, 0};
        CreateVecIndex(dstLocal, (T)0, maskbit, m_repeatTimes, m_dstBlkStride, m_dstRepStride);
        CreateVecIndex(dstLocal, (T)0, m_mask, m_repeatTimes, m_dstBlkStride, m_dstRepStride);
        CreateVecIndex(dstLocal, (T)0, m_repeatTimes * 256 / sizeof(T));
        m_queOut.EnQue(dstLocal);
        m_queTmp.FreeTensor(tmpLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = m_queOut.DeQue<T>();
        
        DataCopy(m_dstGlobal, dstLocal, m_elementCount);
        m_queOut.FreeTensor(dstLocal);
    }
private:
    TPipe m_pipe;
    uint32_t m_elementCount;
    uint32_t m_mask;
    uint32_t m_repeatTimes;
    uint32_t m_dstBlkStride;
    uint32_t m_dstRepStride;
    GlobalTensor<T> m_dstGlobal;
    TQue<TPosition::VECOUT, 1> m_queOut;
    TQue<TPosition::VECIN, 1> m_queTmp;
}; // class CreateVecIndexTest
} // namespace AscendC

template <typename T>
__global__ __aicore__ void testCreateVecIndex(GM_ADDR dstGm, uint64_t mask, uint8_t repeatTimes,
        uint16_t dstBlkStride, uint8_t dstRepStride)
{
    AscendC::CreateVecIndexTest<T> op;
    op.Init(dstGm, mask, repeatTimes, dstBlkStride, dstRepStride);
    op.Process();
}

struct createVecIndexParams {
    uint32_t typeSize;
    uint64_t mask;
    uint8_t repeatTimes;
    uint16_t dstBlkStride;
    uint8_t dstRepStride;
    void (*cal_func)(uint8_t*, uint64_t, uint8_t, uint16_t, uint8_t);
};

class CreateVecIndexTestsuite : public testing::Test,
    public testing::WithParamInterface<createVecIndexParams> {
protected:
    void SetUp() {
        AscendC::SetGCoreType(0);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_CREATEVECINDEX, CreateVecIndexTestsuite,
    ::testing::Values(createVecIndexParams { 2, 256, 2, 1, 8, testCreateVecIndex<half>},
                      createVecIndexParams { 4, 128, 2, 1, 8, testCreateVecIndex<float>},
                      createVecIndexParams { 2, 256, 2, 1, 8, testCreateVecIndex<int16_t>},
                      createVecIndexParams { 4, 128, 2, 1, 8, testCreateVecIndex<int32_t>}));

TEST_P(CreateVecIndexTestsuite, testCreateVecIndex)
{
    auto param = GetParam();
    uint8_t *dstGm = new uint8_t[param.repeatTimes * 256 * 10] {0};
    param.cal_func(dstGm, param.mask, param.repeatTimes, param.dstBlkStride, param.dstRepStride);
    
    for (int32_t i = 0; i < param.repeatTimes * 256 / param.typeSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
    delete []dstGm;
}