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

namespace AscendC {
template <typename T, typename T1>
class DuplicateTest {
public:
    __aicore__ inline DuplicateTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, const uint32_t count)
    {
        m_elementCount = count;
        m_srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        m_dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);

        m_pipe.InitBuffer(m_queIn, 1, m_elementCount * sizeof(T));
        m_pipe.InitBuffer(m_queOut, 1, m_elementCount * sizeof(T));

    }
    __aicore__ inline void Process(uint32_t computeMode)
    {
        CopyIn();
        Compute(computeMode);
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<T> srcLocal = m_queIn.AllocTensor<T>();
        DataCopy(srcLocal, m_srcGlobal, m_elementCount);
        m_queIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute(uint32_t computeMode)
    {
        LocalTensor<T> srcLocal = m_queIn.DeQue<T>();
        LocalTensor<T> dstLocal = m_queOut.AllocTensor<T>();
        __ubuf__ T* srcPtr = (__ubuf__ T*)srcLocal.GetPhyAddr();
        __ubuf__ T* dstPtr = (__ubuf__ T*)dstLocal.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::RegTensor<T> vSrcReg;
            Reg::RegTensor<T> vSrcReg1;
            T1 src = 0;
            Reg::RegTensor<T> vDstReg;
            Reg::RegTensor<T> vDstReg1;
            uint32_t sreg = (uint32_t)m_elementCount;
            Reg::MaskReg preg;
            preg = Reg::UpdateMask<T>(sreg);
            if (computeMode == 0) {
                Duplicate(vDstReg, src);
                Duplicate(vDstReg, src, preg);
                Duplicate(vDstReg, vSrcReg, preg);
            } else if (computeMode == 1) {
                Reg::Interleave(vDstReg, vDstReg1, vSrcReg, vSrcReg1);
            } else if (computeMode == 2) {
                Reg::DeInterleave(vDstReg, vDstReg1, vSrcReg, vSrcReg1);
            }
        }

        m_queIn.FreeTensor(srcLocal);
        m_queOut.EnQue(dstLocal);
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
    GlobalTensor<T> m_srcGlobal;
    GlobalTensor<T> m_dstGlobal;
    TQue<TPosition::VECIN, 1> m_queIn;
    TQue<TPosition::VECOUT, 1> m_queOut;
};
} // namespace AscendC

template <typename T, typename T1>
__global__ __aicore__ void testDuplicate(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, uint32_t elementCount, uint32_t computeMode)
{
    AscendC::DuplicateTest<T, T1> op;
    op.Init(dstGm, srcGm, elementCount);
    op.Process(computeMode);
}

struct duplicateParams {
    uint32_t dstTypeSize;
    uint32_t srcTypeSize;
    uint32_t elementCount;
    uint32_t computeMode;
    void (*cal_func)(uint8_t*, uint8_t*, uint32_t, uint32_t);
};

class DuplicateMicroTestsuite : public testing::Test,
    public testing::WithParamInterface<duplicateParams> {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DUPLICATE, DuplicateMicroTestsuite,
    ::testing::Values(duplicateParams { 2, 1, 128, 0, testDuplicate<half, half>},
        duplicateParams { 4, 1, 64, 0, testDuplicate<float, float>},
        duplicateParams { 4, 1, 64, 0, testDuplicate<uint32_t, uint32_t>},
        duplicateParams { 4, 1, 64, 0, testDuplicate<int32_t, uint32_t>},
        duplicateParams { 2, 1, 128, 0, testDuplicate<uint16_t, uint32_t>},
        duplicateParams { 2, 1, 64, 0, testDuplicate<int16_t, uint32_t>},
        duplicateParams { 1, 1, 64, 0, testDuplicate<bool, uint32_t>},
        duplicateParams { 1, 1, 64, 0, testDuplicate<int8_t, uint32_t>},
        duplicateParams { 1, 1, 128, 0, testDuplicate<uint8_t, uint32_t>},
        duplicateParams { 2, 1, 64, 1, testDuplicate<half, uint32_t>},
        duplicateParams { 4, 1, 64, 1, testDuplicate<uint32_t, uint32_t>},
        duplicateParams { 4, 1, 64, 1, testDuplicate<float, uint32_t>},
        duplicateParams { 2, 1, 128, 1, testDuplicate<uint16_t, uint32_t>},
        duplicateParams { 2, 1, 128, 1, testDuplicate<int16_t, uint32_t>},
        duplicateParams { 1, 1, 128, 1, testDuplicate<bool, uint32_t>},
        duplicateParams { 1, 1, 128, 1, testDuplicate<int8_t, uint32_t>},
        duplicateParams { 1, 1, 128, 1, testDuplicate<uint8_t, uint32_t>},
        duplicateParams { 4, 1, 64, 1, testDuplicate<int32_t, uint32_t>},
        duplicateParams { 2, 1, 64, 2, testDuplicate<half, uint32_t>},
        duplicateParams { 4, 1, 64, 2, testDuplicate<uint32_t, uint32_t>},
        duplicateParams { 4, 1, 64, 2, testDuplicate<float, uint32_t>},
        duplicateParams { 2, 1, 128, 2, testDuplicate<uint16_t, uint32_t>},
        duplicateParams { 2, 1, 128, 2, testDuplicate<int16_t, uint32_t>},
        duplicateParams { 1, 1, 128, 2, testDuplicate<bool, uint32_t>},
        duplicateParams { 1, 1, 128, 2, testDuplicate<int8_t, uint32_t>},
        duplicateParams { 1, 1, 128, 2, testDuplicate<uint8_t, uint32_t>},
        duplicateParams { 4, 1, 64, 2, testDuplicate<int32_t, uint32_t>}));


TEST_P(DuplicateMicroTestsuite, testDuplicate)
{
    auto param = GetParam();
    uint8_t dstGm[param.elementCount * param.dstTypeSize] = {0};
    uint8_t srcGm[param.elementCount * param.srcTypeSize] = {0};
    param.cal_func(dstGm, srcGm, param.elementCount, param.computeMode);
    
    for (int32_t i = 0; i < param.elementCount; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}