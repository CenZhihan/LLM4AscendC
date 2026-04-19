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
#define protected public
#include "kernel_operator.h"
#include "kernel_utils.h"

namespace AscendC {
template <typename InputDataType, typename OutputDataType>
class AntiQuantWeightScalarTest {
public:
    __aicore__ inline AntiQuantWeightScalarTest() {}
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR srcGm,
        uint32_t elementCountOfInput, uint32_t K)
    {
        m_elementCountOfInput = elementCountOfInput;
        m_K = K;

        m_dstGlobal.SetGlobalBuffer((__gm__ OutputDataType*)dstGm);
        m_srcGlobal.SetGlobalBuffer((__gm__ InputDataType*)srcGm);

        m_pipe.InitBuffer(m_queInSrc, 1, elementCountOfInput * sizeof(InputDataType));
        m_pipe.InitBuffer(m_queOut, 1, elementCountOfInput * sizeof(OutputDataType));
        m_pipe.InitBuffer(m_queTmp, 1, 256);
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
        LocalTensor<InputDataType> srcLocal = m_queInSrc.AllocTensor<InputDataType>();
        DataCopy(srcLocal, m_srcGlobal, m_elementCountOfInput);
        m_queInSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<InputDataType> srcLocal = m_queInSrc.DeQue<InputDataType>();
        LocalTensor<OutputDataType> dstLocal = m_queOut.AllocTensor<OutputDataType>();
        LocalTensor<uint8_t> sharedTmpBuffer = m_queTmp.AllocTensor<uint8_t>();

        if constexpr (IsSameType<OutputDataType, half>::value) {
            half offset = 1.2;
            half scale = 3.4;
            AscendAntiQuant<InputDataType, OutputDataType, true>(dstLocal, srcLocal, offset, scale, sharedTmpBuffer, m_K);
        } else {
            float offset = 1.2;
            float scale = 3.4;

            AscendAntiQuant<InputDataType, OutputDataType, true>(dstLocal, srcLocal, ToBfloat16(offset), ToBfloat16(scale), sharedTmpBuffer, m_K);
        }

        m_queInSrc.FreeTensor(srcLocal);
        m_queTmp.FreeTensor(sharedTmpBuffer);
        m_queOut.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<OutputDataType> dstLocal = m_queOut.DeQue<OutputDataType>();
        DataCopy(m_dstGlobal, dstLocal, m_elementCountOfInput);
        m_queOut.FreeTensor(dstLocal);
    }
private:
    TPipe m_pipe;
    TQue<TPosition::VECIN, 1> m_queInSrc;
    TQue<TPosition::VECOUT, 1> m_queTmp;
    TQue<TPosition::VECOUT, 1> m_queOut;

    GlobalTensor<OutputDataType> m_dstGlobal;
    GlobalTensor<InputDataType> m_srcGlobal;

    uint32_t m_elementCountOfInput;
    uint32_t m_K;
}; // class AntiQuantTest
} // namespace AscendC

template <typename InputDataType, typename OutputDataType>
__global__ __aicore__ void testAntiQuantWeightScalar(GM_ADDR dst, GM_ADDR src, uint32_t elementCountOfInput, uint32_t K)
{
    AscendC::AntiQuantWeightScalarTest<InputDataType, OutputDataType> op;
    op.Init(dst, src, elementCountOfInput, K);
    op.Process();
}

struct antiquantWeightScalarParams {
    uint32_t elementCountOfInput;
    uint32_t K;
    void (*cal_func)(GM_ADDR, GM_ADDR, uint32_t, uint32_t);
};

class AntiquantWeightScalarTestsuite : public testing::Test,
    public testing::WithParamInterface<antiquantWeightScalarParams> {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_ANTIQUANTWEIGHTSCALAR, AntiquantWeightScalarTestsuite,
    ::testing::Values(antiquantWeightScalarParams { 2048, 64, testAntiQuantWeightScalar<int8_t, half>},
    antiquantWeightScalarParams { 2048, 64, testAntiQuantWeightScalar<int8_t, bfloat16_t>}));

TEST_P(AntiquantWeightScalarTestsuite, testAntiquantWeightScalar)
{
    auto param = GetParam();
    uint8_t srcGm[param.elementCountOfInput] = {0};
    uint8_t dstGm[param.elementCountOfInput * sizeof(half)] = {0};
    param.cal_func(dstGm, srcGm, param.elementCountOfInput, param.K);

    for (int32_t i = 0; i < param.elementCountOfInput; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}