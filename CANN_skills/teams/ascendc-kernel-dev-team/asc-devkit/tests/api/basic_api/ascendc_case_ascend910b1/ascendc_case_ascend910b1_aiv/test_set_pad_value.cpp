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
#include "kernel_utils.h"

#include "kernel_operator.h"

namespace AscendC {
template <typename T>
class SetPadValueTest {
public:
    __aicore__ inline SetPadValueTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, uint32_t n1, uint32_t n2)
    {
        mN1 = n1;
        mN2 = n2;
        mN2Align = n2 % 32 == 0 ? n2 : (n2 / 32 + 1) * 32;
        mSrcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        mDstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);

        mPipe.InitBuffer(mQueInSrc, 1, mN1 * mN2Align * sizeof(T));
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
        LocalTensor<T> srcLocal = mQueInSrc.AllocTensor<T>();
        DataCopyExtParams dataCopyExtParams;
        DataCopyPadExtParams<T> padParams;

        dataCopyExtParams.blockCount = mN1;
        dataCopyExtParams.blockLen = mN2 * sizeof(T);
        dataCopyExtParams.srcStride = 0;
        dataCopyExtParams.dstStride = 0;

        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 1;

        SetPadValue(static_cast<T>(37));
        pipe_barrier(PIPE_ALL);
        DataCopyPad(srcLocal, mSrcGlobal, dataCopyExtParams, padParams);
        mQueInSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        ;
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = mQueInSrc.DeQue<T>();
        DataCopy(mDstGlobal, dstLocal, mN1 * mN2Align);
        mQueInSrc.FreeTensor(dstLocal);
    }
private:
    TPipe mPipe;
    uint32_t mN1;
    uint32_t mN2;
    uint32_t mN2Align;
    GlobalTensor<T> mSrcGlobal;
    GlobalTensor<T> mDstGlobal;
    TQue<TPosition::VECIN, 1> mQueInSrc;
}; // class SetPadValueTest
} // namespace AscendC

template <typename T>
__global__ __aicore__ void testSetPadValue(GM_ADDR dstGm, GM_ADDR srcGm, uint32_t n1, uint32_t n2)
{
    AscendC::SetPadValueTest<T> op;
    op.Init(dstGm, srcGm, n1, n2);
    op.Process();
}

struct setPadValueParams {
    uint32_t n1;
    uint32_t n2;
    uint32_t typeSize;
    void (*CalFunc)(uint8_t*, uint8_t*, uint32_t, uint32_t);
};

class SetPadValueTestsuite : public testing::Test,
    public testing::WithParamInterface<setPadValueParams> {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_SETPADVALUE, SetPadValueTestsuite,
    ::testing::Values(setPadValueParams { 32, 31, 2, testSetPadValue<half>}));

TEST_P(SetPadValueTestsuite, testSetPadValue)
{
    auto param = GetParam();
    uint32_t n1 = param.n1;
    uint32_t n2 = param.n2;
    uint32_t n2Align = n2 % 32 == 0 ? n2 : (n2 / 32 + 1) * 32;
    uint8_t srcGm[n1 * n2 * param.typeSize] = {0};
    uint8_t dstGm[n1 * n2Align * param.typeSize] = {0};
    param.CalFunc(dstGm, srcGm, n1, n2);
    
    for (int32_t i = 0; i < n1 * n2Align; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}