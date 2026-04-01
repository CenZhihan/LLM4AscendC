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
// #include "model/model_factory.h"
#include "common.h"
#include "kernel_operator.h"

namespace AscendC {
template <typename T>
class GatherTest {
public:
    __aicore__ inline GatherTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
        __gm__ uint8_t* srcOffsetGm, const uint32_t count)
    {
        mElementCount = count;
        mDstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        mSrcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        mSrcOffsetGlobal.SetGlobalBuffer((__gm__ uint32_t*)srcOffsetGm);
        mPipe.InitBuffer(mQueIn, 2, mElementCount * sizeof(uint32_t));
        mPipe.InitBuffer(mQueOut, 2, mElementCount * sizeof(uint32_t));
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
        LocalTensor<T> srcLocal = mQueIn.AllocTensor<T>();
        DataCopy(srcLocal, mSrcGlobal, mElementCount);
        mQueIn.EnQue(srcLocal);

        LocalTensor<uint32_t> srcOffsetLocal = mQueIn.AllocTensor<uint32_t>();
        DataCopy(srcOffsetLocal, mSrcOffsetGlobal, mElementCount);
        mQueIn.EnQue(srcOffsetLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> srcLocal = mQueIn.DeQue<T>();
        LocalTensor<uint32_t> srcOffsetLocal = mQueIn.DeQue<uint32_t>();
        LocalTensor<T> dstLocal = mQueOut.AllocTensor<T>();

        srcLocal.SetSize(mElementCount);
        Gather(dstLocal, srcLocal, srcOffsetLocal, static_cast<uint32_t>(0), mElementCount);
        mQueIn.FreeTensor(srcLocal);
        mQueIn.FreeTensor(srcOffsetLocal);
        mQueOut.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = mQueOut.DeQue<T>();
        DataCopy(mDstGlobal, dstLocal, mElementCount);
        mQueOut.FreeTensor(dstLocal);
    }
private:
    TPipe mPipe;
    TQue<TPosition::VECIN, 1> mQueCalc;
    GlobalTensor<T> mValueGlobal;
    uint32_t mConcatRepeatTimes;
    uint32_t mSortRepeatTimes;
    uint32_t mExtractRepeatTimes;
    uint32_t mElementCount;
    GlobalTensor<uint32_t> mSrcOffsetGlobal;
    GlobalTensor<T> mSrcGlobal;
    GlobalTensor<T> mDstGlobal;
    TQue<TPosition::VECIN, 2> mQueIn;
    TQue<TPosition::VECOUT, 2> mQueOut;
}; // class GatherTest
} // namespace AscendC

template <typename T>
__global__ __aicore__ void testGather(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, __gm__ uint8_t* srcOffsetGm, uint32_t elementCount)
{
    AscendC::GatherTest<T> op;
    op.Init(dstGm, srcGm, srcOffsetGm, elementCount);
    op.Process();
}

struct GatherParams {
    uint32_t typeSize;
    uint32_t elementCount;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint32_t);
};

class GatherTestsuite : public testing::Test,
    public testing::WithParamInterface<GatherParams> {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_GATHER, GatherTestsuite,
    ::testing::Values(GatherParams { 2, 128, testGather<half>},
        GatherParams { 4, 128, testGather<float>},
        GatherParams { 2, 192, testGather<half>},
        GatherParams { 4, 192, testGather<float>},
        GatherParams { 2, 256, testGather<half>},
        GatherParams { 4, 256, testGather<half>}));


TEST_P(GatherTestsuite, testGather)
{
    auto param = GetParam();
    uint8_t srcGm[param.elementCount * param.typeSize] = {0};
    uint8_t srcOffsetGm[param.elementCount * sizeof(uint32_t)] = {0};
    uint8_t dstGm[param.elementCount * param.typeSize] = {0};
    param.cal_func(dstGm, srcGm, srcOffsetGm, param.elementCount);
    
    for (int32_t i = 0; i < param.elementCount; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}