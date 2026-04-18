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
class SyncTest {
public:
    __aicore__ inline SyncTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, __gm__ uint8_t* gmWorkspace, 
    const uint32_t count, const uint32_t datacopyTimes)
    {
        mElementCount = count;
        mDatacopyTimes = datacopyTimes;
        mDstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        mSrcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        mGmWorkspace.SetGlobalBuffer((__gm__ int32_t*)gmWorkspace);
        mPipe.InitBuffer(mQue, 1, mElementCount * sizeof(uint32_t));
        mPipe.InitBuffer(mQueTmp, 1, 32 * sizeof(int32_t));
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
        LocalTensor<T> srcLocal = mQue.AllocTensor<T>();
        DataCopy(srcLocal, mSrcGlobal, mElementCount);
        mQue.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        ;
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = mQue.DeQue<T>();
        LocalTensor<int32_t> ubWorkspace = mQueTmp.AllocTensor<int32_t>();
        InitDetermineComputeWorkspace(mGmWorkspace, ubWorkspace);
        pipe_barrier(PIPE_ALL);
        for (int i = 0; i < mDatacopyTimes; i++) {
            pipe_barrier(PIPE_ALL);
            WaitPreBlock(mGmWorkspace, ubWorkspace);
            pipe_barrier(PIPE_ALL);
            SetAtomicAdd<T>();
            pipe_barrier(PIPE_ALL);
            DataCopy(mDstGlobal, dstLocal, mElementCount);
            pipe_barrier(PIPE_ALL);
            SetAtomicNone();
            pipe_barrier(PIPE_ALL);
            NotifyNextBlock(mGmWorkspace, ubWorkspace);
        }
        mQue.FreeTensor(dstLocal);
        mQueTmp.FreeTensor(ubWorkspace);
    }

private:
    TPipe mPipe;
    uint32_t mElementCount;
    uint32_t mDatacopyTimes;
    GlobalTensor<T> mSrcGlobal;
    GlobalTensor<T> mDstGlobal;
    GlobalTensor<int32_t> mGmWorkspace;
    TQue<TPosition::VECIN, 1> mQue;
    TQue<TPosition::VECIN, 1> mQueTmp;
}; // class SyncTest
} // namespace AscendC

template <typename T>
__global__ __aicore__ void TestSync(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, __gm__ uint8_t *gmWorkspace,
    uint32_t elementCount, uint32_t datacopyTimes)
{
    AscendC::SyncTest<T> op;
    op.Init(dstGm, srcGm, gmWorkspace, elementCount, datacopyTimes);
    op.Process();
}

struct SyncTestParams {
    uint32_t typeSize;
    uint32_t elementCount;
    uint32_t datacopyTimes;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t);
};

class DeterComputeSyncTestsuite : public testing::Test,
    public testing::WithParamInterface<SyncTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_SYNC, DeterComputeSyncTestsuite,
    ::testing::Values(SyncTestParams { 2, 128, 20, TestSync<half>},
        SyncTestParams { 2, 2048, 10, TestSync<half>},
        SyncTestParams { 4, 128, 20, TestSync<float>},
        SyncTestParams { 4, 2048, 10, TestSync<float>}));


TEST_P(DeterComputeSyncTestsuite, TestSync)
{
    auto param = GetParam();
    uint8_t srcGm[param.elementCount * param.typeSize] = {0};
    uint8_t dstGm[param.elementCount * param.typeSize] = {0};
    uint8_t gmWorkspace[8 * sizeof(int32_t)] = {0};
    param.cal_func(dstGm, srcGm, gmWorkspace, param.elementCount, param.datacopyTimes);
    
    for (int32_t i = 0; i < param.elementCount; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}