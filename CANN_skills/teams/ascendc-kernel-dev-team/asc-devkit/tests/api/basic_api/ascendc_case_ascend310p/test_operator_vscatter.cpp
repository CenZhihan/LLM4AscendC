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

#include "kernel_log.h"
static uint8_t g_testRes = 1;      // 全局变量记录运行结果, 如果进入ASCENDC_ASSERT报错，会被置为0
// 重定义ASCENDC_ASSERT，不Abort，仅修改全局变量通知进入报错分支
#undef ASCENDC_ASSERT
#define ASCENDC_ASSERT(cond, behavior) \
    do {                               \
        if (!(cond)) {                 \
            g_testRes = 0;              \
            behavior;                  \
        }                              \
    } while (0)

#undef ASCENDC_REPORT_CHECK_ERROR
#define ASCENDC_REPORT_CHECK_ERROR(apiMsg, funcType)   \
    do {                                                \
        g_testRes = 0;                                   \
    } while (0)

#include "kernel_utils.h"
#include "kernel_operator.h"
#include "kernel_utils.h"
// #include "model/model_factory.h"
#include "common.h"

#include "kernel_operator.h"

namespace AscendC {
template <typename T>
class ScatterTest {
public:
    __aicore__ inline ScatterTest() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
        __gm__ uint8_t* dstOffsetGm, const uint32_t count)
    {
        mElementCount = count;
        mDstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        mSrcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        mDstOffsetGlobal.SetGlobalBuffer((__gm__ uint32_t*)dstOffsetGm);
        mPipe.InitBuffer(mQueIn, 2, mElementCount * sizeof(uint32_t));
        mPipe.InitBuffer(mQueOut, 1, mElementCount * sizeof(uint32_t));
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

        LocalTensor<uint32_t> dstOffsetLocal = mQueIn.AllocTensor<uint32_t>();
        DataCopy(dstOffsetLocal, mDstOffsetGlobal, mElementCount);
        mQueIn.EnQue(dstOffsetLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<T> srcLocal = mQueIn.DeQue<T>();
        LocalTensor<uint32_t> dstOffsetLocal = mQueIn.DeQue<uint32_t>();
        LocalTensor<T> dstLocal = mQueOut.AllocTensor<T>();

        dstLocal.SetSize(mElementCount);
        if (std::is_same<T, float>::value && mElementCount == 128) {
            Scatter(dstLocal, srcLocal, dstOffsetLocal, static_cast<uint32_t>(200), mElementCount);
        }
        if (std::is_same<T, float>::value && mElementCount == 192) {
            Scatter(dstLocal, srcLocal, dstOffsetLocal, static_cast<uint32_t>(0), mElementCount * 2);
        }
        Scatter(dstLocal, srcLocal, dstOffsetLocal, static_cast<uint32_t>(0), mElementCount);
        mQueIn.FreeTensor(srcLocal);
        mQueIn.FreeTensor(dstOffsetLocal);
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
    GlobalTensor<uint32_t> mDstOffsetGlobal;
    GlobalTensor<T> mSrcGlobal;
    GlobalTensor<T> mDstGlobal;
    TQue<TPosition::VECIN, 2> mQueIn;
    TQue<TPosition::VECOUT, 1> mQueOut;
}; // class ScatterTest
} // namespace AscendC

template <typename T>
__global__ __aicore__ void testScatter(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, __gm__ uint8_t* dstOffsetGm, uint32_t elementCount)
{
    AscendC::ScatterTest<T> op;
    op.Init(dstGm, srcGm, dstOffsetGm, elementCount);
    op.Process();
}

struct ScatterParams {
    uint32_t typeSize;
    uint32_t elementCount;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint32_t);
};

class ScatterTestsuite : public testing::Test,
    public testing::WithParamInterface<ScatterParams> {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_SCATTER, ScatterTestsuite,
    ::testing::Values(ScatterParams { 2, 128, testScatter<half>},
        ScatterParams { 4, 128, testScatter<float>},
        ScatterParams { 2, 192, testScatter<half>},
        ScatterParams { 4, 192, testScatter<float>},
        ScatterParams { 2, 256, testScatter<half>},
        ScatterParams { 4, 256, testScatter<half>}));


TEST_P(ScatterTestsuite, testScatter)
{
    auto param = GetParam();
    uint8_t srcGm[param.elementCount * param.typeSize] = {0};
    uint8_t dstOffsetGm[param.elementCount * sizeof(uint32_t)] = {0};
    uint8_t dstGm[param.elementCount * param.typeSize] = {0};
    param.cal_func(dstGm, srcGm, dstOffsetGm, param.elementCount);

    for (int32_t i = 0; i < param.elementCount; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}