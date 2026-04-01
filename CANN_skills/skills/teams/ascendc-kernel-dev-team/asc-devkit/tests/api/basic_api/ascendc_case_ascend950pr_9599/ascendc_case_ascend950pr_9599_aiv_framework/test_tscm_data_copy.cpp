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
#include "mockcpp/mockcpp.hpp"
#include "kernel_tpipe_impl.h"
#include "kernel_operator.h"

namespace AscendC {
// T, U for gm->l1 nd2nz
template <typename T> class KernelDataCopyGm2L12 {
public:
    __aicore__ inline KernelDataCopyGm2L12() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, Nd2NzParams& intriParamsIn)
    {
        intriParams = intriParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueTscm, 1, intriParams.ndNum * intriParams.nValue * intriParams.dValue * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        LocalTensor<T> srcLocal = inQueueTscm.AllocTensor<T>();

        DataCopy(srcLocal, srcGlobal, intriParams);

        uint32_t dstSize = ((intriParams.dValue * sizeof(T) -1 ) / 32 + 1) * intriParams.dstNzC0Stride * 32;

        inQueueTscm.FreeTensor(srcLocal);
    }

private:
    TPipe pipe;
    TSCM<TPosition::VECIN, 1> inQueueTscm;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    Nd2NzParams intriParams;
};

template <typename T>
__global__ __aicore__ void MainDataCopyGm2L1Nd2Nz(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
    Nd2NzParams intriParams)
{
    KernelDataCopyGm2L12<T> op;
    op.Init(dstGm, srcGm, intriParams);
    op.Process();
}

class TscmGm2L1Nd2NzTestSuite : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }
};

struct DataCopyGm2L1Nd2NzTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, Nd2NzParams);
    Nd2NzParams intriParams;
};

class TscmDataCopyGm2L1Nd2NzTestsuite : public testing::Test,
    public testing::WithParamInterface<DataCopyGm2L1Nd2NzTestParams> {
protected:
    void SetUp() {
        SetGCoreType(2);
    }
    void TearDown() {
        SetGCoreType(0);
        GlobalMockObject::verify();
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DATACOPYGM2L1ND2NZ, TscmDataCopyGm2L1Nd2NzTestsuite,
    ::testing::Values(DataCopyGm2L1Nd2NzTestParams { 2, MainDataCopyGm2L1Nd2Nz<half>, { 1, 32, 32, 0, 32, 32, 1, 0 } },
    DataCopyGm2L1Nd2NzTestParams { 4, MainDataCopyGm2L1Nd2Nz<float>, {  1, 32, 32, 0, 32, 32, 1, 0 } }));

TEST_P(TscmDataCopyGm2L1Nd2NzTestsuite, TscmDataCopyGm2L1Nd2NzTestsuiteOpTestCase)
{
    auto param = GetParam();
    MOCKER(ScmDataCopyND2NZMsg, void (*)(void*, void*, const uint8_t, const Nd2NzParams&, int32_t)).expects(atLeast(1));
    Nd2NzParams intriParams = param.intriParams;
    uint8_t srcGm[intriParams.ndNum * intriParams.nValue * intriParams.dValue * param.typeSize] = {0};
    uint8_t dstGm[intriParams.ndNum * intriParams.nValue * intriParams.dValue * param.typeSize] = {0};

    KernelDataCopyGm2L12<half> op;

    param.cal_func(dstGm, srcGm, intriParams);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

// T, U for gm->l1
template <typename T> class KernelDataCopyGm2L1 {
public:
    __aicore__ inline KernelDataCopyGm2L1() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, DataCopyParams& intriParamsIn)
    {
        intriParams = intriParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueTscm, 1, 512 * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        LocalTensor<T> srcLocal = inQueueTscm.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal, intriParams);
        PipeBarrier<PIPE_ALL>();
        inQueueTscm.FreeTensor(srcLocal);
    }

private:
    TPipe pipe;
    TSCM<TPosition::VECIN, 1> inQueueTscm;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    DataCopyParams intriParams;
};

template <typename T>
__global__ __aicore__ void MainDataCopyGm2L1(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
    DataCopyParams &intriParams)
{
    KernelDataCopyGm2L1<T> op;
    op.Init(dstGm, srcGm, intriParams);
    op.Process();
}

struct DataCopyGm2L1TestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, DataCopyParams &);
    DataCopyParams intriParams;
};

class DataCopyGm2L1Testsuite : public testing::Test, public testing::WithParamInterface<DataCopyGm2L1TestParams> {
public:
protected:
    void SetUp() {
        SetGCoreType(2);
    }
    void TearDown() {
        SetGCoreType(0);
        GlobalMockObject::verify();
    }

};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DATACOPYGm2L1, DataCopyGm2L1Testsuite,
    ::testing::Values(DataCopyGm2L1TestParams { 2, MainDataCopyGm2L1<half>, {1, 32, 0, 0} }, 
    DataCopyGm2L1TestParams { 4, MainDataCopyGm2L1<int32_t>, {1, 64, 0, 0} }));

TEST_P(DataCopyGm2L1Testsuite, DataCopyGm2L1OpTestCase)
{
    MOCKER(ScmDataCopyMsg, void (*)(void*, void*, const DataCopyParams&, int32_t)).expects(atLeast(1));
    auto param = GetParam();
    DataCopyParams intriParams = param.intriParams;
    uint8_t srcGm[512 * param.typeSize] = {0};
    uint8_t dstGm[512 * param.typeSize] = {0};
 
    param.cal_func(dstGm, srcGm, intriParams);
    for (int i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0);
    }
}
} // namespace AscendC