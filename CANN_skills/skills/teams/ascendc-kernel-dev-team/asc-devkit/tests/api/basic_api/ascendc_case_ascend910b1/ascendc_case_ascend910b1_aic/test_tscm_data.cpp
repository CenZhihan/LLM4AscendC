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

using namespace std;
using namespace AscendC;

namespace AscendC {
// T, U for gm->l1
template <typename T> class KernelDataCopyGm2L1 {
public:
    __aicore__ inline KernelDataCopyGm2L1() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, DataCopyParams& intriParamsIn)
    {
        intriParams = intriParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrcTscm, 1, 512 * sizeof(T));
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
        LocalTensor<T> srcLocal = inQueueSrcTscm.AllocTensor<T>();

        DataCopy(srcLocal, srcGlobal, intriParams);
        inQueueSrcTscm.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        ;
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = inQueueSrcTscm.DeQue<T>();
        DataCopy(dstGlobal, dstLocal, intriParams);
        inQueueSrcTscm.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TSCM<TPosition::GM> inQueueSrcTscm;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    DataCopyParams intriParams;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void MainDataCopyGm2L1(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
    DataCopyParams &intriParams)
{
    AscendC::KernelDataCopyGm2L1<T> op;
    op.Init(dstGm, srcGm, intriParams);
    op.Process();
}

struct DataCopyGm2L1TestParams {
    int32_t typeSize;
    void (*CalFunc)(uint8_t*, uint8_t*, DataCopyParams &);
    DataCopyParams intriParams;
};

class DataCopyGm2L1Testsuite : public testing::Test, public testing::WithParamInterface<DataCopyGm2L1TestParams> {
protected:
    void SetUp() {
        g_coreType = AIC_TYPE;
    }
    void TearDown() {
        AscendC::CheckSyncState();
        g_coreType = MIX_TYPE;
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DATACOPYGm2L1, DataCopyGm2L1Testsuite,
    ::testing::Values(DataCopyGm2L1TestParams { 2, MainDataCopyGm2L1<half>, {1, 32, 0, 0} }, 
    DataCopyGm2L1TestParams { 4, MainDataCopyGm2L1<int32_t>, {1, 64, 0, 0} }));

TEST_P(DataCopyGm2L1Testsuite, DataCopyGm2L1OpTestCase)
{
    auto param = GetParam();
    DataCopyParams intriParams = param.intriParams;
    uint8_t srcGm[512 * param.typeSize] = {0};
    uint8_t dstGm[512 * param.typeSize] = {0};
 
    param.CalFunc(dstGm, srcGm, intriParams);
    for (int i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0);
    }
}