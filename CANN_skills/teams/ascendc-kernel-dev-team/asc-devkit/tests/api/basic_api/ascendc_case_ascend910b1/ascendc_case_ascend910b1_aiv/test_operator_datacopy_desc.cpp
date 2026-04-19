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
#include <cmath>
#include <iostream>
#include <fstream>
#include "securec.h"

using namespace std;
using namespace AscendC;

template <typename T>
struct SelfTensorDesc : public TensorTrait<T> {
    static constexpr int32_t tensorLen = 256;
};

namespace AscendC {
// T, U for ub->out nz2nd
template <typename T> class KernelDataCopyUb2GmNz2Nd {
public:
    __aicore__ inline KernelDataCopyUb2GmNz2Nd() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, Nz2NdParamsFull& intriParamsIn)
    {
        intriParams = intriParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrcVecIn, 1, intriParams.nValue * intriParams.dValue * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyND2NZ(const LocalTensor<SelfTensorDesc<half>>& dst, const GlobalTensor<SelfTensorDesc<half>>& src, const uint16_t height,
        const uint16_t width)
    {
        for (int i = 0; i < width / 16; ++i) {
            int srcOffset = i * 16;
            int dstOffset = i * 16 * height;
            DataCopy(dst[dstOffset], src[srcOffset], { height, 1, uint16_t(width / 16 - 1), 0 });
        }
    }

    __aicore__ inline void CopyIn()
    {
        LocalTensor<SelfTensorDesc<T>> srcLocal = inQueueSrcVecIn.AllocTensor<SelfTensorDesc<T>>();

        CopyND2NZ(srcLocal, srcGlobal, intriParams.nValue, intriParams.dValue);

        inQueueSrcVecIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        ;
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<SelfTensorDesc<T>> dstLocal = inQueueSrcVecIn.DeQue<SelfTensorDesc<T>>();

        DataCopy(dstGlobal, dstLocal, intriParams);

        inQueueSrcVecIn.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrcVecIn;
    GlobalTensor<SelfTensorDesc<T>> srcGlobal;
    GlobalTensor<SelfTensorDesc<T>> dstGlobal;
    Nz2NdParamsFull intriParams;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void MainDataCopyUb2GmNz2Nd(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
    Nz2NdParamsFull& intriParams)
{
    AscendC::KernelDataCopyUb2GmNz2Nd<T> op;
    op.Init(dstGm, srcGm, intriParams);
    op.Process();
}

struct DataCopyUb2GmNz2NdTestParams {
    int32_t typeSize;
    void (*CalFunc)(uint8_t*, uint8_t*, Nz2NdParamsFull&);
    Nz2NdParamsFull intriParams;
};

class DataCopyDescUb2GmNz2NdTestsuite : public testing::Test,
    public testing::WithParamInterface<DataCopyUb2GmNz2NdTestParams> {
protected:
    void SetUp() {
        g_coreType = AIV_TYPE;
    }
    void TearDown() {
        AscendC::CheckSyncState();
        g_coreType = MIX_TYPE;
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DATACOPYUb2GmNz2Nd, DataCopyDescUb2GmNz2NdTestsuite,
    ::testing::Values(DataCopyUb2GmNz2NdTestParams { 2, MainDataCopyUb2GmNz2Nd<half>, { 1, 32, 32, 1, 32, 32, 1 } },
    DataCopyUb2GmNz2NdTestParams { 2, MainDataCopyUb2GmNz2Nd<half>, { 1, 64, 32, 1, 64, 32, 1 } }));

TEST_P(DataCopyDescUb2GmNz2NdTestsuite, DataCopyUb2GmNz2NdTestsuiteOpTestCase)
{
    auto param = GetParam();
    Nz2NdParamsFull intriParams = param.intriParams;
    uint8_t srcGm[intriParams.nValue * intriParams.dValue * param.typeSize] = {0};
    uint8_t dstGm[intriParams.nValue * intriParams.dValue * param.typeSize] = {0};

    param.CalFunc(dstGm, srcGm, intriParams);
    for (int32_t i = 0; i < (intriParams.nValue * intriParams.dValue); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

class DataCopyDescUb2GmNz2NdTestsuiteCheckGmOverflow : public testing::Test,
    public testing::WithParamInterface<DataCopyUb2GmNz2NdTestParams> {
protected:
    void SetUp() {
        g_coreType = AIV_TYPE;
    }
    void TearDown() {
        AscendC::CheckSyncState();
        g_coreType = MIX_TYPE;
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DATACOPYUb2GmNz2Nd_OOM, DataCopyDescUb2GmNz2NdTestsuiteCheckGmOverflow,
    ::testing::Values(DataCopyUb2GmNz2NdTestParams { 2, MainDataCopyUb2GmNz2Nd<half>, { 1, 32, 32, 1, 32, 32, 1 } },
    DataCopyUb2GmNz2NdTestParams { 2, MainDataCopyUb2GmNz2Nd<half>, { 1, 64, 32, 1, 64, 32, 1 } }));

TEST_P(DataCopyDescUb2GmNz2NdTestsuiteCheckGmOverflow, DataCopyUb2GmNz2NdTestsuiteCheckGmOverflow_1)
{
    constexpr size_t workspaceSize = AscendC::RESERVED_WORKSPACE;
    uint8_t* sysWorkSpacePtr = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    memset_s(sysWorkSpacePtr, workspaceSize, 0, workspaceSize);
    std::cout<<"kernel_util sysWorkSpacePtr "<< reinterpret_cast<uintptr_t>(sysWorkSpacePtr)<<std::endl;
    std::cout<<" workspaceSize "<< reinterpret_cast<uintptr_t>(workspaceSize)<<std::endl;
    if (sysWorkSpacePtr == nullptr) {
        printf("[error]g_sysWorkspaceReserved is null, g_sysWorkspaceReserved has been set or not\n");
    }
    g_sysWorkspaceReserved = sysWorkSpacePtr;
    auto workspace = GetSysWorkSpacePtr();
    auto param = GetParam();
    Nz2NdParamsFull intriParams = param.intriParams;
    uint8_t srcGm[intriParams.nValue * intriParams.dValue * param.typeSize] = {0};
    uint8_t dstGm[intriParams.nValue * intriParams.dValue * param.typeSize] = {0};
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024)) = 1;
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 8)) = 1;
    *((__gm__ uintptr_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 16)) = reinterpret_cast<uintptr_t>(srcGm);
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 24)) = intriParams.nValue * intriParams.dValue * param.typeSize;
    *((__gm__ uintptr_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 32)) = reinterpret_cast<uintptr_t>(dstGm);
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 40)) = intriParams.nValue * intriParams.dValue * param.typeSize;

    param.CalFunc(dstGm, srcGm, intriParams);
    AscendC::GmFree((void*)sysWorkSpacePtr);
    g_sysWorkspaceReserved = nullptr; 
    for (int32_t i = 0; i < (intriParams.nValue * intriParams.dValue); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
