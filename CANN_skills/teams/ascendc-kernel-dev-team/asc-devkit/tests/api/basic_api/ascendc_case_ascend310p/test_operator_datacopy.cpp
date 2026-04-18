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
#include <math.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace AscendC;

namespace AscendC {
// T, U for ub->gm nz2nd
template <typename T> class KernelDataCopyUb2GmNz2Nd310p {
public:
    __aicore__ inline KernelDataCopyUb2GmNz2Nd310p() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, Nz2NdParamsFull& intriParamsIn)
    {
        intriParams = intriParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrcVecIn, 1, intriParams.nValue * intriParams.dValue * sizeof(T));
        pipe.InitBuffer(inQueueSrcUb2Gm, 1, intriParams.nValue * intriParams.dValue * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyND2NZ(const LocalTensor<half>& dst, const GlobalTensor<half>& src, const uint16_t height,
        const uint16_t width)
    {
        for (int i = 0; i < width / 16; ++i) {
            int srcOffset = i * 16;
            int dstOffset = i * 16 * height;
            DataCopy(dst[dstOffset], src[srcOffset], { height, 1, static_cast<uint16_t>(width / 16 - 1), 0 });
        }
    }

    __aicore__ inline void CopyIn()
    {
        LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
        LocalTensor<T> tmpLocal = inQueueSrcUb2Gm.AllocTensor<T>();

        CopyND2NZ(srcLocal, srcGlobal, intriParams.nValue, intriParams.dValue);
        DataCopy(tmpLocal, srcGlobal, intriParams.nValue * intriParams.dValue);

        inQueueSrcVecIn.EnQue(srcLocal);
        inQueueSrcUb2Gm.FreeTensor(tmpLocal);
    }
    __aicore__ inline void Compute()
    {
        ;
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = inQueueSrcVecIn.DeQue<T>();

        DataCopy(dstGlobal, dstLocal, intriParams);

        inQueueSrcVecIn.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrcVecIn;
    TQue<TPosition::VECIN, 1> inQueueSrcUb2Gm;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    Nz2NdParamsFull intriParams;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void MainDataCopyUb2GmNz2Nd310p(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
    Nz2NdParamsFull& intriParams)
{
    AscendC::KernelDataCopyUb2GmNz2Nd310p<T> op;
    op.Init(dstGm, srcGm, intriParams);
    op.Process();
}

struct DataCopyGm2L1Nd2NzTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, Nd2NzParams&);
    Nd2NzParams intriParams;
};

struct DataCopyUb2GmNz2Nd310pTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, Nz2NdParamsFull&);
    Nz2NdParamsFull intriParams;
};

class DataCopyUb2GmNz2Nd310pTestsuite : public testing::Test,
    public testing::WithParamInterface<DataCopyUb2GmNz2Nd310pTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DataCopyUb2GmNz2Nd310p, DataCopyUb2GmNz2Nd310pTestsuite,
    ::testing::Values(DataCopyUb2GmNz2Nd310pTestParams { 2, MainDataCopyUb2GmNz2Nd310p<half>, { 1, 32, 32, 1, 32, 32, 1 } },
    DataCopyUb2GmNz2Nd310pTestParams { 2, MainDataCopyUb2GmNz2Nd310p<half>, { 1, 64, 32, 1, 64, 32, 1 } },
    DataCopyUb2GmNz2Nd310pTestParams { 2, MainDataCopyUb2GmNz2Nd310p<half>, { 1, 4095, 16, 1, 16, 16, 1 } }));

TEST_P(DataCopyUb2GmNz2Nd310pTestsuite, DataCopyUb2GmNz2Nd310pTestsuiteOpTestCase)
{
    auto param = GetParam();
    Nz2NdParamsFull intriParams = param.intriParams;
    uint8_t srcGm[intriParams.nValue * intriParams.dValue * param.typeSize] = {0};
    uint8_t dstGm[intriParams.nValue * intriParams.dValue * param.typeSize] = {0};

    param.cal_func(dstGm, srcGm, intriParams);
    for (int32_t i = 0; i < (intriParams.nValue * intriParams.dValue); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

class DataCopyUb2GmNz2Nd310pTestsuiteCheckGmOverflow : public testing::Test,
    public testing::WithParamInterface<DataCopyUb2GmNz2Nd310pTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DataCopyUb2GmNz2Nd310p_OOM, DataCopyUb2GmNz2Nd310pTestsuiteCheckGmOverflow,
    ::testing::Values(DataCopyUb2GmNz2Nd310pTestParams { 2, MainDataCopyUb2GmNz2Nd310p<half>, { 1, 32, 32, 1, 32, 32, 1 } },
    DataCopyUb2GmNz2Nd310pTestParams { 2, MainDataCopyUb2GmNz2Nd310p<half>, { 1, 64, 32, 1, 64, 32, 1 } },
    DataCopyUb2GmNz2Nd310pTestParams { 2, MainDataCopyUb2GmNz2Nd310p<half>, { 1, 4095, 16, 1, 16, 16, 1 } }));

TEST_P(DataCopyUb2GmNz2Nd310pTestsuiteCheckGmOverflow, DataCopyUb2GmNz2Nd310pTestsuiteCheckGmOverflow_1)
{
    constexpr size_t workspaceSize = AscendC::RESERVED_WORKSPACE;
    uint8_t* sysWorkSpacePtr = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    memset(sysWorkSpacePtr, 0, workspaceSize);
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

    param.cal_func(dstGm, srcGm, intriParams);
    AscendC::GmFree((void*)sysWorkSpacePtr);
    g_sysWorkspaceReserved = nullptr; 
    for (int32_t i = 0; i < (intriParams.nValue * intriParams.dValue); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
