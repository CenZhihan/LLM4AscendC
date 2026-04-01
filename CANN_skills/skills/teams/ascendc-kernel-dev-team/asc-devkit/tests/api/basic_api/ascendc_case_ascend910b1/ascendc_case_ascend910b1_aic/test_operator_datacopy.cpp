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
// T, U for l1->l0c
template <typename T, typename U> class KernelDataCopyL12L0c {
public:
    __aicore__ inline KernelDataCopyL12L0c() = default;
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm)
    {
        srcGlobal.SetGlobalBuffer((__gm__ U*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrcA1, 1, 512 * sizeof(U));
        pipe.InitBuffer(outQueueCO1, 1, 512 * sizeof(T));
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
        LocalTensor<U> srcLocal = inQueueSrcA1.AllocTensor<U>();

        DataCopy(srcLocal, srcGlobal, 512);

        inQueueSrcA1.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<U> srcLocal = inQueueSrcA1.DeQue<U>();
        LocalTensor<T> dstLocal = outQueueCO1.AllocTensor<T>();
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = 2 * sizeof(U) / sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyEnhancedParams enhancedParams;
        DataCopy(dstLocal, srcLocal, dataCopyParams, enhancedParams);
        SetAtomicNone();
        outQueueCO1.EnQue<T>(dstLocal);
        inQueueSrcA1.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = outQueueCO1.DeQue<T>();
        outQueueCO1.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::A1, 1> inQueueSrcA1;
    TQue<TPosition::CO1, 1> outQueueCO1;
    GlobalTensor<U> srcGlobal;
    GlobalTensor<T> dstGlobal;
};
} // namespace AscendC

template <typename T, typename U>
__global__ __aicore__ void MainDataCopyL12L0c(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm)
{
    AscendC::KernelDataCopyL12L0c<T, U> op;
    op.Init(dstGm, srcGm);
    op.Process();
}

struct DataCopyL12L0cTestParams {
    int32_t typeSize;
    void (*CalFunc)(uint8_t*, uint8_t*);
};

class DataCopyL12L0cTestsuite : public testing::Test, public testing::WithParamInterface<DataCopyL12L0cTestParams> {
protected:
    void SetUp() {
        g_coreType = AIC_TYPE;
    }
    void TearDown() {
        AscendC::CheckSyncState();
        g_coreType = MIX_TYPE;
        GlobalMockObject::verify();
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DATACOPYL12L0C, DataCopyL12L0cTestsuite,
   ::testing::Values(
    DataCopyL12L0cTestParams { 2, MainDataCopyL12L0c<bfloat16_t, bfloat16_t> },
    DataCopyL12L0cTestParams { 4, MainDataCopyL12L0c<bfloat16_t, float> },
    DataCopyL12L0cTestParams { 4, MainDataCopyL12L0c<float, float> }, 
    DataCopyL12L0cTestParams { 4, MainDataCopyL12L0c<int32_t, int32_t> } 
    // DataCopyL12L0cTestParams { 4, MainDataCopyL12L0c<uint32_t,uint32_t> }
));

TEST_P(DataCopyL12L0cTestsuite, DataCopyL12L0cOpTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[512 * param.typeSize];
    uint8_t dstGm[512 * param.typeSize];

    param.CalFunc(dstGm, srcGm);
}

namespace AscendC {
// T, U for gm->l1 nd2nz
template <typename T> class KernelDataCopyGm2L1Nd2Nz {
public:
    __aicore__ inline KernelDataCopyGm2L1Nd2Nz() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, Nd2NzParams& intriParamsIn)
    {
        intriParams = intriParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrcA1, 1,
            (((intriParams.dValue * sizeof(T) - 1) / 32 + 1) * intriParams.dstNzC0Stride * 32));
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
        LocalTensor<T> srcLocal = inQueueSrcA1.AllocTensor<T>();

        DataCopy(srcLocal, srcGlobal, intriParams);

        inQueueSrcA1.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        ;
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = inQueueSrcA1.DeQue<T>();
        DataCopy(dstGlobal, dstLocal,
            { 1, (((intriParams.dValue * sizeof(T) - 1) / 32 + 1) * intriParams.dstNzC0Stride * 32) / 32, 0, 0 });

        inQueueSrcA1.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::A1, 1> inQueueSrcA1;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    Nd2NzParams intriParams;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void MainDataCopyGm2L1Nd2Nz(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
    Nd2NzParams& intriParams)
{
    AscendC::KernelDataCopyGm2L1Nd2Nz<T> op;
    op.Init(dstGm, srcGm, intriParams);
    op.Process();
}

struct DataCopyGm2L1Nd2NzTestParams {
    int32_t typeSize;
    void (*CalFunc)(uint8_t*, uint8_t*, Nd2NzParams&);
    Nd2NzParams intriParams;
};

class DataCopyGm2L1Nd2NzTestsuite : public testing::Test,
    public testing::WithParamInterface<DataCopyGm2L1Nd2NzTestParams> {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(AIC_TYPE);
    }
    void TearDown()
    {
        AscendC::CheckSyncState();
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DATACOPYGM2L1ND2NZ, DataCopyGm2L1Nd2NzTestsuite,
    ::testing::Values(DataCopyGm2L1Nd2NzTestParams { 2, MainDataCopyGm2L1Nd2Nz<half>, { 2, 2, 13, 48, 16, 11, 2, 48 } },
    DataCopyGm2L1Nd2NzTestParams { 2, MainDataCopyGm2L1Nd2Nz<bfloat16_t>, { 2, 2, 13, 48, 16, 11, 2, 48 } },
    DataCopyGm2L1Nd2NzTestParams { 2, MainDataCopyGm2L1Nd2Nz<int16_t>, { 2, 2, 13, 48, 16, 11, 2, 48 } },
    DataCopyGm2L1Nd2NzTestParams { 2, MainDataCopyGm2L1Nd2Nz<uint16_t>, { 2, 2, 13, 48, 16, 11, 2, 48 } },
    DataCopyGm2L1Nd2NzTestParams { 4, MainDataCopyGm2L1Nd2Nz<float>, { 2, 2, 13, 48, 16, 11, 2, 40 } },
    DataCopyGm2L1Nd2NzTestParams { 4, MainDataCopyGm2L1Nd2Nz<int32_t>, { 2, 2, 13, 48, 16, 11, 2, 40 } },
    DataCopyGm2L1Nd2NzTestParams { 4, MainDataCopyGm2L1Nd2Nz<uint32_t>, { 2, 2, 13, 48, 16, 11, 2, 40 } },
    DataCopyGm2L1Nd2NzTestParams { 1, MainDataCopyGm2L1Nd2Nz<int8_t>, { 2, 2, 13, 96, 32, 11, 2, 96 } },
    DataCopyGm2L1Nd2NzTestParams { 1, MainDataCopyGm2L1Nd2Nz<uint8_t>, { 2, 2, 13, 96, 32, 11, 2, 96 } }));

TEST_P(DataCopyGm2L1Nd2NzTestsuite, DataCopyGm2L1Nd2NzTestsuiteOpTestCase)
{
    auto param = GetParam();
    Nd2NzParams intriParams = param.intriParams;
    uint8_t srcGm[intriParams.ndNum * intriParams.srcNdMatrixStride * param.typeSize] = {0};
    uint8_t dstGm[((intriParams.dValue * param.typeSize -1 ) / 32 + 1) * intriParams.dstNzC0Stride * 32] = {0};

    param.CalFunc(dstGm, srcGm, intriParams);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

namespace AscendC {
// T, U for l0c->l1
template <typename T, typename U> class KernelDataCopyL0c2L1 {
public:
    __aicore__ inline KernelDataCopyL0c2L1() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, QuantMode_t mode)
    {
        srcGlobal.SetGlobalBuffer((__gm__ U*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrcCO1, 1, 512 * sizeof(U));
        pipe.InitBuffer(outQueueA1, 1, 512 * sizeof(T));
        quantMode = mode;
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
        ;
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<U> srcLocal = inQueueSrcCO1.AllocTensor<U>();
        LocalTensor<T> dstLocal = outQueueA1.AllocTensor<T>();

        DataCopyCO12DstParams dataCopyParams(16, 32, 8, 1, quantMode, 0, false, false);
        DataCopy(dstLocal, srcLocal, dataCopyParams);
        outQueueA1.EnQue<T>(dstLocal);
        inQueueSrcCO1.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = outQueueA1.DeQue<T>();
        outQueueA1.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::CO1, 1> inQueueSrcCO1;
    TQue<TPosition::A1, 1> outQueueA1;
    GlobalTensor<U> srcGlobal;
    GlobalTensor<T> dstGlobal;
    QuantMode_t quantMode;
};
} // namespace AscendC

template <typename T, typename U>
__global__ __aicore__ void MainDataCopyL0c2L1(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, QuantMode_t mode)
{
    AscendC::KernelDataCopyL0c2L1<T, U> op;
    op.Init(dstGm, srcGm, mode);
    op.Process();
}

struct DataCopyL0c2L1TestParams {
    int32_t dstTypeSize;
    int32_t srcTypeSize;
    QuantMode_t quantMode;
    void (*CalFunc)(uint8_t*, uint8_t*, QuantMode_t);
};

class DataCopyL0c2L1Testsuite : public testing::Test, public testing::WithParamInterface<DataCopyL0c2L1TestParams> {
protected:
    void SetUp() {
        g_coreType = AIC_TYPE;
    }
    void TearDown() {
        AscendC::CheckSyncState();
        g_coreType = MIX_TYPE;
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DATACOPYL0C2L1, DataCopyL0c2L1Testsuite,
   ::testing::Values(
    DataCopyL0c2L1TestParams { 1, 4, QuantMode_t::QF322B8_PRE, MainDataCopyL0c2L1<int8_t, float> },
    DataCopyL0c2L1TestParams { 1, 4, QuantMode_t::VQF322B8_PRE, MainDataCopyL0c2L1<int8_t, float> },
    DataCopyL0c2L1TestParams { 2, 4, QuantMode_t::F322F16, MainDataCopyL0c2L1<half, float> },
    DataCopyL0c2L1TestParams { 2, 4, QuantMode_t::F322BF16, MainDataCopyL0c2L1<bfloat16_t, float> },
    DataCopyL0c2L1TestParams { 2, 4, QuantMode_t::DEQF16, MainDataCopyL0c2L1<half, int32_t> },
    DataCopyL0c2L1TestParams { 2, 4, QuantMode_t::VDEQF16, MainDataCopyL0c2L1<half, int32_t> },
    DataCopyL0c2L1TestParams { 1, 4, QuantMode_t::REQ8, MainDataCopyL0c2L1<int8_t, int32_t> },
    DataCopyL0c2L1TestParams { 1, 4, QuantMode_t::VREQ8, MainDataCopyL0c2L1<int8_t, int32_t> }
    ));

TEST_P(DataCopyL0c2L1Testsuite, DataCopyL0c2L1OpTestCase)
{
    int32_t tmp = g_coreType;
    g_coreType = AscendC::AIC_TYPE;
    auto param = GetParam();
    uint8_t srcGm[512 * param.srcTypeSize];
    uint8_t dstGm[512 * param.dstTypeSize];

    param.CalFunc(dstGm, srcGm, param.quantMode);
    g_coreType = tmp;
}

namespace AscendC {
// T, U for l0c->gm
template <typename T, typename U> class KernelDataCopyL0c2Gm {
public:
    __aicore__ inline KernelDataCopyL0c2Gm() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, QuantMode_t mode)
    {
        srcGlobal.SetGlobalBuffer((__gm__ U*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrcCO1, 1, 512 * sizeof(U));
        quantMode = mode;
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
        ;
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<U> srcLocal = inQueueSrcCO1.AllocTensor<U>();

        DataCopyCO12DstParams dataCopyParams(16, 32, 8, 1, quantMode, 0, false, false);
        DataCopy(dstGlobal, srcLocal, dataCopyParams);
        inQueueSrcCO1.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        ;
    }

private:
    TPipe pipe;
    TQue<TPosition::CO1, 1> inQueueSrcCO1;
    GlobalTensor<U> srcGlobal;
    GlobalTensor<T> dstGlobal;
    QuantMode_t quantMode;
};
} // namespace AscendC

template <typename T, typename U>
__global__ __aicore__ void MainDataCopyL0c2Gm(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, QuantMode_t mode)
{
    AscendC::KernelDataCopyL0c2Gm<T, U> op;
    op.Init(dstGm, srcGm, mode);
    op.Process();
}

struct DataCopyL0c2GmTestParams {
    int32_t dstTypeSize;
    int32_t srcTypeSize;
    QuantMode_t quantMode;
    void (*CalFunc)(uint8_t*, uint8_t*, QuantMode_t);
};

class DataCopyL0c2GmTestsuite : public testing::Test, public testing::WithParamInterface<DataCopyL0c2GmTestParams> {
protected:
    void SetUp() {
        g_coreType = AIC_TYPE;
    }
    void TearDown() {
        AscendC::CheckSyncState();
        g_coreType = MIX_TYPE;
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DATACOPYL0C2GM, DataCopyL0c2GmTestsuite,
   ::testing::Values(
    DataCopyL0c2GmTestParams { 4, 4, QuantMode_t::NoQuant, MainDataCopyL0c2Gm<int32_t, int32_t> },
    DataCopyL0c2GmTestParams { 1, 4, QuantMode_t::QF322B8_PRE, MainDataCopyL0c2Gm<int8_t, float> },
    DataCopyL0c2GmTestParams { 1, 4, QuantMode_t::VQF322B8_PRE, MainDataCopyL0c2Gm<int8_t, float> },
    DataCopyL0c2GmTestParams { 2, 4, QuantMode_t::F322F16, MainDataCopyL0c2Gm<half, float> },
    DataCopyL0c2GmTestParams { 2, 4, QuantMode_t::F322BF16, MainDataCopyL0c2Gm<bfloat16_t, float> },
    DataCopyL0c2GmTestParams { 2, 4, QuantMode_t::DEQF16, MainDataCopyL0c2Gm<half, int32_t> },
    DataCopyL0c2GmTestParams { 2, 4, QuantMode_t::VDEQF16, MainDataCopyL0c2Gm<half, int32_t> },
    DataCopyL0c2GmTestParams { 1, 4, QuantMode_t::REQ8, MainDataCopyL0c2Gm<int8_t, int32_t> },
    DataCopyL0c2GmTestParams { 1, 4, QuantMode_t::VREQ8, MainDataCopyL0c2Gm<int8_t, int32_t> }
    ));

TEST_P(DataCopyL0c2GmTestsuite, DataCopyL0c2GmOpTestCase)
{
    int32_t tmp = g_coreType;
    g_coreType = AscendC::AIC_TYPE;
    auto param = GetParam();
    uint8_t srcGm[512 * param.srcTypeSize];
    uint8_t dstGm[512 * param.dstTypeSize];

    param.CalFunc(dstGm, srcGm, param.quantMode);
    g_coreType = tmp;
}

class DataCopyGm2L1Nd2NzTestsuiteCheckGmOverflow : public testing::Test,
    public testing::WithParamInterface<DataCopyGm2L1Nd2NzTestParams> {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(1);
    }
    void TearDown()
    {
        AscendC::CheckSyncState();
        AscendC::SetGCoreType(0);
    }
};

INSTANTIATE_TEST_CASE_P(TestOperationDataCopyGM2L1Nd2NzOom, DataCopyGm2L1Nd2NzTestsuiteCheckGmOverflow,
    ::testing::Values(DataCopyGm2L1Nd2NzTestParams { 2, MainDataCopyGm2L1Nd2Nz<half>, { 2, 2, 13, 48, 16, 11, 2, 48 } },
    DataCopyGm2L1Nd2NzTestParams { 2, MainDataCopyGm2L1Nd2Nz<bfloat16_t>, { 2, 2, 13, 48, 16, 11, 2, 48 } },
    DataCopyGm2L1Nd2NzTestParams { 2, MainDataCopyGm2L1Nd2Nz<int16_t>, { 2, 2, 13, 48, 16, 11, 2, 48 } },
    DataCopyGm2L1Nd2NzTestParams { 2, MainDataCopyGm2L1Nd2Nz<uint16_t>, { 2, 2, 13, 48, 16, 11, 2, 48 } },
    DataCopyGm2L1Nd2NzTestParams { 4, MainDataCopyGm2L1Nd2Nz<float>, { 2, 2, 13, 48, 16, 11, 2, 40 } },
    DataCopyGm2L1Nd2NzTestParams { 4, MainDataCopyGm2L1Nd2Nz<int32_t>, { 2, 2, 13, 48, 16, 11, 2, 40 } },
    DataCopyGm2L1Nd2NzTestParams { 4, MainDataCopyGm2L1Nd2Nz<uint32_t>, { 2, 2, 13, 48, 16, 11, 2, 40 } },
    DataCopyGm2L1Nd2NzTestParams { 1, MainDataCopyGm2L1Nd2Nz<int8_t>, { 2, 2, 13, 96, 32, 11, 2, 96 } },
    DataCopyGm2L1Nd2NzTestParams { 1, MainDataCopyGm2L1Nd2Nz<uint8_t>, { 2, 2, 13, 96, 32, 11, 2, 96 } }));

TEST_P(DataCopyGm2L1Nd2NzTestsuiteCheckGmOverflow, DataCopyGm2L1Nd2NzTestsuiteCheckGmOverflowCheckGmOverflow_1)
{
    constexpr size_t workspaceSize = AscendC::RESERVED_WORKSPACE;
    uint8_t* sysWorkSpacePtr = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    memset(sysWorkSpacePtr, 0, workspaceSize);
    if (sysWorkSpacePtr == nullptr) {
        printf("[error]g_sysWorkspaceReserved is null, g_sysWorkspaceReserved has been set or not\n");
    }
    g_sysWorkspaceReserved = sysWorkSpacePtr;
    auto workspace = GetSysWorkSpacePtr();
    auto param = GetParam();
    Nd2NzParams intriParams = param.intriParams;
    uint8_t srcGm[intriParams.ndNum * intriParams.srcNdMatrixStride * param.typeSize] = {0};
    uint8_t dstGm[((intriParams.dValue * param.typeSize -1 ) / 32 + 1) * intriParams.dstNzC0Stride * 32] = {0};

    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024)) = 1;
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 8)) = 1;
    *((__gm__ uintptr_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 16)) = reinterpret_cast<uintptr_t>(srcGm);
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 24)) = intriParams.ndNum * intriParams.srcNdMatrixStride * param.typeSize;
    *((__gm__ uintptr_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 32)) = reinterpret_cast<uintptr_t>(dstGm);
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 40)) = ((intriParams.dValue * param.typeSize -1 ) / 32 + 1) * intriParams.dstNzC0Stride * 32;

    param.CalFunc(dstGm, srcGm, intriParams);
    AscendC::GmFree((void*)sysWorkSpacePtr);
    g_sysWorkspaceReserved = nullptr; 
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}