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
// #include "api_check/kernel_cpu_check.h"

using namespace std;
using namespace AscendC;

struct TestDataCopyParams {
    uint32_t dataSize;
    uint32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, int32_t);
};

class TestDataCopySuite : public testing::Test, public testing::WithParamInterface<TestDataCopyParams> {
protected:
    void SetUp() {
        SetGCoreType(1);
        BufIdTracker::GetInstance().Reset();
    }
    void TearDown() {
        SetGCoreType(0);
    }
};

template<typename T>
void MainDataCopyKernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;

    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    GlobalTensor<half> testGlobal;
    testGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(dstGm), dataSize);

    TSCM<TPosition::GM> tque;
    tpipe.InitBuffer(tque, 1, 2048);
    Nd2NzParams intriParams;
    intriParams.ndNum = 1;
    intriParams.nValue = 32;
    intriParams.dValue = 32;
    intriParams.srcNdMatrixStride = 0;
    intriParams.srcDValue = 32;
    intriParams.dstNzC0Stride = 32;
    intriParams.dstNzNStride = 1;
    intriParams.dstNzMatrixStride = 0;

    LocalTensor<T> inputLocal = tque.AllocTensor<T>();
    DataCopy(inputLocal, srcGlobal, intriParams);
    tque.EnQue(inputLocal);
    LocalTensor<T> outputLocal = tque.DeQue<T>();
    tque.FreeTensor(outputLocal);
}

template<typename T>
void MainDataCopyKernelFixpipe(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;

    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::C1> tbufL1;
    tpipe.InitBuffer(tbufL1, dataSize * sizeof(uint64_t));
    LocalTensor<uint64_t> l1Local = tbufL1.Get<uint64_t>();

    TBuf<TPosition::C2PIPE2GM> tbufDeq;
    tpipe.InitBuffer(tbufDeq, dataSize * sizeof(uint64_t));
    LocalTensor<uint64_t> deqLocal = tbufDeq.Get<uint64_t>();

    TBuf<TPosition::CO1> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<float> inputLocal = tbuf.Get<float>();

    DataCopyParams dataCopyParams(1, 2, 0, 0);
    DataCopy(deqLocal, l1Local, dataCopyParams);
    SetFixPipeConfig(deqLocal);
    DataCopyCO12DstParams intriParams(16, 16, 0, 0, QuantMode_t::VDEQF16, static_cast<uint8_t>(0), 0, 0);
    DataCopy(dstGlobal, inputLocal, intriParams);
    DataCopyCO12DstParams intriParams2(16, 16, 0, 0, QuantMode_t::VDEQF16, static_cast<uint8_t>(0), 1, 0);
    DataCopy(dstGlobal, inputLocal, intriParams2);
}

template<typename T>
void MainDataCopyDn2nzKernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;

    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    GlobalTensor<half> testGlobal;
    testGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(dstGm), dataSize);

    TBuf<TPosition::A1> tbuf1;
    tpipe.InitBuffer(tbuf1, 2048);

    LocalTensor<T> inputLocal = tbuf1.Get<T>();
    Dn2NzParams intriParams1;
    intriParams1.dnNum = 1;
    intriParams1.nValue = 32;
    intriParams1.dValue = 32;
    intriParams1.srcDnMatrixStride = 0;
    intriParams1.srcDValue = 32;
    intriParams1.dstNzC0Stride = 32;
    intriParams1.dstNzNStride = 1;
    intriParams1.dstNzMatrixStride = 0;
    DataCopy(inputLocal, srcGlobal, intriParams1);
}

template<typename T>
void main_tensortrait_data_copy_kernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<TensorTrait<T>> srcGlobal;
    GlobalTensor<TensorTrait<T>> dstGlobal;

    GlobalTensor<TensorTrait<T>> disableCacheGlobal;
    disableCacheGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    disableCacheGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm));

    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    GlobalTensor<TensorTrait<half>> testGlobal;
    testGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(dstGm), dataSize);

    TSCM<TPosition::GM> tque;
    tpipe.InitBuffer(tque, 1, 2048);
    Nd2NzParams intriParams;
    intriParams.ndNum = 1;
    intriParams.nValue = 32;
    intriParams.dValue = 32;
    intriParams.srcNdMatrixStride = 0;
    intriParams.srcDValue = 32;
    intriParams.dstNzC0Stride = 32;
    intriParams.dstNzNStride = 1;
    intriParams.dstNzMatrixStride = 0;

    LocalTensor<TensorTrait<T>> inputLocal = tque.AllocTensor<TensorTrait<T>>();
    DataCopy(inputLocal, srcGlobal, intriParams);
    tque.EnQue(inputLocal);
    LocalTensor<TensorTrait<T>> outputLocal = tque.DeQue<TensorTrait<T>>();
    tque.FreeTensor(outputLocal);
}

template<typename T>
void main_tensortrait_data_copy_dn2nz_kernel(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm,
    __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<TensorTrait<T>> srcGlobal;
    GlobalTensor<TensorTrait<T>> dstGlobal;

    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    GlobalTensor<TensorTrait<half>> testGlobal;
    testGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(dstGm), dataSize);

    TBuf<TPosition::A1> tbuf1;
    tpipe.InitBuffer(tbuf1, 2048);

    LocalTensor<TensorTrait<T>> inputLocal = tbuf1.Get<TensorTrait<T>>();
    Dn2NzParams intriParams1;
    intriParams1.dnNum = 1;
    intriParams1.nValue = 32;
    intriParams1.dValue = 32;
    intriParams1.srcDnMatrixStride = 0;
    intriParams1.srcDValue = 32;
    intriParams1.dstNzC0Stride = 32;
    intriParams1.dstNzNStride = 1;
    intriParams1.dstNzMatrixStride = 0;
    DataCopy(inputLocal, srcGlobal, intriParams1);
}

template<typename T>
void main_tensortrait_data_copy_kernel_fixpipe(__gm__ uint8_t* __restrict__ srcGm, __gm__ uint8_t* __restrict__ dstGm, __gm__ int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<TensorTrait<T>> srcGlobal;
    GlobalTensor<TensorTrait<T>> dstGlobal;

    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::C1> tbufL1;
    tpipe.InitBuffer(tbufL1, dataSize * sizeof(uint64_t));
    LocalTensor<TensorTrait<uint64_t>> l1Local = tbufL1.Get<TensorTrait<uint64_t>>();

    TBuf<TPosition::C2PIPE2GM> tbufDeq;
    tpipe.InitBuffer(tbufDeq, dataSize * sizeof(uint64_t));
    LocalTensor<TensorTrait<uint64_t>> deqLocal = tbufDeq.Get<TensorTrait<uint64_t>>();

    TBuf<TPosition::CO1> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<TensorTrait<float>> inputLocal = tbuf.Get<TensorTrait<float>>();

    DataCopyParams dataCopyParams(1, 2, 0, 0);
    DataCopy(deqLocal, l1Local, dataCopyParams);
    SetFixPipeConfig(deqLocal);
    DataCopyCO12DstParams intriParams(16, 16, 0, 0, QuantMode_t::VDEQF16, static_cast<uint8_t>(0), 0, 0);
    DataCopy(dstGlobal, inputLocal, intriParams);
    DataCopyCO12DstParams intriParams2(16, 16, 0, 0, QuantMode_t::VDEQF16, static_cast<uint8_t>(0), 1, 0);
    DataCopy(dstGlobal, inputLocal, intriParams2);
}

INSTANTIATE_TEST_CASE_P(TEST_DATA_COPY_AIC, TestDataCopySuite,
    ::testing::Values(
    TestDataCopyParams { 64, 4, MainDataCopyKernel<int32_t> },
    TestDataCopyParams { 64, 4, MainDataCopyKernel<uint32_t> },
    TestDataCopyParams { 64, 2, MainDataCopyKernel<int16_t> },
    TestDataCopyParams { 64, 2, MainDataCopyKernel<uint16_t> },
    TestDataCopyParams { 64, 4, MainDataCopyKernel<float> },
    TestDataCopyParams { 64, 2, MainDataCopyKernel<half> },
    TestDataCopyParams { 64, 1, MainDataCopyKernel<int8_t> },
    TestDataCopyParams { 64, 1, MainDataCopyKernel<uint8_t> },
    TestDataCopyParams { 64, 4, MainDataCopyDn2nzKernel<float> },
    TestDataCopyParams { 64, 2, MainDataCopyDn2nzKernel<half> },
    TestDataCopyParams { 64, 1, MainDataCopyDn2nzKernel<int8_t> },
    TestDataCopyParams { 64, 1, MainDataCopyDn2nzKernel<bfloat16_t> },
    TestDataCopyParams { 64, 1, MainDataCopyDn2nzKernel<uint8_t> },
    TestDataCopyParams { 64, 1, MainDataCopyDn2nzKernel<fp8_e4m3fn_t> },
    TestDataCopyParams { 64, 1, MainDataCopyDn2nzKernel<fp8_e5m2_t> },
    TestDataCopyParams { 64, 1, MainDataCopyDn2nzKernel<hifloat8_t> },
    TestDataCopyParams { 64, 1, MainDataCopyDn2nzKernel<int16_t> },
    TestDataCopyParams { 64, 1, MainDataCopyDn2nzKernel<uint16_t> },
    TestDataCopyParams { 64, 1, MainDataCopyDn2nzKernel<int32_t> },
    TestDataCopyParams { 64, 1, MainDataCopyDn2nzKernel<uint32_t> },
    TestDataCopyParams { 256, 1, MainDataCopyKernelFixpipe<int8_t> },
    TestDataCopyParams { 64, 4, main_tensortrait_data_copy_kernel<int32_t> },
    TestDataCopyParams { 64, 4, main_tensortrait_data_copy_kernel<uint32_t> },
    TestDataCopyParams { 64, 2, main_tensortrait_data_copy_kernel<int16_t> },
    TestDataCopyParams { 64, 2, main_tensortrait_data_copy_kernel<uint16_t> },
    TestDataCopyParams { 64, 4, main_tensortrait_data_copy_kernel<float> },
    TestDataCopyParams { 64, 2, main_tensortrait_data_copy_kernel<half> },
    TestDataCopyParams { 64, 1, main_tensortrait_data_copy_kernel<int8_t> },
    TestDataCopyParams { 64, 1, main_tensortrait_data_copy_kernel<uint8_t> },
    TestDataCopyParams { 64, 4, main_tensortrait_data_copy_dn2nz_kernel<float> },
    TestDataCopyParams { 64, 2, main_tensortrait_data_copy_dn2nz_kernel<half> },
    TestDataCopyParams { 64, 1, main_tensortrait_data_copy_dn2nz_kernel<int8_t> },
    TestDataCopyParams { 64, 1, main_tensortrait_data_copy_dn2nz_kernel<bfloat16_t> },
    TestDataCopyParams { 64, 1, main_tensortrait_data_copy_dn2nz_kernel<uint8_t> },
    TestDataCopyParams { 64, 1, main_tensortrait_data_copy_dn2nz_kernel<fp8_e4m3fn_t> },
    TestDataCopyParams { 64, 1, main_tensortrait_data_copy_dn2nz_kernel<fp8_e5m2_t> },
    TestDataCopyParams { 64, 1, main_tensortrait_data_copy_dn2nz_kernel<hifloat8_t> },
    TestDataCopyParams { 64, 1, main_tensortrait_data_copy_dn2nz_kernel<int16_t> },
    TestDataCopyParams { 64, 1, main_tensortrait_data_copy_dn2nz_kernel<uint16_t> },
    TestDataCopyParams { 64, 1, main_tensortrait_data_copy_dn2nz_kernel<int32_t> },
    TestDataCopyParams { 64, 1, main_tensortrait_data_copy_dn2nz_kernel<uint32_t> },
    TestDataCopyParams { 256, 1, main_tensortrait_data_copy_kernel_fixpipe<int8_t> }
    ));

TEST_P(TestDataCopySuite, TestDataCopyPadCases)
{
    auto param = GetParam();
    uint8_t srcGm[param.dataSize * param.typeSize];
    uint8_t dstGm[param.dataSize * param.typeSize];
    param.cal_func(srcGm, dstGm, param.dataSize);
}

namespace AscendC {
// T, U for gm->l1 nd2nz
template <typename T> class KernelDataCopyGm2UBNd2Nz {
public:
    __aicore__ inline KernelDataCopyGm2UBNd2Nz() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, Nd2NzParams& intriParamsIn)
    {
        intriParams = intriParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrcUB, 1,
            (((intriParams.dValue * sizeof(T) - 1) / 32 + 1) * intriParams.dstNzC0Stride * 32));
        pipe.InitBuffer(inQueueSrcL1, 1,
            (((intriParams.dValue * sizeof(T) - 1) / 32 + 1) * intriParams.dstNzC0Stride * 32));
        pipe.InitBuffer(inQueueSrcVecOut, 1,
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
        LocalTensor<T> srcLocal = inQueueSrcUB.AllocTensor<T>();
        LocalTensor<T> srcLocalL1 = inQueueSrcL1.AllocTensor<T>();
 
        DataCopy(srcLocal, srcGlobal, intriParams);
 
        DataCopyParams params(1, 128, 0, 0);
        DataCopy(srcLocalL1, srcLocal, params);
        inQueueSrcUB.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        ;
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = inQueueSrcUB.DeQue<T>();
        LocalTensor<T> dstLocalVecOut = inQueueSrcVecOut.AllocTensor<T>();
        DataCopy(dstLocalVecOut, dstLocal,
            { 1, (((intriParams.dValue * sizeof(T) - 1) / 32 + 1) * intriParams.dstNzC0Stride * 32) / 32, 0, 0 });
        inQueueSrcVecOut.EnQue(dstLocalVecOut);
        inQueueSrcVecOut.DeQue<T>();
        DataCopy(dstGlobal, dstLocalVecOut,
            { 1, (((intriParams.dValue * sizeof(T) - 1) / 32 + 1) * intriParams.dstNzC0Stride * 32) / 32, 0, 0 });
 
        inQueueSrcVecOut.FreeTensor(dstLocalVecOut);
        inQueueSrcUB.FreeTensor(dstLocal);
    }
 
private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrcUB;
    TQue<TPosition::A1, 1> inQueueSrcL1;
    TQue<TPosition::VECOUT, 1> inQueueSrcVecOut;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    Nd2NzParams intriParams;
};
} // namespace AscendC
 
template <typename T>
__global__ __aicore__ void MainDataCopyGm2UBNd2Nz(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
    Nd2NzParams& intriParams)
{
    KernelDataCopyGm2UBNd2Nz<T> op;
    op.Init(dstGm, srcGm, intriParams);
    op.Process();
}
 
struct DataCopyGm2UBNd2NzTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, Nd2NzParams&);
    Nd2NzParams intriParams;
};
 
class DataCopyGm2UBNd2NzTestsuite : public testing::Test,
    public testing::WithParamInterface<DataCopyGm2UBNd2NzTestParams> {
protected:
    void SetUp() {
        SetGCoreType(2);
        BufIdTracker::GetInstance().Reset();
    }
    void TearDown() {
        SetGCoreType(1);
    }
};
 
INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DATACOPYGM2UBND2NZ, DataCopyGm2UBNd2NzTestsuite,
    ::testing::Values(DataCopyGm2UBNd2NzTestParams { 2, MainDataCopyGm2UBNd2Nz<half>, { 2, 2, 13, 48, 16, 11, 2, 48 } },
    DataCopyGm2UBNd2NzTestParams { 4, MainDataCopyGm2UBNd2Nz<float>, { 2, 2, 13, 48, 16, 11, 2, 40 } }));
 
TEST_P(DataCopyGm2UBNd2NzTestsuite, DataCopyGm2UBNd2NzTestsuiteOpTestCase)
{
    auto param = GetParam();
    Nd2NzParams intriParams = param.intriParams;
    uint8_t srcGm[intriParams.ndNum * intriParams.srcNdMatrixStride * param.typeSize] = {0};
    uint8_t dstGm[((intriParams.dValue * param.typeSize -1 ) / 32 + 1) * intriParams.dstNzC0Stride * 32] = {0};
 
    param.cal_func(dstGm, srcGm, intriParams);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
 
namespace AscendC {
// T, U for gm->l1 nd2nz
template <typename T> class KernelDataCopyGm2L1 {
public:
    __aicore__ inline KernelDataCopyGm2L1() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, Nd2NzParams& intriParamsIn)
    {
        intriParams = intriParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrcL1, 1,
            (((intriParams.dValue * sizeof(T) - 1) / 32 + 1) * intriParams.dstNzC0Stride * 32));
    }
    __aicore__ inline void Process()
    {
        LocalTensor<T> srcLocal = inQueueSrcL1.AllocTensor<T>();
        DataCopyParams params(1, 128, 0, 0);
        DataCopyPadParams padParams;
        DataCopyPad(srcLocal, srcGlobal, params, padParams);
 
        DataCopyExtParams params2(1, 128, 0, 0, 0);
        DataCopyPadExtParams<T> padParams2;
        DataCopyPad(srcLocal, srcGlobal, params2, padParams2);
    }
 
private:
    TPipe pipe;
    TQue<TPosition::A1, 1> inQueueSrcL1;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    Nd2NzParams intriParams;
};
} // namespace AscendC
 
template <typename T>
__global__ __aicore__ void MainDataCopyGm2L1(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
    Nd2NzParams& intriParams)
{
    KernelDataCopyGm2L1<T> op;
    op.Init(dstGm, srcGm, intriParams);
    op.Process();
}
 
struct DataCopyGm2L1TestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, Nd2NzParams&);
    Nd2NzParams intriParams;
};
 
class DataCopyGm2L1Testsuite : public testing::Test,
    public testing::WithParamInterface<DataCopyGm2L1TestParams> {
protected:
    void SetUp() {
        SetGCoreType(1);
        BufIdTracker::GetInstance().Reset();
    }
    void TearDown() {
        SetGCoreType(1);
    }
};
 
INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DATACOPYGM2L1, DataCopyGm2L1Testsuite,
    ::testing::Values(DataCopyGm2L1TestParams { 2, MainDataCopyGm2L1<half>, { 2, 2, 13, 48, 16, 11, 2, 48 } },
    DataCopyGm2L1TestParams { 4, MainDataCopyGm2L1<float>, { 2, 2, 13, 48, 16, 11, 2, 40 } },
    DataCopyGm2L1TestParams { 1, MainDataCopyGm2L1<fp8_e5m2_t>, { 2, 2, 13, 96, 32, 11, 2, 96 } },
    DataCopyGm2L1TestParams { 1, MainDataCopyGm2L1<fp8_e4m3fn_t>, { 2, 2, 13, 96, 32, 11, 2, 96 } }
    ));
 
TEST_P(DataCopyGm2L1Testsuite, DataCopyGm2L1TestsuiteOpTestCase)
{
    auto param = GetParam();
    Nd2NzParams intriParams = param.intriParams;
    uint8_t srcGm[intriParams.ndNum * intriParams.srcNdMatrixStride * param.typeSize] = {0};
    uint8_t dstGm[((intriParams.dValue * param.typeSize -1 ) / 32 + 1) * intriParams.dstNzC0Stride * 32] = {0};
 
    param.cal_func(dstGm, srcGm, intriParams);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}