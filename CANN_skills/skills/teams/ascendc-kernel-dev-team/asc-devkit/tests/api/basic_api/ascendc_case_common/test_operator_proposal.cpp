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
#define private public
#define protected public
#include "kernel_operator.h"
#include "kernel_utils.h"
// #include "model/model_factory.h"
#include "common.h"

using namespace std;
using namespace AscendC;

#define LOCAL_TENSOR_REGISTER(tensorName, type, que_pos, initAddr, dataSize) \
    LocalTensor<type> tensorName;                                              \
    TBuffAddr tbuf_##tensorName;                                               \
    tbuf_##tensorName.logicPos = (uint8_t)(TPosition::que_pos);                \
    tensorName.SetAddr(tbuf_##tensorName);                                    \
    tensorName.InitBuffer(initAddr, dataSize);


template <typename T>
__global__ __aicore__ void MainMrgSort4(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm)
{
    TPipe tpipe;
    uint32_t srcDataSize = 64;
    uint16_t singleDataSize = srcDataSize / 4;
    uint32_t dstDataSize = srcDataSize / 4 * 8;
    uint32_t vmrgsortDstDataSize = srcDataSize / 4 * 8 * 4;

    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

    int repeat = srcDataSize / 4 / 16;
    GlobalTensor<T> srcGlobal1;
    GlobalTensor<T> srcGlobal2;
    GlobalTensor<T> srcGlobal3;
    GlobalTensor<T> srcGlobal4;
    GlobalTensor<T> dstGlobal;
    srcGlobal1.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(srcGm), singleDataSize);
    srcGlobal2.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(srcGm + singleDataSize * 1 * sizeof(PrimT<T>)),
        singleDataSize);
    srcGlobal3.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(srcGm + singleDataSize * 2 * sizeof(PrimT<T>)),
        singleDataSize);
    srcGlobal4.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(srcGm + singleDataSize * 3 * sizeof(PrimT<T>)),
        singleDataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(dstGm), vmrgsortDstDataSize);

    LOCAL_TENSOR_REGISTER(srcUb1, T, CO2, 0, singleDataSize)
    LOCAL_TENSOR_REGISTER(srcUb2, T, CO2, singleDataSize * sizeof(PrimT<T>), singleDataSize)
    LOCAL_TENSOR_REGISTER(srcUb3, T, CO2, (singleDataSize * 2) * sizeof(PrimT<T>), singleDataSize)
    LOCAL_TENSOR_REGISTER(srcUb4, T, CO2, (singleDataSize * 3) * sizeof(PrimT<T>), singleDataSize)

    LOCAL_TENSOR_REGISTER(vconcatDstUb1, T, CO2, srcDataSize * sizeof(PrimT<T>), dstDataSize)
    LOCAL_TENSOR_REGISTER(vconcatDstUb2, T, CO2, (srcDataSize + dstDataSize) * sizeof(PrimT<T>), dstDataSize)
    LOCAL_TENSOR_REGISTER(vconcatDstUb3, T, CO2, (srcDataSize + dstDataSize * 2) * sizeof(PrimT<T>), dstDataSize)
    LOCAL_TENSOR_REGISTER(vconcatDstUb4, T, CO2, (srcDataSize + dstDataSize * 3) * sizeof(PrimT<T>), dstDataSize)
    LOCAL_TENSOR_REGISTER(vmrgsortDstUb, T, CO2, (srcDataSize + dstDataSize * 4) * sizeof(PrimT<T>),
        vmrgsortDstDataSize)
    if constexpr (IsSameType<PrimT<T>, T>::value) {
        PrimT<T> zero(0);
        Duplicate(vconcatDstUb1, zero, dstDataSize);
        Duplicate(vconcatDstUb2, zero, dstDataSize);
        Duplicate(vconcatDstUb3, zero, dstDataSize);
        Duplicate(vconcatDstUb4, zero, dstDataSize);
        Duplicate(vmrgsortDstUb, zero, vmrgsortDstDataSize);
    }
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    DataCopy(srcUb1, srcGlobal1, srcDataSize / 4);
    DataCopy(srcUb2, srcGlobal2, srcDataSize / 4);
    DataCopy(srcUb3, srcGlobal3, srcDataSize / 4);
    DataCopy(srcUb4, srcGlobal4, srcDataSize / 4);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    ProposalConcat(vconcatDstUb1, srcUb1, repeat, 4);
    pipe_barrier(PIPE_V);
    RpSort16(vconcatDstUb1, vconcatDstUb1, repeat);
    pipe_barrier(PIPE_V);

    ProposalConcat(vconcatDstUb2, srcUb2, repeat, 4);
    pipe_barrier(PIPE_V);
    RpSort16(vconcatDstUb2, vconcatDstUb2, repeat);
    pipe_barrier(PIPE_V);

    ProposalConcat(vconcatDstUb3, srcUb3, repeat, 4);
    pipe_barrier(PIPE_V);
    RpSort16(vconcatDstUb3, vconcatDstUb3, repeat);
    pipe_barrier(PIPE_V);

    ProposalConcat(vconcatDstUb4, srcUb4, repeat, 4);
    pipe_barrier(PIPE_V);
    RpSort16(vconcatDstUb4, vconcatDstUb4, repeat);
    pipe_barrier(PIPE_V);

    struct MrgSortSrcList<T> srcList(vconcatDstUb1, vconcatDstUb2, vconcatDstUb3, vconcatDstUb4);
    uint16_t elementLengths[4] = {singleDataSize, singleDataSize, singleDataSize, singleDataSize};
    struct MrgSort4Info srcInfo(elementLengths, false, 15, 1);
    MrgSort4(vmrgsortDstUb, srcList, srcInfo);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(dstGlobal, vmrgsortDstUb, vmrgsortDstDataSize);

    pipe_barrier(PIPE_ALL);
}

struct MrgSort4TestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*);
};

class MrgSort4Testsuite : public testing::Test, public testing::WithParamInterface<MrgSort4TestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_MRGSORT4, MrgSort4Testsuite,
    ::testing::Values(
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201) // dav-m200
    MrgSort4TestParams { 4, MainMrgSort4<float> },
    MrgSort4TestParams { 4, MainMrgSort4<TensorTrait<float>> },
#endif
    MrgSort4TestParams { 2, MainMrgSort4<half> },
    MrgSort4TestParams { 2, MainMrgSort4<TensorTrait<half>> }
    ));

TEST_P(MrgSort4Testsuite, MrgSortOpTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[64 * param.typeSize];
    uint8_t dstGm[512 * param.typeSize];

    param.cal_func(dstGm, srcGm);
    for (int32_t i = 0; i < 512 * param.typeSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

template <typename T>
__global__ __aicore__ void MainRpsort16(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm)
{
    TPipe tpipe;
    int32_t dataSize = 128;

    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(srcGm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(dstGm), dataSize);

    TBuf<TPosition::CO2> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(PrimT<T>));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    TBuf<TPosition::CO2> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(PrimT<T>));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    DataCopy(inputLocal, inputGlobal, dataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    int32_t repeat = 1;
    RpSort16<T>(outputLocal, inputLocal, repeat);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);

    pipe_barrier(PIPE_ALL);
}

struct RpSort16TestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*);
};

class RpSort16Testsuite : public testing::Test, public testing::WithParamInterface<RpSort16TestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_RPSORT16, RpSort16Testsuite,
    ::testing::Values(
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201) // dav-m200
    RpSort16TestParams { 4, MainRpsort16<float> },
    RpSort16TestParams { 4, MainRpsort16<TensorTrait<float>> },
#endif
    RpSort16TestParams { 2, MainRpsort16<half> },
    RpSort16TestParams { 2, MainRpsort16<TensorTrait<half>> }
    ));

TEST_P(RpSort16Testsuite, RpSort16OpTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[128 * param.typeSize];
    uint8_t dstGm[128 * param.typeSize];

    param.cal_func(dstGm, srcGm);
    for (int32_t i = 0; i < 128 * param.typeSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

template <typename T>
__global__ __aicore__ void MainConcatTest(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm,
    __gm__ int32_t srcDataSize, __gm__ int32_t dstDataSize, __gm__ int32_t mode)
{
    TPipe tpipe;
    int32_t repeat = srcDataSize / 16;

    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(srcGm), srcDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(dstGm), dstDataSize);

    TBuf<TPosition::CO2> tbuf;
    tpipe.InitBuffer(tbuf, srcDataSize * sizeof(PrimT<T>));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    TBuf<TPosition::CO2> tbuf1;
    tpipe.InitBuffer(tbuf1, dstDataSize * sizeof(PrimT<T>));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    DataCopy(inputLocal, inputGlobal, srcDataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    ProposalConcat(outputLocal, inputLocal, repeat, mode);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    pipe_barrier(PIPE_ALL);
}

template <typename T>
__global__ __aicore__ void MainExtractTest(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ srcGm,
    __gm__ int32_t srcDataSize, __gm__ int32_t dstDataSize, __gm__ int32_t mode)
{
    TPipe tpipe;
    int32_t repeat = srcDataSize / 128;

    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(srcGm), srcDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ PrimT<T>*>(dstGm), dstDataSize);

    TBuf<TPosition::CO2> tbuf;
    tpipe.InitBuffer(tbuf, srcDataSize * sizeof(PrimT<T>));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    TBuf<TPosition::CO2> tbuf1;
    tpipe.InitBuffer(tbuf1, dstDataSize * sizeof(PrimT<T>));
    LocalTensor<T> outputLocal = tbuf1.Get<T>();

    DataCopy(inputLocal, inputGlobal, srcDataSize);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    ProposalExtract(outputLocal, inputLocal, repeat, mode);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dstDataSize);

    pipe_barrier(PIPE_ALL);
}
struct ProposalTestParams {
    int32_t typeSize;
    void (*calFunc)(uint8_t*, uint8_t*, int32_t, int32_t, int32_t);
    int32_t srcDataSize;
    int32_t dstDataSize;
    int32_t modeNumber;
};
class ProposalTestsuite : public testing::Test, public testing::WithParamInterface<ProposalTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_PROPOSAL, ProposalTestsuite,
    ::testing::Values(ProposalTestParams { 2, MainConcatTest<half>, 32, 256, 5 },
    ProposalTestParams { 2, MainConcatTest<half>, 16, 128, 3 },
    ProposalTestParams { 2, MainExtractTest<half>, 128, 16, 3 },
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201) // dav-m200
    ProposalTestParams { 4, MainConcatTest<float>, 16, 128, 4 },
    ProposalTestParams { 4, MainExtractTest<float>, 256, 32, 4 },
#endif
    ProposalTestParams { 2, MainExtractTest<half>, 256, 32, 4 },
    // TensorTrait
    ProposalTestParams { 2, MainConcatTest<TensorTrait<half>>, 16, 128, 3 },
    ProposalTestParams { 2, MainExtractTest<TensorTrait<half>>, 128, 16, 3 }));

TEST_P(ProposalTestsuite, ProposalApiTestCase)
{
    auto param = GetParam();
    uint8_t srcGm[param.srcDataSize * param.typeSize];
    uint8_t dstGm[param.dstDataSize * param.typeSize];

    param.calFunc(dstGm, srcGm, param.srcDataSize, param.dstDataSize, param.modeNumber);
    for (int32_t i = 0; i < param.dstDataSize * param.typeSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}