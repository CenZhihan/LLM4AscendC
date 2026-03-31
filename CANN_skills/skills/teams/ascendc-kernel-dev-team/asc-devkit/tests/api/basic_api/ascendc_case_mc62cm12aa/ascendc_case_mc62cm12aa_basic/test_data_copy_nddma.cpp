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
#include <type_traits>
#include "kernel_operator.h"
#include "mockcpp/mockcpp.hpp"

using namespace std;
using namespace AscendC;

struct NDDMATestParams;

template <typename T, uint8_t dim, bool enableNearest, bool disablePad = true, bool optimized = false>
void DataCopyNDDMAOp(__gm__ uint8_t *__restrict__ dstGm, __gm__ uint8_t *__restrict__ srcGm,
    const NDDMATestParams &param)
{
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;

    MultiCopyLoopInfo<dim> loopInfo;

    uint32_t srcSize = param.srcSize;
    uint32_t dstSize = param.dstSize;
    for (size_t i = 0; i < dim; i++) {
        loopInfo.loopSize[i] = param.loopSize[i];
        loopInfo.loopSrcStride[i] = param.loopSrcStride[i];
        loopInfo.loopDstStride[i] = param.loopDstStride[i];
        loopInfo.loopLpSize[i] = param.loopLpSize[i];
        loopInfo.loopRpSize[i] = param.loopRpSize[i];
    };
    T constValue = sizeof(T) == 8 ? T{ 0 } : T{ 1 };
    MultiCopyParams<T, dim> params = { loopInfo, constValue};

    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), srcSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), dstSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dstSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf.Get<T>();

    constexpr uint16_t padValue = disablePad ? NdDmaConfig::unsetPad : 0;

    static constexpr NdDmaConfig config = { enableNearest, padValue, padValue, optimized };
    DataCopy<T, dim, config>(outputLocal, inputGlobal, params);

    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dstSize);
    pipe_barrier(PIPE_ALL);
}

struct NDDMATestParams {
    uint32_t srcSize;
    uint32_t dstSize;
    uint32_t loopSize[5];
    uint64_t loopSrcStride[5];
    uint32_t loopDstStride[5];
    uint8_t loopLpSize[5];
    uint8_t loopRpSize[5];
    void (*cal_func)(uint8_t *, uint8_t *, const NDDMATestParams &);
};

class DataCopyNDDMATtest : public testing::Test, public testing::WithParamInterface<NDDMATestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(DataCopyNDDMATtest, DataCopyNDDMATtest,
    ::testing::Values(NDDMATestParams{ 30 * 120 * 4,
    32 * 128 * 4,
    { 30, 120 },
    { 1, 30 },
    { 1, 32 },
    { 1, 3 },
    { 1, 5 },
    DataCopyNDDMAOp<int32_t, 2, true> },
    NDDMATestParams{ 30 * 120 * 4,
    32 * 128 * 4,
    { 30, 120 },
    { 1, 30 },
    { 1, 32 },
    { 1, 3 },
    { 1, 5 },
    DataCopyNDDMAOp<uint32_t, 2, false> },
    NDDMATestParams{ 4 * 15 * 13 * 4,
    2 * 8 * 10 * 4,
    { 10, 8, 2 },
    { 1, 13, 13 * 15 },
    { 1, 10, 8 * 10 },
    { 0 },
    { 0 },
    DataCopyNDDMAOp<float, 3, true> },
    NDDMATestParams{ 4 * 15 * 13 * 2,
    2 * 8 * 10 * 2,
    { 10, 8, 2 },
    { 1, 13, 13 * 15 },
    { 1, 10, 8 * 10 },
    { 0 },
    { 0 },
    DataCopyNDDMAOp<half, 3, false> },
    NDDMATestParams{ 10 * 8 * 6 * 4 * 8,
    10 * 8 * 6 * 4 * 8,
    { 10, 8, 6, 4, 1 },
    { 4 * 6 * 8, 4 * 6, 4, 1, 1 },
    { 1, 10, 10 * 8, 10 * 8 * 6, 10 * 8 * 6 * 4 },
    { 0 },
    { 0 },
    DataCopyNDDMAOp<uint64_t, 5, false> },
    NDDMATestParams{ 10 * 8 * 6 * 4 * 8,
    10 * 8 * 6 * 4 * 8,
    { 10, 8, 6, 4, 1 },
    { 4 * 6 * 8, 4 * 6, 4, 1, 1 },
    { 1, 10, 10 * 8, 10 * 8 * 6, 10 * 8 * 6 * 4 },
    { 0 },
    { 0 },
    DataCopyNDDMAOp<uint64_t, 5, false, true, false> },
    NDDMATestParams{ 10 * 8 * 6 * 4 * 8,
    10 * 8 * 6 * 4 * 8,
    { 10, 8, 6, 4, 1 },
    { 4 * 6 * 8, 4 * 6, 4, 1, 1 },
    { 1, 10, 10 * 8, 10 * 8 * 6, 10 * 8 * 6 * 4 },
    { 0 },
    { 0 },
    DataCopyNDDMAOp<uint64_t, 5, false, false> },
    NDDMATestParams{ 8 * 16, 16 * 8, { 8, 16 }, { 16, 1 }, { 1, 8 }, {}, {}, DataCopyNDDMAOp<int8_t, 2, true> },
    NDDMATestParams{ 8 * 16, 16 * 8, { 8, 16 }, { 16, 1 }, { 1, 8 }, {}, {}, DataCopyNDDMAOp<uint8_t, 2, true> },
    NDDMATestParams{ 16 * 2, 4 * 16 * 2, { 16, 4 }, { 1, 0 }, { 1, 16 }, {}, {}, DataCopyNDDMAOp<int16_t, 2, true> },
    NDDMATestParams{ 16 * 2, 4 * 16 * 2, { 16, 4 }, { 1, 0 }, { 1, 16 }, {}, {}, DataCopyNDDMAOp<uint16_t, 2, true> }));

TEST_P(DataCopyNDDMATtest, DataCopyNDDMATestCase)
{
    TPipe tpipe;
    auto param = GetParam();
    uint8_t* srcGm = new uint8_t[param.srcSize * 4 * 10] {0};
    uint8_t* dstGm = new uint8_t[param.dstSize * 4 * 10] {0};
    param.cal_func(dstGm, srcGm, param);
    for (int32_t i = 0; i < param.dstSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
    delete [] srcGm;
    delete [] dstGm;
}

class DataCopyL12UBTestSuite : public ::testing::Test {
protected:
    virtual void SetUp()
    {
        SetGCoreType(2);
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        SetGCoreType(0);
    }
};
 
TEST_F(DataCopyL12UBTestSuite, testCaseDataCopyL12UB)
{
    LocalTensor<uint16_t> srcGm;
    LocalTensor<uint16_t> dstGm;
    DataCopyParams params;
    MOCKER(raise).stubs().will(returnValue(int(0)));
    DataCopyL12UBIntf(dstGm, srcGm, params);
}