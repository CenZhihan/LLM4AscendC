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
#include <mockcpp/mockcpp.hpp>
#include "kernel_operator.h"
// #include "api_check/kernel_cpu_check.h"

using namespace AscendC;
#define ASCENDC_DUMP
struct TestDumpTensorParams {
    uint32_t dataSize;
    uint32_t typeSize;
    uint32_t dstSize;
    uint64_t dumpSize;
    void (*cal_func)(uint8_t *, uint8_t *, uint8_t *, uint32_t, uint32_t, uint64_t);
};

class TestDumpTensorSuite : public testing::Test, public testing::WithParamInterface<TestDumpTensorParams> {
protected:
    void SetUp()
    {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

template <typename T>
void DataDumpTensor(__gm__ uint8_t *dstGm, __gm__ uint8_t *srcGm, __gm__ uint8_t *workGm, __gm__ uint32_t dataSize,
    __gm__ uint32_t dstSize, __gm__ uint64_t dumpSize)
{
    TPipe tpipe;
    uint32_t desc = 630;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    GlobalTensor<T> workGlobal;

    srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), dataSize);
    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), dstSize);
    workGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workGm), DUMP_WORKSPACE_SIZE * 2);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();
    DataCopyParams copyParams{1, dataSize * sizeof(T) / 32, 0, 0};
    DataCopy(inputLocal, srcGlobal, copyParams);
    pipe_barrier(PIPE_ALL);

    InitDump(workGm, dumpSize);
    DumpTensor(inputLocal, 0, dataSize);
    PRINTF("AAAA %d", 0x1);
}

INSTANTIATE_TEST_CASE_P(TEST_DUMP_TENSOR, TestDumpTensorSuite,
    ::testing::Values(TestDumpTensorParams{64, 4, 96, 96 * 3, DataDumpTensor<uint32_t>}));

int32_t RaiseStubForDumpTensor(int32_t input)
{
    return 0;
}

TEST_P(TestDumpTensorSuite, TestDumpTensorCases)
{
    auto param = GetParam();
    int32_t tmp = g_coreType;
    g_coreType = 2;

    uint8_t srcGm[param.dataSize * param.typeSize];
    uint8_t dstGm[param.dstSize * param.typeSize];
    uint8_t workGm[param.dumpSize * param.typeSize];
    MOCKER(raise, int32_t (*)(int32_t)).stubs().will(invoke(RaiseStubForDumpTensor));

    param.cal_func(dstGm, srcGm, workGm, param.dataSize, param.dstSize, param.dumpSize);

    for (int i = 0; i < 96; i++) {
        printf("* %d dstGm: %d\n ", i, dstGm[i]);
    }
    for (int i = 0; i < 20; i++) {
        printf("* %d srcGm: %d\n ", i, srcGm[i]);
    }
    for (int i = 0; i < 112; i++) {
        printf("* %d workGm: %d\n ", i, workGm[i]);
    }
    g_coreType = tmp;
}