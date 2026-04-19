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
// #include "model/model_factory.h"
#include "common.h"

using namespace std;
using namespace AscendC;


class TEST_DUPLICATE : public testing::Test {
protected:
    void SetUp() {
        g_coreType = AscendC::AIV_TYPE;
    }
    void TearDown() {
        g_coreType = AscendC::MIX_TYPE;
    }
};

template <typename T> __global__ __aicore__ void MainDuplicate(__gm__ uint8_t* __restrict__ dstGm, int32_t dataSize)
{
    TPipe tpipe;
    GlobalTensor<T> outputGlobal;
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf.Get<T>();

    T scalarValue(0);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Duplicate(inputLocal, scalarValue, dataSize);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, inputLocal, dataSize);

    pipe_barrier(PIPE_ALL);
}

struct DuplicateTestParams {
    int32_t dataSize;
    int32_t dstTypeBytes;
    uint64_t finalMaskHigh;
    uint64_t finalMaskLow;
    void (*CalFunc)(uint8_t*, int32_t);
};

class DuplicateTestsuite : public testing::Test, public testing::WithParamInterface<DuplicateTestParams> {
protected:
    void SetUp() {
        g_coreType = AscendC::AIV_TYPE;
    }
    void TearDown() {
        g_coreType = AscendC::MIX_TYPE;
    }
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DUPLICATE, DuplicateTestsuite,
    ::testing::Values(
    DuplicateTestParams { 32, sizeof(bfloat16_t), 0, GenerateMask(32), MainDuplicate<bfloat16_t> },
    DuplicateTestParams { 288, sizeof(half), 0, GenerateMask(32), MainDuplicate<half> }));

TEST_P(DuplicateTestsuite, DuplicateLevel2TestCase)
{
    auto param = GetParam();
    uint8_t dstGm[param.dataSize * param.dstTypeBytes] = {0};

    param.CalFunc(dstGm, param.dataSize);
    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
