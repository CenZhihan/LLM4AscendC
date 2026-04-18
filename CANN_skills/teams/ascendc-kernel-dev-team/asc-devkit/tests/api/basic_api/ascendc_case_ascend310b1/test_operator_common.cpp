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

enum CommonFunc {
    BLOCKIDX,
    BLOCKNUM,
};

void TestCommon(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint32_t dataSize, CommonFunc commonFunc)
{
    TPipe tpipe;
    GlobalTensor<int32_t> outputGlobal;
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(dstGm), dataSize);


    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, dataSize * sizeof(int32_t));
    LocalTensor<int32_t> outputLocal = tbuf.Get<int32_t>();

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    int32_t result;
    if (commonFunc == CommonFunc::BLOCKIDX) {
        result = GetBlockIdx();
    } else if (commonFunc == CommonFunc::BLOCKNUM) {
        result = GetBlockNum();
    }
    Duplicate(outputLocal, result, dataSize);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, dataSize);
    pipe_barrier(PIPE_ALL);
}

struct CommonTestParams {
    uint32_t dataSize;
    int32_t dstDatabitSize;
    CommonFunc commonFunc;
    void (*cal_func)(uint8_t*, uint32_t, CommonFunc);
};

class CommonTestsuite : public testing::Test, public testing::WithParamInterface<CommonTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(CommonTestCase, CommonTestsuite,
    ::testing::Values(
        CommonTestParams { 256, 4, CommonFunc::BLOCKIDX, TestCommon },
        CommonTestParams { 256, 4, CommonFunc::BLOCKNUM, TestCommon }
    ));

TEST_P(CommonTestsuite, CommonTestCase)
{
    TPipe tpipe;
    auto param = GetParam();
    uint8_t dstGm[param.dataSize * param.dstDatabitSize];

    param.cal_func(dstGm, param.dataSize, param.commonFunc);
    for (int32_t i = 0; i < param.dataSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}