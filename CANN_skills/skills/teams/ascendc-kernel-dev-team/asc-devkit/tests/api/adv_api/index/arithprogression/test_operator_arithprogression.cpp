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

using namespace std;
using namespace AscendC;

class TEST_arithprogression : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TEST_arithprogression SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "TEST_arithprogression TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

template <typename dataType>
__aicore__ inline void mainArithprogressionTest(GM_ADDR dstGm, uint32_t firstValue, uint32_t diffValue,
    uint32_t count)
{
    GlobalTensor<dataType> dstGlobal;

    dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType*>(dstGm), count);

    TPipe pipe;
    TQue<TPosition::VECOUT, 1> outDst;
    pipe.InitBuffer(outDst, 1, (sizeof(dataType) * count + 32 - 1) / 32 * 32);

    LocalTensor<dataType> dstLocal = outDst.AllocTensor<dataType>();

    PipeBarrier<PIPE_ALL>();
    ArithProgression<dataType>(dstLocal, (dataType)firstValue, (dataType)diffValue, count);

    PipeBarrier<PIPE_ALL>();
    constexpr int32_t BLOCK_NUM = 32 / sizeof(dataType);
    DataCopy(dstGlobal, dstLocal, (count + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM);
    outDst.FreeTensor(dstLocal);
    PipeBarrier<PIPE_ALL>();

}

struct arithprogressionTestParams {
    uint32_t firstValue;
    uint32_t diffValue;
    uint32_t count;
    uint32_t typeSize;
    void (*cal_func)(uint8_t *, uint32_t, uint32_t, uint32_t);
};

class arithprogressionTestSuite : public testing::Test, public testing::WithParamInterface<arithprogressionTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "arithprogressionTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "arithprogressionTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_arithprogression, arithprogressionTestSuite,
    ::testing::Values(
        arithprogressionTestParams { 1, 2, 15, sizeof(half), mainArithprogressionTest<half> },
        arithprogressionTestParams { 1, 2, 64, sizeof(half), mainArithprogressionTest<half> },
        arithprogressionTestParams { 1, 2, 260, sizeof(half), mainArithprogressionTest<half> }
        ));

TEST_P(arithprogressionTestSuite, arithprogressionTestCase)
{
    auto param = GetParam();
    uint8_t dstGm[param.count * param.typeSize] { 0x00 };
    param.cal_func(dstGm, param.firstValue, param.diffValue, param.count);

    for (int32_t i = 0; i < param.count * param.typeSize; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}