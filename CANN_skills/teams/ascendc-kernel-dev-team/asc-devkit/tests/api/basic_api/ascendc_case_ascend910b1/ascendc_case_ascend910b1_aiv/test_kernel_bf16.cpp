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
#include <math.h>
using namespace std;
using namespace AscendC;

struct TestBf16Params {
    float dataIn1;
    float dataOut1;
    uint32_t dataIn2;
    uint32_t dataOut2;
    uint16_t dataIn5;
    uint16_t dataIn6;
    uint16_t dataOut4;
    bfloat16_t dataIn3;
    bfloat16_t dataIn4;
    bfloat16_t dataOut3;
};

class TestBf16Suite : public testing::Test, public testing::WithParamInterface<TestBf16Params> {
    protected:
    // 每一个case之前
    void SetUp() {}
    // 每一个case之后
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_BF16, TestBf16Suite,
    ::testing::Values(TestBf16Params { 1, 1, 0x7F80, 0x7F800000, 0x7F80, 0, 0x7C00, 1, 1, 2},
    TestBf16Params { 1, 1, 0x7F80, 0x7F800000, 0, 0x7F80, 0x7C00, 1, 1, 2},
    TestBf16Params { 3.4028236692093846346337460743177e+38, INFINITY, 0x7FFF, 0x7FFFFFFF, 0x7FFF, 0, 0x7FFF, 2.30e+38, 1.0e+38, 3.30e+38},
    TestBf16Params { 64, 64, 0, 0, 0, 0x7FFF, 0x7FFF, -1, 1, 0},
    TestBf16Params { -5, -5, 0, 0, 0x03, 0x04, 0x07, 0, 0, 0}));

TEST_P(TestBf16Suite, TestBf16Cases)
{
    auto param = GetParam();
    bfloat16_t m = param.dataIn1;
    float res = m.ToFloat();
    EXPECT_EQ(res, param.dataOut1);

    bfloat16_t n = param.dataIn1;
    EXPECT_EQ(n, param.dataOut1);

    bfloat16_t s;
    s.val = param.dataIn2;
    float sFloat = s.ToFloat();
    uint32_t tmp = *(reinterpret_cast<uint32_t *>(&sFloat));
    EXPECT_EQ(tmp, param.dataOut2);

    bfloat16_t sum0 = param.dataIn3 + param.dataIn4;
    EXPECT_EQ(sum0, param.dataOut3);
    bfloat16_t sum1 = param.dataIn3;
    sum1 += param.dataIn4;
    EXPECT_EQ(sum1, param.dataOut3);
    sum1.val = param.dataIn5;
    sum0.val = param.dataIn6;
    sum1 += sum0;
    tmp = sum1.val;
    EXPECT_EQ(tmp, param.dataOut4);
}