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
#include "test_operator_vec_binary.h"

using namespace std;
using namespace AscendC;

INSTANTIATE_TEST_CASE_P(TEST_VEC_BINARY_SIMPLE, BinarySimpleTestsuite,
    ::testing::Values(BinaryTestParams { 256, 2, main_vec_binary_operator_simple_demo<half> },
    BinaryTestParams { 256, 4, main_vec_binary_operator_simple_demo<float> },
    BinaryTestParams { 256, 2, main_vec_binary_operator_simple_demo<int16_t> },
    BinaryTestParams { 256, 4, main_vec_binary_operator_simple_demo<int32_t> },
    BinaryTestParams { 256, 4, main_vec_binary_operator_div_demo<half> },
    BinaryTestParams { 256, 4, main_vec_binary_operator_div_demo<float> },
    BinaryTestParams { 256, 4, main_vec_binary_operator_and_or_demo<uint16_t> },
    BinaryTestParams { 256, 4, main_vec_binary_operator_and_or_demo<int16_t> },
    BinaryTestParams { 256, 4, main_vec_binary_operator_addrelu_demo<half> },
    BinaryTestParams { 256, 4, main_vec_binary_operator_addrelu_demo<float> },
    BinaryTestParams { 256, 4, main_vec_binary_operator_addrelu_demo<int16_t> },
    BinaryTestParams { 256, 4, main_vec_binary_operator_subrelu_demo<half> },
    BinaryTestParams { 256, 4, main_vec_binary_operator_subrelu_demo<float> },
    BinaryTestParams { 256, 4, main_vec_binary_operator_subrelu_demo<int16_t> }));

TEST_P(BinarySimpleTestsuite, BinarySimpleTestCase)
{
    TPipe tpipe;
    AscendC::SetGCoreType(2);
    auto param = GetParam();
    uint8_t src0Gm[param.data_size * param.data_bit_size];
    uint8_t src1Gm[param.data_size * param.data_bit_size];
    uint8_t dstGm[param.data_size * param.data_bit_size];

    param.cal_func(dstGm, src0Gm, src1Gm, param.data_size);
    for (int32_t i = 0; i < param.data_size; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
    AscendC::SetGCoreType(0);
}