/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "test_symbol_override.h"

using namespace std;
using namespace AscendC;

INSTANTIATE_TEST_CASE_P(TEST_SYMBOL_OVERRIDE, SymbolOverrideTestsuite,
    ::testing::Values(SymbolOverrideTestParams { 192, 4, test_symbol_override_add_kernel<int32_t> },
    SymbolOverrideTestParams { 192, 4, test_symbol_override_add_kernel<float> },
    SymbolOverrideTestParams { 192, 2, test_symbol_override_add_kernel<half> },
    SymbolOverrideTestParams { 192, 4, test_symbol_override_div_kernel<float> },
    SymbolOverrideTestParams { 192, 2, test_symbol_override_div_kernel<half> },
    SymbolOverrideTestParams { 192, 2, test_symbol_override_or_kernel<int16_t> },
    SymbolOverrideTestParams { 192, 2, test_symbol_override_or_kernel<uint16_t> },
    SymbolOverrideTestParams { 192, 4, test_symbol_override_compare_gt_kernel<float> },
    SymbolOverrideTestParams { 256, 2, test_symbol_override_compare_gt_kernel<half> },
    SymbolOverrideTestParams { 192, 4, test_symbol_override_compare_ge_kernel<float> },
    SymbolOverrideTestParams { 256, 2, test_symbol_override_compare_ge_kernel<half> },
    SymbolOverrideTestParams { 192, 4, test_symbol_override_compare_le_kernel<float> },
    SymbolOverrideTestParams { 256, 2, test_symbol_override_compare_le_kernel<half> },
    SymbolOverrideTestParams { 192, 4, test_symbol_override_compare_eq_kernel<float> },
    SymbolOverrideTestParams { 256, 2, test_symbol_override_compare_eq_kernel<half> },
    SymbolOverrideTestParams { 192, 4, test_symbol_override_compare_kernel<float> },
    SymbolOverrideTestParams { 256, 2, test_symbol_override_compare_kernel<half> }));

TEST_P(SymbolOverrideTestsuite, SymbolOverrideTestCase)
{
    TPipe tpipe;
    auto param = GetParam();
    uint8_t src0Gm[param.data_size * param.type_bit_size];
    uint8_t src1Gm[param.data_size * param.type_bit_size];
    uint8_t dstGm[param.data_size * param.type_bit_size];

    param.cal_func(src0Gm, src1Gm, dstGm, param.data_size);
    for (int32_t i = 0; i < param.data_size; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}