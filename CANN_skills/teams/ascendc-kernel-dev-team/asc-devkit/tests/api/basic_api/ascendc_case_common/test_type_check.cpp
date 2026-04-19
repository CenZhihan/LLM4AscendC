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
#include <cstdint>
#include "kernel_utils.h"

namespace AscendC {

template <typename T>
void TestFuncSingleType() {
    static_assert(SupportType<T, int16_t, uint32_t, float>(), "it only supports int16_t, uint32_t, float type.");
}

template <typename T, typename U>
void TestFuncCompositeTypes() {
    static_assert(SupportType<Tuple<T, U>, Tuple<int8_t, half>, Tuple<float, int32_t>>(), "input types must be one of the pair amongs (int8_t, half), (float, int32_t).");
}

TEST(TestAscendCTypeCheck, testSingleType) {
    // MOCKER(TestFuncSingleType<int16_t>).expects(once());
    // MOCKER(TestFuncSingleType<uint32_t>).expects(once());
    // MOCKER(TestFuncSingleType<float>).expects(once());
    TestFuncSingleType<int16_t>();
    TestFuncSingleType<uint32_t>();
    TestFuncSingleType<float>();
    EXPECT_NO_THROW(GlobalMockObject::verify());
}

void TestCompositeTypesWrapper()
{
    TestFuncCompositeTypes<int8_t, half>();
    TestFuncCompositeTypes<float, int32_t>();
}

TEST(TestAscendCTypeCheck, testCompositeTypes) {
    MOCKER(TestCompositeTypesWrapper).expects(once());
    TestCompositeTypesWrapper();
    EXPECT_NO_THROW(GlobalMockObject::verify());
}

} // namespace AscendC