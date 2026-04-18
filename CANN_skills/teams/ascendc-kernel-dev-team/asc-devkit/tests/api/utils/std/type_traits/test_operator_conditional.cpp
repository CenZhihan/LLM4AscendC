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

class ConditionalTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "ConditionalTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ConditionalTest TearDownTestCase" << std::endl;
    }
};

// test AscendC::Std::conditional to select the first type when the boolean condition is true (int & double)
TEST_F(ConditionalTest, ConditionalTrueIntDouble) {
    constexpr bool condition = true;
    using ResultType = typename AscendC::Std::conditional<condition, int, double>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, int>::value));
}

// test AscendC::Std::conditional to select the second type when the boolean condition is false (int & double)
TEST_F(ConditionalTest, ConditionalFalseIntDouble) {
    constexpr bool condition = false;
    using ResultType = typename AscendC::Std::conditional<condition, int, double>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, double>::value));
}

// test AscendC::Std::conditional to select the first type when the boolean condition is true (short & long)
TEST_F(ConditionalTest, ConditionalTrueShortLong) {
    constexpr bool condition = true;
    using ResultType = typename AscendC::Std::conditional<condition, short, long>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, short>::value));
}

// test AscendC::Std::conditional to select the second type when the boolean condition is false (short & long)
TEST_F(ConditionalTest, ConditionalFalseShortLong) {
    constexpr bool condition = false;
    using ResultType = typename AscendC::Std::conditional<condition, short, long>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, long>::value));
}

// test AscendC::Std::conditional to select the first type when the boolean condition is true (char & float)
TEST_F(ConditionalTest, ConditionalTrueCharFloat) {
    constexpr bool condition = true;
    using ResultType = typename AscendC::Std::conditional<condition, char, float>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, char>::value));
}

// test AscendC::Std::conditional to select the second type when the boolean condition is false (char & float)
TEST_F(ConditionalTest, ConditionalFalseCharFloat) {
    constexpr bool condition = false;
    using ResultType = typename AscendC::Std::conditional<condition, char, float>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, float>::value));
}

// the test uses a constant expression function as a condition, and the result is true
constexpr bool constantTrueFunction() { return true; }
TEST_F(ConditionalTest, ConditionalTrueFromFunction) {
    using ResultType = typename AscendC::Std::conditional<constantTrueFunction(), int, long long>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, int>::value));
}

// the test uses a constant expression function as a condition, and the result is false
constexpr bool constantFalseFunction() { return false; }
TEST_F(ConditionalTest, ConditionalFalseFromFunction) {
    using ResultType = typename AscendC::Std::conditional<constantFalseFunction(), int, long long>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, long long>::value));
}

// the test uses a enum expression as a condition, and the result is true
enum class ConditionEnum { TrueValue = true, FalseValue = false };
TEST_F(ConditionalTest, ConditionalTrueFromEnum) {
    using ResultType = typename AscendC::Std::conditional<static_cast<bool>(ConditionEnum::TrueValue), short, double>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, short>::value));
}

// the test uses a enum expression as a condition, and the result is false
TEST_F(ConditionalTest, ConditionalFalseFromEnum) {
    using ResultType = typename AscendC::Std::conditional<static_cast<bool>(ConditionEnum::FalseValue), short, double>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, double>::value));
}

