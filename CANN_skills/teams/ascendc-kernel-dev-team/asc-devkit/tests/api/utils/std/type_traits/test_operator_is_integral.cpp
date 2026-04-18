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

class IsIntegralTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IsIntegralTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IsIntegralTest TearDownTestCase" << std::endl;
    }
};

// Test the judgment of is_integral and is_integral_v for integer types
TEST_F(IsIntegralTest, CheckIntegralTypes) {
    EXPECT_EQ(AscendC::Std::is_integral<int>::value, true);
    EXPECT_EQ(AscendC::Std::is_integral<int>::value, AscendC::Std::is_integral_v<int>);

    EXPECT_EQ(AscendC::Std::is_integral<unsigned int>::value, true);
    EXPECT_EQ(AscendC::Std::is_integral<unsigned int>::value, AscendC::Std::is_integral_v<unsigned int>);

    EXPECT_EQ(AscendC::Std::is_integral<short>::value, true);
    EXPECT_EQ(AscendC::Std::is_integral<short>::value, AscendC::Std::is_integral_v<short>);

    EXPECT_EQ(AscendC::Std::is_integral<unsigned short>::value, true);
    EXPECT_EQ(AscendC::Std::is_integral<unsigned short>::value, AscendC::Std::is_integral_v<unsigned short>);

    EXPECT_EQ(AscendC::Std::is_integral<long>::value, true);
    EXPECT_EQ(AscendC::Std::is_integral<long>::value, AscendC::Std::is_integral_v<long>);

    EXPECT_EQ(AscendC::Std::is_integral<unsigned long>::value, true);
    EXPECT_EQ(AscendC::Std::is_integral<unsigned long>::value, AscendC::Std::is_integral_v<unsigned long>);

    EXPECT_EQ(AscendC::Std::is_integral<long long>::value, true);
    EXPECT_EQ(AscendC::Std::is_integral<long long>::value, AscendC::Std::is_integral_v<long long>);

    EXPECT_EQ(AscendC::Std::is_integral<unsigned long long>::value, true);
    EXPECT_EQ(AscendC::Std::is_integral<unsigned long long>::value, AscendC::Std::is_integral_v<unsigned long long>);

    EXPECT_EQ(AscendC::Std::is_integral<char>::value, true);
    EXPECT_EQ(AscendC::Std::is_integral<char>::value, AscendC::Std::is_integral_v<char>);

    EXPECT_EQ(AscendC::Std::is_integral<signed char>::value, true);
    EXPECT_EQ(AscendC::Std::is_integral<signed char>::value, AscendC::Std::is_integral_v<signed char>);

    EXPECT_EQ(AscendC::Std::is_integral<unsigned char>::value, true);
    EXPECT_EQ(AscendC::Std::is_integral<unsigned char>::value, AscendC::Std::is_integral_v<unsigned char>);

    EXPECT_EQ(AscendC::Std::is_integral<bool>::value, true);
    EXPECT_EQ(AscendC::Std::is_integral<bool>::value, AscendC::Std::is_integral_v<bool>);

    // Test const and volatile qualifiers
    EXPECT_EQ(AscendC::Std::is_integral<const int>::value, true);
    EXPECT_EQ(AscendC::Std::is_integral<const int>::value, AscendC::Std::is_integral_v<const int>);

    EXPECT_EQ(AscendC::Std::is_integral<volatile int>::value, true);
    EXPECT_EQ(AscendC::Std::is_integral<volatile int>::value, AscendC::Std::is_integral_v<volatile int>);

    EXPECT_EQ(AscendC::Std::is_integral<const volatile int>::value, true);
    EXPECT_EQ(AscendC::Std::is_integral<const volatile int>::value, AscendC::Std::is_integral_v<const volatile int>);
}

// Test the judgment of is_integral and is_integral_v for non-integer types
TEST_F(IsIntegralTest, CheckNonIntegralTypes) {
    EXPECT_EQ(AscendC::Std::is_integral<float>::value, false);
    EXPECT_EQ(AscendC::Std::is_integral<float>::value, AscendC::Std::is_integral_v<float>);

    EXPECT_EQ(AscendC::Std::is_integral<double>::value, false);
    EXPECT_EQ(AscendC::Std::is_integral<double>::value, AscendC::Std::is_integral_v<double>);

    EXPECT_EQ(AscendC::Std::is_integral<void>::value, false);
    EXPECT_EQ(AscendC::Std::is_integral<void>::value, AscendC::Std::is_integral_v<void>);

    EXPECT_EQ(AscendC::Std::is_integral<std::string>::value, false);
    EXPECT_EQ(AscendC::Std::is_integral<std::string>::value, AscendC::Std::is_integral_v<std::string>);

    // Test const and volatile qualifiers
    EXPECT_EQ(AscendC::Std::is_integral<const float>::value, false);
    EXPECT_EQ(AscendC::Std::is_integral<const float>::value, AscendC::Std::is_integral_v<const float>);

    EXPECT_EQ(AscendC::Std::is_integral<volatile float>::value, false);
    EXPECT_EQ(AscendC::Std::is_integral<volatile float>::value, AscendC::Std::is_integral_v<volatile float>);

    EXPECT_EQ(AscendC::Std::is_integral<const volatile float>::value, false);
    EXPECT_EQ(AscendC::Std::is_integral<const volatile float>::value, AscendC::Std::is_integral_v<const volatile float>);
}