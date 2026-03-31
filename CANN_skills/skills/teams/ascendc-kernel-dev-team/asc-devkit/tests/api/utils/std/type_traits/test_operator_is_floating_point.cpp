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

class IsFloatingPointTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IsFloatingPointTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IsFloatingPointTest TearDownTestCase" << std::endl;
    }
};

// Testing Floating Point Types
TEST_F(IsFloatingPointTest, FloatingPointTypes) {
    EXPECT_TRUE(AscendC::Std::is_floating_point_v<float>);
    EXPECT_TRUE(AscendC::Std::is_floating_point_v<double>);
    EXPECT_TRUE(AscendC::Std::is_floating_point_v<long double>);
    EXPECT_TRUE(AscendC::Std::is_floating_point_v<half>);
}

// Testing Non-Floating Point Types
TEST_F(IsFloatingPointTest, NonFloatingPointTypes) {
    EXPECT_FALSE(AscendC::Std::is_floating_point_v<int>);
    EXPECT_FALSE(AscendC::Std::is_floating_point_v<char>);
    EXPECT_FALSE(AscendC::Std::is_floating_point_v<bool>);
    EXPECT_FALSE(AscendC::Std::is_floating_point_v<std::string>);
}

// Testing CV Restricted Types (const/volatile)
TEST_F(IsFloatingPointTest, CVQualifiedTypes) {
    EXPECT_TRUE(AscendC::Std::is_floating_point_v<const float>);
    EXPECT_TRUE(AscendC::Std::is_floating_point_v<volatile double>);
    EXPECT_TRUE(AscendC::Std::is_floating_point_v<const volatile long double>);
    EXPECT_TRUE(AscendC::Std::is_floating_point_v<const volatile half>);
}

// Testing Non Floating Point CV Restricted Types (const/volatile)
TEST_F(IsFloatingPointTest, CVQualifiedNonFloatingPointTypes) {
    EXPECT_FALSE(AscendC::Std::is_floating_point_v<const int>);
    EXPECT_FALSE(AscendC::Std::is_floating_point_v<volatile char>);
    EXPECT_FALSE(AscendC::Std::is_floating_point_v<const volatile bool>);
    EXPECT_FALSE(AscendC::Std::is_floating_point_v<const volatile std::string>);
}