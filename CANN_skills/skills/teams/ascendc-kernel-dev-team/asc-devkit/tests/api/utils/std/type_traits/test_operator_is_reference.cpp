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

class IsReferenceTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IsReferenceTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IsReferenceTest TearDownTestCase" << std::endl;
    }
};

// Test lvalue reference types
TEST_F(IsReferenceTest, LvalueReferenceTypes) {
    EXPECT_TRUE(AscendC::Std::is_lvalue_reference_v<int&>);
    EXPECT_TRUE(AscendC::Std::is_lvalue_reference_v<float&>);
    EXPECT_TRUE(AscendC::Std::is_lvalue_reference_v<const char&>);
    EXPECT_TRUE(AscendC::Std::is_lvalue_reference_v<volatile double&>);
    EXPECT_TRUE(AscendC::Std::is_lvalue_reference_v<const volatile int&>);

    EXPECT_FALSE(AscendC::Std::is_rvalue_reference_v<int&>);
    EXPECT_FALSE(AscendC::Std::is_rvalue_reference_v<float&>);
    EXPECT_FALSE(AscendC::Std::is_rvalue_reference_v<const char&>);
    EXPECT_FALSE(AscendC::Std::is_rvalue_reference_v<volatile double&>);
    EXPECT_FALSE(AscendC::Std::is_rvalue_reference_v<const volatile int&>);

    EXPECT_TRUE(AscendC::Std::is_reference_v<int&>);
    EXPECT_TRUE(AscendC::Std::is_reference_v<float&>);
    EXPECT_TRUE(AscendC::Std::is_reference_v<const char&>);
    EXPECT_TRUE(AscendC::Std::is_reference_v<volatile double&>);
    EXPECT_TRUE(AscendC::Std::is_reference_v<const volatile int&>);
}

// Test rvalue reference types
TEST_F(IsReferenceTest, RvalueReferenceTypes) {
    EXPECT_FALSE(AscendC::Std::is_lvalue_reference_v<int&&>);
    EXPECT_FALSE(AscendC::Std::is_lvalue_reference_v<float&&>);
    EXPECT_FALSE(AscendC::Std::is_lvalue_reference_v<const char&&>);
    EXPECT_FALSE(AscendC::Std::is_lvalue_reference_v<volatile double&&>);
    EXPECT_FALSE(AscendC::Std::is_lvalue_reference_v<const volatile int&&>);

    EXPECT_TRUE(AscendC::Std::is_rvalue_reference_v<int&&>);
    EXPECT_TRUE(AscendC::Std::is_rvalue_reference_v<float&&>);
    EXPECT_TRUE(AscendC::Std::is_rvalue_reference_v<const char&&>);
    EXPECT_TRUE(AscendC::Std::is_rvalue_reference_v<volatile double&&>);
    EXPECT_TRUE(AscendC::Std::is_rvalue_reference_v<const volatile int&&>);

    EXPECT_TRUE(AscendC::Std::is_reference_v<int&&>);
    EXPECT_TRUE(AscendC::Std::is_reference_v<float&&>);
    EXPECT_TRUE(AscendC::Std::is_reference_v<const char&&>);
    EXPECT_TRUE(AscendC::Std::is_reference_v<volatile double&&>);
    EXPECT_TRUE(AscendC::Std::is_reference_v<const volatile int&&>);
}
// Test non-reference type
TEST_F(IsReferenceTest, NonReferenceTypes) {
    EXPECT_FALSE(AscendC::Std::is_lvalue_reference_v<int>);
    EXPECT_FALSE(AscendC::Std::is_lvalue_reference_v<float>);
    EXPECT_FALSE(AscendC::Std::is_lvalue_reference_v<std::string>);

    EXPECT_FALSE(AscendC::Std::is_rvalue_reference_v<int>);
    EXPECT_FALSE(AscendC::Std::is_rvalue_reference_v<float>);
    EXPECT_FALSE(AscendC::Std::is_rvalue_reference_v<std::string>);

    EXPECT_FALSE(AscendC::Std::is_reference_v<int>);
    EXPECT_FALSE(AscendC::Std::is_reference_v<float>);
    EXPECT_FALSE(AscendC::Std::is_reference_v<std::string>);
}

// Test CV Qualified reference Types
TEST_F(IsReferenceTest, CVQualifiedReferenceTypes) {
    EXPECT_TRUE(AscendC::Std::is_lvalue_reference_v<const int&>);
    EXPECT_TRUE(AscendC::Std::is_lvalue_reference_v<volatile float&>);
    EXPECT_TRUE(AscendC::Std::is_lvalue_reference_v<const volatile char&>);

    EXPECT_FALSE(AscendC::Std::is_rvalue_reference_v<const int&>);
    EXPECT_FALSE(AscendC::Std::is_rvalue_reference_v<volatile float&>);
    EXPECT_FALSE(AscendC::Std::is_rvalue_reference_v<const volatile char&>);

    EXPECT_TRUE(AscendC::Std::is_reference_v<const int&>);
    EXPECT_TRUE(AscendC::Std::is_reference_v<volatile float&>);
    EXPECT_TRUE(AscendC::Std::is_reference_v<const volatile char&>);

    EXPECT_FALSE(AscendC::Std::is_lvalue_reference_v<const int&&>);
    EXPECT_FALSE(AscendC::Std::is_lvalue_reference_v<volatile float&&>);
    EXPECT_FALSE(AscendC::Std::is_lvalue_reference_v<const volatile char&&>);

    EXPECT_TRUE(AscendC::Std::is_rvalue_reference_v<const int&&>);
    EXPECT_TRUE(AscendC::Std::is_rvalue_reference_v<volatile float&&>);
    EXPECT_TRUE(AscendC::Std::is_rvalue_reference_v<const volatile char&&>);

    EXPECT_TRUE(AscendC::Std::is_reference_v<const int&&>);
    EXPECT_TRUE(AscendC::Std::is_reference_v<volatile float&&>);
    EXPECT_TRUE(AscendC::Std::is_reference_v<const volatile char&&>);
}

// Test nested reference types
TEST_F(IsReferenceTest, NestedReferenceTypes) {
    EXPECT_TRUE(AscendC::Std::is_lvalue_reference_v<int&>);
    EXPECT_TRUE(AscendC::Std::is_rvalue_reference_v<int&&>);
    EXPECT_TRUE(AscendC::Std::is_reference_v<int&>);
}

// Test nested types
TEST_F(IsReferenceTest, NestedTypes) {
    struct TestStruct {
        int& ref;
    };
    EXPECT_TRUE(AscendC::Std::is_reference_v<decltype(TestStruct::ref)>);
    EXPECT_FALSE(AscendC::Std::is_reference_v<TestStruct>);
}