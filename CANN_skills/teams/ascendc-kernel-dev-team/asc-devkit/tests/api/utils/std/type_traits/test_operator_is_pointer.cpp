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

class IsPointerTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IsPointerTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IsPointerTest TearDownTestCase" << std::endl;
    }
};

// Test pointer type
TEST_F(IsPointerTest, PointerTypes) {
    EXPECT_TRUE(AscendC::Std::is_pointer_v<int*>);
    EXPECT_TRUE(AscendC::Std::is_pointer_v<float*>);
    EXPECT_TRUE(AscendC::Std::is_pointer_v<const char*>);
    EXPECT_TRUE(AscendC::Std::is_pointer_v<volatile double*>);
    EXPECT_TRUE(AscendC::Std::is_pointer_v<const volatile int*>);
}

// Test non-pointer type
TEST_F(IsPointerTest, NonPointerTypes) {
    EXPECT_FALSE(AscendC::Std::is_pointer_v<int>);
    EXPECT_FALSE(AscendC::Std::is_pointer_v<float>);
    EXPECT_FALSE(AscendC::Std::is_pointer_v<std::string>);
    EXPECT_FALSE(AscendC::Std::is_pointer_v<int&>);
    EXPECT_FALSE(AscendC::Std::is_pointer_v<const int&>);
}

// Test CV Qualified Pointer Types
TEST_F(IsPointerTest, CVQualifiedPointerTypes) {
    EXPECT_TRUE(AscendC::Std::is_pointer_v<const int*>);
    EXPECT_TRUE(AscendC::Std::is_pointer_v<volatile float*>);
    EXPECT_TRUE(AscendC::Std::is_pointer_v<const volatile char*>);
}

// Test the pointer type of the pointer
TEST_F(IsPointerTest, PointerToPointerTypes) {
    EXPECT_TRUE(AscendC::Std::is_pointer_v<int**>);
    EXPECT_TRUE(AscendC::Std::is_pointer_v<float**>);
    EXPECT_TRUE(AscendC::Std::is_pointer_v<const int**>);
    EXPECT_TRUE(AscendC::Std::is_pointer_v<volatile float**>);
    EXPECT_TRUE(AscendC::Std::is_pointer_v<const volatile char**>);
}

// Test function pointer type
TEST_F(IsPointerTest, FunctionPointerTypes) {
    EXPECT_TRUE(AscendC::Std::is_pointer_v<void(*)()>);
    EXPECT_TRUE(AscendC::Std::is_pointer_v<int(*)(float)>);
    EXPECT_TRUE(AscendC::Std::is_pointer_v<const int(*)()>);
    EXPECT_TRUE(AscendC::Std::is_pointer_v<volatile float(*)(int)>);
    EXPECT_TRUE(AscendC::Std::is_pointer_v<const volatile int(*)(float)>);
}

// Test nested types
TEST_F(IsPointerTest, NestedTypes) {
    struct TestStruct {
        int* ptr;
    };
    EXPECT_TRUE(AscendC::Std::is_pointer_v<decltype(TestStruct::ptr)>);
    EXPECT_FALSE(AscendC::Std::is_pointer_v<TestStruct>);
}