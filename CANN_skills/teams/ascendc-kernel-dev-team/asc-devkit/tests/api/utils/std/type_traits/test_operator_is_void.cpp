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

class IsVoidTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IsVoidTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IsVoidTest TearDownTestCase" << std::endl;
    }
};

// Test the judgment of is_void and is_void_v for the void type
TEST_F(IsVoidTest, CheckVoidTypes) {
    EXPECT_EQ(AscendC::Std::is_void<void>::value, true);
    EXPECT_EQ(AscendC::Std::is_void<void>::value, AscendC::Std::is_void_v<void>);
}

// Test the judgment of is_void and is_void_v for non-void types
TEST_F(IsVoidTest, CheckNonVoidTypes) {
    EXPECT_EQ(AscendC::Std::is_void<int>::value, false);
    EXPECT_EQ(AscendC::Std::is_void<int>::value, AscendC::Std::is_void_v<int>);

    EXPECT_EQ(AscendC::Std::is_void<double>::value, false);
    EXPECT_EQ(AscendC::Std::is_void<double>::value, AscendC::Std::is_void_v<double>);

    EXPECT_EQ(AscendC::Std::is_void<std::string>::value, false);
    EXPECT_EQ(AscendC::Std::is_void<std::string>::value, AscendC::Std::is_void_v<std::string>);

    // Test const and volatile qualifiers
    EXPECT_EQ(AscendC::Std::is_void<const void>::value, true);
    EXPECT_EQ(AscendC::Std::is_void<const void>::value, AscendC::Std::is_void_v<const void>);

    EXPECT_EQ(AscendC::Std::is_void<volatile void>::value, true);
    EXPECT_EQ(AscendC::Std::is_void<volatile void>::value, AscendC::Std::is_void_v<volatile void>);

    EXPECT_EQ(AscendC::Std::is_void<const volatile void>::value, true);
    EXPECT_EQ(AscendC::Std::is_void<const volatile void>::value, AscendC::Std::is_void_v<const volatile void>);

    // Non void type with const/volatile qualifier
    EXPECT_EQ(AscendC::Std::is_void<const int>::value, false);
    EXPECT_EQ(AscendC::Std::is_void<const int>::value, AscendC::Std::is_void_v<const int>);

    EXPECT_EQ(AscendC::Std::is_void<volatile int>::value, false);
    EXPECT_EQ(AscendC::Std::is_void<volatile int>::value, AscendC::Std::is_void_v<volatile int>);

    EXPECT_EQ(AscendC::Std::is_void<const volatile int>::value, false);
    EXPECT_EQ(AscendC::Std::is_void<const volatile int>::value, AscendC::Std::is_void_v<const volatile int>);
}

// Test the handling of pointer types by is-void
TEST_F(IsVoidTest, PointerTypes) {
    ASSERT_FALSE((AscendC::Std::is_void<int*>::value));
    ASSERT_FALSE((AscendC::Std::is_void<const int*>::value));
    ASSERT_FALSE((AscendC::Std::is_void<void*>::value));
    ASSERT_FALSE((AscendC::Std::is_void<const void*>::value));
}