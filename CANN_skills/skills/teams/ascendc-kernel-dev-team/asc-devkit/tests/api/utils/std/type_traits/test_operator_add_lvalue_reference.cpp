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

class AddLvalueReferenceTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "AddLvalueReferenceTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "AddLvalueReferenceTest TearDownTestCase" << std::endl;
    }
};

// Test basic type
TEST_F(AddLvalueReferenceTest, BasicTypes) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_lvalue_reference_t<int>, int&>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_lvalue_reference_t<float>, float&>));
}

// Test pointer type
TEST_F(AddLvalueReferenceTest, PointerTypes) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_lvalue_reference_t<int*>, int*&>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_lvalue_reference_t<const int*>, const int*&>));
}

// Test reference type
TEST_F(AddLvalueReferenceTest, ReferenceTypes) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_lvalue_reference_t<int&>, int&>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_lvalue_reference_t<const int&>, const int&>));
}

// Test void type
TEST_F(AddLvalueReferenceTest, VoidTypes) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_lvalue_reference_t<void>, void>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_lvalue_reference_t<const void>, const void>));
}

// Test function type
TEST_F(AddLvalueReferenceTest, FunctionTypes) {
    using FuncType = void();
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_lvalue_reference_t<FuncType>, FuncType&>));
}

// Test array type
TEST_F(AddLvalueReferenceTest, ArrayTypes) {
    using ArrayType = int[];
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_lvalue_reference_t<ArrayType>, ArrayType&>));
}

// Test class type
TEST_F(AddLvalueReferenceTest, ClassTypes) {
    class MyClass {};
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_lvalue_reference_t<MyClass>, MyClass&>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_lvalue_reference_t<const MyClass>, const MyClass&>));
}