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

class AddPointerTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "AddPointerTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "AddPointerTest TearDownTestCase" << std::endl;
    }
};

// Test basic type
TEST_F(AddPointerTest, BasicTypes) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_pointer<int>::type, int*>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_pointer<float>::type, float*>));
}

// Test void type
TEST_F(AddPointerTest, VoidTypes) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_pointer<void>::type, void*>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_pointer<const void>::type, const void*>));
}

// Test reference type
TEST_F(AddPointerTest, ReferenceTypes) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_pointer<int&>::type, int*>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_pointer<const int&>::type, const int*>));
}

// Test function type
TEST_F(AddPointerTest, FunctionTypes) {
    using FuncType = void();
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_pointer<FuncType>::type, FuncType*>));
}

// Test array type
TEST_F(AddPointerTest, ArrayTypes) {
    using ArrayType = int[];
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_pointer<ArrayType>::type, ArrayType*>));
}

// Test class type
TEST_F(AddPointerTest, ClassTypes) {
    class MyClass {};
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_pointer<MyClass>::type, MyClass*>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_pointer<const MyClass>::type, const MyClass*>));
}