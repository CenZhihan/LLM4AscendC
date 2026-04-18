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

class RemovePointerTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "RemovePointerTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "RemovePointerTest TearDownTestCase" << std::endl;
    }
};

// Test non-pointer type
TEST_F(RemovePointerTest, NonPointerType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer<int>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer<double>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer<char>::type, char>));
}

// Test pointer type
TEST_F(RemovePointerTest, PointerType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer<int*>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer<double*>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer<char*>::type, char>));
}

// Test const pointer type
TEST_F(RemovePointerTest, ConstPointerType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer<int* const>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer<double* const>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer<char* const>::type, char>));
}

// Test volatile type
TEST_F(RemovePointerTest, VolatilePointerType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer<int* volatile>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer<double* volatile>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer<char* volatile>::type, char>));
}

// Test const and volatile type
TEST_F(RemovePointerTest, ConstVolatilePointerType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer<int* const volatile>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer<double* const volatile>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer<char* const volatile>::type, char>));
}

TEST_F(RemovePointerTest, RemovePointerT) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer_t<int>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer_t<double>, double>));

    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer_t<int*>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer_t<double*>, double>));

    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer_t<int* const>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer_t<double* const>, double>));

    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer_t<int* volatile>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer_t<double* volatile>, double>));

    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer_t<int* const volatile>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_pointer_t<double* const volatile>, double>));
}