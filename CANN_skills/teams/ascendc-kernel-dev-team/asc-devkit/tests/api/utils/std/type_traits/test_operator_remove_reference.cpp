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

class RemoveReferenceTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "RemoveReferenceTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "RemoveReferenceTest TearDownTestCase" << std::endl;
    }
};

// Test non-reference type
TEST_F(RemoveReferenceTest, NonReferenceType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_reference<int>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_reference<double>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_reference<char>::type, char>));
}

// Test lvalue reference type
TEST_F(RemoveReferenceTest, LvalueReferenceType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_reference<int&>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_reference<double&>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_reference<char&>::type, char>));
}

// Test rvalue reference type
TEST_F(RemoveReferenceTest, RvalueReferenceType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_reference<int&&>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_reference<double&&>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_reference<char&&>::type, char>));
}

TEST_F(RemoveReferenceTest, RemoveReferenceT) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_reference_t<int>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_reference_t<double>, double>));

    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_reference_t<int&>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_reference_t<double&>, double>));

    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_reference_t<int&&>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_reference_t<double&&>, double>));
}