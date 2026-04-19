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

class IsArrayTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IsArrayTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IsArrayTest TearDownTestCase" << std::endl;
    }
};

// Test basic array types
TEST_F(IsArrayTest, BasicArrayTypes) {
    EXPECT_TRUE(AscendC::Std::is_array_v<int[]>);
    EXPECT_TRUE(AscendC::Std::is_array_v<int[10]>);
    EXPECT_TRUE(AscendC::Std::is_array_v<double[]>);
    EXPECT_TRUE(AscendC::Std::is_array_v<double[5]>);
}

// Testing Non-Array Types
TEST_F(IsArrayTest, NonArrayTypes) {
    EXPECT_FALSE(AscendC::Std::is_array_v<int>);
    EXPECT_FALSE(AscendC::Std::is_array_v<float>);
    EXPECT_FALSE(AscendC::Std::is_array_v<std::string>);
    EXPECT_FALSE(AscendC::Std::is_array_v<int*>);
    EXPECT_FALSE(AscendC::Std::is_array_v<int&>);
}

// Test multidimensional array types
TEST_F(IsArrayTest, MultiDimensionalArrayTypes) {
    EXPECT_TRUE(AscendC::Std::is_array_v<int[10][20]>);
    EXPECT_TRUE(AscendC::Std::is_array_v<double[5][5]>);
}

// Test array references and pointers
TEST_F(IsArrayTest, ArrayRefenercesAndPointers) {
    int arr[10];
    EXPECT_TRUE(AscendC::Std::is_array_v<decltype(arr)>);
    EXPECT_FALSE(AscendC::Std::is_array_v<decltype(&arr)>);
    EXPECT_FALSE(AscendC::Std::is_array_v<decltype(arr) &>);
}

// Test nested types
TEST_F(IsArrayTest, NestedTypes) {
    struct TestStruct {
        int arr[10];
    };
    EXPECT_TRUE(AscendC::Std::is_array_v<decltype(TestStruct::arr)>);
    EXPECT_FALSE(AscendC::Std::is_array_v<TestStruct>);
}