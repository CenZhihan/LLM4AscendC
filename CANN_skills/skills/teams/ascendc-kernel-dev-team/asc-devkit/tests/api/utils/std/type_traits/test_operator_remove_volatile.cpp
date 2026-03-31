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

class RemoveVolatileTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "RemoveVolatileTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "RemoveVolatileTest TearDownTestCase" << std::endl;
    }
};

// Test non-volatile type
TEST_F(RemoveVolatileTest, NonVolatileType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_volatile<int>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_volatile<double>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_volatile<char>::type, char>));
}

// Test volatile type
TEST_F(RemoveVolatileTest, VolatileType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_volatile<volatile int>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_volatile<volatile double>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_volatile<volatile char>::type, char>));
}

TEST_F(RemoveVolatileTest, RemoveVolatileT) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_volatile_t<int>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_volatile_t<double>, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_volatile_t<volatile int>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_volatile_t<volatile double>, double>));
}