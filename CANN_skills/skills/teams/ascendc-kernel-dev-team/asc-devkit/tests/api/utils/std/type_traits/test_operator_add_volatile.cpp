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

class AddVolatileTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "AddVolatileTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "AddVolatileTest TearDownTestCase" << std::endl;
    }
};

// Test non-volatile type
TEST_F(AddVolatileTest, NonVolatileType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_volatile<int>::type, volatile int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_volatile<double>::type, volatile double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_volatile<char>::type, volatile char>));
}

// Test volatile type
TEST_F(AddVolatileTest, VolatileType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_volatile<volatile int>::type, volatile int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_volatile<volatile double>::type, volatile double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_volatile<volatile char>::type, volatile char>));
}

TEST_F(AddVolatileTest, AddVolatileT) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_volatile_t<int>, volatile int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_volatile_t<double>, volatile double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_volatile_t<volatile int>, volatile int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_volatile_t<volatile double>, volatile double>));
}