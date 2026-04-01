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

class AddConstTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "AddConstTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "AddConstTest TearDownTestCase" << std::endl;
    }
};

// Test non-const type
TEST_F(AddConstTest, NonConstType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_const<int>::type, const int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_const<double>::type, const double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_const<char>::type, const char>));
}

// Test const type
TEST_F(AddConstTest, ConstType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_const<const int>::type, const int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_const<const double>::type, const double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_const<const char>::type, const char>));
}

TEST_F(AddConstTest, AddConstT) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_const_t<int>, const int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_const_t<double>, const double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_const_t<const int>, const int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_const_t<const double>, const double>));
}