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

class IsConstTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IsConstTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IsConstTest TearDownTestCase" << std::endl;
    }
};

// Test non-const type
TEST_F(IsConstTest, NonConstType) {
    ASSERT_FALSE((AscendC::Std::is_const<int>::value));
    ASSERT_FALSE((AscendC::Std::is_const<double>::value));
    ASSERT_FALSE((AscendC::Std::is_const<char>::value));
}

// Test const type
TEST_F(IsConstTest, ConstType) {
    ASSERT_TRUE((AscendC::Std::is_const<const int>::value));
    ASSERT_TRUE((AscendC::Std::is_const<const double>::value));
    ASSERT_TRUE((AscendC::Std::is_const<const char>::value));
}

TEST_F(IsConstTest, IsConstV) {
    ASSERT_FALSE(AscendC::Std::is_const_v<int>);
    ASSERT_FALSE(AscendC::Std::is_const_v<double>);
    ASSERT_TRUE(AscendC::Std::is_const_v<const int>);
    ASSERT_TRUE(AscendC::Std::is_const_v<const double>);
}