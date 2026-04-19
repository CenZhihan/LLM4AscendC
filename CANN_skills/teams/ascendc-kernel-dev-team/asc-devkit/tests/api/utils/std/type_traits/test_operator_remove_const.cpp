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

class RemoveConstTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "RemoveConstTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "RemoveConstTest TearDownTestCase" << std::endl;
    }
};

// Test non-const type
TEST_F(RemoveConstTest, NonConstType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_const<int>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_const<double>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_const<char>::type, char>));
}

// Test const type
TEST_F(RemoveConstTest, ConstType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_const<const int>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_const<const double>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_const<const char>::type, char>));
}

TEST_F(RemoveConstTest, RemoveConstT) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_const_t<int>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_const_t<double>, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_const_t<const int>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_const_t<const double>, double>));
}