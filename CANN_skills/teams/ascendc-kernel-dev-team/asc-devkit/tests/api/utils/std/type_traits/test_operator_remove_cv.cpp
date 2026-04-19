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

class RemoveCVTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "RemoveCVTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "RemoveCVTest TearDownTestCase" << std::endl;
    }
};

// Test non-const and non-volatile type
TEST_F(RemoveCVTest, NonCVType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv<int>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv<double>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv<char>::type, char>));
}

// Test const type
TEST_F(RemoveCVTest, ConstType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv<const int>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv<const double>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv<const char>::type, char>));
}

// Test volatile type
TEST_F(RemoveCVTest, VolatileType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv<volatile int>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv<volatile double>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv<volatile char>::type, char>));
}

// Test const and volatile type
TEST_F(RemoveCVTest, ConstVolatileType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv<const volatile int>::type, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv<const volatile double>::type, double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv<const volatile char>::type, char>));
}

TEST_F(RemoveCVTest, RemoveCVT) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv_t<int>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv_t<double>, double>));

    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv_t<const int>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv_t<const double>, double>));

    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv_t<volatile int>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv_t<volatile double>, double>));

    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv_t<const volatile int>, int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::remove_cv_t<const volatile double>, double>));
}