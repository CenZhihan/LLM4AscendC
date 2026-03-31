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

class AddCVTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "AddCVTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "AddCVTest TearDownTestCase" << std::endl;
    }
};

// Test non-const and non-volatile type
TEST_F(AddCVTest, NonCVType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv<int>::type, const volatile int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv<double>::type, const volatile double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv<char>::type, const volatile char>));
}

// Test const type
TEST_F(AddCVTest, ConstType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv<const int>::type, const volatile int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv<const double>::type, const volatile double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv<const char>::type, const volatile char>));
}

// Test volatile type
TEST_F(AddCVTest, VolatileType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv<volatile int>::type, const volatile int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv<volatile double>::type, const volatile double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv<volatile char>::type, const volatile char>));
}

// Test const and volatile type
TEST_F(AddCVTest, ConstVolatileType) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv<const volatile int>::type, const volatile int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv<const volatile double>::type, const volatile double>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv<const volatile char>::type, const volatile char>));
}

TEST_F(AddCVTest, AddCVT) {
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv_t<int>, const volatile int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv_t<double>, const volatile double>));

    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv_t<const int>, const volatile int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv_t<const double>, const volatile double>));

    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv_t<volatile int>, const volatile int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv_t<volatile double>, const volatile double>));

    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv_t<const volatile int>, const volatile int>));
    ASSERT_TRUE((std::is_same_v<AscendC::Std::add_cv_t<const volatile double>, const volatile double>));
}