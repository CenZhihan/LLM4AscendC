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

class EnableIfTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "EnableIfTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "EnableIfTest TearDownTestCase" << std::endl;
    }
};

template <typename T>
struct is_arithmetic {
    static constexpr bool value = AscendC::Std::is_integral<T>::value || AscendC::Std::is_floating_point<T>::value;
};

// example 1: enable function template based on arithmetic type
// the function template will only be enabled when T is an arithmetic type
template <typename T>
typename AscendC::Std::enable_if<is_arithmetic<T>::value, T>::type
multiply(T a, T b) {
    return a * b;
}

// example 2: enable different function overloads based on type size
// enable when the size of T is less than or equal to 4 bytes
template <typename T>
typename AscendC::Std::enable_if<(sizeof(T) <= 4), T>::type
getDefaultValue() {
    return T();
}

// enable when the size of T is more than or equal to 4 bytes
template <typename T>
typename AscendC::Std::enable_if<(sizeof(T) > 4), T>::type
getDefaultValue() {
    return T();
}

// example 3: enable class template based on whether the template parameter is of specific type
// the class template will only be enabled when T is of type int
template <typename T, typename = typename AscendC::Std::enable_if<AscendC::Std::is_same<T, int>::value>::type>
class IntContainer {
public:
    IntContainer(T value) : data(value) {}
    T getData() const { return data; }
private:
    T data;
};

// testing multi of integer types
TEST_F(EnableIfTest, ArithmeticIntegerMultiplication) {
    int result = multiply(3, 4);
    EXPECT_EQ(result, 12);
}

// testing multi of float types
TEST_F(EnableIfTest, ArithmeticFloatingPointMultiplication) {
    double result = multiply(2.5, 3.0);
    EXPECT_EQ(result, 7.5);
}

// testing the default value retrieval for small-sized types
TEST_F(EnableIfTest, SizeSmallTypeDefaultValue) {
    int value = getDefaultValue<int>();
    EXPECT_EQ(value, 0);
}

// testing the retrieval of default values for large-sized types
TEST_F(EnableIfTest, SizeLargeTypeDefaultValue) {
    double value = getDefaultValue<double>();
    EXPECT_EQ(value, 0.0);
}

// test the IntContainer class template
TEST_F(EnableIfTest, SpecificTypeIntContainerFunctionality) {
    IntContainer<int> container(42);
    EXPECT_EQ(container.getData(), 42);
}
