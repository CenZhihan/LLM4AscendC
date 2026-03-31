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

class IsSameTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IsSameTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IsSameTest TearDownTestCase" << std::endl;
    }
};


// test the same basic type
TEST_F(IsSameTest, BasicTypesSame) {
    EXPECT_TRUE((AscendC::Std::is_same<int, int>::value));
    EXPECT_TRUE((AscendC::Std::is_same_v<int, int>));
    EXPECT_TRUE((AscendC::Std::is_same<double, double>::value));
    EXPECT_TRUE((AscendC::Std::is_same_v<double, double>));
}

// test different case of basic type
TEST_F(IsSameTest, BasicTypesDifferent) {
    EXPECT_FALSE((AscendC::Std::is_same<int, double>::value));
    EXPECT_FALSE((AscendC::Std::is_same_v<int, double>));
    EXPECT_FALSE((AscendC::Std::is_same<char, long>::value));
    EXPECT_FALSE((AscendC::Std::is_same_v<char, long>));
}

// test the same custom class
class MyClass1 {};
TEST_F(IsSameTest, CustomClassesSame) {
    EXPECT_TRUE((AscendC::Std::is_same<MyClass1, MyClass1>::value));
    EXPECT_TRUE((AscendC::Std::is_same_v<MyClass1, MyClass1>));
}

// test different case of custom class
class MyClass2 {};
TEST_F(IsSameTest, CustomClassesDifferent) {
    EXPECT_FALSE((AscendC::Std::is_same<MyClass1, MyClass2>::value));
    EXPECT_FALSE((AscendC::Std::is_same_v<MyClass1, MyClass2>));
}

// test the same pointer type
TEST_F(IsSameTest, PointerTypesSame) {
    EXPECT_TRUE((AscendC::Std::is_same<int*, int*>::value));
    EXPECT_TRUE((AscendC::Std::is_same_v<int*, int*>));
    EXPECT_TRUE((AscendC::Std::is_same<MyClass1*, MyClass1*>::value));
    EXPECT_TRUE((AscendC::Std::is_same_v<MyClass1*, MyClass1*>));
}

// test different case of pointer type
TEST_F(IsSameTest, PointerTypesDifferent) {
    EXPECT_FALSE((AscendC::Std::is_same<int*, double*>::value));
    EXPECT_FALSE((AscendC::Std::is_same_v<int*, double*>));
    EXPECT_FALSE((AscendC::Std::is_same<MyClass1*, MyClass2*>::value));
    EXPECT_FALSE((AscendC::Std::is_same_v<MyClass1*, MyClass2*>));
}

// test the same reference type
TEST_F(IsSameTest, ReferenceTypesSame) {
    EXPECT_TRUE((AscendC::Std::is_same<int&, int&>::value));
    EXPECT_TRUE((AscendC::Std::is_same_v<int&, int&>));
    EXPECT_TRUE((AscendC::Std::is_same<MyClass1&, MyClass1&>::value));
    EXPECT_TRUE((AscendC::Std::is_same_v<MyClass1&, MyClass1&>));
}

// test different case of reference type
TEST_F(IsSameTest, ReferenceTypesDifferent) {
    EXPECT_FALSE((AscendC::Std::is_same<int&, double&>::value));
    EXPECT_FALSE((AscendC::Std::is_same_v<int&, double&>));
    EXPECT_FALSE((AscendC::Std::is_same<MyClass1&, MyClass2&>::value));
    EXPECT_FALSE((AscendC::Std::is_same_v<MyClass1&, MyClass2&>));
}

// test the same constant type
TEST_F(IsSameTest, ConstTypesSame) {
    EXPECT_TRUE((AscendC::Std::is_same<const int, const int>::value));
    EXPECT_TRUE((AscendC::Std::is_same_v<const int, const int>));
    EXPECT_TRUE((AscendC::Std::is_same<const MyClass1, const MyClass1>::value));
    EXPECT_TRUE((AscendC::Std::is_same_v<const MyClass1, const MyClass1>));
}

// test different case of constant type and non-constant type
TEST_F(IsSameTest, ConstAndNonConstDifferent) {
    EXPECT_FALSE((AscendC::Std::is_same<int, const int>::value));
    EXPECT_FALSE((AscendC::Std::is_same_v<int, const int>));
    EXPECT_FALSE((AscendC::Std::is_same<MyClass1, const MyClass1>::value));
    EXPECT_FALSE((AscendC::Std::is_same_v<MyClass1, const MyClass1>));
}

// test the same template type
template <typename T>
class TemplateClass {};

TEST_F(IsSameTest, TemplateTypesSame) {
    EXPECT_TRUE((AscendC::Std::is_same<TemplateClass<int>, TemplateClass<int>>::value));
    EXPECT_TRUE((AscendC::Std::is_same_v<TemplateClass<int>, TemplateClass<int>>));
}

// test different case of template type
TEST_F(IsSameTest, TemplateTypesDifferent) {
    EXPECT_FALSE((AscendC::Std::is_same<TemplateClass<int>, TemplateClass<double>>::value));
    EXPECT_FALSE((AscendC::Std::is_same_v<TemplateClass<int>, TemplateClass<double>>));
}

