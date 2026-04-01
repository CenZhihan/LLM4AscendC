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

class IsOneOfTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IsOneOfTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IsOneOfTest TearDownTestCase" << std::endl;
    }
};


// test the same basic type
TEST_F(IsOneOfTest, BasicTypesSame)
{
    EXPECT_TRUE((AscendC::Std::is_one_of<int, int, double>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<int, int, double>));
    EXPECT_TRUE((AscendC::Std::is_one_of<double, double, int>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<double, double, int>));
}

// test different scenarios of basic type
TEST_F(IsOneOfTest, BasicTypesDifferent)
{
    EXPECT_FALSE((AscendC::Std::is_one_of<int, double, long>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<int, double, long>));
    EXPECT_FALSE((AscendC::Std::is_one_of<char, long, int>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<char, long, int>));
}

// test the same scenario for custom class
class MyClass1 {};
class MyClass3 {};
TEST_F(IsOneOfTest, CustomClassesSame)
{
    EXPECT_TRUE((AscendC::Std::is_one_of<MyClass1, MyClass1, MyClass3>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<MyClass1, MyClass1, MyClass3>));
}

// testing different scenarios of custom classes
class MyClass2 {};
TEST_F(IsOneOfTest, CustomClassesDifferent)
{
    EXPECT_FALSE((AscendC::Std::is_one_of<MyClass1, MyClass2, MyClass3>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<MyClass1, MyClass2, MyClass3>));
}

// test the same pointer type
TEST_F(IsOneOfTest, PointerTypesSame)
{
    EXPECT_TRUE((AscendC::Std::is_one_of<int *, int *, double *>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<int *, int *, double *>));
    EXPECT_TRUE((AscendC::Std::is_one_of<MyClass1 *, MyClass1 *, MyClass2 *>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<MyClass1 *, MyClass1 *, MyClass2 *>));
}

// testing different pointer types
TEST_F(IsOneOfTest, PointerTypesDifferent)
{
    EXPECT_FALSE((AscendC::Std::is_one_of<int *, double *, long *>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<int *, double *, long *>));
    EXPECT_FALSE((AscendC::Std::is_one_of<MyClass1 *, MyClass2 *, MyClass3 *>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<MyClass1 *, MyClass2 *, MyClass3 *>));
}

// test the case where the reference types are the same
TEST_F(IsOneOfTest, ReferenceTypesSame)
{
    EXPECT_TRUE((AscendC::Std::is_one_of<int &, int &, double &>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<int &, int &, double &>));
    EXPECT_TRUE((AscendC::Std::is_one_of<MyClass1 &, MyClass1 &, MyClass2 &>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<MyClass1 &, MyClass1 &, MyClass2 &>));
}

// test different scenarios of reference types
TEST_F(IsOneOfTest, ReferenceTypesDifferent)
{
    EXPECT_FALSE((AscendC::Std::is_one_of<int &, double &, long &>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<int &, double &, long &>));
    EXPECT_FALSE((AscendC::Std::is_one_of<MyClass1 &, MyClass2 &, MyClass3 &>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<MyClass1 &, MyClass2 &, MyClass3 &>));
}

// test the same constant type
TEST_F(IsOneOfTest, ConstTypesSame)
{
    EXPECT_TRUE((AscendC::Std::is_one_of<const int, const int, const double>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<const int, const int, const double>));
    EXPECT_TRUE((AscendC::Std::is_one_of<const MyClass1, const MyClass1, const MyClass2>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<const MyClass1, const MyClass1, const MyClass2>));
}

// test the different between constant types and non-const types
TEST_F(IsOneOfTest, ConstAndNonConstDifferent)
{
    EXPECT_FALSE((AscendC::Std::is_one_of<int, const int, const double>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<int, const int, const double>));
    EXPECT_FALSE((AscendC::Std::is_one_of<MyClass1, const MyClass1, const MyClass2>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<MyClass1, const MyClass1, const MyClass2>));
}

// testing scenarios with identical template type
template <typename T> class TemplateClass {};

TEST_F(IsOneOfTest, TemplateTypesSame)
{
    EXPECT_TRUE((AscendC::Std::is_one_of<TemplateClass<int>, TemplateClass<int>, TemplateClass<double>>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_v<TemplateClass<int>, TemplateClass<int>, TemplateClass<double>>));
}

// different test template types
TEST_F(IsOneOfTest, TemplateTypesDifferent)
{
    EXPECT_FALSE((AscendC::Std::is_one_of<TemplateClass<int>, TemplateClass<double>, TemplateClass<long>>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_v<TemplateClass<int>, TemplateClass<double>, TemplateClass<long>>));
}

/*
  is_one_of_value
*/ 
class IsOneOfValueTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IsOneOfValueTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IsOneOfValueTest TearDownTestCase" << std::endl;
    }
};

// test the same
TEST_F(IsOneOfValueTest, BasicTypesSame)
{
    EXPECT_TRUE((AscendC::Std::is_one_of_value<int, 1, 1, 2>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_value_v<int, 1, 1, 2>));
}

// test the different
TEST_F(IsOneOfValueTest, BasicTypesDifferent)
{
    EXPECT_FALSE((AscendC::Std::is_one_of_value<int, 1, 2, 3>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_value_v<int, 1, 2, 3>));
}

// test the enum
enum TestColor {
    Red = 0,
    Green = 1,
    Blue = 2
};

TEST_F(IsOneOfValueTest, CustomType)
{
    EXPECT_TRUE((AscendC::Std::is_one_of_value<TestColor, TestColor::Green, TestColor::Green, TestColor::Blue>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_value<TestColor, TestColor::Green, TestColor::Blue, TestColor::Green>::value));
    EXPECT_TRUE((AscendC::Std::is_one_of_value_v<TestColor, TestColor::Green, TestColor::Green, TestColor::Blue>));
    EXPECT_TRUE((AscendC::Std::is_one_of_value_v<TestColor, TestColor::Green, TestColor::Blue, TestColor::Green>));
    EXPECT_FALSE((AscendC::Std::is_one_of_value<TestColor, TestColor::Red, TestColor::Green, TestColor::Blue>::value));
    EXPECT_FALSE((AscendC::Std::is_one_of_value_v<TestColor, TestColor::Red, TestColor::Green, TestColor::Blue>));
}
