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

class IsBaseOfTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IsBaseOfTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IsBaseOfTest TearDownTestCase" << std::endl;
    }
};

// base class and derived class definitions
class Base1 {};
class Derived1 : public Base1 {};

class Base2 {};
class Derived2 : public Base2 {};

// multi-level inheritance class
class GrandBase {};
class Intermediate : public GrandBase {};
class DeepDerived : public Intermediate {};

// classes related to virtual inheritance
class VirtualBase {};
class VirtualDerived : virtual public VirtualBase {};

// private inheritance class
class PrivateBase {};
class PrivateDerived : private PrivateBase {};

// test standard Base and Derived class relationships
TEST_F(IsBaseOfTest, StandardBaseDerived) {
    EXPECT_TRUE((AscendC::Std::is_base_of<Base1, Derived1>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<Base1, Derived1>));
}

// testing different base classes and derived classes
TEST_F(IsBaseOfTest, DifferentBaseDerivedPair) {
    EXPECT_TRUE((AscendC::Std::is_base_of<Base2, Derived2>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<Base2, Derived2>));
}

// the relationships between the test class and its own base class
TEST_F(IsBaseOfTest, ClassIsSelfBase) {
    EXPECT_TRUE((AscendC::Std::is_base_of<Base1, Base1>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<Base1, Base1>));
}

// testing the base class relationships between unrelated classes
TEST_F(IsBaseOfTest, UnrelatedClasses) {
    EXPECT_FALSE((AscendC::Std::is_base_of<Base1, Base2>::value));
    EXPECT_FALSE((AscendC::Std::is_base_of_v<Base1, Base2>));
}

// testing the inverse relationship between derived classes and base classes
TEST_F(IsBaseOfTest, DerivedToBaseReverse) {
    EXPECT_FALSE((AscendC::Std::is_base_of<Derived1, Base1>::value));
    EXPECT_FALSE((AscendC::Std::is_base_of_v<Derived1, Base1>));
}

// testing multi-level inheritance relationships
TEST_F(IsBaseOfTest, MultiLevelInheritance) {
    EXPECT_TRUE((AscendC::Std::is_base_of<GrandBase, DeepDerived>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<GrandBase, DeepDerived>));
    EXPECT_TRUE((AscendC::Std::is_base_of<Intermediate, DeepDerived>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<Intermediate, DeepDerived>));
}

// testing multi-level inheritance reverse relationships
TEST_F(IsBaseOfTest, MultiLevelInheritanceReverse) {
    EXPECT_FALSE((AscendC::Std::is_base_of<DeepDerived, GrandBase>::value));
    EXPECT_FALSE((AscendC::Std::is_base_of_v<DeepDerived, GrandBase>));
}

// testing virtual inheritance relationships
TEST_F(IsBaseOfTest, VirtualInheritance) {
    EXPECT_TRUE((AscendC::Std::is_base_of<VirtualBase, VirtualDerived>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<VirtualBase, VirtualDerived>));
}

// testing private inheritance relationships (AscendC::Std::is_base_of does not consider access permissions)
TEST_F(IsBaseOfTest, PrivateInheritance) {
    EXPECT_TRUE((AscendC::Std::is_base_of<PrivateBase, PrivateDerived>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<PrivateBase, PrivateDerived>));
}

// test template class as base class and derived class
template <typename T>
class TemplateBase {};

template <typename T>
class TemplateDerived : public TemplateBase<T> {};

TEST_F(IsBaseOfTest, TemplateClasses) {
    EXPECT_TRUE((AscendC::Std::is_base_of<TemplateBase<int>, TemplateDerived<int>>::value));
    EXPECT_TRUE((AscendC::Std::is_base_of_v<TemplateBase<int>, TemplateDerived<int>>));
}

// testing template class with different types of parameters
TEST_F(IsBaseOfTest, TemplateClassesDifferentParams) {
    EXPECT_FALSE((AscendC::Std::is_base_of<TemplateBase<int>, TemplateDerived<double>>::value));
    EXPECT_FALSE((AscendC::Std::is_base_of_v<TemplateBase<int>, TemplateDerived<double>>));
}

