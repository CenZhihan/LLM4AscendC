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

class TupleTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "TupleTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "TupleTest TearDownTestCase" << std::endl;
    }
};

TEST_F(TupleTest, InitTupleValue)
{
    // 1.1 define tuple
    AscendC::Std::tuple<uint32_t, float, bool> testTuple;
    EXPECT_EQ(sizeof(testTuple), 12);

    // 1.2 define and initialize a tuple
    AscendC::Std::tuple<uint32_t, float, bool> testTupleInit{11, 2.2, true};
    EXPECT_EQ(sizeof(testTupleInit), 12);

    // 1.3 define tuple, type: Tensor
    AscendC::Std::tuple<AscendC::LocalTensor<half>, AscendC::GlobalTensor<float>> testTupleInitTensor;
    EXPECT_EQ(sizeof(testTupleInitTensor), sizeof(AscendC::GlobalTensor<float>) + sizeof(AscendC::LocalTensor<half>));

    // 1.4 define tuple, type is Tensor, initialize
    AscendC::LocalTensor<float> srcLocal;
    AscendC::GlobalTensor<half> srcGlobal;
    AscendC::Std::tuple<AscendC::LocalTensor<float>, AscendC::GlobalTensor<half>> testTupleInitTensorInit{srcLocal, srcGlobal};
    EXPECT_EQ(sizeof(testTupleInitTensorInit), sizeof(AscendC::LocalTensor<float>) + sizeof(AscendC::GlobalTensor<half>));

    // 1.5 multiple type definitions
    AscendC::Std::tuple<AscendC::int4b_t, int8_t, uint8_t, half, int16_t, uint16_t, int32_t, uint32_t, uint64_t, int64_t, float, bfloat16_t, double> testMultiType;
    EXPECT_EQ(sizeof(testMultiType), 56);

    // 1.6 multiple type initialization
    AscendC::Std::tuple<AscendC::int4b_t, int8_t, uint8_t, half, int16_t, uint16_t, int32_t, uint32_t, uint64_t, int64_t, float, bfloat16_t, double> testMultiTypeInit{
        1, 2, 3, 4.4, 5, 6, 7, 8, 9, 10, 11.11, 12.12, 13.13 };
    EXPECT_EQ(sizeof(testMultiTypeInit), 56);

    // 1.7 multiple composite type definitions
    AscendC::Std::tuple<AscendC::int4b_t, int8_t, uint8_t, half, int16_t, uint16_t, int32_t, uint32_t, uint64_t, int64_t, float, bfloat16_t, double, \
    AscendC::LocalTensor<AscendC::int4b_t>, AscendC::GlobalTensor<bfloat16_t>> testMultiTensorType;
    EXPECT_EQ(sizeof(testMultiTensorType), 55 + sizeof(AscendC::int4b_t) + sizeof(AscendC::LocalTensor<AscendC::int4b_t>) + sizeof(AscendC::GlobalTensor<bfloat16_t>));

    // 1.7 aggregation of 64 variables
    AscendC::Std::tuple<uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t,\
        uint32_t, float, bool, uint32_t> variableTuple64;
    EXPECT_EQ(sizeof(variableTuple64), 256);

    // 1.8 copy constructor
    AscendC::Std::tuple<uint32_t, float, bool> testCopyTuple = testTupleInit;
    EXPECT_EQ(sizeof(testCopyTuple), 12);

    // 1.9 step-by-step constructor
    AscendC::Std::tuple<uint32_t, float, bool> testCopySplitTuple;
    testCopySplitTuple = testTupleInit;
    EXPECT_EQ(sizeof(testCopySplitTuple), 12);
    
    // 1.10 variable assignment
    uint32_t aaa = 11;
    float bbb = 2.2;
    bool ccc = true;
    AscendC::Std::tuple<uint32_t, float, bool> InitTupleUsingVariable{aaa, bbb, ccc};
    EXPECT_EQ(sizeof(InitTupleUsingVariable), 12);
}

TEST_F(TupleTest, TupleSize)
{
    // 2.1 get the number of elements in a tuple
    AscendC::Std::tuple<uint32_t, float, bool> testTuple;
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(testTuple)>::value, 3);

    // 2.2 get the number of elements in a tuple
    AscendC::Std::tuple<uint32_t, float, bool> testTupleInit{11, 2.2, true};
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(testTupleInit)>::value, 3);

    // 2.3 get the number of elements in a tuple
    AscendC::Std::tuple<AscendC::int4b_t, int8_t, uint8_t, half, int16_t, uint16_t, int32_t, uint32_t, uint64_t, int64_t, float, bfloat16_t, double, \
    AscendC::LocalTensor<AscendC::int4b_t>, AscendC::GlobalTensor<bfloat16_t>> testMultiTensorType;
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(testMultiTensorType)>::value, 15);

    // 2.4 get the number of elements in a tuple
    AscendC::Std::tuple<uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t,\
        uint32_t, float, bool, uint32_t> variableTuple64;
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(variableTuple64)>::value, 64);

    // 2.5 get the number of elements in a tuple
     AscendC::Std::tuple<uint32_t, float, bool> testCopySplitTuple;
    testCopySplitTuple = testTupleInit;
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(testCopySplitTuple)>::value, 3);

    // 2.6 get the number of elements in a tuple, const
    const AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testConstTupleSize;
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(testConstTupleSize)>::value, 4);

    // 2.7 get the number of elements in a tuple, volatile
    volatile AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testVolatileTupleSize;
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(testVolatileTupleSize)>::value, 4);

    // 2.8 get the number of elements in a tuple, const volatile
    const volatile AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testConstVolatileTupleSize;
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(testConstVolatileTupleSize)>::value, 4);
}

TEST_F(TupleTest, TupleSizeV)
{
    // 3.1 get the number of elements in a tuple
    AscendC::Std::tuple<uint32_t, float, bool> testTuple;
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(testTuple)>, 3);

    // 3.2 get the number of elements in a tuple
    AscendC::Std::tuple<uint32_t, float, bool> testTupleInit{11, 2.2, true};
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(testTupleInit)>, 3);

    // 3.3 get the number of elements in a tuple
    AscendC::Std::tuple<AscendC::int4b_t, int8_t, uint8_t, half, int16_t, uint16_t, int32_t, uint32_t, uint64_t, int64_t, float, bfloat16_t, double, \
    AscendC::LocalTensor<AscendC::int4b_t>, AscendC::GlobalTensor<bfloat16_t>> testMultiTensorType;
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(testMultiTensorType)>, 15);

    // 3.4 get the number of elements in a tuple
    AscendC::Std::tuple<uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t, \
        uint32_t, float, bool, uint32_t, float, bool, uint32_t, float, bool, uint32_t,\
        uint32_t, float, bool, uint32_t> variableTuple64;
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(variableTuple64)>, 64);

    // 3.5 get the number of elements in a tuple
     AscendC::Std::tuple<uint32_t, float, bool> testCopySplitTuple;
    testCopySplitTuple = testTupleInit;
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(testCopySplitTuple)>, 3);

    // 3.6 get the number of elements in a tuple, const
    const AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testConstTupleSize;
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(testConstTupleSize)>, 4);

    // 3.7 get the number of elements in a tuple, volatile
    volatile AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testVolatileTupleSize;
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(testVolatileTupleSize)>, 4);

    // 3.8 get the number of elements in a tuple, const volatile
    const volatile AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testConstVolatileTupleSize;
    EXPECT_EQ(AscendC::Std::tuple_size_v<decltype(testConstVolatileTupleSize)>, 4);
}


TEST_F(TupleTest, TupleElement)
{
    // 4.1 get the data type of the first element
    const AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testConstTuple;
    using FirstType = AscendC::Std::tuple_element<0, decltype(testConstTuple)>::type; // const uint32_t
    FirstType first = 88;
    EXPECT_EQ(first, 88);
    EXPECT_EQ(sizeof(FirstType), 4);

    // 4.2  get the data type of the second element
    volatile AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testVolatileTuple;
    using SecondType = AscendC::Std::tuple_element<1, decltype(testVolatileTuple)>::type; // volatile float
    SecondType second = 8.0;
    EXPECT_EQ(second, 8.0);
    EXPECT_EQ(sizeof(SecondType), 4);

    // 4.3 get the data type of the thrid element
    const volatile AscendC::Std::tuple<uint32_t, float, bool, AscendC::LocalTensor<AscendC::int4b_t>> testConstVolatileTuple;
    using ThirdType = AscendC::Std::tuple_element<2, decltype(testConstVolatileTuple)>::type; // const volatile bool
    ThirdType third = false;
    EXPECT_EQ(third, false);
    EXPECT_EQ(sizeof(ThirdType), 1);

    // 4.4 get the data type of the thrid element
    const AscendC::Std::tuple<const uint32_t, const volatile float, volatile uint16_t, const AscendC::LocalTensor<AscendC::int4b_t>> testConstElement;
    using ConstThirdType = AscendC::Std::tuple_element<2, decltype(testConstElement)>::type; // const volatile uint16_t
    ConstThirdType thirdConst = 22;
    EXPECT_EQ(thirdConst, 22);
    EXPECT_EQ(sizeof(ConstThirdType), 2);

    // 4.5 get the data type of the second element
    using ConstSecondType = AscendC::Std::tuple_element<1, decltype(testConstElement)>::type; // const volatile float
    ConstSecondType secondConst = 8.0;
    EXPECT_EQ(secondConst, 8.0);
    EXPECT_EQ(sizeof(ConstSecondType), 4);

    // 4.6 get the data type of the first element
    using ConstFirstType = AscendC::Std::tuple_element<0, decltype(testConstElement)>::type; // const uint32_t
    ConstFirstType firstConst = 88;
    EXPECT_EQ(firstConst, 88);
    EXPECT_EQ(sizeof(ConstFirstType), 4);
}

TEST_F(TupleTest, MakeTuple)
{
    // specified data type aggregation
    auto makeTupleEle = AscendC::Std::make_tuple(55, (float)6.6, true);

    // aggregation of 64 variables
    auto makeTuple64Ele = AscendC::Std::make_tuple( \
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, \
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, \
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, \
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, \
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, \
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, \
    61, 62, 63, 64);

    // Tensor class aggregation
    AscendC::LocalTensor<float> xLocal;
    AscendC::LocalTensor<AscendC::int4b_t> yLocal;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<bfloat16_t> yGm;
    auto makeTupleTensor = AscendC::Std::make_tuple(xLocal, yLocal, xGm, yGm);

    // use make_tuple for initialization
    AscendC::Std::tuple<uint16_t, float> makeTupleInit (AscendC::Std::make_tuple((uint16_t)33, (float)4.4));

    // use make_tuple for copy constructor initialization
    const AscendC::Std::tuple<uint32_t, float, bool> makeTupleCopyInit = AscendC::Std::make_tuple((uint32_t)55, (float)6.6, true);
}

// test the basic functionality of AscendC::Std::tuple_size
TEST_F(TupleTest, BasicFunctionalityTupleSize) {
    using MyTuple = AscendC::Std::tuple<int, double, char, float>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size<MyTuple>::value;
    EXPECT_EQ(tuple_size, 4);
}


// test AscendC::Std::tuple_size handling of empty tuples
TEST_F(TupleTest, EmptyTupleTupleSize) {
    using EmptyTuple = AscendC::Std::tuple<>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size<EmptyTuple>::value;
    EXPECT_EQ(tuple_size, 0);
}


// test AscendC::Std::tuple_size handling of const tuples
TEST_F(TupleTest, ConstTupleTupleSize) {
    using ConstTuple = const AscendC::Std::tuple<int, double, char, float>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size<ConstTuple>::value;
    EXPECT_EQ(tuple_size, 4);
}


// test AscendC::Std::tuple_size handling of reference tuples
TEST_F(TupleTest, TupleWithReferencesTupleSize) {
    using TupleWithRef = AscendC::Std::tuple<int&, double&, char&, float&>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size<TupleWithRef>::value;
    EXPECT_EQ(tuple_size, 4);
}

// test AscendC::Std::tuple_size for handling tuples of different sizes
TEST_F(TupleTest, DifferentTupleSizesTupleSize) {
    using Tuple1 = AscendC::Std::tuple<int>;
    using Tuple2 = AscendC::Std::tuple<int, double>;
    using Tuple3 = AscendC::Std::tuple<int, double, char>;
    using Tuple4 = AscendC::Std::tuple<int, double, char, float>;

    constexpr size_t size1 = AscendC::Std::tuple_size<Tuple1>::value;
    constexpr size_t size2 = AscendC::Std::tuple_size<Tuple2>::value;
    constexpr size_t size3 = AscendC::Std::tuple_size<Tuple3>::value;
    constexpr size_t size4 = AscendC::Std::tuple_size<Tuple4>::value;

    EXPECT_EQ(size1, 1);
    EXPECT_EQ(size2, 2);
    EXPECT_EQ(size3, 3);
    EXPECT_EQ(size4, 4);
}

// test the basic functionality of AscendC::Std::tuple_size_v
TEST_F(TupleTest, BasicFunctionalityTupleSizeV) {
    using MyTuple = AscendC::Std::tuple<int, double, char>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size_v<MyTuple>;
    EXPECT_EQ(tuple_size, 3);
}


// test AscendC::Std::tuple_size_v handling of empty tuples
TEST_F(TupleTest, EmptyTupleTupleSizeV) {
    using EmptyTuple = AscendC::Std::tuple<>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size_v<EmptyTuple>;
    EXPECT_EQ(tuple_size, 0);
}


// test AscendC::Std::tuple_size_v handling of const tuples
TEST_F(TupleTest, ConstTupleTupleSizeV) {
    using ConstTuple = const AscendC::Std::tuple<int, double, char>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size_v<ConstTuple>;
    EXPECT_EQ(tuple_size, 3);
}


// test AscendC::Std::tuple_size_v handling of reference tuples
TEST_F(TupleTest, TupleWithReferencesTupleSizeV) {
    using TupleWithRef = AscendC::Std::tuple<int&, double&, char&>;
    constexpr size_t tuple_size = AscendC::Std::tuple_size_v<TupleWithRef>;
    EXPECT_EQ(tuple_size, 3);
}

// test AscendC::Std::tuple_size_v for handling different tuple sizes
TEST_F(TupleTest, DifferentTupleSizesTupleSizeV) {
    using Tuple1 = AscendC::Std::tuple<int>;
    using Tuple2 = AscendC::Std::tuple<int, double>;
    using Tuple3 = AscendC::Std::tuple<int, double, char>;
    using Tuple4 = AscendC::Std::tuple<int, double, char, float>;

    constexpr size_t size1 = AscendC::Std::tuple_size_v<Tuple1>;
    constexpr size_t size2 = AscendC::Std::tuple_size_v<Tuple2>;
    constexpr size_t size3 = AscendC::Std::tuple_size_v<Tuple3>;
    constexpr size_t size4 = AscendC::Std::tuple_size_v<Tuple4>;

    EXPECT_EQ(size1, 1);
    EXPECT_EQ(size2, 2);
    EXPECT_EQ(size3, 3);
    EXPECT_EQ(size4, 4);
}

// test the basic functionality of make_tuple
TEST_F(TupleTest, BasicFunctionalityMakeTuple) {
    auto t1 = AscendC::Std::make_tuple(1, 2.0, 'h');
    EXPECT_EQ(AscendC::Std::get<0>(t1), 1);
    EXPECT_EQ(AscendC::Std::get<1>(t1), 2.0);
    EXPECT_EQ(AscendC::Std::get<2>(t1), 'h');
}

// test make_tuple to create an empty tuple
TEST_F(TupleTest, EmptyTupleMakeTuple) {
    auto t1 = AscendC::Std::make_tuple();
    // check if the size of the tuple is 0
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(t1)>::value, 0);
}

// testing make_tuple with different types if combinations
TEST_F(TupleTest, DifferentTypes) {
    auto t1 = AscendC::Std::make_tuple(1, 2.0, 'a');
    EXPECT_EQ(AscendC::Std::get<0>(t1), 1);
    EXPECT_EQ(AscendC::Std::get<1>(t1), 2.0);
    EXPECT_EQ(AscendC::Std::get<2>(t1), 'a');
}

// test how make_tuple handles references
TEST_F(TupleTest, TupleWithReferencesMakeTuple) {
    int a = 12;
    double b = 21.5;
    auto t1 = AscendC::Std::make_tuple(a, b);
    AscendC::Std::get<0>(t1)++;
    AscendC::Std::get<1>(t1) += 1.5;
    EXPECT_EQ(a, 12);
    EXPECT_EQ(b, 21.5);
}


// test the handling of const elements in make_tuple
TEST_F(TupleTest, TupleWithConstElements) {
    const int a = 10;
    const double b = 20.0;
    const char s = 't';
    auto t1 = AscendC::Std::make_tuple(a, b, s);
    EXPECT_EQ(AscendC::Std::get<0>(t1), 10);
    EXPECT_EQ(AscendC::Std::get<1>(t1), 20.0);
    EXPECT_EQ(AscendC::Std::get<2>(t1), 't');
}

// test the basic functionality of AscendC::Std::tuple_element
TEST_F(TupleTest, BasicFunctionalityTupleElement) {
    using MyTuple = AscendC::Std::tuple<int, double>;
    using FirstType = AscendC::Std::tuple_element<0, MyTuple>::type;
    using SecondType = AscendC::Std::tuple_element<1, MyTuple>::type;

    // check if the element type obtained by AscendC::Std::tuple_element is correct
    EXPECT_TRUE((std::is_same_v<FirstType, int>));
    EXPECT_TRUE((std::is_same_v<SecondType, double>));
}

// test AscendC::Std::tuple_element's handling of const tuple
TEST_F(TupleTest, ConstTuple) {
    using ConstTuple = const AscendC::Std::tuple<int, double>;
    using FirstType = AscendC::Std::tuple_element<0, ConstTuple>::type;
    using SecondType = AscendC::Std::tuple_element<1, ConstTuple>::type;

    EXPECT_TRUE((std::is_same_v<FirstType, const int>));
    EXPECT_TRUE((std::is_same_v<SecondType, const double>));
}


// test AscendC::Std::tuple_element's handling of reference tuple
TEST_F(TupleTest, TupleWithReferencesTupleElement) {
    using TupleWithRef = AscendC::Std::tuple<int&, double&>;
    using FirstType = AscendC::Std::tuple_element<0, TupleWithRef>::type;
    using SecondType = AscendC::Std::tuple_element<1, TupleWithRef>::type;

    EXPECT_TRUE((std::is_same_v<FirstType, int&>));
    EXPECT_TRUE((std::is_same_v<SecondType, double&>));
}


// test AscendC::Std::tuple_element for handling tuple of different sizes
TEST_F(TupleTest, DifferentTupleSizes) {
    using Tuple1 = AscendC::Std::tuple<int, double, char>;
    using FirstType = AscendC::Std::tuple_element<0, Tuple1>::type;
    using SecondType = AscendC::Std::tuple_element<1, Tuple1>::type;
    using ThirdType = AscendC::Std::tuple_element<2, Tuple1>::type;

    EXPECT_TRUE((std::is_same_v<FirstType, int>));
    EXPECT_TRUE((std::is_same_v<SecondType, double>));
    EXPECT_TRUE((std::is_same_v<ThirdType, char>));
}

// test the basic functionality of AscendC::Std::tie
TEST_F(TupleTest, BasicFunctionalityTie) {
    int a = 10;
    double b = 20.5;
    char c = 'x';
    bool flag = true;

    auto tie_result = AscendC::Std::tie(a, b, c, flag);
    EXPECT_EQ(AscendC::Std::get<0>(tie_result), a);
    EXPECT_EQ(AscendC::Std::get<1>(tie_result), b);
    EXPECT_EQ(AscendC::Std::get<2>(tie_result), c);
    EXPECT_EQ(AscendC::Std::get<3>(tie_result), flag);
}


// test AscendC::Std::tie for modifications to variable references
TEST_F(TupleTest, ModifyThroughTieTie) {
    int a = 10;
    double b = 20.5;
    char c = 'x';
    bool flag = true;

    auto tie_result = AscendC::Std::tie(a, b, c, flag);
    AscendC::Std::get<0>(tie_result) = 15;
    AscendC::Std::get<1>(tie_result) = 25.5;
    AscendC::Std::get<2>(tie_result) = 'y';
    AscendC::Std::get<3>(tie_result) = false;

    EXPECT_EQ(a, 15);
    EXPECT_EQ(b, 25.5);
    EXPECT_EQ(c, 'y');
    EXPECT_EQ(flag, false);
}


// testing AscendC::Std::tie with const variables
TEST_F(TupleTest, TieWithConstTie) {
    const int a = 10;
    const double b = 20.5;
    const char c = 'x';
    const bool flag = true;

    auto tie_result = AscendC::Std::tie(a, b, c, flag);
    // attempting to modify a const variable, should result in a compilation error
    EXPECT_EQ(AscendC::Std::get<0>(tie_result), a);
    EXPECT_EQ(AscendC::Std::get<1>(tie_result), b);
    EXPECT_EQ(AscendC::Std::get<2>(tie_result), c);
    EXPECT_EQ(AscendC::Std::get<3>(tie_result), flag);
}


// test AscendC::Std::tie for handling different types
TEST_F(TupleTest, DifferentTypesTie) {
    int a = 10;
    float f = 1.23f;
    char c = 'z';
    bool flag = false;

    auto tie_result = AscendC::Std::tie(a, f, c, flag);
    EXPECT_EQ(AscendC::Std::get<0>(tie_result), a);
    EXPECT_EQ(AscendC::Std::get<1>(tie_result), f);
    EXPECT_EQ(AscendC::Std::get<2>(tie_result), c);
    EXPECT_EQ(AscendC::Std::get<3>(tie_result), flag);
}


// test AscendC::Std::tie handling of empty bindings
TEST_F(TupleTest, EmptyTieTie) {
    auto tie_result = AscendC::Std::tie();
    // check if the size of the empty tuple binding is 0
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(tie_result)>::value, 0);
}

// test the functionality of forward_as_tuple
TEST_F(TupleTest, BasicUsageForwardAsTuple) {
    // use AscendC::Std::forward_as_tuple to wrap the data
    auto t1 = AscendC::Std::forward_as_tuple(1, 2.0);

    int& aaa = AscendC::Std::get<0>(t1);
    double& bbb = AscendC::Std::get<1>(t1);

    // verify if the size of the tuples is 2
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(t1)>::value, 2);
}

// testing the combination of forward_as_tuple with forwarding references
TEST_F(TupleTest, ForwardingReferencesForwardAsTuple) {
    int x = 5;
    double y = 3.14;
    char c = 'a';

    // use AscendC::Std::forward_as_tuple to wrap forwarding references
    auto t2 = AscendC::Std::forward_as_tuple(AscendC::Std::forward<int>(x), AscendC::Std::forward<double>(y), AscendC::Std::forward<char>(c));

    // verify if the size of the tuple is 3
    EXPECT_EQ(AscendC::Std::tuple_size<decltype(t2)>::value, 3);

    // verify whether the elements in the tuple are corrent
    EXPECT_EQ(AscendC::Std::get<0>(t2), 5);
    EXPECT_EQ(AscendC::Std::get<1>(t2), 3.14);
    EXPECT_EQ(AscendC::Std::get<2>(t2), 'a');
}

// testing the use of std::forward_as_tuple in template functions
template <typename... Args>
void processTupleRight(AscendC::Std::tuple<Args...>&& tuple) {
    EXPECT_EQ(AscendC::Std::get<0>(tuple), 100);
    EXPECT_EQ(AscendC::Std::get<1>(tuple), 3.14159);
}

TEST_F(TupleTest, TemplateFunctionUsage1) {
    // passing rvalues as tuple
    processTupleRight(AscendC::Std::forward_as_tuple(100, 3.14159));
}

// testing the use of std::forward_as_tuple in template functions
template <typename... Args>
void processTupleLeft(AscendC::Std::tuple<Args...>&& tuple) {
    EXPECT_EQ(AscendC::Std::get<0>(tuple), 5);
    EXPECT_EQ(AscendC::Std::get<1>(tuple), 2.71);
}

TEST_F(TupleTest, TemplateFunctionUsage2) {
    int a = 5;
    double b = 2.71;

    // pass parameters as a tuple
    processTupleLeft(AscendC::Std::forward_as_tuple(a, b));
}