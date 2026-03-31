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
#include "mockcpp/mockcpp.hpp"
#include "stub_def.h"
#include "kernel_operator.h"

class TensorTraitTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

template <size_t v>
using Int = AscendC::Std::integral_constant<size_t, v>; 

TEST_F(TensorTraitTest, ShapeAndStrideOperation)
{
    AscendC::Shape<int, int, int> shapeInit{11, 22, 33};
    AscendC::Stride<int, int, int> strideInit{44, 55, 66};

    AscendC::Shape<int, int, int> shapeMake = AscendC::MakeShape(10, 20, 30);
    AscendC::Stride<int, int, int> strideMake = AscendC::MakeStride(40, 50, 60);

    auto shapeMakeInt = AscendC::MakeShape(Int<111>{}, Int<222>{}, Int<333>{});
    auto strideMakeInt = AscendC::MakeStride(Int<444>{}, Int<555>{}, Int<666>{});

    EXPECT_EQ(AscendC::Std::get<0>(shapeInit), 11);
    EXPECT_EQ(AscendC::Std::get<1>(shapeInit), 22);
    EXPECT_EQ(AscendC::Std::get<2>(shapeInit), 33);

    EXPECT_EQ(AscendC::Std::get<0>(strideInit), 44);
    EXPECT_EQ(AscendC::Std::get<1>(strideInit), 55);
    EXPECT_EQ(AscendC::Std::get<2>(strideInit), 66);

    EXPECT_EQ(AscendC::Std::get<0>(shapeMake), 10);
    EXPECT_EQ(AscendC::Std::get<1>(shapeMake), 20);
    EXPECT_EQ(AscendC::Std::get<2>(shapeMake), 30);

    EXPECT_EQ(AscendC::Std::get<0>(strideMake), 40);
    EXPECT_EQ(AscendC::Std::get<1>(strideMake), 50);
    EXPECT_EQ(AscendC::Std::get<2>(strideMake), 60);

    EXPECT_EQ(AscendC::Std::get<0>(shapeMakeInt).value, 111);
    EXPECT_EQ(AscendC::Std::get<1>(shapeMakeInt).value, 222);
    EXPECT_EQ(AscendC::Std::get<2>(shapeMakeInt).value, 333);

    EXPECT_EQ(AscendC::Std::get<0>(strideMakeInt).value, 444);
    EXPECT_EQ(AscendC::Std::get<1>(strideMakeInt).value, 555);
    EXPECT_EQ(AscendC::Std::get<2>(strideMakeInt).value, 666);
}

TEST_F(TensorTraitTest, IsTupleOperation)
{
    AscendC::Shape<int, int, int> shapeInit{11, 22, 33};
    AscendC::Stride<int, int, int> strideInit{44, 55, 66};

    AscendC::Shape<int, int, int> shapeMake = AscendC::MakeShape(10, 20, 30);
    AscendC::Stride<int, int, int> strideMake = AscendC::MakeStride(40, 50, 60);

    auto shapeMakeInt = AscendC::MakeShape(Int<111>{}, Int<222>{}, Int<333>{});
    auto strideMakeInt = AscendC::MakeStride(Int<444>{}, Int<555>{}, Int<666>{});

    EXPECT_EQ(AscendC::Std::is_tuple<decltype(shapeInit)>::value, true);
    EXPECT_EQ(AscendC::Std::is_tuple<decltype(strideInit)>::value, true);
    EXPECT_EQ(AscendC::Std::is_tuple<decltype(shapeMake)>::value, true);
    EXPECT_EQ(AscendC::Std::is_tuple<decltype(strideMake)>::value, true);
    EXPECT_EQ(AscendC::Std::is_tuple<decltype(shapeMakeInt)>::value, true);
    EXPECT_EQ(AscendC::Std::is_tuple<decltype(strideMakeInt)>::value, true);

    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(shapeInit)>, true);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(strideInit)>, true);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(shapeMake)>, true);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(strideMake)>, true);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(shapeMakeInt)>, true);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(strideMakeInt)>, true);
}

TEST_F(TensorTraitTest, LayoutOperation)
{
    AscendC::Shape<int,int,int> shape = AscendC::MakeShape(10, 20, 30);
    AscendC::Stride<int,int,int> stride = AscendC::MakeStride(1, 100, 200);

    auto layoutMake = AscendC::MakeLayout(shape, stride);

    AscendC::Layout<AscendC::Shape<int, int, int>, AscendC::Stride<int, int, int>> layoutInit(shape, stride);

    EXPECT_EQ(AscendC::Std::get<0>(layoutMake.GetShape()), 10);
    EXPECT_EQ(AscendC::Std::get<1>(layoutMake.GetShape()), 20);
    EXPECT_EQ(AscendC::Std::get<2>(layoutMake.GetShape()), 30);

    EXPECT_EQ(AscendC::Std::get<0>(layoutMake.GetStride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(layoutMake.GetStride()), 100);
    EXPECT_EQ(AscendC::Std::get<2>(layoutMake.GetStride()), 200);


    EXPECT_EQ(AscendC::Std::get<0>(layoutInit.GetShape()), 10);
    EXPECT_EQ(AscendC::Std::get<1>(layoutInit.GetShape()), 20);
    EXPECT_EQ(AscendC::Std::get<2>(layoutInit.GetShape()), 30);

    EXPECT_EQ(AscendC::Std::get<0>(layoutInit.GetStride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(layoutInit.GetStride()), 100);
    EXPECT_EQ(AscendC::Std::get<2>(layoutInit.GetStride()), 200);
}

TEST_F(TensorTraitTest, IsLayoutOperation)
{
    AscendC::Shape<int,int,int> shape = AscendC::MakeShape(10, 20, 30);
    AscendC::Stride<int,int,int> stride = AscendC::MakeStride(1, 100, 200);

    auto layoutMake = AscendC::MakeLayout(shape, stride);

    AscendC::Layout<AscendC::Shape<int, int, int>, AscendC::Stride<int, int, int>> layoutInit(shape, stride);

    EXPECT_EQ(AscendC::is_layout<decltype(shape)>::value, false);
    EXPECT_EQ(AscendC::is_layout<decltype(stride)>::value, false);
    EXPECT_EQ(AscendC::is_layout<decltype(layoutMake)>::value, true);
    EXPECT_EQ(AscendC::is_layout<decltype(layoutInit)>::value, true);

    EXPECT_EQ(AscendC::is_layout_v<decltype(shape)>, false);
    EXPECT_EQ(AscendC::is_layout_v<decltype(stride)>, false);
    EXPECT_EQ(AscendC::is_layout_v<decltype(layoutMake)>, true);
    EXPECT_EQ(AscendC::is_layout_v<decltype(layoutInit)>, true);
}

TEST_F(TensorTraitTest, TensorTraitOperation)
{
    AscendC::Shape<int,int,int> shape = AscendC::MakeShape(10, 20, 30);
    AscendC::Stride<int,int,int> stride = AscendC::MakeStride(1, 100, 200);

    auto layoutMake = AscendC::MakeLayout(shape, stride);    

    auto tensorTraitMake = AscendC::MakeTensorTrait<float, AscendC::TPosition::VECIN>(layoutMake);

    EXPECT_EQ(AscendC::Std::get<0>(tensorTraitMake.GetLayout().GetShape()), 10);
    EXPECT_EQ(AscendC::Std::get<1>(tensorTraitMake.GetLayout().GetShape()), 20);
    EXPECT_EQ(AscendC::Std::get<2>(tensorTraitMake.GetLayout().GetShape()), 30);

    EXPECT_EQ(AscendC::Std::get<0>(tensorTraitMake.GetLayout().GetStride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(tensorTraitMake.GetLayout().GetStride()), 100);
    EXPECT_EQ(AscendC::Std::get<2>(tensorTraitMake.GetLayout().GetStride()), 200);

    using TensorTraitType = AscendC::TensorTrait<half, AscendC::TPosition::VECCALC, AscendC::Layout<AscendC::Shape<int, int, int>, AscendC::Stride<int, int, int>>>;

    TensorTraitType tensorTraitInit(layoutMake);

    EXPECT_EQ(AscendC::Std::get<0>(tensorTraitInit.GetLayout().GetShape()), 10);
    EXPECT_EQ(AscendC::Std::get<1>(tensorTraitInit.GetLayout().GetShape()), 20);
    EXPECT_EQ(AscendC::Std::get<2>(tensorTraitInit.GetLayout().GetShape()), 30);

    EXPECT_EQ(AscendC::Std::get<0>(tensorTraitInit.GetLayout().GetStride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(tensorTraitInit.GetLayout().GetStride()), 100);
    EXPECT_EQ(AscendC::Std::get<2>(tensorTraitInit.GetLayout().GetStride()), 200);

    TensorTraitType tensorTraitSet;
    tensorTraitSet.SetLayout(layoutMake);

    EXPECT_EQ(AscendC::Std::get<0>(tensorTraitSet.GetLayout().GetShape()), 10);
    EXPECT_EQ(AscendC::Std::get<1>(tensorTraitSet.GetLayout().GetShape()), 20);
    EXPECT_EQ(AscendC::Std::get<2>(tensorTraitSet.GetLayout().GetShape()), 30);

    EXPECT_EQ(AscendC::Std::get<0>(tensorTraitSet.GetLayout().GetStride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(tensorTraitSet.GetLayout().GetStride()), 100);
    EXPECT_EQ(AscendC::Std::get<2>(tensorTraitSet.GetLayout().GetStride()), 200);
}

TEST_F(TensorTraitTest, IsTensorTraitOperation)
{
    AscendC::Shape<int,int,int> shape = AscendC::MakeShape(10, 20, 30);
    AscendC::Stride<int,int,int> stride = AscendC::MakeStride(1, 100, 200);

    auto layoutMake = AscendC::MakeLayout(shape, stride);    

    auto tensorTraitMake = AscendC::MakeTensorTrait<float, AscendC::TPosition::VECIN>(layoutMake);

    EXPECT_EQ(AscendC::is_tensorTrait<decltype(shape)>::value, false);
    EXPECT_EQ(AscendC::is_tensorTrait<decltype(stride)>::value, false);
    EXPECT_EQ(AscendC::is_tensorTrait<decltype(layoutMake)>::value, false);
    EXPECT_EQ(AscendC::is_tensorTrait<decltype(tensorTraitMake)>::value, true);

    EXPECT_EQ(AscendC::is_tensorTrait_v<decltype(shape)>, false);
    EXPECT_EQ(AscendC::is_tensorTrait_v<decltype(stride)>, false);
    EXPECT_EQ(AscendC::is_tensorTrait_v<decltype(layoutMake)>, false);
    EXPECT_EQ(AscendC::is_tensorTrait_v<decltype(tensorTraitMake)>, true);
}

TEST_F(TensorTraitTest, TensorOperation)
{
    AscendC::Shape<int,int,int> shape = AscendC::MakeShape(10, 20, 30);
    AscendC::Stride<int,int,int> stride = AscendC::MakeStride(1, 100, 200);

    auto layoutMake = AscendC::MakeLayout(shape, stride);

    auto tensorTraitMake = AscendC::MakeTensorTrait<float, AscendC::TPosition::VECIN>(layoutMake);

    AscendC::GlobalTensor<decltype(tensorTraitMake)> gTensor;
    AscendC::LocalTensor<decltype(tensorTraitMake)> lTensor;

    gTensor.SetTensorTrait(tensorTraitMake);
    lTensor.SetTensorTrait(tensorTraitMake);

    EXPECT_EQ(AscendC::Std::get<0>(gTensor.GetTensorTrait().GetLayout().GetShape()), 10);
    EXPECT_EQ(AscendC::Std::get<1>(gTensor.GetTensorTrait().GetLayout().GetShape()), 20);
    EXPECT_EQ(AscendC::Std::get<2>(gTensor.GetTensorTrait().GetLayout().GetShape()), 30);

    EXPECT_EQ(AscendC::Std::get<0>(gTensor.GetTensorTrait().GetLayout().GetStride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(gTensor.GetTensorTrait().GetLayout().GetStride()), 100);
    EXPECT_EQ(AscendC::Std::get<2>(gTensor.GetTensorTrait().GetLayout().GetStride()), 200);

    EXPECT_EQ(AscendC::Std::get<0>(lTensor.GetTensorTrait().GetLayout().GetShape()), 10);
    EXPECT_EQ(AscendC::Std::get<1>(lTensor.GetTensorTrait().GetLayout().GetShape()), 20);
    EXPECT_EQ(AscendC::Std::get<2>(lTensor.GetTensorTrait().GetLayout().GetShape()), 30);

    EXPECT_EQ(AscendC::Std::get<0>(lTensor.GetTensorTrait().GetLayout().GetStride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(lTensor.GetTensorTrait().GetLayout().GetStride()), 100);
    EXPECT_EQ(AscendC::Std::get<2>(lTensor.GetTensorTrait().GetLayout().GetStride()), 200);
}

TEST_F(TensorTraitTest, MakeCoordOperation)
{
    constexpr int M = 11;
    constexpr int N = 12;
    constexpr int blockM = 13;
    constexpr int blockN = 14;

    auto coord = AscendC::MakeCoord(Int<20>{}, Int<30>{});
    auto shape = AscendC::MakeShape(AscendC::MakeShape(Int<blockM>{}, Int<M/blockM>{}), AscendC::MakeShape(Int<blockN>{}, Int<N/blockN>{}));
    auto stride = AscendC::MakeStride(AscendC::MakeStride(Int<blockN>{}, Int<blockM*blockN>{}),AscendC::MakeStride(Int<1>{}, Int<M*blockN>{}));

    auto layout = AscendC::MakeLayout(shape, stride);
    auto index = layout(coord);
    EXPECT_EQ(index, 590);

    index = AscendC::Crd2Idx(coord, layout);
    EXPECT_EQ(index, 590);
}

TEST_F(TensorTraitTest, Crd2IdxOperation)
{
    auto blockCoordM    = Int<11>{};
    auto blockCoordN    = Int<12>{};
    auto baseShapeM     = Int<13>{};
    auto baseShapeN     = Int<14>{};
    auto basestrideM    = Int<15>{};
    auto basestrideN    = Int<16>{};

    auto coord = AscendC::MakeCoord(blockCoordM, blockCoordN);
    auto shape = AscendC::MakeShape(AscendC::MakeShape(baseShapeM, baseShapeM), AscendC::MakeShape(baseShapeN, baseShapeN));
    auto stride = AscendC::MakeStride(AscendC::MakeStride(basestrideM, basestrideM),AscendC::MakeStride(basestrideN, basestrideN));
    
    auto index = AscendC::Crd2Idx(coord, shape, stride);
    EXPECT_EQ(index, 357);
}

TEST_F(TensorTraitTest, Crd2IdxIntZeroOperation)
{
    auto blockCoordM    = Int<11>{};
    auto blockCoordN    = Int<12>{};
    auto baseShapeM     = Int<13>{};
    auto baseShapeN     = Int<14>{};
    auto basestrideM    = Int<15>{};
    auto basestrideN    = Int<16>{};

    auto coord = AscendC::MakeCoord(Int<0>{}, Int<0>{});
    auto shape = AscendC::MakeShape(AscendC::MakeShape(baseShapeM, baseShapeM), AscendC::MakeShape(baseShapeN, baseShapeN));
    auto stride = AscendC::MakeStride(AscendC::MakeStride(basestrideM, basestrideM),AscendC::MakeStride(basestrideN, basestrideN));
    
    auto index = AscendC::Crd2Idx(coord, shape, stride);
    EXPECT_EQ(index, 0);
}

TEST_F(TensorTraitTest, Crd2IdxCoordSingleZeroOperation)
{
    auto blockCoordM    = Int<11>{};
    auto blockCoordN    = Int<12>{};
    auto baseShapeM     = Int<13>{};
    auto baseShapeN     = Int<14>{};
    auto basestrideM    = Int<15>{};
    auto basestrideN    = Int<16>{};

    auto coord = AscendC::MakeCoord(Int<0>{}, blockCoordN);
    auto shape = AscendC::MakeShape(AscendC::MakeShape(baseShapeM, baseShapeM), AscendC::MakeShape(baseShapeN, baseShapeN));
    auto stride = AscendC::MakeStride(AscendC::MakeStride(basestrideM, basestrideM),AscendC::MakeStride(basestrideN, basestrideN));
    
    auto index = AscendC::Crd2Idx(coord, shape, stride);
    EXPECT_EQ(index, 192);
}

TEST_F(TensorTraitTest, UserFriendlyInterface)
{
    AscendC::Shape<int, int, int> shape = AscendC::MakeShape(2, 5, 32);
    AscendC::Stride<int, int, int> stride = AscendC::MakeStride(1, 9, 200);
    auto layoutMake = AscendC::MakeLayout(shape, stride);
    auto tensorTraitMake = AscendC::MakeTensorTrait<float, AscendC::TPosition::VECIN>(layoutMake);

    AscendC::GlobalTensor<decltype(tensorTraitMake)> gTensor;
    AscendC::LocalTensor<decltype(tensorTraitMake)> lTensor;

    gTensor.SetTensorTrait(tensorTraitMake);
    lTensor.SetTensorTrait(tensorTraitMake);

    using GoldenLayout = decltype(layoutMake);
    using GoldenShape = decltype(shape);
    using GoldenStride = decltype(stride);

    //===---------------- TensorTrait ----------------===//
    EXPECT_EQ(AscendC::Std::get<0>(tensorTraitMake.GetShape()), 2);
    EXPECT_EQ(AscendC::Std::get<1>(tensorTraitMake.GetShape()), 5);
    EXPECT_EQ(AscendC::Std::get<2>(tensorTraitMake.GetShape()), 32);

    EXPECT_EQ(AscendC::Std::get<0>(tensorTraitMake.GetStride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(tensorTraitMake.GetStride()), 9);
    EXPECT_EQ(AscendC::Std::get<2>(tensorTraitMake.GetStride()), 200);

    //===---------------- GlobalTensor ----------------===//
    auto gLayout = gTensor.GetLayout();
    EXPECT_TRUE((AscendC::Std::is_same<decltype(gLayout), GoldenLayout>::value));
    EXPECT_EQ(AscendC::Std::get<0>(gLayout.GetShape()), 2);
    EXPECT_EQ(AscendC::Std::get<1>(gLayout.GetShape()), 5);
    EXPECT_EQ(AscendC::Std::get<2>(gLayout.GetShape()), 32);
    EXPECT_EQ(AscendC::Std::get<0>(gLayout.GetStride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(gLayout.GetStride()), 9);
    EXPECT_EQ(AscendC::Std::get<2>(gLayout.GetStride()), 200);

    EXPECT_EQ(AscendC::Std::get<0>(gTensor.GetShape()), 2);
    EXPECT_EQ(AscendC::Std::get<1>(gTensor.GetShape()), 5);
    EXPECT_EQ(AscendC::Std::get<2>(gTensor.GetShape()), 32);
    EXPECT_EQ(AscendC::Std::get<0>(gTensor.GetStride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(gTensor.GetStride()), 9);
    EXPECT_EQ(AscendC::Std::get<2>(gTensor.GetStride()), 200);

    //===---------------- LocalTensor ----------------===//
    auto lLayout = lTensor.GetLayout();
    EXPECT_TRUE((AscendC::Std::is_same<decltype(lLayout), GoldenLayout>::value));
    EXPECT_EQ(AscendC::Std::get<0>(lLayout.GetShape()), 2);
    EXPECT_EQ(AscendC::Std::get<1>(lLayout.GetShape()), 5);
    EXPECT_EQ(AscendC::Std::get<2>(lLayout.GetShape()), 32);
    EXPECT_EQ(AscendC::Std::get<0>(lLayout.GetStride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(lLayout.GetStride()), 9);
    EXPECT_EQ(AscendC::Std::get<2>(lLayout.GetStride()), 200);

    EXPECT_EQ(AscendC::Std::get<0>(lTensor.GetShape()), 2);
    EXPECT_EQ(AscendC::Std::get<1>(lTensor.GetShape()), 5);
    EXPECT_EQ(AscendC::Std::get<2>(lTensor.GetShape()), 32);
    EXPECT_EQ(AscendC::Std::get<0>(lTensor.GetStride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(lTensor.GetStride()), 9);
    EXPECT_EQ(AscendC::Std::get<2>(lTensor.GetStride()), 200);
}

TEST_F(TensorTraitTest, UserFriendlyGetTypes)
{
    AscendC::Shape<int, int, int> shape = AscendC::MakeShape(2, 5, 32);
    AscendC::Stride<int, int, int> stride = AscendC::MakeStride(1, 9, 200);
    auto layoutMake = AscendC::MakeLayout(shape, stride);
    auto tensorTraitMake = AscendC::MakeTensorTrait<float, AscendC::TPosition::VECIN>(layoutMake);

    AscendC::GlobalTensor<decltype(tensorTraitMake)> gTensor;
    AscendC::LocalTensor<decltype(tensorTraitMake)> lTensor;

    gTensor.SetTensorTrait(tensorTraitMake);
    lTensor.SetTensorTrait(tensorTraitMake);

    using GoldenLayout = decltype(layoutMake);
    using GoldenShape = decltype(shape);
    using GoldenStride = decltype(stride);

    using TraitType = decltype(tensorTraitMake);
    EXPECT_TRUE((std::is_same<AscendC::GetLayoutType<TraitType>, GoldenLayout>::value));
    EXPECT_TRUE((std::is_same<AscendC::GetShapeType<TraitType>, GoldenShape>::value));
    EXPECT_TRUE((std::is_same<AscendC::GetStrideType<TraitType>, GoldenStride>::value));

    using GlobalTraitTensor = decltype(gTensor);
    EXPECT_TRUE((std::is_same<AscendC::GetLayoutType<GlobalTraitTensor>, GoldenLayout>::value));
    EXPECT_TRUE((std::is_same<AscendC::GetShapeType<GlobalTraitTensor>, GoldenShape>::value));
    EXPECT_TRUE((std::is_same<AscendC::GetStrideType<GlobalTraitTensor>, GoldenStride>::value));

    using LocalTraitTensor = decltype(lTensor);
    EXPECT_TRUE((std::is_same<AscendC::GetLayoutType<LocalTraitTensor>, GoldenLayout>::value));
    EXPECT_TRUE((std::is_same<AscendC::GetShapeType<LocalTraitTensor>, GoldenShape>::value));
    EXPECT_TRUE((std::is_same<AscendC::GetStrideType<LocalTraitTensor>, GoldenStride>::value));
}