/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include "tensor_api/stub/cce_stub.h"
#include "include/experimental/tensor_api/tensor.h"


class Tensor_Api_Layout : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

TEST_F(Tensor_Api_Layout, ShapeAndStrideOperation)
{
    using namespace AscendC::Te;

    Shape<int, int, int> shapeInit{11, 22, 33};
    Stride<int, int, int> strideInit{44, 55, 66};

    Shape<int, int, int> shapeMake = MakeShape(10, 20, 30);
    Stride<int, int, int> strideMake = MakeStride(40, 50, 60);

    auto shapeMakeInt = MakeShape(AscendC::Std::Int<111>{}, AscendC::Std::Int<222>{}, AscendC::Std::Int<333>{});
    auto strideMakeInt = MakeStride(AscendC::Std::Int<444>{}, AscendC::Std::Int<555>{}, AscendC::Std::Int<666>{});

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

TEST_F(Tensor_Api_Layout, IsTupleOperation)
{
    using namespace AscendC::Te;

    Shape<int, int, int> shapeInit{11, 22, 33};
    Stride<int, int, int> strideInit{44, 55, 66};

    Shape<int, int, int> shapeMake = MakeShape(10, 20, 30);
    Stride<int, int, int> strideMake = MakeStride(40, 50, 60);

    auto shapeMakeInt = MakeShape(AscendC::Std::Int<111>{}, AscendC::Std::Int<222>{}, AscendC::Std::Int<333>{});
    auto strideMakeInt = MakeStride(AscendC::Std::Int<444>{}, AscendC::Std::Int<555>{}, AscendC::Std::Int<666>{});

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

TEST_F(Tensor_Api_Layout, LayoutOperation)
{
    using namespace AscendC::Te;

    Shape<int, int, int> shape = MakeShape(10, 20, 30);
    Stride<int, int, int> stride = MakeStride(1, 100, 200);

    auto layoutMake = MakeLayout(shape, stride);

    Layout<Shape<int, int, int>, Stride<int, int, int>> layoutInit(shape, stride);

    EXPECT_EQ(AscendC::Std::get<0>(layoutMake.Shape()), 10);
    EXPECT_EQ(AscendC::Std::get<1>(layoutMake.Shape()), 20);
    EXPECT_EQ(AscendC::Std::get<2>(layoutMake.Shape()), 30);

    EXPECT_EQ(AscendC::Std::get<0>(layoutMake.Stride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(layoutMake.Stride()), 100);
    EXPECT_EQ(AscendC::Std::get<2>(layoutMake.Stride()), 200);

    EXPECT_EQ(AscendC::Std::get<0>(layoutInit.Shape()), 10);
    EXPECT_EQ(AscendC::Std::get<1>(layoutInit.Shape()), 20);
    EXPECT_EQ(AscendC::Std::get<2>(layoutInit.Shape()), 30);

    EXPECT_EQ(AscendC::Std::get<0>(layoutInit.Stride()), 1);
    EXPECT_EQ(AscendC::Std::get<1>(layoutInit.Stride()), 100);
    EXPECT_EQ(AscendC::Std::get<2>(layoutInit.Stride()), 200);

    EXPECT_EQ(layoutMake.Rank(), 3);
    EXPECT_EQ(layoutMake.Rank<0>(), 1);
    EXPECT_EQ(Rank(layoutMake), 3);

    auto shapeTuple = GetShape(Select<1, 2>(layoutMake));
    EXPECT_EQ(AscendC::Std::get<0>(shapeTuple), 20);
    EXPECT_EQ(Size(layoutMake), 6000);
    EXPECT_EQ(layoutMake.Size(), 6000);
    EXPECT_EQ(layoutMake.Size(), 6000);
    EXPECT_EQ(Coshape(layoutMake), 7710);
    EXPECT_EQ(Cosize(layoutMake), 7710);
}

TEST_F(Tensor_Api_Layout, IsLayoutOperation)
{
    using namespace AscendC::Te;

    Shape<int, int, int> shape = MakeShape(10, 20, 30);
    Stride<int, int, int> stride = MakeStride(1, 100, 200);

    auto layoutMake = MakeLayout(shape, stride);

    Layout<Shape<int, int, int>, Stride<int, int, int>> layoutInit(shape, stride);

    EXPECT_EQ(is_layout<decltype(shape)>::value, false);
    EXPECT_EQ(is_layout<decltype(stride)>::value, false);
    EXPECT_EQ(is_layout<decltype(layoutMake)>::value, true);
    EXPECT_EQ(is_layout<decltype(layoutInit)>::value, true);

    EXPECT_EQ(is_layout_v<decltype(shape)>, false);
    EXPECT_EQ(is_layout_v<decltype(stride)>, false);
    EXPECT_EQ(is_layout_v<decltype(layoutMake)>, true);
    EXPECT_EQ(is_layout_v<decltype(layoutInit)>, true);
}

TEST_F(Tensor_Api_Layout, MakeLayoutByShapeOperation)
{
    using namespace AscendC::Te;

    Shape<int, int, int> shape = MakeShape(2, 3, 4);
    Stride<int, int, int> stride = MakeStride(1, 2, 6);

    auto layoutMake1 = MakeLayout(shape, stride);
    auto layoutMake2 = MakeLayout(shape);

    EXPECT_EQ(AscendC::Std::get<1>(layoutMake1.Stride()), 2);
    EXPECT_EQ(AscendC::Std::get<1>(layoutMake2.Stride()), 4);
    EXPECT_EQ(Size(layoutMake1), 24);
    EXPECT_EQ(Size(layoutMake2), 24);
    EXPECT_EQ(layoutMake1.Size(), 24);
    EXPECT_EQ(layoutMake2.Size(), 24);
    EXPECT_EQ(layoutMake2.Capacity(), 24);
}

TEST_F(Tensor_Api_Layout, LayoutSizeOperation)
{
    using namespace AscendC::Te;

    using shape = Shape<AscendC::Std::Int<16>, AscendC::Std::Int<16>>;
    using stride = Stride<AscendC::Std::Int<1>, AscendC::Std::Int<16>>;
    Layout<shape, stride> layoutMake;
    Tile<int, int> tile = MakeTile(1, 2);
    Coord<int, int> coord = MakeCoord(1, 2);

    EXPECT_EQ(layoutMake.size, 256);
    EXPECT_EQ(AscendC::Std::get<0>(layoutMake.Shape()), 16);
    EXPECT_EQ(AscendC::Std::get<0>(GetShape(layoutMake)), 16);
    EXPECT_EQ(AscendC::Std::get<0>(layoutMake.Stride()), 1);
    EXPECT_EQ(AscendC::Std::get<0>(GetStride(layoutMake)), 1);
    EXPECT_EQ(layoutMake.Size(), 256);
}

TEST_F(Tensor_Api_Layout, StaticLayoutOperation)
{
    using namespace AscendC::Te;

    using TwoDimT = AscendC::Std::tuple<AscendC::Std::Int<3>, AscendC::Std::Int<4>>;
    using TwoDimU = AscendC::Std::tuple<AscendC::Std::Int<2>, AscendC::Std::Int<1>>;
    EXPECT_EQ((StaticLayoutSize<TwoDimT, TwoDimU>::size), 6);

    using FourDimT = AscendC::Std::tuple<AscendC::Std::tuple<AscendC::Std::Int<3>, AscendC::Std::Int<4>>,
                                         AscendC::Std::tuple<AscendC::Std::Int<1>, AscendC::Std::Int<2>>>;
    using FourDimU = AscendC::Std::tuple<AscendC::Std::tuple<AscendC::Std::Int<4>, AscendC::Std::Int<5>>,
                                         AscendC::Std::tuple<AscendC::Std::Int<2>, AscendC::Std::Int<3>>>;
    EXPECT_EQ((StaticLayoutSize<FourDimT, FourDimU>::size), 20);
}

TEST_F(Tensor_Api_Layout, MakeMxLayout)
{
    using namespace AscendC::Te;
    
    auto NnLayout = MakeNnLayout<fp8_e8m0_t>(16, 48); // (scaleK, n)
    auto Zzlayout = MakeZzLayout<fp8_e8m0_t>(48, 16); // (m, scaleK)
    auto MxANDLayout = MakeScaleANDLayout<fp8_e8m0_t>(48, 16); // (m, scaleK) 不转置
    auto MxADNLayout = MakeScaleADNLayout<fp8_e8m0_t>(48, 16); // (m, scaleK) 转置
    auto MxBNDLayout = MakeScaleBNDLayout<fp8_e8m0_t>(16, 48); // (scaleK, n) 不转置
    auto MxBDNLayout = MakeScaleBDNLayout<fp8_e8m0_t>(16, 48); // (scaleK, n) 转置

    EXPECT_EQ(AscendC::Std::get<0>(GetShape<0>(NnLayout)), 2);
    EXPECT_EQ(AscendC::Std::get<0>(GetShape<1>(NnLayout)), 16);
    EXPECT_EQ(AscendC::Std::get<0>(GetShape<0>(Zzlayout)), 16);
    EXPECT_EQ(AscendC::Std::get<0>(GetShape<1>(Zzlayout)), 2);
    EXPECT_EQ(AscendC::Std::get<1>(GetShape<0>(MxANDLayout)), 48);
    EXPECT_EQ(AscendC::Std::get<1>(GetShape<1>(MxANDLayout)), 16);
    EXPECT_EQ(AscendC::Std::get<0>(GetShape<0>(MxADNLayout)), 1); // 48
    EXPECT_EQ(AscendC::Std::get<0>(GetShape<1>(MxADNLayout)), 2);
    EXPECT_EQ(AscendC::Std::get<1>(GetShape<0>(MxBNDLayout)), 8); // 1
    EXPECT_EQ(AscendC::Std::get<1>(GetShape<1>(MxBNDLayout)), 48); // 8
    EXPECT_EQ(AscendC::Std::get<0>(GetShape<0>(MxBDNLayout)), 1);
    EXPECT_EQ(AscendC::Std::get<0>(GetShape<1>(MxBDNLayout)), 1);
}

TEST_F(Tensor_Api_Layout, GetOperation)
{
    using namespace AscendC::Te;

    Shape<int, int, int> shape = MakeShape(10, 20, 30);
    Stride<int, int, int> stride = MakeStride(1, 100, 200);
    auto layout = MakeLayout(shape, stride);

    auto getAll = Get(layout);
    auto getShapeAll = GetShape(getAll);
    auto getStrideAll = GetStride(getAll);
    EXPECT_EQ(AscendC::Std::get<0>(getShapeAll), 10);
    EXPECT_EQ(AscendC::Std::get<0>(getStrideAll), 1);

    auto get1 = Get<1>(layout);
    auto get1Shape = GetShape(get1);
    auto get1Stride = GetStride(get1);
    EXPECT_EQ(AscendC::Std::get<0>(get1Shape), 20);
    EXPECT_EQ(AscendC::Std::get<0>(get1Stride), 100);

    auto nestedShape = MakeShape(MakeShape(2, 3), MakeShape(4, 5));
    auto nestedStride = MakeStride(MakeStride(1, 2), MakeStride(3, 4));
    auto nestedLayout = MakeLayout(nestedShape, nestedStride);

    auto nestedGetAll = Get(nestedLayout);
    auto nestedGetShape = GetShape(nestedGetAll);
    auto nestedGetStride = GetStride(nestedGetAll);
    auto nestedGetShape0 = AscendC::Std::get<0>(nestedGetShape);
    auto nestedGetStride1 = AscendC::Std::get<1>(nestedGetStride);
    EXPECT_EQ(AscendC::Std::get<0>(nestedGetShape0), 2);
    EXPECT_EQ(AscendC::Std::get<1>(nestedGetShape0), 3);
    EXPECT_EQ(AscendC::Std::get<0>(nestedGetStride1), 3);
    EXPECT_EQ(AscendC::Std::get<1>(nestedGetStride1), 4);

    auto nestedGet1 = Get<1>(nestedLayout);
    auto nestedGet1Shape = GetShape(nestedGet1);
    auto nestedGet1Stride = GetStride(nestedGet1);
    EXPECT_EQ(AscendC::Std::get<0>(nestedGet1Shape), 4);
    EXPECT_EQ(AscendC::Std::get<1>(nestedGet1Stride), 4);

    auto nestedGet10 = Get<1, 0>(nestedLayout);
    auto nestedGet10Shape = GetShape(nestedGet10);
    auto nestedGet10Stride = GetStride(nestedGet10);
    EXPECT_EQ(AscendC::Std::get<0>(nestedGet10Shape), 4);
    EXPECT_EQ(AscendC::Std::get<0>(nestedGet10Stride), 3);
}

TEST_F(Tensor_Api_Layout, GetOperationInt)
{
    using namespace AscendC::Te;

    auto shape = MakeShape(AscendC::Std::Int<40>{}, AscendC::Std::Int<50>{}, AscendC::Std::Int<60>{});
    auto stride = MakeStride(AscendC::Std::Int<100>{}, AscendC::Std::Int<200>{}, AscendC::Std::Int<300>{});
    auto layout = MakeLayout(shape, stride);

    auto getAll = Get(layout);
    auto getShapeAll = GetShape(getAll);
    auto getStrideAll = GetStride(getAll);
    EXPECT_EQ(AscendC::Std::get<0>(getShapeAll).value, 40);
    EXPECT_EQ(AscendC::Std::get<0>(getStrideAll).value, 100);

    auto get1 = Get<1>(layout);
    auto get1Shape = GetShape(get1);
    auto get1Stride = GetStride(get1);
    EXPECT_EQ(AscendC::Std::get<0>(get1Shape).value, 50);
    EXPECT_EQ(AscendC::Std::get<0>(get1Stride).value, 200);

    auto nestedShape = MakeShape(MakeShape(AscendC::Std::Int<40>{}, AscendC::Std::Int<50>{}),
                                 MakeShape(AscendC::Std::Int<60>{}, AscendC::Std::Int<70>{}));
    auto nestedStride = MakeStride(MakeStride(AscendC::Std::Int<100>{}, AscendC::Std::Int<200>{}),
                                   MakeStride(AscendC::Std::Int<300>{}, AscendC::Std::Int<400>{}));
    auto nestedLayout = MakeLayout(nestedShape, nestedStride);

    auto nestedGetAll = Get(nestedLayout);
    auto nestedGetShape = GetShape(nestedGetAll);
    auto nestedGetStride = GetStride(nestedGetAll);
    auto nestedGetShape0 = AscendC::Std::get<0>(nestedGetShape);
    auto nestedGetStride1 = AscendC::Std::get<1>(nestedGetStride);
    EXPECT_EQ(AscendC::Std::get<0>(nestedGetShape0).value, 40);
    EXPECT_EQ(AscendC::Std::get<1>(nestedGetShape0).value, 50);
    EXPECT_EQ(AscendC::Std::get<0>(nestedGetStride1).value, 300);
    EXPECT_EQ(AscendC::Std::get<1>(nestedGetStride1).value, 400);

    auto nestedGet1 = Get<1>(nestedLayout);
    auto nestedGet1Shape = GetShape(nestedGet1);
    auto nestedGet1Stride = GetStride(nestedGet1);
    EXPECT_EQ(AscendC::Std::get<0>(nestedGet1Shape).value, 60);
    EXPECT_EQ(AscendC::Std::get<1>(nestedGet1Stride).value, 400);

    auto nestedGet11 = Get<1, 1>(nestedLayout);
    auto nestedGet11Shape = GetShape(nestedGet11);
    auto nestedGet11Stride = GetStride(nestedGet11);
    EXPECT_EQ(AscendC::Std::get<0>(nestedGet11Shape), 70);
    EXPECT_EQ(AscendC::Std::get<0>(nestedGet11Stride), 400);
}

TEST_F(Tensor_Api_Layout, SelectOperation)
{
    using namespace AscendC::Te;

    Shape<int, int, int> shape = MakeShape(10, 20, 30);
    Stride<int, int, int> stride = MakeStride(1, 100, 200);
    auto layout = MakeLayout(shape, stride);

    auto selectAll = Select(layout);
    auto selectAllShape = GetShape(selectAll);
    auto selectAllStride = GetStride(selectAll);
    EXPECT_EQ(AscendC::Std::get<0>(selectAllShape), 10);
    EXPECT_EQ(AscendC::Std::get<2>(selectAllStride), 200);

    auto select1 = Select<1>(layout);
    auto select1Shape = GetShape(select1);
    auto select1Stride = GetStride(select1);
    EXPECT_EQ(AscendC::Std::get<0>(select1Shape), 20);
    EXPECT_EQ(AscendC::Std::get<0>(select1Stride), 100);

    auto select12 = Select<1, 2>(layout);
    auto select12Shape = GetShape(select12);
    auto select12Stride = GetStride(select12);
    EXPECT_EQ(AscendC::Std::get<1>(select12Shape), 30);
    EXPECT_EQ(AscendC::Std::get<1>(select12Stride), 200);

    auto nestedShape = MakeShape(MakeShape(2, 3), MakeShape(4, 5));
    auto nestedStride = MakeStride(MakeStride(1, 2), MakeStride(3, 4));
    auto nestedLayout = MakeLayout(nestedShape, nestedStride);

    auto nestedSelectAll = Select(nestedLayout);
    auto nestedSelectAllShape = GetShape(nestedSelectAll);
    auto nestedSelectAllStride = GetStride(nestedSelectAll);
    auto nestedSelectAllShape0 = AscendC::Std::get<0>(nestedSelectAllShape);
    auto nestedSelectAllShape1 = AscendC::Std::get<1>(nestedSelectAllShape);
    auto nestedSelectAllStride0 = AscendC::Std::get<0>(nestedSelectAllStride);
    auto nestedSelectAllStride1 = AscendC::Std::get<1>(nestedSelectAllStride);

    EXPECT_EQ(AscendC::Std::get<0>(nestedSelectAllShape0), 2);
    EXPECT_EQ(AscendC::Std::get<0>(nestedSelectAllShape1), 4);
    EXPECT_EQ(AscendC::Std::get<1>(nestedSelectAllStride0), 2);
    EXPECT_EQ(AscendC::Std::get<1>(nestedSelectAllStride1), 4);

    auto nestedSelect0 = Select<0>(nestedLayout);
    auto nestedSelect0Shape = GetShape(nestedSelect0);
    auto nestedSelect0Stride = GetStride(nestedSelect0);
    EXPECT_EQ(AscendC::Std::get<0>(nestedSelect0Shape), 2);
    EXPECT_EQ(AscendC::Std::get<1>(nestedSelect0Stride), 2);

    auto nestedSelect01 = Select<0, 1>(nestedLayout);
    auto nestedSelect01Shape = GetShape(nestedSelect01);
    auto nestedSelect01Stride = GetStride(nestedSelect01);
    auto nestedSelect01Shape0 = AscendC::Std::get<0>(nestedSelect01Shape);
    auto nestedSelect01Shape1 = AscendC::Std::get<1>(nestedSelect01Shape);
    auto nestedSelect01Stride0 = AscendC::Std::get<0>(nestedSelect01Stride);
    auto nestedSelect01Stride1 = AscendC::Std::get<1>(nestedSelect01Stride);
    EXPECT_EQ(AscendC::Std::get<0>(nestedSelect01Shape0), 2);
    EXPECT_EQ(AscendC::Std::get<0>(nestedSelect01Shape1), 4);
    EXPECT_EQ(AscendC::Std::get<1>(nestedSelect01Stride0), 2);
    EXPECT_EQ(AscendC::Std::get<1>(nestedSelect01Stride1), 4);
}

TEST_F(Tensor_Api_Layout, SelectOperationInt)
{
    using namespace AscendC::Te;

    auto shape = MakeShape(AscendC::Std::Int<70>{}, AscendC::Std::Int<80>{}, AscendC::Std::Int<90>{});
    auto stride = MakeStride(AscendC::Std::Int<10>{}, AscendC::Std::Int<200>{}, AscendC::Std::Int<300>{});
    auto layout = MakeLayout(shape, stride);

    auto selectAll = Select(layout);
    auto selectAllShape = GetShape(selectAll);
    auto selectAllStride = GetStride(selectAll);
    EXPECT_EQ(AscendC::Std::get<0>(selectAllShape).value, 70);
    EXPECT_EQ(AscendC::Std::get<2>(selectAllStride).value, 300);

    auto select1 = Select<1>(layout);
    auto select1Shape = GetShape(select1);
    auto select1Stride = GetStride(select1);
    EXPECT_EQ(AscendC::Std::get<0>(select1Shape).value, 80);
    EXPECT_EQ(AscendC::Std::get<0>(select1Stride).value, 200);

    auto select12 = Select<1, 2>(layout);
    auto select12Shape = GetShape(select12);
    auto select12Stride = GetStride(select12);
    EXPECT_EQ(AscendC::Std::get<1>(select12Shape).value, 90);
    EXPECT_EQ(AscendC::Std::get<1>(select12Stride).value, 300);

    auto nestedShape = MakeShape(MakeShape(AscendC::Std::Int<20>{}, AscendC::Std::Int<30>{}),
                                 MakeShape(AscendC::Std::Int<40>{}, AscendC::Std::Int<50>{}));
    auto nestedStride = MakeStride(MakeStride(AscendC::Std::Int<10>{}, AscendC::Std::Int<20>{}),
                                   MakeStride(AscendC::Std::Int<30>{}, AscendC::Std::Int<40>{}));
    auto nestedLayout = MakeLayout(nestedShape, nestedStride);

    auto nestedSelectAll = Select(nestedLayout);
    auto nestedSelectAllShape = GetShape(nestedSelectAll);
    auto nestedSelectAllStride = GetStride(nestedSelectAll);
    auto nestedSelectAllShape0 = AscendC::Std::get<0>(nestedSelectAllShape);
    auto nestedSelectAllShape1 = AscendC::Std::get<1>(nestedSelectAllShape);
    auto nestedSelectAllStride0 = AscendC::Std::get<0>(nestedSelectAllStride);
    auto nestedSelectAllStride1 = AscendC::Std::get<1>(nestedSelectAllStride);

    EXPECT_EQ(AscendC::Std::get<0>(nestedSelectAllShape0).value, 20);
    EXPECT_EQ(AscendC::Std::get<0>(nestedSelectAllShape1).value, 40);
    EXPECT_EQ(AscendC::Std::get<1>(nestedSelectAllStride0).value, 20);
    EXPECT_EQ(AscendC::Std::get<1>(nestedSelectAllStride1).value, 40);

    auto nestedSelect0 = Select<0>(nestedLayout);
    auto nestedSelect0Shape = GetShape(nestedSelect0);
    auto nestedSelect0Stride = GetStride(nestedSelect0);
    EXPECT_EQ(AscendC::Std::get<0>(nestedSelect0Shape).value, 20);
    EXPECT_EQ(AscendC::Std::get<1>(nestedSelect0Stride).value, 20);

    auto nestedSelect01 = Select<0, 1>(nestedLayout);
    auto nestedSelect01Shape = GetShape(nestedSelect01);
    auto nestedSelect01Stride = GetStride(nestedSelect01);
    auto nestedSelect01Shape0 = AscendC::Std::get<0>(nestedSelect01Shape);
    auto nestedSelect01Shape1 = AscendC::Std::get<1>(nestedSelect01Shape);
    auto nestedSelect01Stride0 = AscendC::Std::get<0>(nestedSelect01Stride);
    auto nestedSelect01Stride1 = AscendC::Std::get<1>(nestedSelect01Stride);
    EXPECT_EQ(AscendC::Std::get<0>(nestedSelect01Shape0).value, 20);
    EXPECT_EQ(AscendC::Std::get<0>(nestedSelect01Shape1).value, 40);
    EXPECT_EQ(AscendC::Std::get<1>(nestedSelect01Stride0).value, 20);
    EXPECT_EQ(AscendC::Std::get<1>(nestedSelect01Stride1).value, 40);
}

// GetCapacity()方法输入Shape与Stride，根据数据类型存在不同的处理方式：
//    1) 若Shape与Stride均为tuple类型，取的是每个维度shape * stride最大值
//    2) 若为int，则返回乘积
TEST_F(Tensor_Api_Layout, TestCapacity)
{
    using namespace AscendC::Te;
    // 两个维度，按维度乘积为{9, 8}
    auto shape1Dim0 = 3;
    auto shape1Dim1 = 2;
    auto stride1Dim0 = 3;
    auto stride1Dim1 = 4;
    auto shape1 = MakeShape(shape1Dim0, shape1Dim1);
    auto stride1 = MakeStride(stride1Dim0, stride1Dim1);
    auto layout1 = MakeLayout(shape1, stride1);
    EXPECT_EQ(layout1.Capacity(), AscendC::Std::max(shape1Dim0 * stride1Dim0, shape1Dim1 * stride1Dim1));
    EXPECT_EQ(GetCapacity(shape1, stride1), AscendC::Std::max(shape1Dim0 * stride1Dim0, shape1Dim1 * stride1Dim1));

    // 三个维度，按维度乘积为{21, 20, 24}
    auto shape2Dim0 = 7;
    auto shape2Dim1 = 2;
    auto shape2Dim2 = 6;
    auto stride2Dim0 = 3;
    auto stride2Dim1 = 10;
    auto stride2Dim2 = 4;
    auto shape2 = MakeShape(shape2Dim0, shape2Dim1, shape2Dim2);
    auto stride2 = MakeStride(stride2Dim0, stride2Dim1, stride2Dim2);
    auto layout2 = MakeLayout(shape2, stride2);
    EXPECT_EQ(layout2.Capacity(),
              AscendC::Std::max(AscendC::Std::max(shape2Dim0 * stride2Dim0, shape2Dim1 * stride2Dim1), shape2Dim2 * stride2Dim2));
    EXPECT_EQ(GetCapacity(shape2, stride2),
              AscendC::Std::max(AscendC::Std::max(shape2Dim0 * stride2Dim0, shape2Dim1 * stride2Dim1), shape2Dim2 * stride2Dim2));

    int shape3 = 7;
    int stride3 = 17;
    EXPECT_EQ(GetCapacity(shape3, stride3), shape3 * stride3);

    // 按维度乘积为{{3, 60}, {40, 480}}
    auto shape4Dim00 = 3;
    auto shape4Dim01 = 4;
    auto shape4Dim10 = 5;
    auto shape4Dim11 = 8;
    auto stride4Dim00 = 1;
    auto stride4Dim01 = 15;
    auto stride4Dim10 = 8;
    auto stride4Dim11 = 60;
    auto shape4 = MakeShape(MakeShape(shape4Dim00, shape4Dim01), MakeShape(shape4Dim10, shape4Dim11));
    auto stride4 = MakeShape(MakeShape(stride4Dim00, stride4Dim01), MakeShape(stride4Dim10, stride4Dim11));
    auto layout4 = MakeLayout(shape4, stride4);
    EXPECT_EQ(layout4.Capacity(),
              AscendC::Std::max(AscendC::Std::max(AscendC::Std::max(shape4Dim00 * stride4Dim00, shape4Dim01 * stride4Dim01), shape4Dim10 * stride4Dim10),
                  shape4Dim11 * stride4Dim11));
    EXPECT_EQ(GetCapacity(shape4, stride4),
              AscendC::Std::max(AscendC::Std::max(AscendC::Std::max(shape4Dim00 * stride4Dim00, shape4Dim01 * stride4Dim01), shape4Dim10 * stride4Dim10),
                  shape4Dim11 * stride4Dim11));
}

// 对const对象，layout方法会返回const类型的引用
TEST_F(Tensor_Api_Layout, TestLayout)
{
    using namespace AscendC::Te;
    Shape<int, int> shape1 = MakeShape(3, 2);
    Stride<int, int> stride1 = MakeStride(3, 4);

    auto layoutObj = MakeLayout(shape1, stride1);
    auto& layoutRes = layoutObj.layout();
    EXPECT_EQ(&layoutObj, &layoutRes);
    auto checkLayoutType = AscendC::Std::is_same_v<decltype(layoutRes), Layout<Shape<int, int>, Stride<int, int>>&>;
    EXPECT_TRUE(checkLayoutType);

    const auto constLayoutObj = MakeLayout(shape1, stride1);
    auto& constLayoutRes = constLayoutObj.layout();
    EXPECT_EQ(&constLayoutObj, &constLayoutRes);
    auto checkConstLayoutType =
        AscendC::Std::is_same_v<decltype(constLayoutRes), const Layout<Shape<int, int>, Stride<int, int>>&>;
    auto checkConstLayoutConst = AscendC::Std::is_const_v<std::remove_reference_t<decltype(constLayoutRes)>>;
    EXPECT_TRUE(checkConstLayoutType);
    EXPECT_TRUE(checkConstLayoutConst);
}

TEST_F(Tensor_Api_Layout, TestShapeStride)
{
    using namespace AscendC::Te;
    auto shape1 = MakeShape(3, 2);
    auto stride1 = MakeStride(1, 4);
    auto layoutObj = MakeLayout(shape1, stride1);
    auto layoutShape = layoutObj.Shape();
    auto layoutStride = layoutObj.Stride();
    EXPECT_EQ(AscendC::Std::get<0>(layoutShape), 3);
    EXPECT_EQ(AscendC::Std::get<1>(layoutShape), 2);
    EXPECT_EQ(AscendC::Std::get<0>(layoutStride), 1);
    EXPECT_EQ(AscendC::Std::get<1>(layoutStride), 4);

    auto shape2 = MakeShape(AscendC::Std::Int<7>{}, AscendC::Std::Int<3>{}, AscendC::Std::Int<6>{});
    auto stride2 = MakeStride(AscendC::Std::Int<2>{}, AscendC::Std::Int<1>{}, AscendC::Std::Int<5>{});
    const auto constLayoutObj = MakeLayout(shape2, stride2);
    auto constLayoutShape = constLayoutObj.Shape();
    auto constLayoutStride = constLayoutObj.Stride();
    EXPECT_EQ(AscendC::Std::get<0>(constLayoutShape), AscendC::Std::Int<7>{});
    EXPECT_EQ(AscendC::Std::get<1>(constLayoutShape), AscendC::Std::Int<3>{});
    EXPECT_EQ(AscendC::Std::get<2>(constLayoutShape), AscendC::Std::Int<6>{});
    EXPECT_EQ(AscendC::Std::get<0>(constLayoutStride), AscendC::Std::Int<2>{});
    EXPECT_EQ(AscendC::Std::get<1>(constLayoutStride), AscendC::Std::Int<1>{});
    EXPECT_EQ(AscendC::Std::get<2>(constLayoutStride), AscendC::Std::Int<5>{});
}

// 1) tuple tuple tuple:计算coord和stride在每个维度上的乘积之和
// 2) int tuple tuple:
//        若只剩最后一个维度：直接计算 coord * stride[0]
//        反之，mod = coord % shape[i], coord = coord / shape[i], res = res + mod * stride[i]
// 3) int int int: 返回coord * stride
TEST_F(Tensor_Api_Layout, TestOperator)
{
    using namespace AscendC::Te;
    // 输出为1 * 1 + 2 * 4 = 9
    Coord<int, int> coord1 = MakeCoord(1, 2);
    Shape<int, int> shape1 = MakeShape(3, 2);
    Stride<int, int> stride1 = MakeStride(1, 4);
    auto layoutObj1 = MakeLayout(shape1, stride1);
    EXPECT_EQ(layoutObj1(coord1), 9);

    // 首轴: div = 30 / 7 = 4, mod = 30 % 7 = 2, res = 2 * 2 = 4
    // 第2轴: div = 4 / 3 = 1, mod = 4 % 3 = 1, res = 4 + 1 * 1 = 5
    // 末轴: res = 5 + 1 * 5 = 10
    auto coord2 = AscendC::Std::Int<30>{};
    Shape<int, int, int> shape2 = MakeShape(7, 3, 6);
    Stride<int, int, int> stride2 = MakeStride(2, 1, 5);
    auto layoutObj2 = MakeLayout(shape2, stride2);
    EXPECT_EQ(layoutObj2(coord2), 10);

    // 输出为7 * 5 = 35
    auto coord3 = AscendC::Std::Int<7>{};
    auto shape3 = AscendC::Std::Int<2>{};
    auto stride3 = AscendC::Std::Int<5>{};
    auto idxRes3 = Crd2Idx(coord3, shape3, stride3);
    EXPECT_EQ(idxRes3, 35);

    // 2*2维度的layout
    auto shape4 = MakeShape(MakeShape(3, 4), MakeShape(5, 8));
    auto stride4 = MakeShape(MakeShape(1, 15), MakeShape(8, 60));
    auto layoutObj4 = MakeLayout(shape4, stride4);
    // 输出为1 * 1 + 2 * 15 + 8 * 3 + 60 * 4 = 295
    auto coord4 = MakeCoord(MakeCoord(1, 2), MakeCoord(3, 4));
    EXPECT_EQ(layoutObj4(coord4), 295);
    // stride[0]与coord[0]的idx: 首轴: div=8, mod=1, res=1; 第2轴: remain=8, res=8*15+1=121
    // stride[1]与coord[1]的idx: 首轴: div=6, mod=0, res=0; 第2轴: remain=6, res=360
    auto coord5 = MakeCoord(25, 30);
    EXPECT_EQ(layoutObj4(coord5), 481);
    // 首轴: prod=12, div=40, mod=11, 首轴: div=3, mod=2, res=2; 第2轴: remain=3, res=3*15+2=47
    // 第2轴: 使用40，首轴: div=8, mod=0, res=0; 第2轴: remain=8, res=8*60=480
    auto coord6 = AscendC::Std::Int<491>{};
    EXPECT_EQ(layoutObj4(coord6), 527);
}

// Rank()方法:调用GetRank获取Shape的rank信息（维度）
// Size()方法:调用GetSize获取Shape的size信息（维度的乘积）
TEST_F(Tensor_Api_Layout, TestRankSize)
{
    using namespace AscendC::Te;
    auto shape1Dim0 = 3;
    auto shape1Dim1 = 2;
    auto stride1Dim0 = 1;
    auto stride1Dim1 = 4;
    auto shape1 = MakeShape(shape1Dim0, shape1Dim1);
    auto stride1 = MakeStride(stride1Dim0, stride1Dim1);
    auto layoutObj1 = MakeLayout(shape1, stride1);
    EXPECT_EQ(layoutObj1.Size(), shape1Dim0 * shape1Dim1);
    EXPECT_EQ(layoutObj1.Rank(), 2);

    auto shape2Dim0 = 7;
    auto shape2Dim1 = 3;
    auto shape2Dim2 = 6;
    auto stride2Dim0 = 2;
    auto stride2Dim1 = 1;
    auto stride2Dim2 = 5;
    auto shape2 = MakeShape(shape2Dim0, shape2Dim1, shape2Dim2);
    auto stride2 = MakeStride(stride2Dim0, stride2Dim1, stride2Dim2);
    auto layoutObj2 = MakeLayout(shape2, stride2);
    EXPECT_EQ(layoutObj2.Size(), shape2Dim0 * shape2Dim1 * shape2Dim2);
    EXPECT_EQ(layoutObj2.Rank(), 3);

    auto shape3Dim00 = 3;
    auto shape3Dim01 = 4;
    auto shape3Dim10 = 5;
    auto shape3Dim11 = 8;
    auto stride3Dim00 = 1;
    auto stride3Dim01 = 15;
    auto stride3Dim10 = 8;
    auto stride3Dim11 = 60;
    auto shape3 = MakeShape(MakeShape(shape3Dim00, shape3Dim01), MakeShape(shape3Dim10, shape3Dim11));
    auto stride3 = MakeShape(MakeShape(stride3Dim00, stride3Dim01), MakeShape(stride3Dim10, stride3Dim11));
    auto layoutObj3 = MakeLayout(shape3, stride3);
    EXPECT_EQ(layoutObj3.Size(), shape3Dim00 * shape3Dim01 * shape3Dim10 * shape3Dim11);
    EXPECT_EQ(layoutObj3.Size<0>(), shape3Dim00 * shape3Dim01);
    EXPECT_EQ(layoutObj3.Size<1>(), shape3Dim10 * shape3Dim11);
    EXPECT_EQ(layoutObj3.Rank(), 2);
    EXPECT_EQ(layoutObj3.Rank<0>(), 2);
    EXPECT_EQ(layoutObj3.Rank<1>(), 2);
}

TEST_F(Tensor_Api_Layout, TestMakeShapeStride)
{
    using namespace AscendC::Te;
    auto shape1 = MakeShape(3, 2);
    auto stride1 = MakeStride(1, 4);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(shape1)>, true);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(stride1)>, true);
    EXPECT_EQ(GetShape<0>(shape1), 3);
    EXPECT_EQ(GetShape<1>(shape1), 2);

    auto shape2 = MakeShape(AscendC::Std::Int<11>{}, AscendC::Std::Int<23>{}, AscendC::Std::Int<37>{});
    auto stride2 = MakeStride(AscendC::Std::Int<3>{}, AscendC::Std::Int<7>{}, AscendC::Std::Int<5>{});
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(shape2)>, true);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(stride2)>, true);
    EXPECT_EQ(GetShape<0>(shape2), AscendC::Std::Int<11>{});
    EXPECT_EQ(GetShape<1>(shape2), AscendC::Std::Int<23>{});
    EXPECT_EQ(GetShape<2>(shape2), AscendC::Std::Int<37>{});

    auto shape3 = MakeShape(MakeShape(AscendC::Std::Int<11>{}, AscendC::Std::Int<22>{}),
                            MakeShape(AscendC::Std::Int<33>{}, AscendC::Std::Int<44>{}));
    auto stride3 = MakeStride(MakeShape(AscendC::Std::Int<55>{}, AscendC::Std::Int<66>{}),
                              MakeShape(AscendC::Std::Int<77>{}, AscendC::Std::Int<88>{}));
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(shape3)>, true);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(stride3)>, true);
    EXPECT_EQ((GetShape<0, 0>(shape3)), AscendC::Std::Int<11>{});
    EXPECT_EQ((GetShape<0, 1>(shape3)), AscendC::Std::Int<22>{});
    EXPECT_EQ((GetShape<1, 0>(shape3)), AscendC::Std::Int<33>{});
    EXPECT_EQ((GetShape<1, 1>(shape3)), AscendC::Std::Int<44>{});
}

TEST_F(Tensor_Api_Layout, TestMakeTileCoord)
{
    using namespace AscendC::Te;
    auto tile1 = MakeTile(3, 2);
    auto coord1 = MakeCoord(1, 4);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(tile1)>, true);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(coord1)>, true);
    EXPECT_EQ(AscendC::Std::get<0>(tile1), 3);
    EXPECT_EQ(AscendC::Std::get<1>(tile1), 2);
    EXPECT_EQ(AscendC::Std::get<0>(coord1), 1);
    EXPECT_EQ(AscendC::Std::get<1>(coord1), 4);

    auto tile2Dim0 = AscendC::Std::Int<11>{};
    auto tile2Dim1 = AscendC::Std::Int<23>{};
    auto tile2Dim2 = AscendC::Std::Int<37>{};
    auto coord2Dim0 = AscendC::Std::Int<3>{};
    auto coord2Dim1 = AscendC::Std::Int<7>{};
    auto coord2Dim2 = AscendC::Std::Int<5>{};
    auto tile2 = MakeTile(tile2Dim0, tile2Dim1, tile2Dim2);
    auto coord2 = MakeCoord(coord2Dim0, coord2Dim1, coord2Dim2);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(tile2)>, true);
    EXPECT_EQ(AscendC::Std::is_tuple_v<decltype(coord2)>, true);
    EXPECT_EQ(AscendC::Std::get<0>(tile2), tile2Dim0);
    EXPECT_EQ(AscendC::Std::get<1>(tile2), tile2Dim1);
    EXPECT_EQ(AscendC::Std::get<2>(tile2), tile2Dim2);
    EXPECT_EQ(AscendC::Std::get<0>(coord2), coord2Dim0);
    EXPECT_EQ(AscendC::Std::get<1>(coord2), coord2Dim1);
    EXPECT_EQ(AscendC::Std::get<2>(coord2), coord2Dim2);
}

// MakeLayout有两种形式，一种传入Shape与Stride来构造，另一种是传入Shape，Stride自动计算
// stride的计算逻辑:
//   若shape[0]为tuple(总维度为2且shape[0].size() == shape[1].size())，调用ComputeStride：
//        stride[0][0] = 1; stride[0][i] = shape[0][i-1]*shape[1][i-1]*stride[0][i-1]
//        stride[1][i] = shape[0][i]*stride[0][i]
//   若shape[0]非tuple，则调用ComputeFlatStride（尾轴为1，外轴为内侧轴shape的乘积）
TEST_F(Tensor_Api_Layout, TestMakeLayout)
{
    using namespace AscendC::Te;
    auto shape1Dim0 = AscendC::Std::Int<3>{};
    auto shape1Dim1 = AscendC::Std::Int<2>{};
    auto stride1Dim0 = AscendC::Std::Int<1>{};
    auto stride1Dim1 = AscendC::Std::Int<4>{};
    auto shape1 = MakeShape(shape1Dim0, shape1Dim1);
    auto stride1 = MakeStride(stride1Dim0, stride1Dim1);
    auto layout1 = MakeLayout(shape1, stride1);
    EXPECT_EQ(GetShape<0>(layout1), shape1Dim0);
    EXPECT_EQ(GetShape<1>(layout1), shape1Dim1);
    EXPECT_EQ(GetStride<0>(layout1), stride1Dim0);
    EXPECT_EQ(GetStride<1>(layout1), stride1Dim1);

    // 此时最内侧stride为1，次内侧为13，外侧为13 * 11 = 143
    auto shape2Dim0 = AscendC::Std::Int<7>{};
    auto shape2Dim1 = AscendC::Std::Int<11>{};
    auto shape2Dim2 = AscendC::Std::Int<13>{};
    auto shape2 = MakeShape(shape2Dim0, shape2Dim1, shape2Dim2);
    auto layout2 = MakeLayout(shape2);
    EXPECT_EQ(GetShape<0>(layout2), shape2Dim0);
    EXPECT_EQ(GetShape<1>(layout2), shape2Dim1);
    EXPECT_EQ(GetShape<2>(layout2), shape2Dim2);
    EXPECT_EQ(GetStride<0>(layout2), shape2Dim1 * shape2Dim2);
    EXPECT_EQ(GetStride<1>(layout2), shape2Dim2);
    EXPECT_EQ(GetStride<2>(layout2), AscendC::Std::Int<1>{});

    // stride[0][0] = 1; stride[0][1] = 2 * 3 * 1 = 6;
    // stride[1][0] = 1 * 2 = 2; stride[1][1] = 3 * 6 = 18;
    auto shape3Dim00 = AscendC::Std::Int<2>{};
    auto shape3Dim01 = AscendC::Std::Int<3>{};
    auto shape3Dim10 = AscendC::Std::Int<3>{};
    auto shape3Dim11 = AscendC::Std::Int<6>{};
    auto shape3 = MakeShape(MakeShape(shape3Dim00, shape3Dim01), MakeShape(shape3Dim10, shape3Dim11));
    auto layout3 = MakeLayout(shape3);
    EXPECT_EQ(AscendC::Std::get<0>(GetShape<0>(layout3)), shape3Dim00);
    EXPECT_EQ(AscendC::Std::get<1>(GetShape<0>(layout3)), shape3Dim01);
    EXPECT_EQ(AscendC::Std::get<0>(GetShape<1>(layout3)), shape3Dim10);
    EXPECT_EQ(AscendC::Std::get<1>(GetShape<1>(layout3)), shape3Dim11);
    EXPECT_EQ(AscendC::Std::get<0>(GetStride<0>(layout3)), AscendC::Std::Int<1>{});
    EXPECT_EQ(AscendC::Std::get<1>(GetStride<0>(layout3)), shape3Dim00 * shape3Dim01);
    EXPECT_EQ(AscendC::Std::get<0>(GetStride<1>(layout3)), shape3Dim00 * AscendC::Std::get<0>(GetStride<0>(layout3)));
    EXPECT_EQ(AscendC::Std::get<1>(GetStride<1>(layout3)), shape3Dim01 * AscendC::Std::get<1>(GetStride<0>(layout3)));
}

// Shape为{{16, row / 16}, {32 / SizeOf(T), col / (32 / SizeOf(T))}}
// Stride为{{32 / SizeOf(T), 32 / SizeOf(T) * 16}, {1, row * (32 / SizeOf(T))}}
TEST_F(Tensor_Api_Layout, TestMakeNzLayout)
{
    using namespace AscendC::Te;
    size_t row1 = 256UL;
    size_t col1 = 16UL;
    auto layout1 = MakeNzLayout<float>(row1, col1);
    auto shapeRow1 = AscendC::Std::get<0>(GetShape(layout1));
    auto shapeCol1 = AscendC::Std::get<1>(GetShape(layout1));
    auto strideRow1 = AscendC::Std::get<0>(GetStride(layout1));
    auto strideCol1 = AscendC::Std::get<1>(GetStride(layout1));
    EXPECT_EQ(AscendC::Std::get<0>(shapeRow1), 16);
    EXPECT_EQ(AscendC::Std::get<1>(shapeRow1), row1 / 16);
    EXPECT_EQ(AscendC::Std::get<0>(shapeCol1), 32 / sizeof(float));
    EXPECT_EQ(AscendC::Std::get<1>(shapeCol1), col1 / (32 / sizeof(float)));
    EXPECT_EQ(AscendC::Std::get<0>(strideRow1), 32 / sizeof(float));
    EXPECT_EQ(AscendC::Std::get<1>(strideRow1), 32 / sizeof(float) * 16);
    EXPECT_EQ(AscendC::Std::get<0>(strideCol1), 1);
    EXPECT_EQ(AscendC::Std::get<1>(strideCol1), row1 * 32 / sizeof(float));

    size_t row2 = 128UL;
    size_t col2 = 65536UL;
    auto layout2 = MakeNzLayout<uint8_t>(row2, col2);
    auto shapeRow2 = AscendC::Std::get<0>(GetShape(layout2));
    auto shapeCol2 = AscendC::Std::get<1>(GetShape(layout2));
    auto strideRow2 = AscendC::Std::get<0>(GetStride(layout2));
    auto strideCol2 = AscendC::Std::get<1>(GetStride(layout2));
    EXPECT_EQ(AscendC::Std::get<0>(shapeRow2), 16);
    EXPECT_EQ(AscendC::Std::get<1>(shapeRow2), row2 / 16);
    EXPECT_EQ(AscendC::Std::get<0>(shapeCol2), 32 / sizeof(uint8_t));
    EXPECT_EQ(AscendC::Std::get<1>(shapeCol2), col2 / (32 / sizeof(uint8_t)));
    EXPECT_EQ(AscendC::Std::get<0>(strideRow2), 32 / sizeof(uint8_t));
    EXPECT_EQ(AscendC::Std::get<1>(strideRow2), 32 / sizeof(uint8_t) * 16);
    EXPECT_EQ(AscendC::Std::get<0>(strideCol2), 1);
    EXPECT_EQ(AscendC::Std::get<1>(strideCol2), row2 * 32 / sizeof(uint8_t));
}

// MakeL0CLayout本质上就是构建一个uint16_t类型的NZLayout
TEST_F(Tensor_Api_Layout, TestMakeL0CLayout)
{
    using namespace AscendC::Te;
    size_t row1 = 128UL;
    size_t col1 = 32UL;
    auto layout1 = MakeL0CLayout(row1, col1);
    auto shapeRow1 = AscendC::Std::get<0>(GetShape(layout1));
    auto shapeCol1 = AscendC::Std::get<1>(GetShape(layout1));
    auto strideRow1 = AscendC::Std::get<0>(GetStride(layout1));
    auto strideCol1 = AscendC::Std::get<1>(GetStride(layout1));
    EXPECT_EQ(AscendC::Std::get<0>(shapeRow1), 16);
    EXPECT_EQ(AscendC::Std::get<1>(shapeRow1), row1 / 16);
    EXPECT_EQ(AscendC::Std::get<0>(shapeCol1), 32 / sizeof(uint16_t));
    EXPECT_EQ(AscendC::Std::get<1>(shapeCol1), col1 / (32 / sizeof(uint16_t)));
    EXPECT_EQ(AscendC::Std::get<0>(strideRow1), 32 / sizeof(uint16_t));
    EXPECT_EQ(AscendC::Std::get<1>(strideRow1), 32 / sizeof(uint16_t) * 16);
    EXPECT_EQ(AscendC::Std::get<0>(strideCol1), 1);
    EXPECT_EQ(AscendC::Std::get<1>(strideCol1), row1 * 32 / sizeof(uint16_t));

    size_t row2 = 2048UL;
    size_t col2 = 1024UL;
    auto layout2 = MakeL0CLayout(row2, col2);
    auto shapeRow2 = AscendC::Std::get<0>(GetShape(layout2));
    auto shapeCol2 = AscendC::Std::get<1>(GetShape(layout2));
    auto strideRow2 = AscendC::Std::get<0>(GetStride(layout2));
    auto strideCol2 = AscendC::Std::get<1>(GetStride(layout2));
    EXPECT_EQ(AscendC::Std::get<0>(shapeRow2), 16);
    EXPECT_EQ(AscendC::Std::get<1>(shapeRow2), row2 / 16);
    EXPECT_EQ(AscendC::Std::get<0>(shapeCol2), 32 / sizeof(uint16_t));
    EXPECT_EQ(AscendC::Std::get<1>(shapeCol2), col2 / (32 / sizeof(uint16_t)));
    EXPECT_EQ(AscendC::Std::get<0>(strideRow2), 32 / sizeof(uint16_t));
    EXPECT_EQ(AscendC::Std::get<1>(strideRow2), 32 / sizeof(uint16_t) * 16);
    EXPECT_EQ(AscendC::Std::get<0>(strideCol2), 1);
    EXPECT_EQ(AscendC::Std::get<1>(strideCol2), row2 * 32 / sizeof(uint16_t));
}

// Shape为{{1, row},{1, column}}
// Stride为{{0, column}, {0, 1}}
TEST_F(Tensor_Api_Layout, TestMakeNDLayout)
{
    using namespace AscendC::Te;
    size_t row1 = 128UL;
    size_t col1 = 32UL;
    auto layout1 = MakeNDLayout<uint32_t>(row1, col1);
    auto shapeRow1 = AscendC::Std::get<0>(GetShape(layout1));
    auto shapeCol1 = AscendC::Std::get<1>(GetShape(layout1));
    auto strideRow1 = AscendC::Std::get<0>(GetStride(layout1));
    auto strideCol1 = AscendC::Std::get<1>(GetStride(layout1));
    EXPECT_EQ(AscendC::Std::get<0>(shapeRow1), 1);
    EXPECT_EQ(AscendC::Std::get<1>(shapeRow1), row1);
    EXPECT_EQ(AscendC::Std::get<0>(shapeCol1), 1);
    EXPECT_EQ(AscendC::Std::get<1>(shapeCol1), col1);
    EXPECT_EQ(AscendC::Std::get<0>(strideRow1), 0);
    EXPECT_EQ(AscendC::Std::get<1>(strideRow1), col1);
    EXPECT_EQ(AscendC::Std::get<0>(strideCol1), 0);
    EXPECT_EQ(AscendC::Std::get<1>(strideCol1), 1);

    size_t row2 = 2048UL;
    size_t col2 = 1024UL;
    auto layout2 = MakeNDLayout<float>(row2, col2);
    auto shapeRow2 = AscendC::Std::get<0>(GetShape(layout2));
    auto shapeCol2 = AscendC::Std::get<1>(GetShape(layout2));
    auto strideRow2 = AscendC::Std::get<0>(GetStride(layout2));
    auto strideCol2 = AscendC::Std::get<1>(GetStride(layout2));
    EXPECT_EQ(AscendC::Std::get<0>(shapeRow2), 1);
    EXPECT_EQ(AscendC::Std::get<1>(shapeRow2), row2);
    EXPECT_EQ(AscendC::Std::get<0>(shapeCol2), 1);
    EXPECT_EQ(AscendC::Std::get<1>(shapeCol2), col2);
    EXPECT_EQ(AscendC::Std::get<0>(strideRow2), 0);
    EXPECT_EQ(AscendC::Std::get<1>(strideRow2), col2);
    EXPECT_EQ(AscendC::Std::get<0>(strideCol2), 0);
    EXPECT_EQ(AscendC::Std::get<1>(strideCol2), 1);
}

// Shape为{{1, row},{1, column}}
// Stride为{{0, 1}, {0, row}}
TEST_F(Tensor_Api_Layout, TestMakeDNLayout)
{
    using namespace AscendC::Te;
    size_t row1 = 128UL;
    size_t col1 = 32UL;
    auto layout1 = MakeDNLayout<uint32_t>(row1, col1);
    auto shapeRow1 = AscendC::Std::get<0>(GetShape(layout1));
    auto shapeCol1 = AscendC::Std::get<1>(GetShape(layout1));
    auto strideRow1 = AscendC::Std::get<0>(GetStride(layout1));
    auto strideCol1 = AscendC::Std::get<1>(GetStride(layout1));
    EXPECT_EQ(AscendC::Std::get<0>(shapeRow1), 1);
    EXPECT_EQ(AscendC::Std::get<1>(shapeRow1), row1);
    EXPECT_EQ(AscendC::Std::get<0>(shapeCol1), 1);
    EXPECT_EQ(AscendC::Std::get<1>(shapeCol1), col1);
    EXPECT_EQ(AscendC::Std::get<0>(strideRow1), 0);
    EXPECT_EQ(AscendC::Std::get<1>(strideRow1), 1);
    EXPECT_EQ(AscendC::Std::get<0>(strideCol1), 0);
    EXPECT_EQ(AscendC::Std::get<1>(strideCol1), row1);

    size_t row2 = 2048UL;
    size_t col2 = 1024UL;
    auto layout2 = MakeDNLayout<float>(row2, col2);
    auto shapeRow2 = AscendC::Std::get<0>(GetShape(layout2));
    auto shapeCol2 = AscendC::Std::get<1>(GetShape(layout2));
    auto strideRow2 = AscendC::Std::get<0>(GetStride(layout2));
    auto strideCol2 = AscendC::Std::get<1>(GetStride(layout2));
    EXPECT_EQ(AscendC::Std::get<0>(shapeRow2), 1);
    EXPECT_EQ(AscendC::Std::get<1>(shapeRow2), row2);
    EXPECT_EQ(AscendC::Std::get<0>(shapeCol2), 1);
    EXPECT_EQ(AscendC::Std::get<1>(shapeCol2), col2);
    EXPECT_EQ(AscendC::Std::get<0>(strideRow2), 0);
    EXPECT_EQ(AscendC::Std::get<1>(strideRow2), 1);
    EXPECT_EQ(AscendC::Std::get<0>(strideCol2), 0);
    EXPECT_EQ(AscendC::Std::get<1>(strideCol2), row2);
}

// Shape为{{32 / size, row / (32 / size)},{16, col / 16}}
// Stride为{{1, 32 / size * col}, {32 / size, 32 / size * 16}}
TEST_F(Tensor_Api_Layout, TestMakeZnLayout)
{
    using namespace AscendC::Te;
    size_t row1 = 32UL;
    size_t col1 = 64UL;
    auto layout1 = MakeZnLayout<uint32_t>(row1, col1);
    auto shapeRow1 = AscendC::Std::get<0>(GetShape(layout1));
    auto shapeCol1 = AscendC::Std::get<1>(GetShape(layout1));
    auto strideRow1 = AscendC::Std::get<0>(GetStride(layout1));
    auto strideCol1 = AscendC::Std::get<1>(GetStride(layout1));
    EXPECT_EQ(AscendC::Std::get<0>(shapeRow1), 32 / sizeof(uint32_t));
    EXPECT_EQ(AscendC::Std::get<1>(shapeRow1), row1 / (32 / sizeof(uint32_t)));
    EXPECT_EQ(AscendC::Std::get<0>(shapeCol1), 16);
    EXPECT_EQ(AscendC::Std::get<1>(shapeCol1), col1 / 16);
    EXPECT_EQ(AscendC::Std::get<0>(strideRow1), 1);
    EXPECT_EQ(AscendC::Std::get<1>(strideRow1), 32 / sizeof(uint32_t) * col1);
    EXPECT_EQ(AscendC::Std::get<0>(strideCol1), 32 / sizeof(uint32_t));
    EXPECT_EQ(AscendC::Std::get<1>(strideCol1), 32 / sizeof(uint32_t) * 16);

    size_t row2 = 2048UL;
    size_t col2 = 1024UL;
    auto layout2 = MakeZnLayout<float>(row2, col2);
    auto shapeRow2 = AscendC::Std::get<0>(GetShape(layout2));
    auto shapeCol2 = AscendC::Std::get<1>(GetShape(layout2));
    auto strideRow2 = AscendC::Std::get<0>(GetStride(layout2));
    auto strideCol2 = AscendC::Std::get<1>(GetStride(layout2));
    EXPECT_EQ(AscendC::Std::get<0>(shapeRow2), 32 / sizeof(float));
    EXPECT_EQ(AscendC::Std::get<1>(shapeRow2), row2 / (32 / sizeof(float)));
    EXPECT_EQ(AscendC::Std::get<0>(shapeCol2), 16);
    EXPECT_EQ(AscendC::Std::get<1>(shapeCol2), col2 / 16);
    EXPECT_EQ(AscendC::Std::get<0>(strideRow2), 1);
    EXPECT_EQ(AscendC::Std::get<1>(strideRow2), 32 / sizeof(float) * col2);
    EXPECT_EQ(AscendC::Std::get<0>(strideCol2), 32 / sizeof(float));
    EXPECT_EQ(AscendC::Std::get<1>(strideCol2), 32 / sizeof(float) * 16);
}

// Shape为{{16, row / 16},{32 / size, col / (32 / size)}}
// Stride为{{32 / size, 16 * col}, {1, 32 / size * 16}}
TEST_F(Tensor_Api_Layout, TestMakeZzLayout)
{
    using namespace AscendC::Te;
    size_t row1 = 32UL;
    size_t col1 = 64UL;
    auto layout1 = MakeZzLayout<uint32_t>(row1, col1);
    auto shapeRow1 = AscendC::Std::get<0>(GetShape(layout1));
    auto shapeCol1 = AscendC::Std::get<1>(GetShape(layout1));
    auto strideRow1 = AscendC::Std::get<0>(GetStride(layout1));
    auto strideCol1 = AscendC::Std::get<1>(GetStride(layout1));
    EXPECT_EQ(AscendC::Std::get<0>(shapeRow1), 16);
    EXPECT_EQ(AscendC::Std::get<1>(shapeRow1), row1 / 16);
    EXPECT_EQ(AscendC::Std::get<0>(shapeCol1), 32 / sizeof(uint32_t));
    EXPECT_EQ(AscendC::Std::get<1>(shapeCol1), col1 / (32 / sizeof(uint32_t)));
    EXPECT_EQ(AscendC::Std::get<0>(strideRow1), 32 / sizeof(uint32_t));
    EXPECT_EQ(AscendC::Std::get<1>(strideRow1), 16 * col1);
    EXPECT_EQ(AscendC::Std::get<0>(strideCol1), 1);
    EXPECT_EQ(AscendC::Std::get<1>(strideCol1), 32 / sizeof(uint32_t) * 16);

    size_t row2 = 2048UL;
    size_t col2 = 1024UL;
    auto layout2 = MakeZzLayout<float>(row2, col2);
    auto shapeRow2 = AscendC::Std::get<0>(GetShape(layout2));
    auto shapeCol2 = AscendC::Std::get<1>(GetShape(layout2));
    auto strideRow2 = AscendC::Std::get<0>(GetStride(layout2));
    auto strideCol2 = AscendC::Std::get<1>(GetStride(layout2));
    EXPECT_EQ(AscendC::Std::get<0>(shapeRow2), 16);
    EXPECT_EQ(AscendC::Std::get<1>(shapeRow2), row2 / 16);
    EXPECT_EQ(AscendC::Std::get<0>(shapeCol2), 32 / sizeof(float));
    EXPECT_EQ(AscendC::Std::get<1>(shapeCol2), col2 / (32 / sizeof(float)));
    EXPECT_EQ(AscendC::Std::get<0>(strideRow2), 32 / sizeof(float));
    EXPECT_EQ(AscendC::Std::get<1>(strideRow2), 16 * col2);
    EXPECT_EQ(AscendC::Std::get<0>(strideCol2), 1);
    EXPECT_EQ(AscendC::Std::get<1>(strideCol2), 32 / sizeof(float) * 16);
}

TEST_F(Tensor_Api_Layout, TestAttributes)
{
    using namespace AscendC::Te;
    auto shape1Dim0 = AscendC::Std::Int<3>{};
    auto shape1Dim1 = AscendC::Std::Int<2>{};
    auto stride1Dim0 = AscendC::Std::Int<7>{};
    auto stride1Dim1 = AscendC::Std::Int<1>{};
    auto shape1 = MakeShape(shape1Dim0, shape1Dim1);
    auto stride1 = MakeStride(stride1Dim0, stride1Dim1);
    auto layoutObj1 = MakeLayout(shape1, stride1);
    auto shape2Dim0 = AscendC::Std::Int<7>{};
    auto shape2Dim1 = AscendC::Std::Int<3>{};
    auto shape2Dim2 = AscendC::Std::Int<11>{};
    auto stride2Dim0 = AscendC::Std::Int<34>{};
    auto stride2Dim1 = AscendC::Std::Int<11>{};
    auto stride2Dim2 = AscendC::Std::Int<1>{};
    auto shape2 = MakeShape(shape2Dim0, shape2Dim1, shape2Dim2);
    auto stride2 = MakeStride(stride2Dim0, stride2Dim1, stride2Dim2);
    auto layoutObj2 = MakeLayout(shape2, stride2);
    auto shape3Dim00 = AscendC::Std::Int<2>{};
    auto shape3Dim01 = AscendC::Std::Int<3>{};
    auto shape3Dim10 = AscendC::Std::Int<3>{};
    auto shape3Dim11 = AscendC::Std::Int<6>{};
    auto stride3Dim00 = AscendC::Std::Int<1>{};
    auto stride3Dim01 = AscendC::Std::Int<6>{};
    auto stride3Dim10 = AscendC::Std::Int<6>{};
    auto stride3Dim11 = AscendC::Std::Int<18>{};
    auto shape3 = MakeShape(MakeShape(shape3Dim00, shape3Dim01), MakeShape(shape3Dim10, shape3Dim11));
    auto stride3 = MakeShape(MakeShape(stride3Dim00, stride3Dim01), MakeShape(stride3Dim10, stride3Dim11));
    auto layoutObj3 = MakeLayout(shape3, stride3);

    // Size:作用同layout的方法Size
    EXPECT_EQ(Size(layoutObj1), shape1Dim0 * shape1Dim1);
    EXPECT_EQ(Size(layoutObj2), shape2Dim0 * shape2Dim1 * shape2Dim2);
    EXPECT_EQ(Size<0>(layoutObj3), shape3Dim00 * shape3Dim01);
    EXPECT_EQ(Size<1>(layoutObj3), shape3Dim10 * shape3Dim11);
    EXPECT_EQ(Size(layoutObj3), shape3Dim00 * shape3Dim01 * shape3Dim10 * shape3Dim11);

    // Capacity:作用同layout的方法Capacity
    EXPECT_EQ(Capacity(layoutObj1), GetMax(shape1Dim0 * stride1Dim0, shape1Dim1 * stride1Dim1));
    EXPECT_EQ(Capacity(layoutObj2),
              GetMax(GetMax(shape2Dim0 * stride2Dim0, shape2Dim1 * stride2Dim1), shape2Dim2 * stride2Dim2));
    EXPECT_EQ(Capacity(layoutObj3),
              GetMax(GetMax(GetMax(shape3Dim00 * stride3Dim00, shape3Dim01 * stride3Dim01), shape3Dim10 * stride3Dim10),
                     shape3Dim11 * stride3Dim11));

    // Coshape:返回实际Shape空间，可以通过template设置Idx
    // 计算逻辑：逐维度(shape - 1) * stride之和+1
    // layout1的结果: (3 - 1) * 7 + (2 - 1) * 1 + 1= 16
    // layout2的结果: (7 - 1) * 34 + (3 - 1) * 11 + (11 - 1) * 1 + 1 = 237
    // layout3[0]的结果: (2 - 1) * 1 + (3 - 1) * 6 + 1 = 14
    // layout3[1]的结果: (3 - 1) * 6 + (6 - 1) * 18 + 1 = 103
    // layout3的结果: 13 + 102 + 1 = 116
    EXPECT_EQ(Coshape(layoutObj1), (shape1Dim0 - 1) * stride1Dim0 + (shape1Dim1 - 1) * stride1Dim1 + 1);
    EXPECT_EQ(Coshape(layoutObj2),
              (shape2Dim0 - 1) * stride2Dim0 + (shape2Dim1 - 1) * stride2Dim1 + (shape2Dim2 - 1) * stride2Dim2 + 1);
    EXPECT_EQ(Coshape<0>(layoutObj3), (shape3Dim00 - 1) * stride3Dim00 + (shape3Dim01 - 1) * stride3Dim01 + 1);
    EXPECT_EQ(Coshape<1>(layoutObj3), (shape3Dim10 - 1) * stride3Dim10 + (shape3Dim11 - 1) * stride3Dim11 + 1);
    EXPECT_EQ(Coshape(layoutObj3), (Coshape<0>(layoutObj3) - 1) + (Coshape<1>(layoutObj3) - 1) + 1);

    // Cosize:返回实际占用的内存，由于coshape维度为1，所以等价于cosize
    EXPECT_EQ(Cosize(layoutObj1), Coshape(layoutObj1));
    EXPECT_EQ(Cosize(layoutObj2), Coshape(layoutObj2));
    EXPECT_EQ(Cosize<0>(layoutObj3), Coshape<0>(layoutObj3));
    EXPECT_EQ(Cosize<1>(layoutObj3), Coshape<1>(layoutObj3));
    EXPECT_EQ(Cosize(layoutObj3), Coshape(layoutObj3));

    // Crd2Idx:等价于operator()
    auto coord1 = MakeCoord(1, 2);
    auto coord2 = MakeCoord(0, 1, 2);
    auto coord3 = MakeCoord(MakeCoord(0, 1), MakeCoord(2, 3));
    EXPECT_EQ(Crd2Idx(coord1, layoutObj1), 9);
    EXPECT_EQ(Crd2Idx(coord2, layoutObj2), 13);
    EXPECT_EQ(Crd2Idx(coord3, layoutObj3), 72);
}

TEST_F(Tensor_Api_Layout, TestStaticLayout)
{
    using namespace AscendC::Te;
    auto NDLayout = MakeNDLayout<half>(AscendC::Std::Int<256>{}, AscendC::Std::Int<256>{});
    auto StaticNDLayout = MakeNDLayout<half>(AscendC::Std::Int<16>{}, AscendC::Std::Int<16>{});
    auto DNLayout = MakeDNLayout<half>(AscendC::Std::Int<256>{}, AscendC::Std::Int<256>{});
    auto NzLayout = MakeNzLayout<half>(AscendC::Std::Int<256>{}, AscendC::Std::Int<256>{});
    auto ZnLayout = MakeZnLayout<half>(AscendC::Std::Int<256>{}, AscendC::Std::Int<256>{});
    auto NnLayout = MakeNnLayout<fp8_e8m0_t>(AscendC::Std::Int<16>{}, AscendC::Std::Int<32>{});
    auto scaleANDLayout = MakeScaleANDLayout<fp8_e8m0_t>(AscendC::Std::Int<16>{}, AscendC::Std::Int<32>{});
    auto scaleADNLayout = MakeScaleADNLayout<fp8_e8m0_t>(AscendC::Std::Int<16>{}, AscendC::Std::Int<32>{});
    auto scaleBNDLayout = MakeScaleBNDLayout<fp8_e8m0_t>(AscendC::Std::Int<32>{}, AscendC::Std::Int<16>{});
    auto scaleBDNLayout = MakeScaleBDNLayout<fp8_e8m0_t>(AscendC::Std::Int<32>{}, AscendC::Std::Int<16>{});

    auto NDShape = GetShape<0>(NDLayout);
    auto staticNDShape = GetShape<0>(StaticNDLayout);
    auto DNShape = GetShape<0>(DNLayout);
    auto NzShape = GetShape<0>(NzLayout);
    auto ZnShape = GetShape<0>(ZnLayout);
    auto NnShape = GetShape<1>(NnLayout);
    auto scaleANDShape = GetShape<0>(scaleANDLayout);
    auto scaleADNShape = GetShape<0>(scaleADNLayout);
    auto scaleBNDShape = GetShape<0>(scaleBNDLayout);
    auto scaleBDNShape = GetShape<0>(scaleBDNLayout);

    EXPECT_EQ(AscendC::Std::get<1>(NDShape), 256);
    EXPECT_EQ(AscendC::Std::get<1>(staticNDShape), 16);
    EXPECT_EQ(AscendC::Std::get<1>(DNShape), 256);
    EXPECT_EQ(AscendC::Std::get<1>(NzShape), 16);
    EXPECT_EQ(AscendC::Std::get<1>(ZnShape), 16);
    EXPECT_EQ(AscendC::Std::get<0>(NnShape), 16);
    EXPECT_EQ(AscendC::Std::get<0>(scaleANDShape), 1);
    EXPECT_EQ(AscendC::Std::get<0>(scaleADNShape), 1);
    EXPECT_EQ(AscendC::Std::get<0>(scaleBNDShape), 2);
    EXPECT_EQ(AscendC::Std::get<0>(scaleBDNShape), 1);
}

TEST_F(Tensor_Api_Layout, TestStructLayout)
{
    using namespace AscendC::Te;
    auto NDLayout = NDLayoutFormat<half>{}(256, 256);
    auto StaticNDLayout = NDLayoutFormat<half>::type<AscendC::Std::Int<16>{}, AscendC::Std::Int<16>{}>{};
    auto DNLayout = DNLayoutFormat<half>{}(256, 256);
    auto NzLayout = NzLayoutFormat<half>{}(256, 256);
    auto ZnLayout = ZnLayoutFormat<half>{}(256, 256);
    auto NnLayout = NnLayoutFormat<fp8_e8m0_t>{}(16, 32);
    auto scaleANDLayout = ScaleANDLayoutFormat<fp8_e8m0_t>{}(16, 32);
    auto scaleADNLayout = ScaleADNLayoutFormat<fp8_e8m0_t>{}(16, 32);
    auto scaleBNDLayout = ScaleBNDLayoutFormat<fp8_e8m0_t>{}(16, 32);
    auto scaleBDNLayout = ScaleBDNLayoutFormat<fp8_e8m0_t>{}(16, 32);

    auto NDShape = GetShape<0>(NDLayout);
    auto staticNDShape = GetShape<0>(StaticNDLayout);
    auto DNShape = GetShape<0>(DNLayout);
    auto NzShape = GetShape<0>(NzLayout);
    auto ZnShape = GetShape<0>(ZnLayout);
    auto NnShape = GetShape<1>(NnLayout);
    auto scaleANDShape = GetShape<0>(scaleANDLayout);
    auto scaleADNShape = GetShape<0>(scaleADNLayout);
    auto scaleBNDShape = GetShape<0>(scaleBNDLayout);
    auto scaleBDNShape = GetShape<0>(scaleBDNLayout);

    EXPECT_EQ(AscendC::Std::get<1>(NDShape), 256);
    EXPECT_EQ(AscendC::Std::get<1>(staticNDShape), 16);
    EXPECT_EQ(AscendC::Std::get<1>(DNShape), 256);
    EXPECT_EQ(AscendC::Std::get<1>(NzShape), 16);
    EXPECT_EQ(AscendC::Std::get<1>(ZnShape), 16);
    EXPECT_EQ(AscendC::Std::get<0>(NnShape), 16);
    EXPECT_EQ(AscendC::Std::get<0>(scaleANDShape), 1);
    EXPECT_EQ(AscendC::Std::get<0>(scaleADNShape), 1);
    EXPECT_EQ(AscendC::Std::get<0>(scaleBNDShape), 2);
    EXPECT_EQ(AscendC::Std::get<0>(scaleBDNShape), 1);
}