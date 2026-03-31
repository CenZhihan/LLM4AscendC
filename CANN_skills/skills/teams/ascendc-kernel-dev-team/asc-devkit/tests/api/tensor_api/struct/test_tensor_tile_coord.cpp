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


class Tensor_Api_Coord : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

template <size_t v>
using Int = AscendC::Std::integral_constant<size_t, v>; 

TEST_F(Tensor_Api_Coord, MakeCoordOperation)
{
    using namespace AscendC::Te;

    constexpr int M = 11;
    constexpr int N = 12;
    constexpr int blockM = 13;
    constexpr int blockN = 14;

    auto coord = MakeCoord(Int<20>{}, Int<30>{});
    auto shape = MakeShape(MakeShape(Int<blockM>{}, Int<M/blockM>{}), MakeShape(Int<blockN>{}, Int<N/blockN>{}));
    auto stride = MakeStride(MakeStride(Int<blockN>{}, Int<blockM*blockN>{}),MakeStride(Int<1>{}, Int<M*blockN>{}));

    auto layout = MakeLayout(shape, stride);
    auto index = layout(coord);
    EXPECT_EQ(index, 590);

    index = Crd2Idx(coord, layout);
    EXPECT_EQ(index, 590);
}

TEST_F(Tensor_Api_Coord, Crd2IdxOperation)
{
    using namespace AscendC::Te;

    auto blockCoordM    = Int<11>{};
    auto blockCoordN    = Int<12>{};
    auto baseShapeM     = Int<13>{};
    auto baseShapeN     = Int<14>{};
    auto basestrideM    = Int<15>{};
    auto basestrideN    = Int<16>{};

    auto coord = MakeCoord(blockCoordM, blockCoordN);
    auto shape = MakeShape(MakeShape(baseShapeM, baseShapeM), MakeShape(baseShapeN, baseShapeN));
    auto stride = MakeStride(MakeStride(basestrideM, basestrideM),MakeStride(basestrideN, basestrideN));
    
    auto index = Crd2Idx(coord, shape, stride);
    EXPECT_EQ(index, 357);
}

TEST_F(Tensor_Api_Coord, Crd2IdxIntZeroOperation)
{
    using namespace AscendC::Te;

    auto blockCoordM    = Int<11>{};
    auto blockCoordN    = Int<12>{};
    auto baseShapeM     = Int<13>{};
    auto baseShapeN     = Int<14>{};
    auto basestrideM    = Int<15>{};
    auto basestrideN    = Int<16>{};

    auto coord = MakeCoord(Int<0>{}, Int<0>{});
    auto shape = MakeShape(MakeShape(baseShapeM, baseShapeM), MakeShape(baseShapeN, baseShapeN));
    auto stride = MakeStride(MakeStride(basestrideM, basestrideM),MakeStride(basestrideN, basestrideN));
    
    auto index = Crd2Idx(coord, shape, stride);
    EXPECT_EQ(index, 0);
}

TEST_F(Tensor_Api_Coord, Crd2IdxCoordSingleZeroOperation)
{
    using namespace AscendC::Te;
    
    auto blockCoordM    = Int<11>{};
    auto blockCoordN    = Int<12>{};
    auto baseShapeM     = Int<13>{};
    auto baseShapeN     = Int<14>{};
    auto basestrideM    = Int<15>{};
    auto basestrideN    = Int<16>{};

    auto coord = MakeCoord(Int<0>{}, blockCoordN);
    auto shape = MakeShape(MakeShape(baseShapeM, baseShapeM), MakeShape(baseShapeN, baseShapeN));
    auto stride = MakeStride(MakeStride(basestrideM, basestrideM),MakeStride(basestrideN, basestrideN));
    
    auto index = Crd2Idx(coord, shape, stride);
    EXPECT_EQ(index, 192);
}
