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

class Tensor_Api_Atom : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

TEST_F(Tensor_Api_Atom, CopyGM2L1Operation)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 1024;

    __gm__ float src[TILE_LENGTH] = {0};
    __cbuf__ float dst[TILE_LENGTH] = {0};

    auto coord = MakeCoord(Int<20>{}, Int<30>{});

    auto gmSrc = MakeTensor(MakeGMmemPtr(src), MakeNDLayout<float>(11, 12));
    auto l1Dst = MakeTensor(MakeL1memPtr(dst), MakeNzLayout<float>(11, 12));

    auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});
    atomCopy.Call(l1Dst, gmSrc);

    atomCopy.Call(l1Dst, gmSrc, coord);

    CopyAtom<CopyTraits<CopyGM2L1, DataCopyTraitDefault>>{}.Call(l1Dst, gmSrc);

    CopyAtom<CopyTraits<CopyGM2L1, DataCopyTraitDefault>>{}.Call(l1Dst, gmSrc, coord);

    Copy(CopyAtom<CopyTraits<CopyGM2L1, DataCopyTraitDefault>>{}, l1Dst, gmSrc);

    Copy(CopyAtom<CopyTraits<CopyGM2L1, DataCopyTraitDefault>>{}, l1Dst, gmSrc, coord);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyL12L0Operation)
{
    using namespace AscendC::Te;
    using namespace AscendC::Std;

    constexpr uint32_t TILE_LENGTH = 128;

    __cbuf__ float l1Src[TILE_LENGTH] = {0};
    __ca__ float l0aDst[TILE_LENGTH] = {0};
    __cb__ float l0bDst[TILE_LENGTH] = {0};

    auto coord = MakeCoord(Int<20>{}, Int<30>{});
    auto shape = MakeShape(MakeShape(Int<11>{}, Int<12>{}), MakeShape(Int<13>{}, Int<14>{}));
    auto stride = MakeStride(MakeStride(Int<15>{}, Int<16>{}), MakeStride(Int<17>{}, Int<18>{}));

    auto srcL1 = MakeTensor(MakeL1memPtr(l1Src), MakeLayout(shape, stride));
    auto dstL0A = MakeTensor(MakeL0AmemPtr(l0aDst), MakeLayout(shape, stride));
    auto dstL0B = MakeTensor(MakeL0BmemPtr(l0bDst), MakeLayout(shape, stride));

    auto atomCopy = MakeCopy(CopyL12L0{}, LoadDataTraitDefault{});
    atomCopy.Call(dstL0A, srcL1);

    atomCopy.Call(dstL0A, srcL1, coord);

    CopyAtom<CopyTraits<CopyL12L0, LoadDataTraitDefault>>{}.Call(dstL0A, srcL1);

    CopyAtom<CopyTraits<CopyL12L0, LoadDataTraitDefault>>{}.Call(dstL0A, srcL1, coord);

    Copy(CopyAtom<CopyTraits<CopyL12L0, LoadDataTraitDefault>>{}, dstL0A, srcL1);

    Copy(CopyAtom<CopyTraits<CopyL12L0, LoadDataTraitDefault>>{}, dstL0A, srcL1, coord);


    atomCopy.Call(dstL0B, srcL1);
    atomCopy.Call(dstL0B, srcL1, coord);

    CopyAtom<CopyTraits<CopyL12L0, LoadDataTraitDefault>>{}.Call(dstL0B, srcL1);

    CopyAtom<CopyTraits<CopyL12L0, LoadDataTraitDefault>>{}.Call(dstL0B, srcL1, coord);

    Copy(CopyAtom<CopyTraits<CopyL12L0, LoadDataTraitDefault>>{}, dstL0B, srcL1);

    Copy(CopyAtom<CopyTraits<CopyL12L0, LoadDataTraitDefault>>{}, dstL0B, srcL1, coord);

    EXPECT_EQ(l0aDst[0], 0);
    EXPECT_EQ(l0bDst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyL0C2GMOperation)
{
    using namespace AscendC::Te;
    using namespace AscendC::Std;

    constexpr uint32_t TILE_LENGTH = 128;

    __cc__ float src[TILE_LENGTH] = {0};
    __gm__ float dst[TILE_LENGTH] = {0};

    auto coord = MakeCoord(Int<20>{}, Int<30>{});
    auto shape = MakeShape(MakeShape(Int<11>{}, Int<12>{}), MakeShape(Int<13>{}, Int<14>{}));
    auto stride = MakeStride(MakeStride(Int<15>{}, Int<16>{}), MakeStride(Int<17>{}, Int<18>{}));

    auto l0cSrc = MakeTensor(MakeL0CmemPtr(src), MakeLayout(shape, stride));
    auto gmDst = MakeTensor(MakeGMmemPtr(dst), MakeLayout(shape, stride));

    auto atomCopy = MakeCopy(CopyL0C2GM{}, FixpipeTraitDefault{});
    atomCopy.Call(gmDst, l0cSrc);

    atomCopy.Call(gmDst, l0cSrc, coord);

    CopyAtom<CopyTraits<CopyL0C2GM, FixpipeTraitDefault>>{}.Call(gmDst, l0cSrc);

    CopyAtom<CopyTraits<CopyL0C2GM, FixpipeTraitDefault>>{}.Call(gmDst, l0cSrc, coord);

    Copy(CopyAtom<CopyTraits<CopyL0C2GM, FixpipeTraitDefault>>{}, gmDst, l0cSrc);

    Copy(CopyAtom<CopyTraits<CopyL0C2GM, FixpipeTraitDefault>>{}, gmDst, l0cSrc, coord);
    
    FixpipeParams params;
    Copy(CopyAtom<CopyTraits<CopyL0C2GM, FixpipeTraitDefault>>{}, gmDst, l0cSrc, params);

    Copy(CopyAtom<CopyTraits<CopyL0C2GM, FixpipeTraitDefault>>{}, gmDst, l0cSrc, coord, params);

    EXPECT_EQ(dst[0], 0);
}

TEST_F(Tensor_Api_Atom, CopyL0C2GMWithOperation)
{
    using namespace AscendC;
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 128;

    __cc__ float src[TILE_LENGTH] = {0};
    __gm__ float dst[TILE_LENGTH] = {0};

    auto coord = MakeCoord(Int<20>{}, Int<30>{});
    auto shape = MakeShape(MakeShape(Int<11>{}, Int<12>{}), MakeShape(Int<13>{}, Int<14>{}));
    auto stride = MakeStride(MakeStride(Int<15>{}, Int<16>{}), MakeStride(Int<17>{}, Int<18>{}));

    auto l0cSrc = MakeTensor(MakeL0CmemPtr(src), MakeLayout(shape, stride));
    auto gmDst = MakeTensor(MakeGMmemPtr(dst), MakeLayout(shape, stride));

    auto atomCopy = MakeCopy(CopyL0C2GM{});
    atomCopy.with(12).Call(gmDst, l0cSrc);

    atomCopy.with(23).Call(gmDst, l0cSrc, coord);

    CopyAtom<CopyTraits<CopyL0C2GM>>{}.with(34).Call(gmDst, l0cSrc);

    CopyAtom<CopyTraits<CopyL0C2GM>>{}.with(45).Call(gmDst, l0cSrc, coord);

    Copy(CopyAtom<CopyTraits<CopyL0C2GM>>{}.with(56), gmDst, l0cSrc);

    Copy(CopyAtom<CopyTraits<CopyL0C2GM>>{}.with(67), gmDst, l0cSrc, coord);

    EXPECT_EQ(dst[0], 0);
}