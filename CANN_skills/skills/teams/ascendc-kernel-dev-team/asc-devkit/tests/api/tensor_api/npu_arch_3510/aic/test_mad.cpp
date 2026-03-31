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

class Tensor_Api_Mad_Atom : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

TEST_F(Tensor_Api_Mad_Atom, Mmadperation)
{
    using namespace AscendC;
    using namespace AscendC::Std;
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 128;

    __ca__ float L0ABuffer[TILE_LENGTH] = {0};
    __cb__ float L0BBuffer[TILE_LENGTH] = {0};
    __cc__ float L0CBuffer[TILE_LENGTH] = {0};
    __biasbuf__ float BiasBuffer[TILE_LENGTH] = {0};

    auto dstShape = MakeShape(MakeShape(Int<16>{}, Int<12>{}), MakeShape(Int<16>{}, Int<14>{}));
    auto dstStride = MakeStride(MakeStride(Int<16>{}, Int<16>{}), MakeStride(Int<1>{}, Int<18>{}));

    auto fmShape = MakeShape(MakeShape(Int<16>{}, Int<12>{}), MakeShape(Int<32/sizeof(float)>{}, Int<14>{}));
    auto fmStride = MakeStride(MakeStride(Int<32/sizeof(float)>{}, Int<16>{}), MakeStride(Int<1>{}, Int<18>{}));

    auto filterShape = MakeShape(MakeShape(Int<32/sizeof(float)>{}, Int<12>{}), MakeShape(Int<16>{}, Int<14>{}));
    auto filterStride = MakeStride(MakeStride(Int<1>{}, Int<16>{}), MakeStride(Int<32/sizeof(float)>{}, Int<18>{}));

    auto biasShape = MakeShape(MakeShape(Int<1>{}, Int<12>{}), MakeShape(Int<1>{}, Int<14>{}));
    auto biasStride = MakeStride(MakeStride(Int<0>{}, Int<16>{}), MakeStride(Int<0>{}, Int<1>{}));

    auto L0ATensor = MakeTensor(MakeL0AmemPtr(L0ABuffer), MakeLayout(fmShape, fmStride));
    auto L0BTensor = MakeTensor(MakeL0BmemPtr(L0BBuffer), MakeLayout(filterShape, filterStride));
    auto L0CTensor = MakeTensor(MakeL0CmemPtr(L0CBuffer), MakeLayout(dstShape, dstStride));
    auto BiasTensor = MakeTensor(MakeBiasmemPtr(BiasBuffer), MakeLayout(biasShape, biasStride));

    auto atomMad = MakeMad(MmadOperation{}, MmadTraitDefault{});
    atomMad.Call(L0CTensor, L0BTensor, L0ATensor, defaultMmadParams);

    atomMad.Call(L0CTensor, L0BTensor, L0ATensor, BiasTensor, defaultMmadWithBiasParams);

    MmadAtom<MmadTraits<MmadOperation, MmadTraitDefault>>{}.Call(L0CTensor, L0BTensor, L0ATensor, defaultMmadParams);

    MmadAtom<MmadTraits<MmadOperation, MmadTraitDefault>>{}.Call(L0CTensor, L0BTensor, L0ATensor, BiasTensor, defaultMmadWithBiasParams);

    Mad(MmadAtom<MmadTraits<MmadOperation, MmadTraitDefault>>{}, L0CTensor, L0BTensor, L0ATensor, defaultMmadParams);

    Mad(MmadAtom<MmadTraits<MmadOperation, MmadTraitDefault>>{}, L0CTensor, L0BTensor, L0ATensor, BiasTensor, defaultMmadWithBiasParams);

    EXPECT_EQ(L0CBuffer[0], 0);
}