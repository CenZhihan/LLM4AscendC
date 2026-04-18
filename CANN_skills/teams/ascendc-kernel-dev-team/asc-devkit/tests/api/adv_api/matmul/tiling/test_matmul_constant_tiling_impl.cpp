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
#include "include/adv_api/matmul/tiling.h"
#include "include/adv_api/matmul/constant_tiling.h"

using namespace std;
using namespace AscendC;
namespace {
__aicore__ inline constexpr MatmulConfig GetStaticCfg(uint32_t singleM, uint32_t singleN, uint32_t singleK,
    uint32_t baseM, uint32_t baseN, uint32_t baseK)
{
    MatmulShapeParams shapeParams {singleM, singleN, singleK, baseM, baseN, baseK};
    auto mmCfg = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(shapeParams);
    return mmCfg;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ constexpr MatmulApiStaticTiling GetCustomMatmulApiTiling(const MatmulConfig &mmCFG)
{
    MatmulApiStaticTiling tiling;
    tiling = GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG);
    return tiling;
}
}

class TestMatmulMxMatmulApiTiling : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE = MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::TSCM, CubeFormat::ND,
        uint8_t, true, AscendC::TPosition::GM, CubeFormat::ND, false, AscendC::TPosition::GM>;
    using B_TYPE = MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::TSCM, CubeFormat::ND,
        uint8_t, false, AscendC::TPosition::GM, CubeFormat::ND, true, AscendC::TPosition::GM>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    constexpr MatmulConfig static CFG_MX = GetStaticCfg(64, 64, 64, 32, 32, 32);
    constexpr MatmulApiStaticTiling static staticTiling =
        GetCustomMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(CFG_MX);
    bool TestMxMatmulApiTiling(const MatmulApiStaticTiling &tiling) {
        constexpr int32_t l1Size = 512*1024;
        GetMxScaleFactor<A_TYPE, B_TYPE, BIAS_TYPE>(tiling, l1Size);
        return true;
    }
};

TEST_F(TestMatmulMxMatmulApiTiling, MxMatmulApiTiling) {
    ASSERT_TRUE(TestMxMatmulApiTiling(staticTiling));
}
