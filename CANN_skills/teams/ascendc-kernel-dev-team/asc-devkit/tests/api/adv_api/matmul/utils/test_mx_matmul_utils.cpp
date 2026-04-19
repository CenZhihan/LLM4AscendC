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
#include "impl/adv_api/detail/matmul/utils/matmul_param.h"
#include "impl/adv_api/detail/matmul/policy/matmul_policy.h"
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/utils/matmul_call_back.h"
#include "impl/adv_api/detail/matmul/utils/matmul_utils.h"

using namespace std;
using namespace AscendC;

class TestMatmulUtilsMx : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using AS_TYPE_GM = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using INPUT_AS_TYPE_GM = MatmulInputScaleAType<AS_TYPE_GM, fp8_e8m0_t>;

    using AS_TYPE_UB = MatmulTypeWithScale<TPosition::VECOUT, TPosition::VECOUT, CubeFormat::ND,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using INPUT_AS_TYPE_UB = MatmulInputScaleAType<AS_TYPE_UB, fp8_e8m0_t>;

    using AS_TYPE_L1 = MatmulTypeWithScale<TPosition::TSCM, TPosition::TSCM, CubeFormat::ND,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using INPUT_AS_TYPE_L1 = MatmulInputScaleAType<AS_TYPE_L1, fp8_e8m0_t>;

    using A_TYPE_GM = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
    using INPUT_A_TYPE_GM = MatmulInputAType<A_TYPE_GM, typename A_TYPE_GM::T>;

    using A_TYPE_UB = MatmulType<TPosition::VECOUT, CubeFormat::ND, half, false>;
    using INPUT_A_TYPE_UB = MatmulInputAType<A_TYPE_UB, typename A_TYPE_UB::T>;

    using A_TYPE_L1 = MatmulType<TPosition::TSCM, CubeFormat::ND, half, false>;
    using INPUT_A_TYPE_L1 = MatmulInputAType<A_TYPE_L1, typename A_TYPE_L1::T>;

    using AS_TYPE_GM_L1 = MatmulTypeWithScale<TPosition::GM, TPosition::TSCM, CubeFormat::ND,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using AS_TYPE_GM_TP = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e4m3fn_t, true, TPosition::GM, CubeFormat::ND, true, TPosition::GM>;

    using BS_TYPE_GM = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using INPUT_BS_TYPE_GM = MatmulInputScaleBType<BS_TYPE_GM, fp8_e8m0_t>;

    using BS_TYPE_GM_TP = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e4m3fn_t, true, TPosition::GM, CubeFormat::ND, true, TPosition::GM>;
    using BS_TYPE_GM_F4 = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp4x2_e1m2_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;

    using B_TYPE_GM = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    using INPUT_B_TYPE_GM = MatmulInputBType<B_TYPE_GM, typename B_TYPE_GM::T>;
};

TEST_F(TestMatmulUtilsMx, test_mx_matmul_utils_AuxGetC0Size) {
    EXPECT_EQ(AuxGetC0Size<float>(), 8);
    EXPECT_EQ(AuxGetC0Size<int32_t>(), 8);
    EXPECT_EQ(AuxGetC0Size<uint32_t>(), 8);

    EXPECT_EQ(AuxGetC0Size<half>(), 16);
    EXPECT_EQ(AuxGetC0Size<bfloat16_t>(), 16);
    EXPECT_EQ(AuxGetC0Size<int16_t>(), 16);
    EXPECT_EQ(AuxGetC0Size<uint16_t>(), 16);

    EXPECT_EQ(AuxGetC0Size<int8_t>(), 32);
    EXPECT_EQ(AuxGetC0Size<uint8_t>(), 32);
    EXPECT_EQ(AuxGetC0Size<hifloat8_t>(), 32);
    EXPECT_EQ(AuxGetC0Size<fp8_e4m3fn_t>(), 32);
    EXPECT_EQ(AuxGetC0Size<fp8_e5m2_t>(), 32);
    EXPECT_EQ(AuxGetC0Size<fp8_e8m0_t>(), 32);

    EXPECT_EQ(AuxGetC0Size<int4b_t>(), 64);
    EXPECT_EQ(AuxGetC0Size<fp4x2_e1m2_t>(), 64);
    EXPECT_EQ(AuxGetC0Size<fp4x2_e2m1_t>(), 64);
}

TEST_F(TestMatmulUtilsMx, test_mx_matmul_utils_IsSupportB8) {
    EXPECT_FALSE(IsSupportB8<float>());
    EXPECT_FALSE(IsSupportB8<int32_t>());
    EXPECT_FALSE(IsSupportB8<uint32_t>());

    EXPECT_FALSE(IsSupportB8<half>());
    EXPECT_FALSE(IsSupportB8<bfloat16_t>());
    EXPECT_FALSE(IsSupportB8<int16_t>());
    EXPECT_FALSE(IsSupportB8<uint16_t>());

    EXPECT_TRUE(IsSupportB8<int8_t>());
    EXPECT_FALSE(IsSupportB8<uint8_t>());
    EXPECT_TRUE(IsSupportB8<hifloat8_t>());
    EXPECT_TRUE(IsSupportB8<fp8_e4m3fn_t>());
    EXPECT_TRUE(IsSupportB8<fp8_e5m2_t>());
    EXPECT_FALSE(IsSupportB8<fp8_e8m0_t>());

    EXPECT_FALSE(IsSupportB8<int4b_t>());
    EXPECT_FALSE(IsSupportB8<fp4x2_e1m2_t>());
    EXPECT_FALSE(IsSupportB8<fp4x2_e2m1_t>());
}

TEST_F(TestMatmulUtilsMx, test_mx_matmul_utils_IsSupportB4) {
    EXPECT_FALSE(IsSupportB4<float>());
    EXPECT_FALSE(IsSupportB4<int32_t>());
    EXPECT_FALSE(IsSupportB4<uint32_t>());

    EXPECT_FALSE(IsSupportB4<half>());
    EXPECT_FALSE(IsSupportB4<bfloat16_t>());
    EXPECT_FALSE(IsSupportB4<int16_t>());
    EXPECT_FALSE(IsSupportB4<uint16_t>());

    EXPECT_FALSE(IsSupportB4<int8_t>());
    EXPECT_FALSE(IsSupportB4<uint8_t>());
    EXPECT_FALSE(IsSupportB4<hifloat8_t>());
    EXPECT_FALSE(IsSupportB4<fp8_e4m3fn_t>());
    EXPECT_FALSE(IsSupportB4<fp8_e5m2_t>());
    EXPECT_FALSE(IsSupportB4<fp8_e8m0_t>());

    EXPECT_TRUE(IsSupportB4<int4b_t>());
    EXPECT_TRUE(IsSupportB4<fp4x2_e1m2_t>());
    EXPECT_TRUE(IsSupportB4<fp4x2_e2m1_t>());
}

TEST_F(TestMatmulUtilsMx, test_mx_matmul_utils_PhyMxScalePosIsL1) {
    EXPECT_FALSE(PhyMxScalePosIsL1<INPUT_AS_TYPE_GM>());
    EXPECT_FALSE(PhyMxScalePosIsL1<INPUT_AS_TYPE_UB>());
    EXPECT_TRUE(PhyMxScalePosIsL1<INPUT_AS_TYPE_L1>());

    EXPECT_FALSE(PhyMxScalePosIsL1<INPUT_A_TYPE_GM>());
    EXPECT_FALSE(PhyMxScalePosIsL1<INPUT_A_TYPE_UB>());
    EXPECT_FALSE(PhyMxScalePosIsL1<INPUT_A_TYPE_L1>());
}

TEST_F(TestMatmulUtilsMx, test_mx_matmul_utils_PhyMxScalePosIsUB) {
    EXPECT_FALSE(PhyMxScalePosIsUB<INPUT_AS_TYPE_GM>());
    EXPECT_TRUE(PhyMxScalePosIsUB<INPUT_AS_TYPE_UB>());
    EXPECT_FALSE(PhyMxScalePosIsUB<INPUT_AS_TYPE_L1>());

    EXPECT_FALSE(PhyMxScalePosIsUB<INPUT_A_TYPE_GM>());
    EXPECT_FALSE(PhyMxScalePosIsUB<INPUT_A_TYPE_UB>());
    EXPECT_FALSE(PhyMxScalePosIsUB<INPUT_A_TYPE_L1>());
}

TEST_F(TestMatmulUtilsMx, test_mx_matmul_utils_PhyMxScalePosIsGM) {
    EXPECT_TRUE(PhyMxScalePosIsGM<INPUT_AS_TYPE_GM>());
    EXPECT_FALSE(PhyMxScalePosIsGM<INPUT_AS_TYPE_UB>());
    EXPECT_FALSE(PhyMxScalePosIsGM<INPUT_AS_TYPE_L1>());

    EXPECT_FALSE(PhyMxScalePosIsGM<INPUT_A_TYPE_GM>());
    EXPECT_FALSE(PhyMxScalePosIsGM<INPUT_A_TYPE_UB>());
    EXPECT_FALSE(PhyMxScalePosIsGM<INPUT_A_TYPE_L1>());
}

TEST_F(TestMatmulUtilsMx, test_mx_matmul_utils_GetBitSize) {
    EXPECT_EQ(GetBitSize<float>(), 32);
    EXPECT_EQ(GetBitSize<int32_t>(), 32);
    EXPECT_EQ(GetBitSize<uint32_t>(), 32);

    EXPECT_EQ(GetBitSize<half>(), 16);
    EXPECT_EQ(GetBitSize<bfloat16_t>(), 16);
    EXPECT_EQ(GetBitSize<int16_t>(), 16);
    EXPECT_EQ(GetBitSize<uint16_t>(), 16);

    EXPECT_EQ(GetBitSize<int8_t>(), 8);
    EXPECT_EQ(GetBitSize<uint8_t>(), 8);
    EXPECT_EQ(GetBitSize<hifloat8_t>(), 8);
    EXPECT_EQ(GetBitSize<fp8_e4m3fn_t>(), 8);
    EXPECT_EQ(GetBitSize<fp8_e5m2_t>(), 8);
    EXPECT_EQ(GetBitSize<fp8_e8m0_t>(), 8);

    EXPECT_EQ(GetBitSize<int4b_t>(), 4);
    EXPECT_EQ(GetBitSize<fp4x2_e1m2_t>(), 4);
    EXPECT_EQ(GetBitSize<fp4x2_e2m1_t>(), 4);
}

TEST_F(TestMatmulUtilsMx, test_mx_matmul_utils_IsScaleTransWithInlv) {
    EXPECT_FALSE(IsScaleTransWithInlv<AS_TYPE_GM>);
    EXPECT_FALSE(IsScaleTransWithInlv<AS_TYPE_UB>);
    EXPECT_FALSE(IsScaleTransWithInlv<AS_TYPE_L1>);
    EXPECT_TRUE(IsScaleTransWithInlv<AS_TYPE_GM_L1>);
}

TEST_F(TestMatmulUtilsMx, test_mx_matmul_utils_IsL1BNeedTrans) {
    EXPECT_FALSE((IsL1BNeedTrans<AS_TYPE_GM, BS_TYPE_GM, CFG_NORM>()));
    EXPECT_TRUE((IsL1BNeedTrans<AS_TYPE_GM, BS_TYPE_GM_F4, CFG_NORM>()));
    EXPECT_FALSE((IsL1BNeedTrans<AS_TYPE_GM, BS_TYPE_GM, CFG_MDL>()));
    EXPECT_TRUE((IsL1BNeedTrans<AS_TYPE_GM, BS_TYPE_GM_F4, CFG_MDL>()));
}

TEST_F(TestMatmulUtilsMx, test_mx_matmul_utils_GetTransBDataType) {
    GetTransBDataType<AS_TYPE_GM, BS_TYPE_GM, CFG_NORM>();
    GetTransBDataType<AS_TYPE_GM, BS_TYPE_GM_F4, CFG_NORM>();
    GetTransBDataType<AS_TYPE_GM, BS_TYPE_GM, CFG_MDL>();
    GetTransBDataType<AS_TYPE_GM, BS_TYPE_GM_F4, CFG_MDL>();

    GetTransBDataType<A_TYPE_GM, B_TYPE_GM, CFG_NORM>();
    GetTransBDataType<A_TYPE_GM, BS_TYPE_GM_F4, CFG_NORM>();
    GetTransBDataType<A_TYPE_GM, B_TYPE_GM, CFG_MDL>();
    GetTransBDataType<A_TYPE_GM, BS_TYPE_GM_F4, CFG_MDL>();
}

TEST_F(TestMatmulUtilsMx, test_mx_matmul_utils_IsScaleTag) {
    EXPECT_TRUE(IsScaleTag<INPUT_AS_TYPE_GM>());
    EXPECT_FALSE(IsScaleTag<INPUT_A_TYPE_GM>());

    EXPECT_TRUE(IsScaleTag<INPUT_BS_TYPE_GM>());
    EXPECT_FALSE(IsScaleTag<INPUT_B_TYPE_GM>());
}

TEST_F(TestMatmulUtilsMx, test_mx_matmul_utils_InputPhyPosIsL1) {
    EXPECT_FALSE(InputPhyPosIsL1<INPUT_AS_TYPE_GM>());
    EXPECT_FALSE(InputPhyPosIsL1<INPUT_AS_TYPE_UB>());
    EXPECT_TRUE(InputPhyPosIsL1<INPUT_AS_TYPE_L1>());

    EXPECT_FALSE(InputPhyPosIsL1<INPUT_A_TYPE_GM>());
    EXPECT_FALSE(InputPhyPosIsL1<INPUT_A_TYPE_UB>());
    EXPECT_TRUE(InputPhyPosIsL1<INPUT_A_TYPE_L1>());
}

TEST_F(TestMatmulUtilsMx, test_mx_matmul_utils_InputPhyPosIsUB) {
    EXPECT_FALSE(InputPhyPosIsUB<INPUT_AS_TYPE_GM>());
    EXPECT_TRUE(InputPhyPosIsUB<INPUT_AS_TYPE_UB>());
    EXPECT_FALSE(InputPhyPosIsUB<INPUT_AS_TYPE_L1>());

    EXPECT_FALSE(InputPhyPosIsUB<INPUT_A_TYPE_GM>());
    EXPECT_TRUE(InputPhyPosIsUB<INPUT_A_TYPE_UB>());
    EXPECT_FALSE(InputPhyPosIsUB<INPUT_A_TYPE_L1>());
}

TEST_F(TestMatmulUtilsMx, test_mx_matmul_utils_SupportMXFP8) {
    EXPECT_EQ(SupportMXFP8<int8_t>, 0);
    EXPECT_EQ(SupportMXFP8<uint8_t>, 0);
    EXPECT_EQ(SupportMXFP8<hifloat8_t>, 0);
    EXPECT_EQ(SupportMXFP8<fp8_e4m3fn_t>, 1);
    EXPECT_EQ(SupportMXFP8<fp8_e5m2_t>, 1);
    EXPECT_EQ(SupportMXFP8<fp8_e8m0_t>, 0);
}

TEST_F(TestMatmulUtilsMx, test_mx_matmul_utils_IsMxEnableUnitFlag) {
    EXPECT_TRUE((IsMxDisableUnitFlag<AS_TYPE_GM, BS_TYPE_GM, CFG_NORM>));
    EXPECT_TRUE((IsMxDisableUnitFlag<AS_TYPE_GM_TP, BS_TYPE_GM, CFG_NORM>));

    EXPECT_FALSE((IsMxDisableUnitFlag<AS_TYPE_GM, BS_TYPE_GM_TP, CFG_NORM>));
    EXPECT_TRUE((IsMxDisableUnitFlag<AS_TYPE_GM_TP, BS_TYPE_GM_TP, CFG_NORM>));
}
