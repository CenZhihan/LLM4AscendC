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
#define private public
#define protected public
#include "impl/adv_api/detail/matmul/scheduler/iterator/k_loop/k_loop.h"
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/utils/matmul_call_back.h"
#include "../copy_cube_in/base_tiling_struct.h"

using namespace std;
using namespace AscendC;


namespace {
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, class MM_CB,
    MATMUL_POLICY_DEFAULT_OF(MatmulWithScalePolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE_PRIVATE(MLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(NLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(KLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    MATMUL_ALLOW_USING_PRIVATE(MLoop);
    MATMUL_ALLOW_USING_PRIVATE(NLoop);
    MATMUL_ALLOW_USING_PRIVATE(KLoop);

public:
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    using POLICY = MATMUL_POLICY<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>;

    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);

    MatmulImpl() {}

    VAR_PARAMS& GetVar() {
        return var;
    }

    void InitVar(const TCubeTiling &tiling) {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
    }

    void SetRuntimeParams() {
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        MATMUL_MODULE(MatmulShapeInfo)->SetSingleShape(tiling.GetSingleCoreM(), tiling.GetSingleCoreN(), tiling.GetSingleCoreK());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgM(tiling.GetSingleCoreM());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgN(tiling.GetSingleCoreN());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgKa(tiling.GetSingleCoreK());
        MATMUL_MODULE(MatmulShapeInfo)->SetOrgKb(tiling.GetSingleCoreK());

        MATMUL_MODULE(MLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM());
        MATMUL_MODULE(NLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN());
        MATMUL_MODULE(KLoop)->Init(MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK());
    }

    void RunCase() {
        MATMUL_MODULE(KLoop)->template OuterStart();
        EXPECT_EQ((MATMUL_MODULE(KLoop)->template OuterNext()), 1);
        EXPECT_EQ((MATMUL_MODULE(KLoop)->template GetOuterScaleKaIdx()), 1);
        EXPECT_EQ((MATMUL_MODULE(KLoop)->template GetOuterScaleKbIdx()), 0);
        EXPECT_EQ((MATMUL_MODULE(KLoop)->template GetNextOuterScaleKaIdx()), 2);
        EXPECT_EQ((MATMUL_MODULE(KLoop)->template GetNextOuterScaleKbIdx()), 0);
        EXPECT_EQ((MATMUL_MODULE(KLoop)->template GetTileShapeScaleKa()), 3);
        EXPECT_EQ((MATMUL_MODULE(KLoop)->template GetTileShapeScaleKb()), 8);
        EXPECT_EQ((MATMUL_MODULE(KLoop)->template GetScaleFactorKa()), 1);
        EXPECT_EQ((MATMUL_MODULE(KLoop)->template GetScaleFactorKb()), 1);
    }

private:
    TCubeTiling tiling;
    TPipe pipe;
    VAR_PARAMS var;
};
}

class TestKLoopMDLMx : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using AS_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e5m2_t, false, TPosition::GM, CubeFormat::ND, false, TPosition::GM>;
    using BS_TYPE = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::ND,
        fp8_e4m3fn_t, true, TPosition::GM, CubeFormat::ND, true, TPosition::GM>;
    using C_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<AS_TYPE, BS_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, Impl::Detail::MatmulWithScalePolicy> mm1;
};

TEST_F(TestKLoopMDLMx, k_loop_mdl_mx_all) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, mxTypePara
    TilingParamsMx tilingParamsMx = {1, 64, 48, 256, 64, 48, 256, 32, 48, 96, 2, 4, 1, 2, 1, 3, 1, 1, 16843009};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);

    mm1.InitVar(tiling);
    mm1.SetRuntimeParams();
    mm1.RunCase();
}