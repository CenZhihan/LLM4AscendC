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
#include "impl/adv_api/detail/matmul/utils/matmul_param.h"
#include "impl/adv_api/detail/matmul/policy/matmul_policy.h"
#include "impl/adv_api/detail/matmul/scheduler/scheduler.h"
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/utils/matmul_call_back.h"
#include "../copy_cube_in/base_tiling_struct.h"

using namespace std;

namespace AscendC {
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulWithScalePolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using TBufPoolL0 = Impl::Detail::TBufPoolL0<IMPL, A_TYPE, B_TYPE, MM_CFG>;
    using MatmulSubBlockInfo = Impl::Detail::MatmulSubBlockInfo<IMPL, MM_CFG>;
    // using MatmulShapeTiling = AscendC::Impl::Detail::MatmulShapeTiling<IMPL, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulWithScalePolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(MatmulSubBlockInfo)
, MATMUL_IMPORT_MODULE(TBufPoolL0)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulCrossCoreSync)
{
    MATMUL_ALLOW_USING(MatmulSubBlockInfo);
    MATMUL_ALLOW_USING(TBufPoolL0);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    MATMUL_ALLOW_USING_PRIVATE(MatmulCrossCoreSync);

    MATMUL_USE_MODULE(MatmulSubBlockInfo);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulCrossCoreSync);

public:
    using VAR_PARAMS = typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;

    using POLICY = MATMUL_POLICY<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>;
    using CallBack = MM_CB;

    MatmulImpl() {}

    void InitVar(const TCubeTiling &tiling) {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
        MATMUL_MODULE(TBufPoolL0)->Init();
    }

    void CopyIn()
    {
        MATMUL_MODULE(MatmulCrossCoreSync)->WaitL1Ready();
    }

    void End()
    {
        MATMUL_MODULE(MatmulCrossCoreSync)->End();
    }
private:
    TPipe pipe;
    VAR_PARAMS var;
};

class TestMatmulCrossCoreSync : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE = MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND,
        uint8_t, false, AscendC::TPosition::GM, CubeFormat::ND, false, AscendC::TPosition::GM>;
    using B_TYPE = MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND,
        uint8_t, true, AscendC::TPosition::GM, CubeFormat::ND, true, AscendC::TPosition::GM>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm;
};

TEST_F(TestMatmulCrossCoreSync, WaitL1Ready) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParamsMx tilingParamsMx = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 0, 0};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm.InitVar(tiling);
    mm.CopyIn();
}

TEST_F(TestMatmulCrossCoreSync, End) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParamsMx tilingParamsMx = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 0, 0};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm.InitVar(tiling);
    mm.End();
}
} // namespace
