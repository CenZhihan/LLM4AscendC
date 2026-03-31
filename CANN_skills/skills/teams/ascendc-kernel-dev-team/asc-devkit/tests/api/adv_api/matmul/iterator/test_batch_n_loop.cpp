/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file test_batch_n_loop.cpp
 * \brief n loop ut for batch
 */

#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "include/adv_api/matmul/tiling.h"
#include "impl/adv_api/detail/matmul/utils/matmul_module.h"
#include "impl/adv_api/detail/matmul/policy/matmul_policy.h"
#define private public
#define protected public
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/param/matmul_tensor_info.h"
#include "impl/adv_api/detail/matmul/param/matmul_shape_tiling.h"
#include "impl/adv_api/detail/matmul/scheduler/iterator/n_loop/n_loop_batch.h"

using namespace std;
using namespace AscendC;

namespace {
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using NLoop = Impl::Detail::NLoop<IMPL, B_TYPE, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl :
    MATMUL_IMPORT_MODULE(NLoop),
    MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo),
    MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING(NLoop);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);

public:
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);

    MatmulImpl() {
        InitVar();
    }

    void SetTiling(TCubeTiling &cubeTiling) {
        tiling = cubeTiling;
    }

    void InitVar() {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
    }

    void SetInitParams(int32_t singleCoreN, int32_t baseN, int32_t stepN) {
        MATMUL_MODULE(MatmulShapeInfo)->SetSingleCoreN(singleCoreN);
        tiling.singleCoreN = singleCoreN;
        tiling.baseN = baseN;
        tiling.stepN = stepN;
        tiling.iterateOrder = 1;
    }

    int32_t GetSingleShape()
    {
        return MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN();
    }

private:
    TCubeTiling tiling;
    TPipe pipe;
    VAR_PARAMS var;
};
}

class TestNLoopBatch : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:

    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NORMAL>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NORMAL>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::NORMAL>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::NORMAL>;

    static constexpr MatmulConfig mm_cfg = GetNormalConfig(false, false, false, BatchMode::BATCH_LARGE_THAN_L1);
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, mm_cfg, void, CustomMatmulPolicy> mm;
};

TEST_F(TestNLoopBatch, batch_n_loop) {
    mm.SetInitParams(77, 80, 1);
    mm.Init(mm.GetSingleShape());
    for (mm.OuterStart(); !mm.OuterEnd(); mm.OuterNext()) {
    }
    EXPECT_EQ(mm.GetOuterIter(), 1);
    EXPECT_EQ(mm.GetOuterIdx(), 1);
    EXPECT_EQ(mm.GetBaseShape(), 77);
    EXPECT_EQ(mm.GetBaseBlockShape(), 5);
}
