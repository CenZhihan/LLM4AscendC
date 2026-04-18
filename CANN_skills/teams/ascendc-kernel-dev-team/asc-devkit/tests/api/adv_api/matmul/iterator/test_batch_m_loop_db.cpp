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
 * \file test_batch_m_loop.cpp
 * \brief m loop ut for batch
 */

#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "include/adv_api/matmul/tiling.h"
#include "impl/adv_api/detail/matmul/utils/matmul_module.h"
#include "impl/adv_api/detail/matmul/policy/matmul_policy.h"
#define private public
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/param/matmul_tensor_info.h"
#include "impl/adv_api/detail/matmul/param/matmul_shape_tiling.h"
#include "impl/adv_api/detail/matmul/scheduler/iterator/m_loop/m_loop_batch_db.h"

using namespace std;
using namespace AscendC;

namespace {
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using MLoop = Impl::Detail::MLoop<IMPL, B_TYPE, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl :
    MATMUL_IMPORT_MODULE_PRIVATE(MLoop),
    MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo),
    MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING_PRIVATE(MLoop);
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

    void SetInitParams(int32_t singleCoreM, int32_t baseM, int32_t stepM) {
        MATMUL_MODULE(MatmulShapeInfo)->SetSingleCoreM(singleCoreM);
        tiling.singleCoreM = singleCoreM;
        tiling.baseM = baseM;
        tiling.stepM = stepM;
        tiling.iterateOrder = 0;
    }

    int32_t GetSingleShape()
    {
        return MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM();
    }

private:
    TCubeTiling tiling;
    TPipe pipe;
    VAR_PARAMS var;
};
}

class TestMLoopBatchDB : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:

    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NORMAL>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NORMAL>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::NORMAL>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::NORMAL>;

    constexpr MatmulConfig static CFG_NORM_OUTER_PRODUCT = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1,
        true, IterateOrder::ORDER_M, ScheduleType::OUTER_PRODUCT);
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM_OUTER_PRODUCT, void, CustomMatmulPolicy> mm;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM_OUTER_PRODUCT, void, CustomMatmulPolicy> mm1;
};

TEST_F(TestMLoopBatchDB, batch_m_loop) {
    mm.SetInitParams(77, 80, 1);
    mm.Init(mm.GetSingleShape());
    for (mm.OuterStart(); !mm.OuterEnd(); mm.OuterNext()) {
    }
    EXPECT_EQ(mm.GetTotalIter(), 1);
    EXPECT_EQ(mm.GetOuterIdx(), 1);
    EXPECT_EQ(mm.GetBaseShape(), 80);
    EXPECT_EQ(mm.GetBaseBlockShape(), 5);
}

TEST_F(TestMLoopBatchDB, batch_m_loop_case2) {
    mm.SetInitParams(81, 32, 1);
    mm.Init(mm.GetSingleShape());
    for (mm.OuterStart(); !mm.OuterEnd(); mm.OuterNext()) {
    }
    EXPECT_EQ(mm.GetTotalIter(), 3);
    EXPECT_EQ(mm.GetOuterIdx(), 3);
    EXPECT_EQ(mm.GetBaseShape(), 32);
    EXPECT_EQ(mm.GetBaseBlockShape(), 2);
}

TEST_F(TestMLoopBatchDB, batch_m_loop_inner_case1) {
    mm1.SetInitParams(81, 32, 1);
    mm1.Init(mm1.GetSingleShape());
    for (mm1.InnerStart(); !mm1.InnerEnd(); mm1.InnerNext()) {
    }
    EXPECT_EQ(mm1.GetTotalIter(), 3);
    EXPECT_EQ(mm1.GetOuterIdx(), 3);
    EXPECT_EQ(mm1.GetBaseShape(), 32);
    EXPECT_EQ(mm1.GetBaseBlockShape(), 2);
}
