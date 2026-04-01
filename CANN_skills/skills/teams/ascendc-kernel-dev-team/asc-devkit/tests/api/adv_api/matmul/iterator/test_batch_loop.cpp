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
 * \file test_batch_loop.cpp
 * \brief batch loop ut
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
#include "impl/adv_api/detail/matmul/scheduler/iterator/batch_loop/batch_loop.h"

using namespace std;
using namespace AscendC;

namespace {
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using BatchLoop = Impl::Detail::BatchLoop<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, BIAS_TYPE, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl :
    MATMUL_IMPORT_MODULE(BatchLoop),
    MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo),
    MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING(BatchLoop);
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

    void SetSingleCoreParams(int32_t singleCoreM, int32_t singleCoreN, int32_t singleCoreK,
        int32_t aLayoutInfoB, int32_t bLayoutInfoB, int32_t batchNum) {
        MATMUL_MODULE(MatmulShapeInfo)->SetSingleShape(singleCoreM, singleCoreN, singleCoreK);
        tiling.singleCoreM = singleCoreM;
        tiling.singleCoreN = singleCoreN;
        tiling.singleCoreK = singleCoreK;
        tiling.iterateOrder = 0;
        tiling.ALayoutInfoB = aLayoutInfoB;
        tiling.BLayoutInfoB = bLayoutInfoB;
        tiling.BatchNum = batchNum;
    }

private:
    TCubeTiling tiling;
    TPipe pipe;
    VAR_PARAMS var;
};
}

class TestBatchLoop : public testing::Test {
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

    static constexpr MatmulConfig mm_cfg1 = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1);
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, mm_cfg1, void, CustomMatmulPolicy> mm1;
};

TEST_F(TestBatchLoop, batch_loop) {
    mm.SetSingleCoreParams(144, 77, 64, 3, 1, 1);
    mm.Init();
    for (mm.OuterStart(); !mm.OuterEnd();mm.OuterNext()) {
        for (mm.SplitStart(); !mm.SplitEnd(); mm.SplitNext()) {
            for (mm.InnerStart(); !mm.InnerEnd(); mm.InnerNext()) {
            }
        }
    }
    EXPECT_EQ(mm.GetOuterIndex(), 1);
    EXPECT_EQ(mm.GetDstOffset(), 33264);
    EXPECT_EQ(mm.GetBatchNum(), 3);
    EXPECT_EQ(mm.GetBiasBatchSrcOffset(), 231);
    EXPECT_EQ(mm.GetSplitIndex(), 1);
    EXPECT_EQ(mm.GetSplitSize(), 1);
    EXPECT_EQ(mm.GetSplitBatchNum(), 3);
    EXPECT_EQ(mm.GetInnerIndex(), 3);
    EXPECT_EQ(mm.GetBatchIndex(), 3);
}

TEST_F(TestBatchLoop, batch_loop_db) {
    mm1.SetSingleCoreParams(32, 256, 64, 2, 6, 6);
    mm1.Init();
    for (mm1.OuterStart(); !mm1.OuterEnd();mm1.OuterNext()) {
        for (mm1.SplitStart(); !mm1.SplitEnd(); mm1.SplitNext()) {
            for (mm1.InnerStart(); !mm1.InnerEnd(); mm1.InnerNext()) {
            }
        }
    }
    EXPECT_EQ(mm1.GetOuterIndex(), 1);
    EXPECT_EQ(mm1.GetDstOffset(), 49152);
    EXPECT_EQ(mm1.GetBatchNum(), 6);
    EXPECT_EQ(mm1.GetBiasBatchSrcOffset(), 1536);
    EXPECT_EQ(mm1.GetSplitIndex(), 2);
    EXPECT_EQ(mm1.GetSplitSize(), 2);
    EXPECT_EQ(mm1.GetSplitBatchNum(), 3);
    EXPECT_EQ(mm1.GetInnerIndex(), 3);
    EXPECT_EQ(mm1.GetBatchIndex(), 6);
}

TEST_F(TestBatchLoop, batch_loop_bias) {
    mm.SetSingleCoreParams(256, 256, 256, 4, 4, 4);
    mm.Init();
    for (mm.OuterStart(); !mm.OuterEnd();mm.OuterNext()) {
        for (mm.SplitStart(); !mm.SplitEnd(); mm.SplitNext()) {
            for (mm.InnerStart(); !mm.InnerEnd(); mm.InnerNext()) {
            }
        }
    }
    EXPECT_EQ(mm.GetOuterIndex(), 4);
    EXPECT_EQ(mm.GetDstOffset(), 262144);
    EXPECT_EQ(mm.GetBatchNum(), 1);
    EXPECT_EQ(mm.GetBiasBatchSrcOffset(), 1024);
    EXPECT_EQ(mm.GetSplitIndex(), 1);
    EXPECT_EQ(mm.GetSplitSize(), 1);
    EXPECT_EQ(mm.GetSplitBatchNum(), 1);
    EXPECT_EQ(mm.GetInnerIndex(), 1);
    EXPECT_EQ(mm.GetBatchIndex(), 1);
    EXPECT_EQ(mm.GetMainBatchBlockA(), 1);
}
