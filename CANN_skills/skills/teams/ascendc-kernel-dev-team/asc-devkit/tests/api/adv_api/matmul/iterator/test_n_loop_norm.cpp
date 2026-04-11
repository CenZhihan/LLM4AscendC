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
 * \file test_n_loop_norm.cpp
 * \brief n loop ut for norm
 */

#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "include/adv_api/matmul/tiling.h"
#include "impl/adv_api/detail/matmul/utils/matmul_param.h"
#include "impl/adv_api/detail/matmul/policy/matmul_policy.h"
#define private public
#define protected public
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/param/matmul_tensor_info.h"
#include "impl/adv_api/detail/matmul/param/matmul_shape_tiling.h"
#include "impl/adv_api/detail/matmul/scheduler/iterator/n_loop/n_loop.h"
#include "impl/adv_api/detail/matmul/scheduler/iterator/n_loop/n_loop_norm.h"

using namespace std;
using namespace AscendC;


namespace {
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG,
          class MM_CB, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl :
    MATMUL_IMPORT_MODULE_PRIVATE(NLoop),
    MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo),
    MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING_PRIVATE(NLoop);
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

class TestNLoopNorm : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM, void> mm;
};

TEST_F(TestNLoopNorm, first_iter) {
    mm.SetInitParams(1793, 128, 4);
    mm.Init(mm.GetSingleShape());           // SetSingleShape
    mm.OuterStart();
    mm.InnerStart();
    // Outer
    EXPECT_EQ(mm.GetTotalIter(), 15);       // totalIter
    EXPECT_EQ(mm.GetOuterIdx(), 0);
    EXPECT_EQ(mm.GetOuterIter(), 4);        // outerIter
    EXPECT_EQ(mm.GetTileShape(), 128);      // mainTileShape
    EXPECT_EQ(mm.GetTileBlockShape(), 8);   // Ceil(mainTileShape, BLOCK_CUBE)
    // Inner
    EXPECT_EQ(mm.GetInnerIdx(), 0);
    EXPECT_EQ(mm.GetInnerIter(), 4);        // stepN
    EXPECT_EQ(mm.GetBaseShape(), 128);      // baseShape
    EXPECT_EQ(mm.GetBaseBlockShape(), 8);   // Ceil(baseShape, BLOCK_CUBE)
}

TEST_F(TestNLoopNorm, inner_end) {
    mm.SetInitParams(1793, 128, 4);
    mm.Init(mm.GetSingleShape());
    mm.OuterStart();
    mm.InnerStart();

    EXPECT_EQ(mm.GetInnerIdx(), 0);
    mm.InnerNext();
    EXPECT_EQ(mm.GetInnerIdx(), 1);
    mm.InnerNext();
    EXPECT_EQ(mm.GetInnerIdx(), 2);
    mm.InnerNext();
    EXPECT_EQ(mm.GetInnerIdx(), 3);
    EXPECT_FALSE(mm.InnerEnd());
    // tail
    mm.InnerNext();
    EXPECT_TRUE(mm.InnerEnd());
}

TEST_F(TestNLoopNorm, outer_end) {
    mm.SetInitParams(1793, 128, 4);
    mm.Init(mm.GetSingleShape());
    mm.OuterStart();
    mm.InnerStart();
    mm.InnerNext();

    EXPECT_EQ(mm.GetOuterIdx(), 0);
    mm.OuterNext();
    EXPECT_EQ(mm.GetOuterIdx(), 1);
    mm.OuterNext();
    EXPECT_EQ(mm.GetOuterIdx(), 2);
    mm.OuterNext();
    EXPECT_EQ(mm.GetOuterIdx(), 3);
    EXPECT_FALSE(mm.OuterEnd());
    mm.OuterNext();
    // tail
    EXPECT_EQ(mm.GetTileShape(), 128);      // tailTileShape
    EXPECT_EQ(mm.GetTileBlockShape(), 8);   // Ceil(tailTileShape, BLOCK_CUBE)
    EXPECT_TRUE(mm.OuterEnd());
}
