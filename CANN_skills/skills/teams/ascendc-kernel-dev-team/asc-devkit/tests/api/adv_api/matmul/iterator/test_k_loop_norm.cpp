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
#include "impl/adv_api/detail/matmul/scheduler/iterator/k_loop/k_loop_norm.h"
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"

using namespace std;
using namespace AscendC;


namespace {
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, class MM_CB,
MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE_PRIVATE(KLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    MATMUL_ALLOW_USING_PRIVATE(KLoop);

public:
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);

    MatmulImpl() {
        InitVar();
    }

    VAR_PARAMS& GetVar() {
        return var;
    }

    void InitVar() {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
    }

    void SetInitParams(int32_t stepKa, int32_t stepKb, int32_t baseK) {
        tiling.stepKa = stepKa;
        tiling.stepKb = stepKb;
        tiling.baseK = baseK;
    }

    void SetRuntimeParams(int32_t singleCoreK) {
        MATMUL_MODULE(MatmulShapeInfo)->SetSingleCoreK(singleCoreK);
    }

private:
    TCubeTiling tiling;
    TPipe pipe;
    VAR_PARAMS var;
};
}

class test_k_loop_norm : public testing::Test {
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

TEST_F(test_k_loop_norm, get_loop_cnt) {
    int32_t stepKa = 2;
    int32_t stepKb = 3;
    int32_t baseK = 32;
    int32_t singleCoreK = 512;
    mm.SetInitParams(stepKa, stepKb, baseK);
    mm.SetRuntimeParams(singleCoreK);
    mm.Init(singleCoreK);
    ASSERT_EQ(mm.GetTotalIter(), 16);
    int32_t outerIter = 0;
    int32_t innerIter = 0;
    mm.OuterStart();
    do {
        outerIter++;
        ASSERT_EQ(mm.GetTileShapeA(), baseK);
        mm.InnerStart();
        innerIter++;
        ASSERT_EQ(mm.GetBaseShape(), baseK);
    } while (mm.OuterNext());
    ASSERT_EQ(outerIter, 16);
    ASSERT_EQ(innerIter, 16);
}

TEST_F(test_k_loop_norm, loop_with_tail) {
    int32_t stepKa = 2;
    int32_t stepKb = 3;
    int32_t baseK = 32;
    int32_t singleCoreK = 500;
    mm.SetInitParams(stepKa, stepKb, baseK);
    mm.SetRuntimeParams(singleCoreK);
    mm.Init(singleCoreK);
    int32_t outerIter = 0;
    int32_t innerIter = 0;
    mm.OuterStart();
    do {
        outerIter++;
        if (mm.GetOuterIdx() == 15) {
            ASSERT_EQ(mm.GetTileShapeA(), 20);
        } else {
            ASSERT_EQ(mm.GetTileShapeA(), baseK);
        }
        mm.InnerStart();
        innerIter++;
        if (mm.GetOuterIdx() == 15) {
            ASSERT_EQ(mm.GetBaseShape(), 20);
        } else {
            ASSERT_EQ(mm.GetBaseShape(), baseK);
        }
    } while (mm.OuterNext());
    ASSERT_EQ(outerIter, 16);
    ASSERT_EQ(innerIter, 16);
}
