
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
 * \file test_scheduler_norm_outer_product.cpp
 * \brief scheduler ut for norm outer product scene
 */
#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "include/adv_api/matmul/tiling.h"
#include "impl/adv_api/detail/matmul/utils/matmul_param.h"
#include "impl/adv_api/detail/matmul/policy/matmul_policy.h"
#include "impl/adv_api/detail/matmul/scheduler/scheduler.h"
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/utils/matmul_call_back.h"
#include "../copy_cube_in/base_tiling_struct.h"
#include "fake_modules.h"

using namespace std;
using namespace AscendC;

namespace {

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type;
    using CopyCubeInA = CustomCopyCubeIn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = CustomCopyCubeIn<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using LoadToA2 = CustomLoadToL0<IMPL, A_TYPE, MM_CFG>;
    using LoadToB2 = CustomLoadToL0<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using TBufPoolL0 = CustomTBufPoolL0<IMPL, A_TYPE, B_TYPE, MM_CFG>;
    using MmadCompute = CustomMmadCompute<IMPL, L0cT, A_TYPE, typename A_TYPE::T, MM_CFG>;
    using CubeOutBuffer = CustomCubeOutBufferForMNDB<IMPL, L0cT, MM_CFG>;
    using CopyCubeOut = CustomCopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG>;
    using BiasScheduler = CustomBiasScheduler<IMPL, A_TYPE, B_TYPE, BIAS_TYPE, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(Scheduler)
, MATMUL_IMPORT_MODULE_PRIVATE(MLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(NLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(KLoop)
, MATMUL_IMPORT_MODULE(CopyCubeInA)
, MATMUL_IMPORT_MODULE(CopyCubeInB)
, MATMUL_IMPORT_MODULE(LoadToA2)
, MATMUL_IMPORT_MODULE(LoadToB2)
, MATMUL_IMPORT_MODULE(TBufPoolL0)
, MATMUL_IMPORT_MODULE(MmadCompute)
, MATMUL_IMPORT_MODULE(CubeOutBuffer)
, MATMUL_IMPORT_MODULE(CopyCubeOut)
, MATMUL_IMPORT_MODULE(BiasScheduler)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulUnitFlag)
{
    MATMUL_ALLOW_USING_PRIVATE(MLoop);
    MATMUL_ALLOW_USING_PRIVATE(NLoop);
    MATMUL_ALLOW_USING_PRIVATE(KLoop);
    MATMUL_ALLOW_USING(Scheduler);
    MATMUL_ALLOW_USING(LoadToA2);
    MATMUL_ALLOW_USING(LoadToB2);
    MATMUL_ALLOW_USING(TBufPoolL0);
    MATMUL_ALLOW_USING(MmadCompute);
    MATMUL_ALLOW_USING(CopyCubeInA);
    MATMUL_ALLOW_USING(CopyCubeInB);
    MATMUL_ALLOW_USING(CubeOutBuffer);
    MATMUL_ALLOW_USING(CopyCubeOut);
    MATMUL_ALLOW_USING(BiasScheduler);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    MATMUL_ALLOW_USING_PRIVATE(MatmulUnitFlag);

    using SrcT = typename A_TYPE::T;

    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(MatmulShapeTiling);

public:
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    using CallBack = MM_CB;

    using Scheduler::ScheduleOnce;

    MatmulImpl() {}

    VAR_PARAMS& GetVar() {
        return var;
    }

    void Init(const TCubeTiling &tiling) {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
        MATMUL_MODULE(CubeOutBuffer)->Init(tiling.baseM * tiling.baseN * 2);
        MATMUL_MODULE(MLoop)->Init(tiling.singleCoreM);
        MATMUL_MODULE(NLoop)->Init(tiling.singleCoreN);
        MATMUL_MODULE(KLoop)->Init(tiling.singleCoreK);
        MATMUL_MODULE(TBufPoolL0)->Init();
    }

    void SetBias(bool) {
        MATMUL_MODULE(BiasScheduler)->SetBias();
    }

private:
    TPipe pipe;
    VAR_PARAMS var;
};
}

class TestSchedulerNormOuterProduct : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    constexpr static MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
    // ORDER_M
    constexpr static MatmulFuncParams funcParamsOrderM{false, false, false, false, 0, IterateOrder::ORDER_M, ScheduleType::OUTER_PRODUCT, true, true};
    constexpr static MatmulConfig CFG_NORM_OUTER_PRODUCT_ORDER_M = GetMMConfig<configMode>(funcParamsOrderM);
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM_OUTER_PRODUCT_ORDER_M, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mmOrderM;

    // ORDER_N
    constexpr static MatmulFuncParams funcParamsOrderN{false, false, false, false, 0, IterateOrder::ORDER_N, ScheduleType::OUTER_PRODUCT, true, true};
    constexpr static MatmulConfig CFG_NORM_OUTER_PRODUCT_ORDER_N = GetMMConfig<configMode>(funcParamsOrderN);
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM_OUTER_PRODUCT_ORDER_N, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mmOrderN;
};

// N axis L0 DB
TEST_F(TestSchedulerNormOuterProduct, ScheduleOnce_OrderM_NIter_15_StepN_5) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 16, 1920, 16, 16, 1920, 16, 16, 128, 16, 1, 5, 1, 5, 1, 1, 1, 0};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mmOrderM.Init(tiling);
    mmOrderM.SetBias(1);
    
    // [NLoop] totalIter=15, outerIter=4, innerIter=5
    // [NLoop] totalIndex: |0, 1, 2, 3, 4,| 5, 6, 7, 8, 9,| 10, 11, 12, 13, 14|
    // [NLoop] innerIndex: 0 -> 2 -> 4 -> 5 -> 7 -> 9 -> 10 -> 12 -> 14 (9次)
    int32_t loopNum = 9;
    for (int32_t i = 0; i < loopNum; ++i) {
        ASSERT_TRUE(mmOrderM.ScheduleOnce(false));
    }
    ASSERT_FALSE(mmOrderM.ScheduleOnce(false));
}

// M axis L0 DB
TEST_F(TestSchedulerNormOuterProduct, ScheduleOnce_OrderN_MIter_15_StepM_5) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 1920, 16, 16, 1920, 16, 16, 128, 16, 16, 5, 1, 5, 1, 1, 1, 1, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mmOrderN.Init(tiling);
    mmOrderN.SetBias(1);

    // [MLoop] totalIter=15, outerIter=4, innerIter=5
    // [MLoop] totalIndex: |0, 1, 2, 3, 4,| 5, 6, 7, 8, 9,| 10, 11, 12, 13, 14|
    // [MLoop] innerIndex: 0 -> 2 -> 4 -> 5 -> 7 -> 9 -> 10 -> 12 -> 14 (9次)
    int32_t loopNum = 9;
    for (int32_t i = 0; i < loopNum; ++i) {
        ASSERT_TRUE(mmOrderN.ScheduleOnce(false));
    }
    ASSERT_FALSE(mmOrderN.ScheduleOnce(false));
}
