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
#include "fake_modules_mx.h"
#include "test_custom_mx_loop.h"

using namespace std;
using namespace AscendC;
using namespace TestCustomModules;
namespace {
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulWithScalePolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using TRANS_B_TYPE = decltype(GetTransBDataType<A_TYPE, B_TYPE, MM_CFG>());
    using L0bT = typename Conditional<HasScalePosition<B_TYPE>::value, \
                 typename GetL0DataType<typename TRANS_B_TYPE::T, true>::Type, \
                 typename GetL0DataType<typename TRANS_B_TYPE::T, false>::Type>::type;
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type;
    using CopyCubeInA = CustomCopyCubeIn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = CustomCopyCubeIn<IMPL, MatmulInputBType<B_TYPE, typename B_TYPE::T>, MM_CFG>;
    using CopyCubeInScaleA = CustomCopyCubeIn<IMPL, MatmulInputScaleAType<A_TYPE, fp8_e8m0_t>, MM_CFG>;
    using CopyCubeInScaleB = CustomCopyCubeIn<IMPL, MatmulInputScaleBType<B_TYPE, fp8_e8m0_t>, MM_CFG>;
    using MLoop = CustomMxLoop<IMPL, A_TYPE, MM_CFG>;
    using NLoop = CustomMxLoop<IMPL, B_TYPE, MM_CFG>;
    using KLoop = CustomMxKLoop<IMPL, typename A_TYPE::T, A_TYPE, MM_CFG>;
    using LoadToA2 = CustomMxLoadToL0A<IMPL, A_TYPE, MM_CFG>;
    using LoadToB2 = CustomMxLoadToL0B<IMPL, B_TYPE, MM_CFG>;
    using TBufPoolL0 = CustomTBufPoolL0<IMPL, A_TYPE, B_TYPE, MM_CFG>;
    using MmadCompute = CustomMxMmadCompute<IMPL, L0cT, A_TYPE, L0bT, MM_CFG>;
    using CubeOutBuffer = CustomCubeOutBuffer<IMPL, L0cT, MM_CFG>;
    using CopyCubeOut = CustomCopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG>;
    using BiasScheduler = CustomBiasScheduler<IMPL, A_TYPE, B_TYPE, BIAS_TYPE, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulWithScalePolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(Scheduler)
, MATMUL_IMPORT_MODULE(MLoop)
, MATMUL_IMPORT_MODULE(NLoop)
, MATMUL_IMPORT_MODULE(KLoop)
, MATMUL_IMPORT_MODULE(CopyCubeInA)
, MATMUL_IMPORT_MODULE(CopyCubeInB)
, MATMUL_IMPORT_MODULE(CopyCubeInScaleA)
, MATMUL_IMPORT_MODULE(CopyCubeInScaleB)
, MATMUL_IMPORT_MODULE(LoadToA2)
, MATMUL_IMPORT_MODULE(LoadToB2)
, MATMUL_IMPORT_MODULE(TBufPoolL0)
, MATMUL_IMPORT_MODULE(MmadCompute)
, MATMUL_IMPORT_MODULE(CubeOutBuffer)
, MATMUL_IMPORT_MODULE(CopyCubeOut)
, MATMUL_IMPORT_MODULE(BiasScheduler)
, MATMUL_IMPORT_MODULE(CubeInBufferA)
, MATMUL_IMPORT_MODULE(CubeInBufferB)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulUnitFlag)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulCrossCoreSync)
{
    MATMUL_ALLOW_USING(MLoop);
    MATMUL_ALLOW_USING(NLoop);
    MATMUL_ALLOW_USING(KLoop);
    MATMUL_ALLOW_USING(Scheduler);
    MATMUL_ALLOW_USING(LoadToA2);
    MATMUL_ALLOW_USING(LoadToB2);
    MATMUL_ALLOW_USING(TBufPoolL0);
    MATMUL_ALLOW_USING(MmadCompute);
    MATMUL_ALLOW_USING(CopyCubeInA);
    MATMUL_ALLOW_USING(CopyCubeInB);
    MATMUL_ALLOW_USING(CopyCubeInScaleA);
    MATMUL_ALLOW_USING(CopyCubeInScaleB);
    MATMUL_ALLOW_USING(CubeOutBuffer);
    MATMUL_ALLOW_USING(CopyCubeOut);
    MATMUL_ALLOW_USING(BiasScheduler);
    MATMUL_ALLOW_USING(CubeInBufferA);
    MATMUL_ALLOW_USING(CubeInBufferB);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    MATMUL_ALLOW_USING_PRIVATE(MatmulUnitFlag);
    MATMUL_ALLOW_USING_PRIVATE(MatmulCrossCoreSync);

    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulShapeTiling);


public:
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;

    using POLICY = MATMUL_POLICY<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>;
    using CallBack = MM_CB;
    using Scheduler::ScheduleOnce;

    MatmulImpl() {}

    VAR_PARAMS& GetVar() {
        return var;
    }

    void InitVar(const TCubeTiling &tiling) {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
        MATMUL_MODULE(TBufPoolL0)->Init();
    }

    void SetTranspose(bool isTranA, bool isTranB) {
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeA(isTranA);
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeB(isTranB);
    }

private:
    TPipe pipe;
    VAR_PARAMS var;
};
}

class TestSchedulerNORMMx : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using AS_TYPE_GM = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::NZ,
        fp4x2_e1m2_t, false, TPosition::GM, CubeFormat::NZ, false, TPosition::GM>;
    using AS_TYPE_L1 = MatmulTypeWithScale<TPosition::TSCM, TPosition::TSCM, CubeFormat::NZ,
        fp8_e4m3fn_t, false, TPosition::GM, CubeFormat::NZ, false, TPosition::GM>;

    using BS_TYPE_GM = MatmulTypeWithScale<TPosition::GM, TPosition::GM, CubeFormat::NZ,
        fp4x2_e2m1_t, true, TPosition::GM, CubeFormat::NZ, true, TPosition::GM>;
    using BS_TYPE_l1 = MatmulTypeWithScale<TPosition::TSCM, TPosition::TSCM, CubeFormat::NZ,
        fp8_e5m2_t, true, TPosition::GM, CubeFormat::NZ, true, TPosition::GM>;

    using C_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    MatmulImpl<AS_TYPE_GM, BS_TYPE_GM, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm1;
    MatmulImpl<AS_TYPE_L1, BS_TYPE_l1, C_TYPE, BIAS_TYPE, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm2;

    constexpr static MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
    constexpr static MatmulShapeParams shapeParams = {64, 64, 64, 64, 64, 64};
    constexpr static MatmulFuncParams funcParams = {false, false, false, false, 0, IterateOrder::ORDER_N, ScheduleType::INNER_PRODUCT, true, true};
    constexpr static MatmulConfig mmStaticConfig = GetMMConfig<configMode>(shapeParams, funcParams);

    constexpr static MatmulApiStaticTiling staticTilingL1 = GetMatmulApiTiling<AS_TYPE_L1, BS_TYPE_l1, C_TYPE, BIAS_TYPE>(mmStaticConfig);
    MatmulImpl<AS_TYPE_L1, BS_TYPE_l1, C_TYPE, BIAS_TYPE, staticTilingL1, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm3;

    constexpr static MatmulConfig staticMmCfg = {
        .doNorm = true,
        .doBasicBlock = false,
        .doMultiDataLoad = false,
        .basicM = 64,
        .basicN = 64,
        .basicK = 64,
        .intrinsicsCheck = false,
        .isNBatch = false,
        .enVecND2NZ = false,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = 0,
        .singleCoreM = 64,
        .singleCoreN = 64,
        .singleCoreK = 64,
        .stepM = 1,
        .stepN = 1,
        .baseMN = 4096,
        .singleCoreMN = 4096,
        .enUnitFlag = false,
        .isPerTensor = false,
        .hasAntiQuantOffset = false,
        .doIBShareNorm = false,
        .doSpecialMDL = false,
        .enableInit = true,
        .batchMode = BatchMode::SINGLE_LARGE_THAN_L1,
        .enableEnd = true,
        .enableGetTensorC = true,
        .enableSetOrgShape = true,
        .enableSetBias = true,
        .enableSetTail = true,
        .enableQuantVector = true,
        .enableSetDefineData = true,
        .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
        .enableReuse = true,
        .enableUBReuse = true,
        .enableL1CacheUB = false,
        .intraBlockPartSum = false,
        .iterateOrder = IterateOrder::UNDEF,
        .scheduleType = ScheduleType::INNER_PRODUCT,
        .enableDoubleCache = false,
        .isBiasBatch = true,
        .enableStaticPadZeros = true
    };
    constexpr static MatmulApiStaticTiling staticTilingGM = GetMatmulApiTiling<AS_TYPE_GM, BS_TYPE_GM, C_TYPE, BIAS_TYPE>(staticMmCfg);
    MatmulImpl<AS_TYPE_GM, BS_TYPE_GM, C_TYPE, BIAS_TYPE, staticTilingGM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm4;
};

TEST_F(TestSchedulerNORMMx, ScheduleOnce_TransA_OrderM_Mx) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, mxTypePara
    TilingParamsMx tilingParamsMx = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 0, 0};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm1.InitVar(tiling);
    mm1.SetTranspose(true, false);
    ASSERT_TRUE(mm1.ScheduleOnce(false));
    ASSERT_FALSE(mm1.ScheduleOnce(false));
}

TEST_F(TestSchedulerNORMMx, ScheduleOnce_TransB_OrderN_Mx) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, mxTypePara
    TilingParamsMx tilingParamsMx = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 1, 0};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm2.InitVar(tiling);
    mm2.SetTranspose(false, true);
    ASSERT_TRUE(mm2.ScheduleOnce(false));
    ASSERT_FALSE(mm2.ScheduleOnce(false));
}

TEST_F(TestSchedulerNORMMx, ScheduleOnce_TransA_OrderN_Mx) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, mxTypePara
    TilingParamsMx tilingParamsMx = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 1, 0};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm2.InitVar(tiling);
    mm2.SetTranspose(true, false);
    ASSERT_TRUE(mm2.ScheduleOnce(false));
    ASSERT_FALSE(mm2.ScheduleOnce(false));
}

TEST_F(TestSchedulerNORMMx, ScheduleOnce_TransA_OrderN_Basic_Mx) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, mxTypePara
    TilingParamsMx tilingParamsMx = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 1, 0};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm3.InitVar(tiling);
    mm3.SetTranspose(true, false);
    ASSERT_TRUE(mm3.ScheduleOnce(false));
    ASSERT_FALSE(mm3.ScheduleOnce(false));
}

TEST_F(TestSchedulerNORMMx, ScheduleOnce_TransA_OrderM_StaticPadZeros_Mx) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, mxTypePara
    TilingParamsMx tilingParamsMx = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 0, 0};
    TCubeTiling tiling;
    tilingParamsMx.GetTiling(tiling);
    mm4.InitVar(tiling);
    mm4.SetTranspose(true, false);
    ASSERT_TRUE(mm4.ScheduleOnce(false));
    ASSERT_FALSE(mm4.ScheduleOnce(false));
}
