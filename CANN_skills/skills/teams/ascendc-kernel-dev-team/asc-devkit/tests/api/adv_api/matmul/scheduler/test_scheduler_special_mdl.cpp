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
#include "impl/adv_api/detail/matmul/scheduler/scheduler.h"
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/utils/matmul_call_back.h"
#include "../copy_cube_in/base_tiling_struct.h"
#include "fake_modules.h"

#include "test_custom_loop.h"
 
using namespace std;
using namespace AscendC;
using namespace TestCustomModules;
 
namespace {

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type;
    using CopyCubeInA = CustomCopyCubeIn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = CustomCopyCubeIn<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Scheduler = Impl::Detail::MatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>;
    using MLoop = CustomLoop<IMPL, A_TYPE, MM_CFG>;
    using NLoop = CustomLoop<IMPL, B_TYPE, MM_CFG>;
    using KLoop = CustomKLoop<IMPL, typename A_TYPE::T, A_TYPE, MM_CFG>;
    using LoadToA2 = CustomLoadToL0<IMPL, A_TYPE, MM_CFG>;
    using LoadToB2 = CustomLoadToL0<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using TBufPoolL0 = CustomTBufPoolL0<IMPL, A_TYPE, B_TYPE, MM_CFG>;
    using MmadCompute = CustomMmadCompute<IMPL, L0cT, A_TYPE, typename A_TYPE::T, MM_CFG>;
    using CubeOutBuffer = CustomCubeOutBuffer<IMPL, L0cT, MM_CFG>;
    using CopyCubeOut = CustomCopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG>;
    using LoadBias2C2 = CustomLoadBias2C2<IMPL, A_TYPE, BIAS_TYPE, MM_CFG>;
    using BiasScheduler = CustomBiasScheduler<IMPL, A_TYPE, B_TYPE, BIAS_TYPE, MM_CFG>;
};
 
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(Scheduler)
, MATMUL_IMPORT_MODULE(MLoop)
, MATMUL_IMPORT_MODULE(NLoop)
, MATMUL_IMPORT_MODULE(KLoop)
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
, MATMUL_IMPORT_MODULE(LoadBias2C2)
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
    MATMUL_ALLOW_USING(CubeOutBuffer);
    MATMUL_ALLOW_USING(CopyCubeOut);
    MATMUL_ALLOW_USING(BiasScheduler);

    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    MATMUL_ALLOW_USING_PRIVATE(MatmulUnitFlag);
 
    using SrcT = typename A_TYPE::T;
 
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(TBufPoolL0);
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
 
    void SetBias(bool) {
        MATMUL_MODULE(BiasScheduler)->SetBias();
    }
 
private:
    TPipe pipe;
    VAR_PARAMS var;
};
}

#define MATMUL_TEST_VALUE_WRAPPER()

// currently gtest does not support compile-time value-templated tests, so wrap them in a type is necessary
class MatmulConfigModeWrapper {
    static constexpr MatmulConfigMode mode = MatmulConfigMode::CONFIG_SPECIALMDL;
};
 
constexpr MatmulConfig CFG_SPECIAL_MDL = GetSpecialMDLConfig();
template <typename MatmulConfigMode = MatmulConfigModeWrapper>
class TestSchedulerSpecialMDL : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
 
private:
    static constexpr auto configMode = MatmulConfigMode::mode;

    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
 
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_SPECIAL_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm;
};

// you can list as many types as you want in ::testing::Types<...>
// this serves as a fundamental idea to automatically test different matmul configs
using MatmulConfigs = ::testing::Types<MatmulConfigModeWrapper>;
TYPED_TEST_SUITE(TestSchedulerSpecialMDL, MatmulConfigs);
 
TYPED_TEST(TestSchedulerSpecialMDL, ScheduleOnce_OrderM) {
    TilingParams tilingParams = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 0};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    this->mm.InitVar(tiling);
    this->mm.SetBias(1);
    ASSERT_TRUE(this->mm.ScheduleOnce(false));
    ASSERT_FALSE(this->mm.ScheduleOnce(false));
}
 
TYPED_TEST(TestSchedulerSpecialMDL, ScheduleOnce_OrderN) {
    TilingParams tilingParams = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    this->mm.InitVar(tiling);
    this->mm.SetBias(1);
    ASSERT_TRUE(this->mm.ScheduleOnce(false));
    ASSERT_FALSE(this->mm.ScheduleOnce(false));
}
