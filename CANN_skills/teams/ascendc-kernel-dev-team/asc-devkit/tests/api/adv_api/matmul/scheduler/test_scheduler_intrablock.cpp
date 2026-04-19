
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
#include "fake_modules.h"

using namespace std;
using namespace AscendC;

namespace {
__aicore__ inline constexpr MatmulConfig GetMmConfig()
{
    auto cfg = GetNormalConfig();
    cfg.intraBlockPartSum = true;
    cfg.enableSetBias = false;
    return cfg;
}
constexpr MatmulConfig MM_CFG_INTRABLOCK = GetMmConfig();
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type;
    using CopyCubeInA = CustomCopyCubeIn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = CustomCopyCubeIn<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Scheduler = Impl::Detail::MatmulScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>;
    using MLoop = Impl::Detail::MLoop<IMPL, A_TYPE, MM_CFG>;
    using NLoop = Impl::Detail::NLoop<IMPL, B_TYPE, MM_CFG>;
    using KLoop = Impl::Detail::KLoop<IMPL, typename A_TYPE::T, A_TYPE, MM_CFG>;
    using LoadToA2 = CustomLoadToL0<IMPL, A_TYPE, MM_CFG>;
    using LoadToB2 = CustomLoadToL0<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using TBufPoolL0 = CustomTBufPoolL0<IMPL, A_TYPE, B_TYPE, MM_CFG>;
    using MmadCompute = CustomMmadCompute<IMPL, L0cT, A_TYPE, typename A_TYPE::T, MM_CFG>;
    using CubeOutBuffer = CustomCubeOutBuffer<IMPL, L0cT, MM_CFG>;
    using CopyCubeOut = CustomCopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG>;
    using BiasScheduler = CustomBiasScheduler<IMPL, A_TYPE, B_TYPE, BIAS_TYPE, MM_CFG>;
    using MatmulQuantProcessor = Impl::Detail::MatmulQuantProcessor<IMPL, A_TYPE, C_TYPE, MM_CFG>;
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
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulSubBlockInfo)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulUnitFlag)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulQuantProcessor)
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
    MATMUL_ALLOW_USING_PRIVATE(MatmulSubBlockInfo);
    MATMUL_ALLOW_USING_PRIVATE(MatmulUnitFlag);
    MATMUL_ALLOW_USING_PRIVATE(MatmulQuantProcessor);

    using SrcT = typename A_TYPE::T;

    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(TBufPoolL0);
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulSubBlockInfo);

public:
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    using CallBack = MM_CB;

    using Scheduler::Schedule;

    MatmulImpl() {}

    VAR_PARAMS& GetVar() {
        return var;
    }

    void InitVar(const TCubeTiling &tiling) {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeA(false);
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeB(false);
        var.tpipe_ = &pipe;
        MATMUL_MODULE(MatmulSubBlockInfo)->SetSubBlockIdx(0);
        MATMUL_MODULE(MLoop)->Init(tiling.singleCoreM);
        MATMUL_MODULE(NLoop)->Init(tiling.singleCoreN);
        MATMUL_MODULE(KLoop)->Init(tiling.singleCoreK);
        MATMUL_MODULE(MatmulSubBlockInfo)->SetSubBlockIdx(1);
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

class TestSchedulerIntrablock : public testing::Test {
protected:
    void SetUp() {
        SetGCoreType(1);
    }
    void TearDown() {
        SetGCoreType(0);
    }
};

TEST_F(TestSchedulerIntrablock, ScheduleOnce_FakeIntra) {
    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_INTRABLOCK, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm;
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 0};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mm.InitVar(tiling);
    mm.SetBias(1);
    GlobalTensor<float> gm;
    mm.Schedule(gm, 0, false, true);
}

TEST_F(TestSchedulerIntrablock, ScheduleOnce_RealIntra) {
    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_INTRABLOCK, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm;
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 64, 64, 64, 64, 64, 64, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 0};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mm.InitVar(tiling);
    mm.SetBias(1);
    GlobalTensor<float> gm;
    mm.Schedule(gm, 0, false, false);
}
