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
#include "impl/adv_api/detail/matmul/policy/matmul_private_modules.h"
#include "impl/adv_api/detail/matmul/stage/copy_cube_in/bias/copy_bias_in.h"
#include "impl/adv_api/detail/matmul/scheduler/bias/bias_scheduler.h"
#include "../../copy_cube_in/base_tiling_struct.h"

using namespace std;
using namespace AscendC;

namespace {
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using BiasScheduler = Impl::Detail::BiasScheduler<IMPL, A_TYPE, B_TYPE, BIAS_TYPE, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, class MM_CB,
MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(CubeOutBuffer)
, MATMUL_IMPORT_MODULE_PRIVATE(C1Buffer)
, MATMUL_IMPORT_MODULE_PRIVATE(C2Buffer)
, MATMUL_IMPORT_MODULE_PRIVATE(CopyBiasIn)
, MATMUL_IMPORT_MODULE_PRIVATE(LoadBias2C2)
, MATMUL_IMPORT_MODULE_PRIVATE(MLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(NLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(KLoop)
, MATMUL_IMPORT_MODULE(BiasScheduler)
, MATMUL_IMPORT_MODULE_PRIVATE(LocalWorkspace)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING(CubeOutBuffer);
    MATMUL_ALLOW_USING(BiasScheduler);
    MATMUL_ALLOW_USING_PRIVATE(MLoop);
    MATMUL_ALLOW_USING_PRIVATE(NLoop);
    MATMUL_ALLOW_USING_PRIVATE(KLoop);
    MATMUL_ALLOW_USING_PRIVATE(C1Buffer);
    MATMUL_ALLOW_USING_PRIVATE(C2Buffer);
    MATMUL_ALLOW_USING_PRIVATE(CopyBiasIn);
    MATMUL_ALLOW_USING_PRIVATE(LoadBias2C2);
    MATMUL_ALLOW_USING_PRIVATE(LocalWorkspace);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);

    using BiasT = typename BIAS_TYPE::T;
    using BiasScheduler::IsBias;
    using BiasScheduler::SetBias;
public:
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    using POLICY = MATMUL_POLICY<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>;

    using CallBack = MM_CB;

    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MatmulImpl() {
        SetGCoreType(1);
    }

    ~MatmulImpl() {
        SetGCoreType(0);
    }

    VAR_PARAMS& GetVar() {
        return var;
    }

    void InitVar(const TCubeTiling &tiling) {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
    }

    void SetGMInput()
    {
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        uint8_t biasGlobal[tiling.GetN() * sizeof(BiasT)] = {0};
        GlobalTensor<BiasT> fakeInput;
        fakeInput.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(biasGlobal), tiling.GetN());
        MATMUL_MODULE(BiasScheduler)->SetInput(fakeInput);
    }

    void RunCase(bool isBias = false, bool isUBIn = false) {
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        auto mLoop = MATMUL_MODULE(MLoop);
        auto nLoop = MATMUL_MODULE(NLoop);
        auto kLoop = MATMUL_MODULE(KLoop);
        MATMUL_MODULE(BiasScheduler)->Init();

        nLoop->Init(tiling.GetSingleCoreN());
        mLoop->Init(tiling.GetSingleCoreM());
        kLoop->Init(tiling.GetSingleCoreK());
        SetGMInput();
        MATMUL_MODULE(BiasScheduler)->SetBias(isBias);
        for (nLoop->OuterStart(); !nLoop->OuterEnd(); nLoop->OuterNext()) {
            for (mLoop->OuterStart(); !mLoop->OuterEnd(); mLoop->OuterNext()) {
                for (mLoop->InnerStart(); !mLoop->InnerEnd(); mLoop->InnerNext()) {
                    for (nLoop->InnerStart(); !nLoop->InnerEnd(); nLoop->InnerNext()) {
                        auto baseUseN = nLoop->GetBaseShape();
                        auto srcOffset = nLoop->GetInnerIdx() * tiling.GetBaseN();
                        for (kLoop->OuterStart(); !kLoop->OuterEnd(); kLoop->OuterNext()) {
                            auto bias = MATMUL_MODULE(BiasScheduler)->CopyIn(baseUseN, 1, srcOffset);
                            MATMUL_MODULE(BiasScheduler)->SplitLoad(bias, baseUseN);
                            MATMUL_MODULE(BiasScheduler)->Free(bias);
                            MATMUL_MODULE(BiasScheduler)->Free();
                        }
                    }
                }
            }
        }
        MATMUL_MODULE(BiasScheduler)->End();
        pipe.Destroy();
    }

private:
    TPipe pipe;
    VAR_PARAMS var;
    TQue<TPosition::VECIN, 2> qidBias_;
};
}

class TestBiasScheduler : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE_GM = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE_UB = MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float>;

    // MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE_GM, CFG_NORM, void> mm1_;
    // MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE_UB, CFG_NORM, void> mm2_;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE_GM, CFG_MDL, void, CustomMatmulPolicy> mm1_;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE_UB, CFG_MDL, void, CustomMatmulPolicy> mm2_;
};

TEST_F(TestBiasScheduler, copy_bias_from_gm) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 64, 256, 256, 64, 256, 256, 32, 64, 128, 2, 4, 1, 1, 1, 1, 1, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mm1_.InitVar(tiling);
    mm1_.RunCase(true, false);
    ASSERT_TRUE(mm1_.IsBias());
    mm1_.SetBias(false);
    ASSERT_FALSE(mm1_.IsBias());
}

TEST_F(TestBiasScheduler, copy_bias_from_ub) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 64, 256, 256, 64, 256, 256, 32, 64, 128, 2, 4, 1, 1, 1, 1, 1, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mm2_.InitVar(tiling);
    mm2_.SetBias(false);
    ASSERT_FALSE(mm2_.IsBias());
    mm2_.RunCase(true, true);
    ASSERT_TRUE(mm2_.IsBias());
}
