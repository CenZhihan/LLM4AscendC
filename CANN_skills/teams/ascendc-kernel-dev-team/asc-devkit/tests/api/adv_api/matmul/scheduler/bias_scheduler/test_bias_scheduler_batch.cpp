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
#include "impl/adv_api/detail/matmul/resource/bias_buffer/c1_buffer/c1_buffer.h"
#include "impl/adv_api/detail/matmul/resource/bias_buffer/c2_buffer/c2_buffer.h"
#include "impl/adv_api/detail/matmul/stage/copy_cube_in/bias/copy_bias_in.h"
#include "impl/adv_api/detail/matmul/scheduler/bias/bias_scheduler.h"
#include "../../copy_cube_in/base_tiling_struct.h"

using namespace std;
using namespace AscendC;


namespace {
__aicore__ inline constexpr MatmulConfig GetMmConfig()
{
    auto cfg = GetNormalConfig();
    cfg.batchMode = BatchMode::BATCH_LESS_THAN_L1;
    return cfg;
}
constexpr MatmulConfig CFG_NORM_BATCH = GetMmConfig();

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type; 
    using CopyBiasIn = Impl::Detail::CopyBiasIn<IMPL, A_TYPE, BIAS_TYPE, MM_CFG>;
    using C1Buffer = Impl::Detail::C1Buffer<IMPL, BIAS_TYPE, A_TYPE, MM_CFG>;
    using C2Buffer = Impl::Detail::C2Buffer<IMPL, L0cT, A_TYPE, MM_CFG>;
    using LoadBias2C2 = Impl::Detail::LoadBias2C2<IMPL, A_TYPE, BIAS_TYPE, MM_CFG>;
    using BiasScheduler = Impl::Detail::BiasScheduler<IMPL, A_TYPE, B_TYPE, BIAS_TYPE, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, class MM_CB,
MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(CubeOutBuffer)
, MATMUL_IMPORT_MODULE(C1Buffer)
, MATMUL_IMPORT_MODULE(C2Buffer)
, MATMUL_IMPORT_MODULE(CopyBiasIn)
, MATMUL_IMPORT_MODULE(LoadBias2C2)
, MATMUL_IMPORT_MODULE(BiasScheduler)
, MATMUL_IMPORT_MODULE_PRIVATE(MLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(NLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(KLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(LocalWorkspace)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
{
    MATMUL_ALLOW_USING(CubeOutBuffer);
    MATMUL_ALLOW_USING(CopyBiasIn);
    MATMUL_ALLOW_USING(LoadBias2C2);
    MATMUL_ALLOW_USING(C1Buffer);
    MATMUL_ALLOW_USING(C2Buffer);
    MATMUL_ALLOW_USING(BiasScheduler);
    MATMUL_ALLOW_USING_PRIVATE(MLoop);
    MATMUL_ALLOW_USING_PRIVATE(NLoop);
    MATMUL_ALLOW_USING_PRIVATE(KLoop);
    MATMUL_ALLOW_USING_PRIVATE(LocalWorkspace);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);

    using BiasT = typename BIAS_TYPE::T;
    using BiasScheduler::IsBias;
    using BiasScheduler::SetBias;
public:
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;

    using CallBack = MM_CB;

    MATMUL_USE_MODULE(CopyBiasIn);
    MATMUL_USE_MODULE(LoadBias2C2);
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MatmulImpl() {}

    VAR_PARAMS& GetVar() {
        return var;
    }

    void InitVar(const TCubeTiling &tiling) {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        var.tpipe_ = &pipe;
    }

    void SetRuntimeParams(int32_t batchA = 4, int32_t batchB = 2, int32_t batchOuter = 1) {
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        MATMUL_MODULE(MatmulShapeInfo)->SetSingleShape(tiling.GetSingleCoreM(),
            tiling.GetSingleCoreN(), tiling.GetSingleCoreK());
        var.tailN_ = MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN() % tiling.GetBaseN();
        if (var.tailN_ == 0) {
            var.tailN_ = tiling.GetBaseN();
        }
        batchOuter_ = batchOuter;
        batchB_ = batchB;
        batchA_ = batchA;
    }

    void SetGMInput()
    {
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        auto batchNum = batchA_ > batchB_ ? batchA_ : batchB_;
        uint8_t biasGlobal[batchNum * tiling.GetN() * sizeof(BiasT)] = {0};
        GlobalTensor<BiasT> fakeInput;
        fakeInput.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(biasGlobal), batchNum * tiling.GetN());
        MATMUL_MODULE(BiasScheduler)->SetInput(fakeInput);
    }

    void RunCase(bool isBias = false, bool isUBIn = false) {
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        auto mLoop = MATMUL_MODULE(MLoop);
        auto nLoop = MATMUL_MODULE(NLoop);
        auto kLoop = MATMUL_MODULE(KLoop);
        auto batchNum = batchA_ > batchB_ ? batchA_ : batchB_;
        MATMUL_MODULE(BiasScheduler)->Init(batchNum);
        // not support UB input, need copy to GM
        SetGMInput();

        MATMUL_MODULE(BiasScheduler)->SetBias(isBias);
        nLoop->Init(tiling.GetSingleCoreN());
        mLoop->Init(tiling.GetSingleCoreM());
        kLoop->Init(tiling.GetSingleCoreK());

        for (int outer = 0; outer < batchOuter_; ++outer) {
            auto srcOffset = outer * batchNum * MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN();
            auto bias = MATMUL_MODULE(BiasScheduler)->CopyIn(
                        MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN(), batchNum, srcOffset);
            mLoop->GetTotalIter();
            for (int m = 0; m < mLoop->GetTotalIter(); ++m) {
                for (int n = 0; n < nLoop->GetTotalIter(); ++n) {
                    auto baseUseN =  (n + 1 == nLoop->GetTotalIter()) ? var.tailN_ : tiling.GetBaseN();
                    for (int k = 0; k < kLoop->GetTotalIter(); ++k) {
                         bool biasValid = false;
                        if (k == 0) {
                            biasValid = true;
                            MATMUL_MODULE(BiasScheduler)->SplitLoad(bias, baseUseN);
                        }
                        if (biasValid) {
                            MATMUL_MODULE(BiasScheduler)->Free();
                        }
                    }
                }
            }
            MATMUL_MODULE(BiasScheduler)->Destroy(bias);
        }
    }

private:
    TPipe pipe;
    VAR_PARAMS var;
    int32_t batchA_ = 1;
    int32_t batchB_ = 1;
    int32_t batchOuter_ = 1;
};
}

class TestBatchBiasScheduler : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE_BSNGD = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::BSNGD>;
    using A_TYPE_NORMAL = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NORMAL>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE_GM = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE_UB = MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float>;

    MatmulImpl<A_TYPE_BSNGD, B_TYPE, C_TYPE, BIAS_TYPE_GM, CFG_NORM_BATCH, void, CustomMatmulPolicy> mm1_;
    MatmulImpl<A_TYPE_NORMAL, B_TYPE, C_TYPE, BIAS_TYPE_UB, CFG_NORM_BATCH, void, CustomMatmulPolicy> mm2_;
};

TEST_F(TestBatchBiasScheduler, batch_scheduler_bias_for_gm) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN,
    // stepKa, stepKb, isBias, iterateOrder, batchNum
    TilingParams tilingParams = {1, 64, 256, 256, 64, 256, 256, 32, 64, 128, 2, 4, 1, 2, 1, 1, 1, 1, 4};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mm1_.InitVar(tiling);
    mm1_.SetRuntimeParams();
    mm1_.SetBias(false);
    ASSERT_FALSE(mm1_.IsBias());
    mm1_.RunCase(true, false);
    ASSERT_TRUE(mm1_.IsBias());
}

TEST_F(TestBatchBiasScheduler, batch_scheduler_bias_for_ub) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN,
    // stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 64, 256, 256, 64, 256, 256, 32, 64, 128, 2, 4, 1, 2, 1, 1, 1, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mm2_.InitVar(tiling);
    mm2_.SetRuntimeParams();
    mm2_.SetBias(false);
    ASSERT_FALSE(mm2_.IsBias());
    mm2_.RunCase(true, false);
    ASSERT_TRUE(mm2_.IsBias());
}
