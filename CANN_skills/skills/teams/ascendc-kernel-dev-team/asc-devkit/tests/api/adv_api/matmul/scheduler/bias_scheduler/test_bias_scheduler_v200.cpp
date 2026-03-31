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
#include "impl/adv_api/detail/matmul/scheduler/iterator/m_loop/m_loop.h"
#include "impl/adv_api/detail/matmul/scheduler/iterator/n_loop/n_loop.h"
#include "impl/adv_api/detail/matmul/resource/bias_buffer/c1_buffer/c1_buffer.h"
#include "impl/adv_api/detail/matmul/resource/bias_buffer/c2_buffer/c2_buffer.h"
#include "impl/adv_api/detail/matmul/scheduler/bias/bias_scheduler.h"
#include "impl/adv_api/detail/matmul/feature_trait/matmul_feature_trait.h"
#include "../../copy_cube_in/base_tiling_struct.h"


using namespace std;
using namespace AscendC;


namespace {
// Mock of CubeOutBuffer
template <typename T>
const LocalTensor<T> EMPTY_TENSOR;
template <typename IMPL, typename L0cT, const auto& MM_CFG>
class CustomCubeOutBuffer {
public:
    __aicore__ inline CustomCubeOutBuffer() {}
    __aicore__ inline ~CustomCubeOutBuffer() {}
    __aicore__ inline void Init(int32_t cacheSize = 1, uint32_t lenFactor = 1)
    {
        GetTPipePtr()->InitBuffer(CO1_, 1, cacheSize * lenFactor * sizeof(L0cT));
    }

    __aicore__ inline LocalTensor<L0cT> AllocTensor()
    {
        cMatrix_ = CO1_.template AllocTensor<L0cT>();
        return cMatrix_;
    }

    __aicore__ inline LocalTensor<L0cT> GetTensor()
    {
        return cMatrix_;
    }

    __aicore__ inline void EnQue(LocalTensor<L0cT>& tensor)
    {
        CO1_.EnQue(tensor);
    }

    __aicore__ inline LocalTensor<L0cT> DeQue()
    {
        return CO1_.template DeQue<L0cT>();
    }

    __aicore__ inline void FreeTensor(LocalTensor<L0cT>& co1Local)
    {
        CO1_.FreeTensor(co1Local);
    }

    __aicore__ inline void Destroy()
    {
        CO1_.FreeAllEvent();
    }

private:
    TQue<TPosition::CO1, 1> CO1_;
    LocalTensor<L0cT> cMatrix_;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type;
    using CubeOutBuffer = CustomCubeOutBuffer<IMPL, L0cT, MM_CFG>;
    using CopyBiasIn = Impl::Detail::CopyBiasIn<IMPL, A_TYPE, BIAS_TYPE, MM_CFG>;
    using LoadBias2C2 = Impl::Detail::LoadBias2C2<IMPL, A_TYPE, BIAS_TYPE, MM_CFG>;
    using BiasScheduler = Impl::Detail::BiasScheduler<IMPL, A_TYPE, B_TYPE, BIAS_TYPE, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const MatmulConfig& MM_CFG, class MM_CB,
MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE_PRIVATE(MLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(NLoop)
, MATMUL_IMPORT_MODULE_PRIVATE(KLoop)
, MATMUL_IMPORT_MODULE(CubeOutBuffer)
, MATMUL_IMPORT_MODULE_PRIVATE(C1Buffer)
, MATMUL_IMPORT_MODULE_PRIVATE(C2Buffer)
, MATMUL_IMPORT_MODULE(CopyBiasIn)
, MATMUL_IMPORT_MODULE(LoadBias2C2)
, MATMUL_IMPORT_MODULE(BiasScheduler)
, MATMUL_IMPORT_MODULE_PRIVATE(LocalWorkspace)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeTiling)
, MATMUL_IMPORT_MODULE_PRIVATE(MatmulShapeInfo)
{
    MATMUL_ALLOW_USING_PRIVATE(MLoop);
    MATMUL_ALLOW_USING_PRIVATE(NLoop);
    MATMUL_ALLOW_USING_PRIVATE(KLoop);
    MATMUL_ALLOW_USING(CubeOutBuffer);
    MATMUL_ALLOW_USING(CopyBiasIn);
    MATMUL_ALLOW_USING(LoadBias2C2);
    MATMUL_ALLOW_USING(BiasScheduler);
    MATMUL_ALLOW_USING_PRIVATE(C1Buffer);
    MATMUL_ALLOW_USING_PRIVATE(C2Buffer);
    MATMUL_ALLOW_USING_PRIVATE(LocalWorkspace);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeTiling);
    MATMUL_ALLOW_USING_PRIVATE(MatmulShapeInfo);

    using BiasT = typename BIAS_TYPE::T;
    using BiasScheduler::IsBias;
    using BiasScheduler::SetBias;
public:
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    using POLICY = MATMUL_POLICY<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>;
    using CallBack = MM_CB;

    MATMUL_USE_MODULE(CopyBiasIn);
    MATMUL_USE_MODULE(BiasScheduler);
    MATMUL_USE_MODULE(LoadBias2C2);
    MATMUL_USE_MODULE(LocalWorkspace);
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MatmulImpl() {}

    VAR_PARAMS& GetVar() {
        return var;
    }

    void InitLocalWorkspace() {
        LocalTensor<uint8_t> workspaceBuffer;
        pipe.InitBuffer(ubWorkspace_, 1, 81920);
        workspaceBuffer = ubWorkspace_.AllocTensor<uint8_t>();
        MATMUL_MODULE(LocalWorkspace)->Init(workspaceBuffer);
    }

    void InitVar(const TCubeTiling &tiling) {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeA(false);
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeB(false);
        var.tpipe_ = &pipe;
        InitLocalWorkspace();
    }

    void SetGMInput()
    {
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        uint8_t biasGlobal[tiling.GetN() * sizeof(BiasT)] = {0};
        GlobalTensor<BiasT> fakeInput;
        fakeInput.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(biasGlobal), tiling.GetN());
        MATMUL_MODULE(BiasScheduler)->SetInput(fakeInput);
    }

    void SetUBInput()
    {
        pipe.InitBuffer(qidBias_, 1, 1024);
        LocalTensor<BiasT> fakeInput = qidBias_.AllocTensor<BiasT>();
        MATMUL_MODULE(BiasScheduler)->SetInput(fakeInput);
    }

    void RunCase(bool isBias = false) {
        const auto tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        auto mLoop = MATMUL_MODULE(MLoop);
        auto nLoop = MATMUL_MODULE(NLoop);
        auto kLoop = MATMUL_MODULE(KLoop);
        MATMUL_MODULE(CubeOutBuffer)->Init(5120);
        MATMUL_MODULE(BiasScheduler)->Init();
        nLoop->Init(tiling.GetSingleCoreN());
        mLoop->Init(tiling.GetSingleCoreM());
        kLoop->Init(tiling.GetSingleCoreK());
        if constexpr (PhyPosIsGM(BIAS_TYPE::pos) ||
            !Impl::Detail::MatmulFeatureTrait<MM_CFG>::IsSupportUBToL1()) {
            SetGMInput();
        } else {
            SetUBInput();
        }

        MATMUL_MODULE(BiasScheduler)->SetBias(isBias);
        for (nLoop->OuterStart(); !nLoop->OuterEnd(); nLoop->OuterNext()) {
            for (mLoop->OuterStart(); !mLoop->OuterEnd(); mLoop->OuterNext()) {
                for (mLoop->InnerStart(); !mLoop->InnerEnd(); mLoop->InnerNext()) {
                    for (nLoop->InnerStart(); !nLoop->InnerEnd(); nLoop->InnerNext()) {
                        auto n = nLoop->GetInnerIdx();
                        auto co1Local = MATMUL_MODULE(CubeOutBuffer)->AllocTensor();
                        auto baseUseN =  nLoop->GetBaseShape();
                        auto srcOffset = n * tiling.GetBaseN();
                        for (kLoop->OuterStart(); !kLoop->OuterEnd(); kLoop->OuterNext()) {
                            auto bias = MATMUL_MODULE(BiasScheduler)->CopyIn(baseUseN, 1, srcOffset);
                            MATMUL_MODULE(BiasScheduler)->SplitLoad(bias, tiling.GetBaseN());
                            MATMUL_MODULE(BiasScheduler)->Free();
                        }
                        MATMUL_MODULE(CubeOutBuffer)->FreeTensor(co1Local);
                    }
                }
            }
        }
        MATMUL_MODULE(CubeOutBuffer)->Destroy();
        MATMUL_MODULE(BiasScheduler)->End();
    }

private:
    TPipe pipe;
    VAR_PARAMS var;
    TQue<TPosition::VECIN, 2> qidBias_;
    TQue<TPosition::VECIN, 2> ubWorkspace_;
};
}

class TestCopyBiasIn310P : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE_GM = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE_UB = MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float>;

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE_GM, CFG_MDL, void, CustomMatmulPolicy> mm1_;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE_UB, CFG_MDL, void, CustomMatmulPolicy> mm2_;
};

TEST_F(TestCopyBiasIn310P, copy_bias_from_gm_to_l0c) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN,
    // stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 64, 256, 256, 64, 256, 256, 32, 64, 128, 2, 4, 1, 1, 1, 1, 1, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mm1_.InitVar(tiling);
    mm1_.RunCase(true);
    ASSERT_TRUE(mm1_.IsBias());
    mm1_.SetBias(false);
    ASSERT_FALSE(mm1_.IsBias());
}

TEST_F(TestCopyBiasIn310P, copy_bias_from_ub_to_l0c) {
    // coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN,
    // stepKa, stepKb, isBias, iterateOrder
    TilingParams tilingParams = {1, 64, 256, 256, 64, 256, 256, 32, 64, 128, 2, 4, 1, 1, 1, 1, 1, 1};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mm2_.InitVar(tiling);
    mm2_.SetBias(false);
    ASSERT_FALSE(mm2_.IsBias());
    mm2_.RunCase(true);
    ASSERT_TRUE(mm2_.IsBias());
}
