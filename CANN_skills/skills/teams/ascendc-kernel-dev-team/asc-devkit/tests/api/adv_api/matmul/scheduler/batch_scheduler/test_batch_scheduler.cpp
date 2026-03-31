
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
#include "../../copy_cube_in/base_tiling_struct.h"
#include "../fake_modules.h"

#include "test_custom_loop.h"

using namespace std;
using namespace AscendC;
using namespace TestCustomModules;

namespace {
template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG , typename = void>
class CustomBatchLoop
{
public:
    __aicore__ inline void Init(int32_t batchA, int32_t batchB, int32_t batchOuter) {}

    __aicore__ inline void SetBatchNum(int32_t batchA, int32_t batchB, int32_t batchOuter) {}

    // Outer Loop
    __aicore__ inline void OuterStart() {
        outerIdx_ = 0;
    }

    __aicore__ inline void OuterNext() {
        outerIdx_++;
    }

    __aicore__ inline bool OuterEnd()
    {
        return outerIdx_ >= 1;
    }

    __aicore__ inline uint32_t GetOuterIndex() const
    {
        return 0;
    }

    __aicore__ inline uint32_t GetDstOffset() const
    {
        return 0;
    }

    __aicore__ inline int32_t GetBatchNum() const
    {
        return 1;
    }

    __aicore__ inline int32_t GetMainBatchBlockA() const
    {
        return 1;
    }

    __aicore__ inline int32_t GetBatchA() const
    {
        return 1;
    }

        __aicore__ inline int32_t GetBatchB() const
    {
        return 1;
    }

    __aicore__ inline int32_t GetBiasBatchSrcOffset() const
    {
        return 0;
    }

    // Double Buffer Loop
    __aicore__ inline void SplitStart() {
        splitIdx_ = 0;
    }

    __aicore__ inline void SplitNext() {
        splitIdx_++;
    }

    __aicore__ inline bool SplitEnd()
    {
        return splitIdx_ >= 1;
    }

    template <InputTypeTag tag>
    __aicore__ inline uint32_t GetSplitIndex() const
    {
        return 0;
    }

    __aicore__ inline uint32_t GetSplitIndex() const
    {
        return 0;
    }

    __aicore__ inline int32_t GetSplitSize() const
    {
        return 1;
    }

    __aicore__ inline int32_t GetSplitBatchNum() const
    {
        return 1;
    }

    // Inner Loop
    __aicore__ inline void InnerStart() {
        innerIdx_ = 0;
    }

    __aicore__ inline void InnerNext() {
        innerIdx_++;
    }

    __aicore__ inline bool InnerEnd()
    {
        return innerIdx_ >= 1;
    }

    __aicore__ inline uint32_t GetInnerIndex() const
    {
        return 0;
    }

    __aicore__ inline uint32_t GetBatchIndex() const
    {
        return 0;
    }

    __aicore__ inline void CalcBatchOuterIdx(uint32_t& outerIdxA, uint32_t& outerIdxB)
    {
    }

private:
    uint32_t outerIdx_;
    uint32_t splitIdx_;
    uint32_t innerIdx_;
};

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CustomMatmulPolicy : public Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>
{
public:
    using L0cT = typename GetMmDstType<typename A_TYPE::T>::Type;
    using BatchCopyCubeInA = CustomCopyCubeIn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using BatchCopyCubeInB = CustomCopyCubeIn<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using BatchScheduler = Impl::Detail::BatchScheduler<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>;
    using BatchLoop = CustomBatchLoop<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using MLoop = CustomLoop<IMPL, A_TYPE, MM_CFG>;
    using NLoop = CustomLoop<IMPL, B_TYPE, MM_CFG>;
    using KLoop = CustomKLoop<IMPL, typename A_TYPE::T, A_TYPE, MM_CFG>;
    using LoadToA2 = CustomLoadToL0<IMPL, A_TYPE, MM_CFG>;
    using LoadToB2 = CustomLoadToL0<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using TBufPoolL0 = CustomTBufPoolL0<IMPL, A_TYPE, B_TYPE, MM_CFG>;
    using MmadCompute = CustomMmadCompute<IMPL, L0cT, A_TYPE, typename A_TYPE::T, MM_CFG>;
    using CubeOutBuffer = CustomCubeOutBuffer<IMPL, L0cT, MM_CFG>;
    using CopyCubeOut = CustomCopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG>;
    using BiasScheduler = CustomBiasScheduler<IMPL, A_TYPE, B_TYPE, BIAS_TYPE, MM_CFG>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG,
class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>, MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
class MatmulImpl
: MATMUL_IMPORT_MODULE(BatchScheduler)
, MATMUL_IMPORT_MODULE(BatchLoop)
, MATMUL_IMPORT_MODULE(MLoop)
, MATMUL_IMPORT_MODULE(NLoop)
, MATMUL_IMPORT_MODULE(KLoop)
, MATMUL_IMPORT_MODULE(BatchCopyCubeInA)
, MATMUL_IMPORT_MODULE(BatchCopyCubeInB)
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
    MATMUL_ALLOW_USING(BatchLoop);
    MATMUL_ALLOW_USING(MLoop);
    MATMUL_ALLOW_USING(NLoop);
    MATMUL_ALLOW_USING(KLoop);
    MATMUL_ALLOW_USING(BatchScheduler);
    MATMUL_ALLOW_USING(LoadToA2);
    MATMUL_ALLOW_USING(LoadToB2);
    MATMUL_ALLOW_USING(TBufPoolL0);
    MATMUL_ALLOW_USING(MmadCompute);
    MATMUL_ALLOW_USING(BatchCopyCubeInA);
    MATMUL_ALLOW_USING(BatchCopyCubeInB);
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
    MATMUL_USE_MODULE(MatmulShapeInfo);

public:
    using VAR_PARAMS =
        typename Impl::Detail::MatmulParams<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, GetMatmulMode(MM_CFG)>::PARAMS;
    using IMPL = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;
    using CallBack = MM_CB;

    using BatchScheduler::Schedule;

    MatmulImpl() {}

    VAR_PARAMS& GetVar() {
        return var;
    }

    void InitVar(const TCubeTiling &tiling) {
        MATMUL_MODULE(MatmulShapeTiling)->SetTiling(&tiling);
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeA(false);
        MATMUL_MODULE(MatmulShapeInfo)->SetTransposeB(false);
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

class TestBatchScheduler : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

private:
    using A_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NORMAL>;
    using B_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true, LayoutMode::NORMAL>;
    using C_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::NORMAL>;
    using BIAS_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    static constexpr MatmulConfig mm_cfg = GetNormalConfig(false, false, false, BatchMode::BATCH_LARGE_THAN_L1);
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, mm_cfg, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm;

    constexpr MatmulConfig static staticMmCfg = {
        .doNorm = true,
        .doBasicBlock = false,
        .doMultiDataLoad = false,
        .basicM = 32,
        .basicN = 32,
        .basicK = 32,
        .intrinsicsCheck = false,
        .isNBatch = false,
        .enVecND2NZ = false,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = 0,
        .singleCoreM = 32,
        .singleCoreN = 32,
        .singleCoreK = 32,
        .stepM = 0,
        .stepN = 0,
        .baseMN = 0,
        .singleCoreMN = 0,
        .enUnitFlag = true,
        .isPerTensor = false,
        .hasAntiQuantOffset = false,
        .doIBShareNorm = false,
        .doSpecialMDL = false,
        .enableInit = true,
        .batchMode = BatchMode::BATCH_LESS_THAN_L1,
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
    constexpr static MatmulApiStaticTiling staticTiling = AscendC::GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(staticMmCfg);
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, staticTiling, MatmulCallBackFunc<nullptr, nullptr, nullptr>, CustomMatmulPolicy> mm1;
};

// coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
// batchM batchN batchNum ALayoutInfoB ALayoutInfoS ALayoutInfoN ALayoutInfoG ALayoutInfoD
// BLayoutInfoB BLayoutInfoS BLayoutInfoN BLayoutInfoG BLayoutInfoD
// CLayoutInfoB CLayoutInfoS1 CLayoutInfoN CLayoutInfoG CLayoutInfoS2
TEST_F(TestBatchScheduler, Schedule_ComputeMultiIter) {
    TilingParamsBatch tilingParams = {1, 144, 77, 64, 144, 77, 64, 48, 80, 64, 3, 1, 3, 1, 3, 1, 0, 0, 1, 1, 1, 1, 144, 1, 1, 64, 1, 77, 1, 1, 64, 1, 144, 1, 1, 77};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mm.InitVar(tiling);
    mm.SetBias(1);
    GlobalTensor<float> cGlobal;
    mm.Schedule(cGlobal, false, false, false, 0, 0, 0);
}

TEST_F(TestBatchScheduler, DISABLED_Schedule_ComputeOneIter) {
    TilingParamsBatch tilingParams = {1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 0, 2, 2, 2, 1, 32, 2, 1, 32, 1, 32, 2, 1, 32, 1, 32, 2, 1, 32};
    TCubeTiling tiling;
    tilingParams.GetTiling(tiling);
    mm1.InitVar(tiling);
    mm1.SetBias(1);
    GlobalTensor<float> cGlobal;
    mm1.Schedule(cGlobal, false, false, false, 0, 0, 0);
}
