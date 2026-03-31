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
#include "kernel_event.h"
#include "kernel_tiling/kernel_tiling.h"
#include "include/adv_api/matmul/tiling.h"
#include "include/adv_api/matmul/matmul.h"
#include "include/adv_api/matmul/matmul_intf.h"
#include "copy_cube_in/base_tiling_struct.h"

using namespace std;
using namespace AscendC;



template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void CalcGMOffset(int blockIdx, TCubeTiling& param, int& offsetA, int& offsetB,
    int& offsetC, int& offsetBias)
{
    using SrcT = typename A_TYPE::T;
    using DstT = typename C_TYPE::T;
    if(A_TYPE::layout == LayoutMode::BSNGD) {
        // BSNGD多核切B轴
        offsetA = blockIdx * param.ALayoutInfoS * param.ALayoutInfoN * param.ALayoutInfoG * param.ALayoutInfoD;
        // BSNGD多核切N轴
    }

    if(B_TYPE::layout == LayoutMode::BSNGD) {
        // BSNGD多核切B轴
        offsetB = blockIdx * param.BLayoutInfoS * param.BLayoutInfoN * param.BLayoutInfoG * param.BLayoutInfoD;
        // BSNGD多核切N轴
    }

    if(C_TYPE::layout == LayoutMode::BSNGD) {
        // BSNGD多核切B轴
        offsetC = blockIdx * param.CLayoutInfoS1 * param.CLayoutInfoN * param.CLayoutInfoG * param.CLayoutInfoS2;
        offsetBias = blockIdx * param.CLayoutInfoN * param.CLayoutInfoG * param.CLayoutInfoS2;
        // BSNGD多核切N轴
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG>
__aicore__ inline void main_kernel_matmul_batch(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
    TilingParamsBatch &tilingParam, int32_t isTransposeAIn, int32_t isTransposeBIn, bool enSequentialWrite)
{
    if (g_coreType == AIV && GetBlockIdx() != 0) { 
        return; 
    }

    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

    SetAtomicNone();
    TCubeTiling tiling;
    tilingParam.GetTiling(tiling);

    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;
    GlobalTensor<BiasT> biasGlobal;

    int32_t sizeA = tiling.ALayoutInfoB * tiling.ALayoutInfoS * tiling.ALayoutInfoN * tiling.ALayoutInfoG * tiling.ALayoutInfoD * sizeof(A_T);
    int32_t sizeB = tiling.BLayoutInfoB * tiling.BLayoutInfoS * tiling.BLayoutInfoN * tiling.BLayoutInfoG * tiling.BLayoutInfoD * sizeof(B_T);
    int32_t sizeC = tiling.CLayoutInfoB * tiling.CLayoutInfoS1 * tiling.CLayoutInfoN * tiling.CLayoutInfoG * tiling.CLayoutInfoS2 * sizeof(C_T);
    int32_t sizebias = tiling.CLayoutInfoB * tiling.CLayoutInfoN * tiling.CLayoutInfoG * tiling.CLayoutInfoS2 * sizeof(C_T);

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(aGM), sizeA);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(bGM), sizeB);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T*>(cGM), sizeC);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(biasGM), sizebias);

    tiling.shareMode = 0;             // 0, share mode
    tiling.shareL1Size = 512 * 1024;  // full L1
    tiling.shareL0CSize = 128 * 1024; // full L0C
    tiling.shareUbSize = 0;           // no UB

    int offset_a = 0, offset_b = 0, offset_c = 0, offset_bias = 0;
    CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(GetBlockIdx() / 2, tiling, offset_a, offset_b, offset_c, offset_bias);

    int batchA = tilingParam.batchM_;
    int batchB = tilingParam.batchN_;
    int batchC = batchA > batchB ? batchA : batchB;

    GlobalTensor<A_T> gm_a;
    gm_a.SetGlobalBuffer(const_cast<__gm__ A_T*>(aGlobal[offset_a].GetPhyAddr()), tiling.ALayoutInfoS * tiling.ALayoutInfoN * tiling.ALayoutInfoG * tiling.ALayoutInfoD);
    GlobalTensor<B_T> gm_b;
    gm_b.SetGlobalBuffer(const_cast<__gm__ B_T*>(bGlobal[offset_b].GetPhyAddr()), tiling.BLayoutInfoS * tiling.BLayoutInfoN * tiling.BLayoutInfoG * tiling.BLayoutInfoD);
    GlobalTensor<C_T> gm_c;
    gm_c.SetGlobalBuffer(const_cast<__gm__ C_T*>(cGlobal[offset_c].GetPhyAddr()), tiling.CLayoutInfoS1 * tiling.CLayoutInfoN * tiling.CLayoutInfoG * tiling.CLayoutInfoS2) ;
    GlobalTensor<BiasT> gm_bias;
    gm_bias.SetGlobalBuffer(const_cast<__gm__ BiasT*>(biasGlobal[offset_bias].GetPhyAddr()), tiling.CLayoutInfoN * tiling.CLayoutInfoG * tiling.CLayoutInfoS2);

    AscendC::MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm1;
    TPipe pipe;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#ifdef __DAV_CUBE__
    g_cubeTPipePtr = &pipe;
#elif defined(__DAV_VEC__)
    g_vecTPipePtr = &pipe;
#else
    g_tPipePtr = &pipe;
#endif
#else
    g_tPipePtr = &pipe;
#endif

    mm1.Init(&tiling, &pipe);
    int g_lay = tiling.ALayoutInfoG > tiling.BLayoutInfoG ? tiling.ALayoutInfoG : tiling.BLayoutInfoG;
    int for_exent = tiling.ALayoutInfoB * tiling.ALayoutInfoN * g_lay / batchC;
    for(int i=0; i<for_exent; ++i) {
        int batchOffsetA = i * tiling.ALayoutInfoD * batchA;
        int batchOffsetB = i * tiling.BLayoutInfoD * batchB;
        if (A_TYPE::layout == LayoutMode::BNGS1S2) {
            batchOffsetA = i * tiling.ALayoutInfoD * batchA * tiling.ALayoutInfoS;
        }
        if (B_TYPE::layout == LayoutMode::BNGS1S2) {
            batchOffsetB = i * tiling.BLayoutInfoD * batchB * tiling.BLayoutInfoS;
        }
        mm1.SetTensorA(gm_a[batchOffsetA], isTransposeAIn);
        mm1.SetTensorB(gm_b[batchOffsetB], isTransposeBIn);
        int idx_c = i * batchC;
        if (tiling.CLayoutInfoG == 1 && (tiling.BLayoutInfoG != 1 || tiling.ALayoutInfoG != 1)) {
            idx_c = idx_c / (tiling.BLayoutInfoG > tiling.ALayoutInfoG ? tiling.BLayoutInfoG : tiling.ALayoutInfoG);
        }
        if(tiling.isBias) {
            int batchOffsetBias = idx_c * tiling.CLayoutInfoS2;
            mm1.SetBias(gm_bias[batchOffsetBias]);
        }

        int batchOffsetC = idx_c * tiling.CLayoutInfoS2;
        if (C_TYPE::layout == LayoutMode::BNGS1S2) {
            batchOffsetC = idx_c * tiling.CLayoutInfoS2 * tiling.CLayoutInfoS1;
        }

        mm1.IterateBatch(gm_c[batchOffsetC], batchA, batchB, false);
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
}

class TEST_KERNEL_MATMUL : public testing::Test {
protected:
    void SetUp() {
        SetGCoreType(1);
    }
    void TearDown() {
        SetGCoreType(0);
    }
};

#define KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, tilingParams, M, N, K, baseM, baseN, baseK, A_Pos, B_Pos, C_Pos, BIAS_Pos, A_Format, B_Format, C_Format, BIAS_Format, \
    A_DType, B_DType, C_DType, BIAS_DType, isTransposeA, isTransposeB, CFG_Mode, enSequentialWrite, enTiling, enOuter, enOrderM) \
    namespace Kernel_Matmul_Case_##tilingParams##_##M##_##N##_##K##_##A_Pos##_##B_Pos##_##C_Pos##_##BIAS_Pos##_##A_Format##_##B_Format##_##C_Format##_##BIAS_Format##_##A_DType##_##B_DType##_##C_DType##_##BIAS_DType##_##isTransposeA##_##isTransposeB##_##CFG_Mode##_##enSequentialWrite##_##enTiling##_##enOuter##_##enOrderM{ \
    typedef AscendC::MatmulType<AscendC::TPosition::A_Pos, CubeFormat::A_Format, A_DType, isTransposeA, LayoutMode::NORMAL> aType;  \
    typedef AscendC::MatmulType<AscendC::TPosition::B_Pos, CubeFormat::B_Format, B_DType, isTransposeB, LayoutMode::NORMAL> bType;   \
    typedef AscendC::MatmulType<AscendC::TPosition::C_Pos, CubeFormat::C_Format, C_DType, false, LayoutMode::NORMAL> cType;       \
    typedef AscendC::MatmulType<AscendC::TPosition::BIAS_Pos, CubeFormat::BIAS_Format, BIAS_DType> biasType;       \
    TEST_F(TEST_KERNEL_MATMUL, Kernel_Matmul_Case_##tilingParams##_##M##_##N##_##K##_##A_Pos##_##B_Pos##_##C_Pos##_##BIAS_Pos##_##A_Format##_##B_Format##_##C_Format##_##BIAS_Format##_##A_DType##_##B_DType##_##C_DType##_##BIAS_DType##_##isTransposeA##_##isTransposeB##_##CFG_Mode##_##enSequentialWrite##_##enTiling##_##enOuter##_##enOrderM) \
    {                                                                                                                    \
        const int32_t batchNum_ = tilingParams.batchM_ > tilingParams.batchN_ ? tilingParams.batchM_ : tilingParams.batchN_;     \
        const int32_t left_data_size = tilingParams.M_ * tilingParams.K_ * tilingParams.batchM_;                                 \
        const int32_t right_data_size = tilingParams.K_ * tilingParams.N_ * tilingParams.batchN_;                                \
        const int32_t bias_data_size = tilingParams.N_ * batchNum_;                                  \
        const int32_t output_data_size = tilingParams.M_ * tilingParams.N_ * batchNum_;                               \
        uint8_t left_global[left_data_size * sizeof(A_DType)] = {0};     \
        uint8_t right_global[right_data_size * sizeof(B_DType)] = {0};  \
        uint8_t bias_global[bias_data_size * sizeof(BIAS_DType)] = {0};    \
        uint8_t output_global[output_data_size * sizeof(C_DType)] = {0};    \
        if (enTiling) { \
            constexpr MatmulConfig static CFG_NORM_NEED_PAD = { \
                .doNorm = true, \
                .doBasicBlock = false,  \
                .doMultiDataLoad = false,   \
                .basicM = baseM,  \
                .basicN = baseN,   \
                .basicK = baseK,   \
                .intrinsicsCheck = false,   \
                .isNBatch = false,  \
                .enVecND2NZ = false,    \
                .doSpecialBasicBlock = false,   \
                .doMTE2Preload = 0, \
                .singleCoreM = M,    \
                .singleCoreN = N,    \
                .singleCoreK = K,    \
                .stepM = 0, \
                .stepN = 0,    \
                .baseMN = 0,    \
                .singleCoreMN = 0,  \
                .enUnitFlag = true, \
                .isPerTensor = false,   \
                .hasAntiQuantOffset = false,    \
                .doIBShareNorm = false, \
                .doSpecialMDL = false,  \
                .enableInit = true, \
                .batchMode = BatchMode::BATCH_LESS_THAN_L1, \
                .enableEnd = true,  \
                .enableGetTensorC = true,   \
                .enableSetOrgShape = true,  \
                .enableSetBias = true,      \
                .enableSetTail = true,  \
                .enableQuantVector = true,  \
                .enableSetDefineData = true,    \
                .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,   \
                .enableReuse = true,    \
                .enableUBReuse = true,      \
                .enableL1CacheUB = false,   \
                .intraBlockPartSum = false, \
                .iterateOrder = IterateOrder::UNDEF,    \
                .scheduleType = ScheduleType::INNER_PRODUCT,    \
                .enableDoubleCache = false, \
                .isBiasBatch = true,    \
                .enableStaticPadZeros = true    \
            };  \
            constexpr static MatmulApiStaticTiling mmTiling = AscendC::GetMatmulApiTiling<aType, bType, cType, biasType>(CFG_NORM_NEED_PAD);    \
            main_kernel_matmul_batch<aType, bType, cType, biasType, mmTiling>(left_global, right_global, output_global, bias_global, tilingParams, isTransposeA, isTransposeB, enSequentialWrite);  \
        } else {    \
            main_kernel_matmul_batch<aType, bType, cType, biasType, CFG_Mode>(left_global, right_global, output_global, bias_global, tilingParams, isTransposeA, isTransposeB, enSequentialWrite);  \
        }   \
    } \
    }


// coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
// batchM batchN batchNum ALayoutInfoB ALayoutInfoS ALayoutInfoN ALayoutInfoG ALayoutInfoD 
// BLayoutInfoB BLayoutInfoS BLayoutInfoN BLayoutInfoG BLayoutInfoD 
// CLayoutInfoB CLayoutInfoS1 CLayoutInfoN CLayoutInfoG CLayoutInfoS2
TilingParamsBatch tiling_params_case1_910B1_batch = {1, 144, 77, 64, 144, 77, 64, 48, 80, 64, 3, 1, 3, 1, 3, 1, 0, 0, 1, 1, 1, 1, 144, 1, 1, 64, 1, 77, 1, 1, 64, 1, 144, 1, 1, 77};
TilingParamsBatch tiling_params_case2_910B1_batch = {1, 144, 64, 77, 144, 64, 77, 48, 32, 80, 3, 2, 1, 1, 3, 2, 0, 0, 1, 1, 1, 1, 144, 1, 1, 77, 1, 77, 1, 1, 64, 1, 144, 1, 1, 64};
TilingParamsBatch tiling_params_case3_910B1_batch = {1, 32, 32, 32, 32, 32, 32, 16, 16, 32, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 32, 1, 2, 32, 1, 32, 1, 1, 32, 1, 32, 1, 2, 32};
TilingParamsBatch tiling_params_case4_910B1_batch = {1, 64, 256, 64, 64, 256, 64, 32, 64, 64, 2, 4, 1, 1, 2, 4, 0, 0, 1, 1, 1, 1, 64, 2, 3, 64, 1, 256, 2, 1, 64, 1, 64, 2, 3, 256};
TilingParamsBatch tiling_params_case5_910B1_batch = {1, 32, 32, 32, 32, 32, 32, 16, 16, 32, 2, 2, 1, 1, 2, 2, 1, 0, 2, 2, 2, 1, 32, 1, 2, 32, 1, 32, 1, 2, 32, 1, 32, 1, 1, 32};
TilingParamsBatch tiling_params_case6_910B1_batch = {1, 32, 256, 64, 32, 256, 64, 32, 32, 64, 1, 2, 1, 1, 1, 2, 1, 0, 3, 3, 3, 1, 32, 2, 3, 64, 1, 256, 2, 3, 64, 1, 32, 2, 3, 256};
TilingParamsBatch tiling_params_case7_910B1_batch = {1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 2, 2, 1, 1, 2, 2, 1, 0, 2, 2, 2, 1, 32, 2, 1, 32, 1, 32, 2, 1, 32, 1, 32, 2, 1, 32};
TilingParamsBatch tiling_params_case8_910B1_batch = {1, 144, 77, 64, 144, 77, 64, 48, 80, 64, 3, 1, 3, 1, 3, 1, 0, 0, 1, 1, 1, 1, 144, 1, 1, 64, 1, 77, 1, 1, 64, 1, 144, 1, 1, 77};
TilingParamsBatch tiling_params_case9_910B1_batch = {1, 32, 256, 64, 32, 256, 64, 32, 32, 64, 1, 2, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 32, 2, 3, 64, 1, 256, 2, 3, 64, 1, 32, 2, 3, 256};

constexpr MatmulConfig mm_cfg = GetNormalConfig(false, false, false, BatchMode::BATCH_LARGE_THAN_L1);
// TEST_KERNEL_MATMUL, tilingParams, A_Pos, B_Pos, C_Pos, BIAS_Pos, A_Format, B_Format, C_Format, BIAS_Format, A_DType, B_DType, C_DType, BIAS_DType, isTransposeA, isTransposeB, CFG_Mode, enSequentialWrite, enTiling, enOuter, enOrderM
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, tiling_params_case1_910B1_batch, 144, 77, 64, 48, 80, 64, GM, GM, GM, GM, ND, ND, ND, ND, half, half, half, float, 0, 0, mm_cfg, false, false, false, false);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, tiling_params_case2_910B1_batch, 144, 64, 77, 48, 32, 80, GM, GM, GM, GM, ND, ND, ND, ND, half, half, half, float, 0, 0, mm_cfg, false, false, false, false);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, tiling_params_case3_910B1_batch, 32, 32, 32, 16, 16, 32, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 0, 1, mm_cfg, false, false, false, false);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, tiling_params_case4_910B1_batch, 64, 256, 64, 32, 64, 64, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 0, 1, mm_cfg, false, false, false, false);
// test batch split loop
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, tiling_params_case5_910B1_batch, 32, 32, 32, 16, 16, 32, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 0, 1, mm_cfg, false, true, false, false);
// test const
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, tiling_params_case7_910B1_batch, 32, 32, 32, 32, 32, 32, GM, GM, GM, GM, ND, ND, ND, ND, half, half, half, float, 0, 0, mm_cfg, false, true, false, false);
// test SINGLE_LARGE_THAN_L1
constexpr MatmulConfig single_mm_cfg = GetNormalConfig(false, false, false, BatchMode::SINGLE_LARGE_THAN_L1);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, tiling_params_case8_910B1_batch, 144, 77, 64, 48, 80, 64, GM, GM, GM, GM, ND, ND, ND, ND, half, half, half, float, 0, 0, single_mm_cfg, false, false, false, false);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, tiling_params_case9_910B1_batch, 32, 256, 64, 32, 32, 64, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 0, 1, single_mm_cfg, false, false, false, false);

constexpr MatmulConfig CFG_NORM_OUTER_PRODUCT = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1, true, IterateOrder::ORDER_M, ScheduleType::OUTER_PRODUCT);
TilingParamsBatch tiling_params_case10_910B1_batch = {1, 81, 256, 64, 81, 256, 64, 32, 32, 64, 1, 2, 1, 1, 1, 2, 0, 0, 1, 1, 1, 1, 81, 1, 1, 64, 1, 256, 1, 1, 64, 1, 81, 1, 1, 256};
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, tiling_params_case10_910B1_batch, 81, 256, 64, 32, 32, 64, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 0, 0, CFG_NORM_OUTER_PRODUCT, false, false, false, false);
 
constexpr MatmulConfig CFG_NORM_OUTER_PRODUCT_N = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1, true, IterateOrder::ORDER_N, ScheduleType::OUTER_PRODUCT);
TilingParamsBatch tiling_params_case11_910B1_batch = {1, 81, 256, 64, 81, 256, 64, 32, 32, 64, 1, 2, 1, 1, 1, 2, 0, 0, 1, 1, 1, 1, 81, 1, 1, 64, 1, 256, 1, 1, 64, 1, 81, 1, 1, 256};
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, tiling_params_case11_910B1_batch, 81, 256, 64, 32, 32, 64, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 0, 0, CFG_NORM_OUTER_PRODUCT_N, false, false, false, false);
