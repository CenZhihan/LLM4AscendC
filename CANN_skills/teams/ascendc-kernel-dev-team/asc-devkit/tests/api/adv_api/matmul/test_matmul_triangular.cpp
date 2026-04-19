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

using namespace std;
using namespace AscendC;

struct TilingParams {
    __aicore__ TilingParams() {}
    __aicore__ TilingParams(uint32_t coreNum, uint32_t M, uint32_t N, uint32_t K, uint32_t singleCoreM,
        uint32_t singleCoreN, uint32_t singleCoreK, uint32_t baseM, uint32_t baseN, uint32_t baseK, uint32_t depthA1,
        uint32_t depthB1, uint32_t stepM, uint32_t stepN, uint32_t stepKa, uint32_t stepKb, uint32_t isbias,
        uint32_t iterateOrder) : coreNum_(coreNum), M_(M), N_(N), K_(K),
        singleCoreM_(singleCoreM), singleCoreN_(singleCoreN), singleCoreK_(singleCoreK), baseM_(baseM), baseN_(baseN),
        baseK_(baseK), depthA1_(depthA1), depthB1_(depthB1), stepM_(stepM), stepN_(stepN), stepKa_(stepKa),
        stepKb_(stepKb), isbias_(isbias), iterateOrder_(iterateOrder) {}
    __aicore__ void GetTiling(TCubeTiling &tiling)
    {
        tiling.usedCoreNum = coreNum_;
        tiling.M = M_;
        tiling.N = N_;
        tiling.Ka = K_;
        tiling.Kb = K_;
        tiling.singleCoreM = singleCoreM_;
        tiling.singleCoreN = singleCoreN_;
        tiling.singleCoreK = singleCoreK_;
        tiling.baseM = baseM_;
        tiling.baseN = baseN_;
        tiling.baseK = baseK_;
        tiling.depthA1 = depthA1_;
        tiling.depthB1 = depthB1_;
        tiling.stepM = stepM_;
        tiling.stepN = stepN_;
        tiling.stepKa = stepKa_;
        tiling.stepKb = stepKb_;
        tiling.isBias = isbias_;
        tiling.iterateOrder = iterateOrder_;
    }
    uint32_t coreNum_;
    uint32_t M_;
    uint32_t N_;
    uint32_t K_;
    uint32_t singleCoreM_;
    uint32_t singleCoreN_;
    uint32_t singleCoreK_;
    uint32_t baseM_;
    uint32_t baseN_;
    uint32_t baseK_;
    uint32_t depthA1_;
    uint32_t depthB1_;
    uint32_t stepM_;
    uint32_t stepN_;
    uint32_t stepKa_;
    uint32_t stepKb_;
    uint32_t isbias_;
    uint32_t iterateOrder_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline int32_t CalcGMOffset(int blockIdx, int usedCoreNum, TCubeTiling& param, int& offsetA, int& offsetB,
    int& offsetC, int& offsetBias, int32_t isTransposeAIn, int32_t isTransposeBIn)
{
    auto temp0 = ConstCeil(param.M, param.singleCoreM);
    auto temp1 = ConstCeil(param.N, param.singleCoreN);
    auto temp2 = ConstCeil(param.Ka, param.singleCoreK); // is 1 when k-axis is full loaded

    auto divideKcoreNum = usedCoreNum / temp2;

    auto mCoreIndx = (blockIdx % divideKcoreNum) % temp0;
    auto nCoreIndx = (blockIdx % divideKcoreNum) / temp0;
    auto subKindx = blockIdx / divideKcoreNum; // default 0

    if constexpr (A_TYPE::format == CubeFormat::ND) {
        if (isTransposeAIn > 0) {
            offsetA = mCoreIndx * param.singleCoreM + subKindx * param.M * param.singleCoreK;
        } else {
            offsetA = mCoreIndx * param.Ka * param.singleCoreM + subKindx * param.singleCoreK;
        }
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        offsetA = subKindx * param.singleCoreK * param.M + mCoreIndx * param.singleCoreM * BLOCK_CUBE;
    } else if constexpr (A_TYPE::format == CubeFormat::SCALAR) {
        // no need offsetA
    } else if constexpr (A_TYPE::format == CubeFormat::VECTOR) {
        // m only support 1, no need offsetA?
    } else {
        return -1;
    }

    if constexpr (B_TYPE::format == CubeFormat::ND) {
        if (isTransposeBIn > 0) {
            offsetB = subKindx * param.singleCoreK + nCoreIndx * param.Ka * param.singleCoreN;
        } else {
            offsetB = subKindx * param.singleCoreK * param.N + nCoreIndx * param.singleCoreN;
        }
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        offsetB = param.Kb * nCoreIndx * param.singleCoreN + subKindx * param.singleCoreK * BLOCK_CUBE;
    } else {
        return -1;
    }

    if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
        offsetC = mCoreIndx * param.N * param.singleCoreM + nCoreIndx * param.singleCoreN;
    } else if constexpr (C_TYPE::format == CubeFormat::NZ) {
        offsetC = param.M * nCoreIndx * param.singleCoreN + mCoreIndx * param.singleCoreM * BLOCK_CUBE;
    } else {
        return -1;
    }

    if constexpr (BIAS_TYPE::format == CubeFormat::ND) {
        offsetBias = nCoreIndx * param.singleCoreN;
    } else {
        return -1;
    }

    // tail M
    int gmUseM = param.M - mCoreIndx * param.singleCoreM;
    param.singleCoreM = gmUseM < param.singleCoreM ? gmUseM : param.singleCoreM;

    // tail N
    int gmUseN = param.N - nCoreIndx * param.singleCoreN;
    param.singleCoreN = gmUseN < param.singleCoreN ? gmUseN : param.singleCoreN;

    // tail K
    int gmUseK = param.Ka - subKindx * param.singleCoreK;
    param.singleCoreK = gmUseK < param.singleCoreK ? gmUseK : param.singleCoreK;
    return 0;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG>
__aicore__ inline void main_kernel_matmul(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
    TilingParams &tilingParam, int32_t isTransposeAIn, int32_t isTransposeBIn, int32_t trianMode,
    bool enSequentialWrite)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

    SetAtomicNone();
    TPipe que;
    TCubeTiling tiling;
    tilingParam.GetTiling(tiling);

    bool isTransposeA = isTransposeAIn > 0 ? true : false;
    bool isTransposeB = isTransposeBIn > 0 ? true : false;
    if (block_idx >= tiling.usedCoreNum) {
        return;
    }

    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;
    GlobalTensor<BiasT> biasGlobal;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(aGM), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(bGM), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T*>(cGM), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(biasGM), tiling.N);

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    int offsetBias = 0;
    CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(block_idx, tiling.usedCoreNum, tiling, offsetA, offsetB, offsetC,
        offsetBias, isTransposeAIn, isTransposeBIn);

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];
    auto gmBias = biasGlobal[offsetBias];

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::TrianUpperMatmulPolicy> mm1;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::TrianLowerMatmulPolicy> mm2;

    if (trianMode == 0) {
        mm1.SetSubBlockIdx(0);
        mm1.Init(&tiling, &que);

        mm1.SetTensorA(gmA, isTransposeA);
        mm1.SetTensorB(gmB, isTransposeB);
        if (tiling.isBias) {
            mm1.SetBias(gmBias);
        }
        mm1.IterateAll(gmC);
    } else {
        mm2.SetSubBlockIdx(0);
        mm2.Init(&tiling, &que);

        mm2.SetTensorA(gmA, isTransposeA);
        mm2.SetTensorB(gmB, isTransposeB);
        if (tiling.isBias) {
            mm2.SetBias(gmBias);
        }
        mm2.IterateAll(gmC);
    }

    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
}

class TEST_KERNEL_TRIAN_MATMUL : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

#define KERNEL_TRIAN_MATMUL_TESTCASE(TEST_KERNEL_TRIAN_MATMUL, tilingParams, A_Pos, B_Pos, C_Pos, BIAS_Pos, A_Format, B_Format, C_Format, BIAS_Format, \
    A_DType, B_DType, C_DType, BIAS_DType, isTransposeA, isTransposeB, CFG_Mode, trianMode, enSequentialWrite) \
    namespace Kernel_Matmul_Case_##tilingParams##_##A_Pos##_##B_Pos##_##C_Pos##_##BIAS_Pos##_##A_Format##_##B_Format##_##C_Format##_##BIAS_Format##_##A_DType##_##B_DType##_##C_DType##_##BIAS_DType##_##isTransposeA##_##isTransposeB##_##CFG_Mode##_##trianMode##_##enSequentialWrite{ \
    typedef MatmulType<AscendC::TPosition::A_Pos, CubeFormat::A_Format, A_DType, isTransposeA> aType;  \
    typedef MatmulType<AscendC::TPosition::B_Pos, CubeFormat::B_Format, B_DType, isTransposeB> bType;   \
    typedef MatmulType<AscendC::TPosition::C_Pos, CubeFormat::C_Format, C_DType> cType;       \
    typedef MatmulType<AscendC::TPosition::BIAS_Pos, CubeFormat::BIAS_Format, BIAS_DType> biasType;  \
    constexpr static MatmulConfig mmCFG = CFG_Mode;                                                          \
    TEST_F(TEST_KERNEL_TRIAN_MATMUL, Kernel_Trian_Matmul_Case_##tilingParams##_##A_Pos##_##B_Pos##_##C_Pos##_##BIAS_Pos##_##A_Format##_##B_Format##_##C_Format##_##BIAS_Format##_##A_DType##_##B_DType##_##C_DType##_##BIAS_DType##_##isTransposeA##_##isTransposeB##_##CFG_Mode##_##trianMode##_##enSequentialWrite) \
    {                                                                                                        \
        const int32_t left_data_size = tilingParams.M_ * tilingParams.K_;                                        \
        const int32_t right_data_size = tilingParams.K_ * tilingParams.N_;                                \
        const int32_t bias_data_size = tilingParams.N_;                                  \
        const int32_t output_data_size = tilingParams.M_ * tilingParams.N_;                               \
        uint8_t left_global[left_data_size * sizeof(A_DType)] = {0};     \
        uint8_t right_global[right_data_size * sizeof(B_DType)] = {0};  \
        uint8_t bias_global[bias_data_size * sizeof(BIAS_DType)] = {0};    \
        uint8_t output_global[output_data_size * sizeof(C_DType)] = {0};\
        main_kernel_matmul<aType, bType, cType, biasType, mmCFG>(left_global, right_global, output_global, bias_global, tilingParams, isTransposeA, isTransposeB, trianMode, enSequentialWrite);\
        for (int32_t i = 0; i < output_data_size * sizeof(C_DType); i++) {                                          \
            EXPECT_EQ(output_global[i], 0x00);                                                                      \
        }                                                                                                           \
    }                                                                                                               \
    }


// coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder
TilingParams tiling_params_trian_case1_910B1 = {1, 64, 256, 256, 32, 64, 256, 32, 64, 128, 1, 1, 1, 1, 1, 1, 1, 0};
TilingParams tiling_params_trian_case2_910B1 = {1, 60, 510, 250, 64, 64, 256, 64, 64, 128, 1, 1, 1, 1, 1, 1, 1, 0};
TilingParams tiling_params_trian_case3_910B1 = {1, 512, 512, 768, 512, 512, 768, 16, 32, 256, 3, 3, 1, 1, 3, 3, 0, 1};
TilingParams tiling_params_trian_case4_910B1 = {1, 64, 256, 256, 64, 256, 256, 32, 64, 256, 2, 4, 1, 2, 1, 1, 1, 0};
TilingParams tiling_params_trian_case5_910B1 = {1, 128, 1024, 128, 128, 1024, 128, 64, 256, 32, 4, 8, 1, 2, 4, 4, 0, 0};
TilingParams tiling_params_trian_case6_910B1 = {1, 1009, 188, 115, 1009, 188, 115, 256, 64, 32, 8, 4, 2, 1, 4, 4, 1, 1};

// TEST_KERNEL_TRIAN_MATMUL, tilingParams, A_Pos, B_Pos, C_Pos, BIAS_Pos, A_Format, B_Format, C_Format, BIAS_Format, A_DType, B_DType, C_DType, BIAS_DType, isTransposeA, isTransposeB, CFG_Mode, trianMode, enSequentialWrite
// Norm
KERNEL_TRIAN_MATMUL_TESTCASE(TEST_KERNEL_TRIAN_MATMUL, tiling_params_trian_case1_910B1, GM, GM, GM, GM, ND, ND, ND, ND, float, float, float, float, 0, 0, CFG_NORM, 0, false);
KERNEL_TRIAN_MATMUL_TESTCASE(TEST_KERNEL_TRIAN_MATMUL, tiling_params_trian_case2_910B1, GM, GM, GM, GM, ND, ND, ND, ND, float, float, float, float, 1, 0, CFG_NORM, 1, false);
KERNEL_TRIAN_MATMUL_TESTCASE(TEST_KERNEL_TRIAN_MATMUL, tiling_params_trian_case3_910B1, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 0, 1, CFG_NORM, 1, false);
KERNEL_TRIAN_MATMUL_TESTCASE(TEST_KERNEL_TRIAN_MATMUL, tiling_params_trian_case4_910B1, GM, GM, GM, GM, NZ, NZ, ND, ND, half, half, float, float, 1, 1, CFG_NORM, 0, false);
KERNEL_TRIAN_MATMUL_TESTCASE(TEST_KERNEL_TRIAN_MATMUL, tiling_params_trian_case5_910B1, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 0, 0, CFG_NORM, 0, false);
KERNEL_TRIAN_MATMUL_TESTCASE(TEST_KERNEL_TRIAN_MATMUL, tiling_params_trian_case6_910B1, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 0, 0, CFG_NORM, 1, false);

// MDL
KERNEL_TRIAN_MATMUL_TESTCASE(TEST_KERNEL_TRIAN_MATMUL, tiling_params_trian_case1_910B1, GM, GM, GM, GM, ND, ND, ND, ND, float, float, float, float, 0, 0, CFG_MDL, 0, false);
KERNEL_TRIAN_MATMUL_TESTCASE(TEST_KERNEL_TRIAN_MATMUL, tiling_params_trian_case2_910B1, GM, GM, GM, GM, ND, ND, ND, ND, float, float, float, float, 1, 0, CFG_MDL, 1, false);
KERNEL_TRIAN_MATMUL_TESTCASE(TEST_KERNEL_TRIAN_MATMUL, tiling_params_trian_case3_910B1, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 0, 1, CFG_MDL, 1, false);
KERNEL_TRIAN_MATMUL_TESTCASE(TEST_KERNEL_TRIAN_MATMUL, tiling_params_trian_case4_910B1, GM, GM, GM, GM, NZ, NZ, ND, ND, half, half, float, float, 1, 1, CFG_MDL, 0, false);
KERNEL_TRIAN_MATMUL_TESTCASE(TEST_KERNEL_TRIAN_MATMUL, tiling_params_trian_case5_910B1, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 0, 0, CFG_MDL, 0, false);
KERNEL_TRIAN_MATMUL_TESTCASE(TEST_KERNEL_TRIAN_MATMUL, tiling_params_trian_case6_910B1, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 0, 0, CFG_MDL, 1, false);
