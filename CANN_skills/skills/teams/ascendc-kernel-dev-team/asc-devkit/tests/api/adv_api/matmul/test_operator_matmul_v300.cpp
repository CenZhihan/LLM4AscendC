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

using namespace std;
using namespace AscendC;
struct TilingParams {
    __aicore__ TilingParams() {}
    __aicore__ TilingParams(uint32_t coreNum, uint32_t M, uint32_t N, uint32_t K, uint32_t singleCoreM,
        uint32_t singleCoreN, uint32_t singleCoreK, uint32_t baseM, uint32_t baseN, uint32_t baseK, uint32_t depthA1,
        uint32_t depthB1, uint32_t stepM, uint32_t stepN, uint32_t stepKa, uint32_t stepKb, uint32_t isbias,
        uint32_t iterateOrder)
        : coreNum_(coreNum),
          M_(M),
          N_(N),
          K_(K),
          singleCoreM_(singleCoreM),
          singleCoreN_(singleCoreN),
          singleCoreK_(singleCoreK),
          baseM_(baseM),
          baseN_(baseN),
          baseK_(baseK),
          depthA1_(depthA1),
          depthB1_(depthB1),
          stepM_(stepM),
          stepN_(stepN),
          stepKa_(stepKa),
          stepKb_(stepKb),
          isbias_(isbias),
          iterateOrder_(iterateOrder)
    {}
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
__aicore__ inline int32_t CalcGMOffset(int blockIdx, int usedCoreNum, TCubeTiling &param, int &offsetA, int &offsetB,
    int &offsetC, int &offsetBias, int32_t isTransposeAIn, int32_t isTransposeBIn)
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

    // tail block M
    int gmUseM = param.M - mCoreIndx * param.singleCoreM;
    param.singleCoreM = gmUseM < param.singleCoreM ? gmUseM : param.singleCoreM;

    // tail block N
    int gmUseN = param.N - nCoreIndx * param.singleCoreN;
    param.singleCoreN = gmUseN < param.singleCoreN ? gmUseN : param.singleCoreN;

    // tail block K
    int gmUseK = param.Ka - subKindx * param.singleCoreK;
    param.singleCoreK = gmUseK < param.singleCoreK ? gmUseK : param.singleCoreK;
    return 0;
}

class TEST_KERNEL_MATMUL : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
    }
};

template <typename T, TPosition> struct ExtractTensor {
    using type = LocalTensor<T>;
};

template <typename T> struct ExtractTensor<T, TPosition::GM> {
    using type = GlobalTensor<T>;
};

template <class aType, class bType, class cType, class biasType, const MatmulConfig &MM_CFG, bool mmMatmul>
__aicore__ inline void kernel_matmul(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, TilingParams &tilingParam,
    int32_t isTransposeAIn, int32_t isTransposeBIn)
{
    using A_T = typename aType::T;
    using B_T = typename bType::T;
    using C_T = typename cType::T;
    using BiasT = typename biasType::T;

    using A_Tensor = typename ExtractTensor<A_T, aType::pos>::type;
    using B_Tensor = typename ExtractTensor<B_T, bType::pos>::type;
    using C_Tensor = typename ExtractTensor<C_T, cType::pos>::type;

    SetAtomicNone();
    TPipe que;
    TCubeTiling tiling;
    tilingParam.GetTiling(tiling);

    bool isTransposeA = isTransposeAIn > 0 ? true : false;
    bool isTransposeB = isTransposeBIn > 0 ? true : false;

    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;
    GlobalTensor<BiasT> biasGlobal;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), tiling.N);

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    int offsetBias = 0;

    CalcGMOffset<aType, bType, cType, biasType>(block_idx, tiling.usedCoreNum, tiling, offsetA, offsetB, offsetC,
        offsetBias, 0, 0);

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];
    auto gmBias = biasGlobal[offsetBias];

    A_Tensor inputA;
    B_Tensor inputB;
    C_Tensor outputC;

    TQue<TPosition::VECIN, 2> leftMatrix;
    TQue<TPosition::VECIN, 2> rightMatrix;
    TQue<TPosition::VECIN, 2> resultCMatrix;

    TQue<TPosition::A1, 2> qidA1;
    TQue<TPosition::B1, 2> qidB1;
    TQue<TPosition::C1, 2> qidRes;


    SetAtomicNone();
    MatmulImpl<aType, bType, cType, biasType> mm;
    if constexpr(mmMatmul) {
        REGIST_MATMUL_OBJ(&que, GetSysWorkSpacePtr(), mm);
        mm.Init(&tiling);
    } else {
        mm.SetSubBlockIdx(0);
        mm.Init(&tiling, &que);
    }

    if constexpr (TPosition::GM == aType::pos) {
        inputA = aGlobal[offsetA];
    } else if constexpr (TPosition::VECCALC == aType::pos) {
        que.InitBuffer(leftMatrix, 1, tiling.M * tiling.Ka * sizeof(A_T));
        inputA = leftMatrix.AllocTensor<A_T>();
    } else {
        que.InitBuffer(qidA1, 1, tiling.M * tiling.Ka * sizeof(A_T));
        inputA = qidA1.AllocTensor<A_T>();
    }
    if (isTransposeA) {
        mm.SetTensorA(inputA, isTransposeA);
    } else {
        mm.SetTensorA(inputA);
    }

    if constexpr (TPosition::GM == bType::pos) {
        inputB = bGlobal[offsetB];
    } else if constexpr (TPosition::VECCALC == bType::pos) {
        que.InitBuffer(rightMatrix, 1, tiling.Kb * tiling.N * sizeof(B_T));
        inputB = rightMatrix.AllocTensor<B_T>();
    } else {
        que.InitBuffer(qidB1, 1, tiling.Kb * tiling.N * sizeof(B_T));
        inputB = qidB1.AllocTensor<B_T>();
    }
    if (isTransposeB) {
        mm.SetTensorB(inputB, isTransposeB);
    } else {
        mm.SetTensorB(inputB);
    }
    if constexpr (TPosition::GM == cType::pos) {
        outputC = cGlobal[offsetC];
    } else if constexpr (TPosition::SHM == cType::pos) {
        que.InitBuffer(qidRes, 1, tiling.M * tiling.N * sizeof(C_T));
        outputC = qidRes.AllocTensor<C_T>();
    } else {
        que.InitBuffer(resultCMatrix, 1, tiling.M * tiling.N * sizeof(C_T));
        outputC = resultCMatrix.AllocTensor<C_T>();
    }

    // MOCKER(Fixpipe, void (*)(const C_Tensor &, const LocalTensor<C_T> &, const FixpipeParams<C_T> &))
    //     .expects(atLeast(1));

    // MOCKER((&MacroMatmul<C_T, A_T, B_T, BiasT, EnUnitFlag(MM_CFG), 0, 0>::MmadMacro)).expects(atLeast(1));

    // MOCKER((&MacroMatmul<C_T, A_T, B_T, BiasT, EnUnitFlag(MM_CFG), 0, 0>::LoadL12L0A)).expects(atLeast(1));
    // MOCKER((&MacroMatmul<C_T, A_T, B_T, BiasT, EnUnitFlag(MM_CFG), 0, 0>::LoadL12L0B)).expects(atLeast(1));

    mm.IterateAll(outputC);
    if constexpr (aType::pos == TPosition::VECCALC) {
        leftMatrix.FreeTensor(inputA);
    }
    if constexpr (aType::pos == TPosition::SHM) {
        qidA1.FreeTensor(inputA);
    }
    if constexpr (bType::pos == TPosition::VECCALC) {
        rightMatrix.FreeTensor(inputB);
    }
    if constexpr (bType::pos == TPosition::SHM) {
        qidB1.FreeTensor(inputB);
    }
    if constexpr (cType::pos == TPosition::VECCALC) {
        resultCMatrix.FreeTensor(outputC);
    }
    if constexpr (cType::pos == TPosition::SHM) {
        qidRes.FreeTensor(outputC);
    }
    SetAtomicNone();
    PipeBarrier<PIPE_ALL>();
}

// coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa,
// stepKb, isBias, iterateOrder
TilingParams g_tilingParams = { 1, 16, 32, 32, 16, 32, 32, 16, 32, 32, 1, 1, 1, 1, 1, 1, 1, 0 };

#define KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, tilingParams, A_Pos, B_Pos, C_Pos, BIAS_Pos, A_Format, B_Format,                                                                                                         \
    C_Format, BIAS_Format, A_DType, B_DType, C_DType, BIAS_DType, isTransposeA, isTransposeB, CFG_Mode, MM_Matmul)                                                                                                          \
    TEST_F(TEST_KERNEL_MATMUL,                                                                                                                                                                                              \
        Kernel_Matmul_Case##tilingParams##_##A_Pos##_##B_Pos##_##C_Pos##_##BIAS_Pos##_##A_Format##_##B_Format##_##C_Format##_##BIAS_Format##_##A_DType##_##B_DType##_##C_DType##_##BIAS_DType##_##isTransposeA##_##isTransposeB##_##CFG_Mode##_##MM_Matmul)   \
    {                                                                                                                                                                                                                       \
        typedef MatmulType<AscendC::TPosition::A_Pos, CubeFormat::A_Format, A_DType, isTransposeA> aType;                                                                                                           \
        typedef MatmulType<AscendC::TPosition::B_Pos, CubeFormat::B_Format, B_DType, isTransposeB> bType;                                                                                                           \
        typedef MatmulType<AscendC::TPosition::C_Pos, CubeFormat::C_Format, C_DType> cType;                                                                                                                         \
        typedef MatmulType<AscendC::TPosition::BIAS_Pos, CubeFormat::BIAS_Format, BIAS_DType> biasType;                                                                                                             \
        TilingParams tilingParam = tilingParams;                                                                                                                                                                            \
        const int32_t left_data_size = tilingParam.M_ * tilingParam.K_;                                                                                                                                                     \
        const int32_t right_data_size = tilingParam.K_ * tilingParam.N_;                                                                                                                                                    \
        const int32_t bias_data_size = tilingParam.N_;                                                                                                                                                                      \
        const int32_t output_data_size = tilingParam.M_ * tilingParam.N_;                                                                                                                                                   \
        uint8_t left_global[left_data_size * sizeof(A_DType)] = {0};                                                                                                                                                        \
        uint8_t right_global[right_data_size * sizeof(B_DType)] = {0};                                                                                                                                                      \
        uint8_t bias_global[bias_data_size * sizeof(BIAS_DType)] = {0};                                                                                                                                                     \
        uint8_t output_global[output_data_size * sizeof(C_DType)] = {0};                                                                                                                                                    \
        kernel_matmul<aType, bType, cType, biasType, CFG_Mode, MM_Matmul>(left_global, right_global, output_global,                                                                                                         \
            bias_global, tilingParams, isTransposeA, isTransposeB);                                                                                                                                                         \
        for (int32_t i = 0; i < output_data_size * sizeof(C_DType); i++) {                                                                                                                                                  \
            EXPECT_EQ(output_global[i], 0x00);                                                                                                                                                                              \
        }                                                                                                                                                                                                                   \
    }

KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, SHM, GM, ND, ND, ND, ND, half, half, float,
    float, 0, 0, CFG_NORM, false);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, SHM, GM, ND, ND, ND, ND, int8_t, int8_t, int32_t,
    int32_t, 0, 0, CFG_NORM, false);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 0, 0,
    CFG_NORM, false);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, GM, GM, NZ, NZ, ND, ND, half, half, float, float, 0, 0,
    CFG_NORM, false);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, GM, GM, ND, ND, ND, ND, float, float, float, float, 0, 0,
    CFG_NORM, false);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, VECCALC, VECCALC, GM, GM, ND, ND, ND, ND, half, half, float,
    float, 0, 0, CFG_NORM, false);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, VECCALC, VECCALC, GM, GM, NZ, NZ, ND, ND, half, half, float,
    float, 0, 0, CFG_NORM, false);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, A1, B1, GM, GM, NZ, NZ, ND, ND, half, half, float, float, 0, 0,
    CFG_NORM, false);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, VECCALC, GM, ND, ND, ND, ND, half, half, float,
    float, 0, 0, CFG_NORM, false);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, VECCALC, GM, ND, ND, NZ, ND, half, half, float,
    float, 0, 0, CFG_NORM, false);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, GM, GM, NZ, NZ, NZ, ND, half, half, float, float, 0, 0,
    CFG_NORM, false);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, SHM, GM, ND, ND, ND, ND, half, half, float,
    float, 0, 0, CFG_NORM, true);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, SHM, GM, ND, ND, ND, ND, int8_t, int8_t, int32_t,
    int32_t, 0, 0, CFG_NORM, true);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 0, 0,
    CFG_NORM, true);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, GM, GM, NZ, NZ, ND, ND, half, half, float, float, 0, 0,
    CFG_NORM, true);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, GM, GM, ND, ND, ND, ND, float, float, float, float, 0, 0,
    CFG_NORM, true);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, VECCALC, VECCALC, GM, GM, ND, ND, ND, ND, half, half, float,
    float, 0, 0, CFG_NORM, true);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, VECCALC, VECCALC, GM, GM, NZ, NZ, ND, ND, half, half, float,
    float, 0, 0, CFG_NORM, true);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, A1, B1, GM, GM, NZ, NZ, ND, ND, half, half, float, float, 0, 0,
    CFG_NORM, true);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, VECCALC, GM, ND, ND, ND, ND, half, half, float,
    float, 0, 0, CFG_NORM, true);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, VECCALC, GM, ND, ND, NZ, ND, half, half, float,
    float, 0, 0, CFG_NORM, true);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, GM, GM, GM, GM, NZ, NZ, NZ, ND, half, half, float, float, 0, 0,
    CFG_NORM, true);
KERNEL_MATMUL_TESTCASE(TEST_KERNEL_MATMUL, g_tilingParams, VECCALC, VECCALC, GM, GM, ND, ND, ND, ND, half, half, float,
    float, 1, 1, CFG_NORM, false);
