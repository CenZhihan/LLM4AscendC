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
#include "copy_cube_in/base_tiling_struct.h"

using namespace std;
using namespace AscendC;

static constexpr int K_ALIGN_64 = 64;

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, const auto& MM_CFG>
__aicore__ inline void main_kernel_matmul(GM_ADDR aGM, GM_ADDR amxGM, GM_ADDR bGM, GM_ADDR bmxGM, GM_ADDR cGM, GM_ADDR biasGM,
    TilingParamsMx &tilingParam, int32_t isTransposeAIn, int32_t isTransposeASIn, int32_t isTransposeBIn, int32_t isTransposeBSIn)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    typedef fp8_e8m0_t MX_T;

    SetAtomicNone();
    TPipe pipe;
    TCubeTiling tiling;
    tilingParam.GetTiling(tiling);

    bool isTransposeA = isTransposeAIn > 0 ? true : false;
    bool isTransposeAS = isTransposeASIn > 0 ? true : false;
    bool isTransposeB = isTransposeBIn > 0 ? true : false;
    bool isTransposeBS = isTransposeBSIn > 0 ? true : false;
    if (block_idx >= tiling.usedCoreNum) {
        return;
    }

    GlobalTensor<A_T> aGlobal;
    GlobalTensor<MX_T> amxGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<MX_T> bmxGlobal;
    GlobalTensor<C_T> cGlobal;
    GlobalTensor<BiasT> biasGlobal;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(aGM), tiling.M * tiling.Ka);
    amxGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ MX_T*>(amxGM), tiling.M * ((tiling.Ka + 63) / 64 * 2));
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(bGM), tiling.Kb * tiling.N);
    bmxGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ MX_T*>(bmxGM), ((tiling.Kb + 63) / 64 * 2) * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T*>(cGM), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(biasGM), tiling.N);

    auto gmA = aGlobal[0];
    auto gmAMX = amxGlobal[0];
    auto gmB = bGlobal[0];
    auto gmBMX = bmxGlobal[0];
    auto gmC = cGlobal[0];
    auto gmBias = biasGlobal[0];

    TQue<TPosition::VECIN, 1> leftMatrix;
    TQue<TPosition::VECIN, 1> leftMatrixScale;
    TQue<TPosition::VECIN, 1> rightMatrix;
    TQue<TPosition::VECIN, 1> rightMatrixScale;
    TQue<TPosition::VECIN, 1> biasQue;
    TQue<TPosition::VECIN, 1> resultCMatrix;

    TQue<TPosition::A1, 1, 0> qidA1;
    TQue<TPosition::A1, 1, 0> qidMxA1;
    TQue<TPosition::B1, 1, 0> qidB1;
    TQue<TPosition::B1, 1, 0> qidMxB1;
    TQue<TPosition::C1, 1, 0> qidBias;

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>,
               AscendC::Impl::Detail::MatmulWithScalePolicy> mm;

    mm.SetSubBlockIdx(0);

    uint32_t inputTypeBit = 8;
    uint32_t inputTypeC0Size = 32;
    if constexpr (IsSameType<A_T, fp4x2_e1m2_t>::value || IsSameType<A_T, fp4x2_e2m1_t>::value) {
        inputTypeBit = 4;
        inputTypeC0Size = 64;
    }

    uint32_t scaleFactorKa = 1;
    uint32_t scaleFactorKb = 1;
    if (tiling.mxTypePara) { // MDL
         uint32_t scaleFactorKa = tiling.mxTypePara & 0x0000007f;
         uint32_t scaleFactorKb = (tiling.mxTypePara & 0x00007f00) >> 8U;
    }

    int alignSingleCoreM = Ceil(tiling.singleCoreM, BLOCK_CUBE) * BLOCK_CUBE;
    int alignSingleCoreN = Ceil(tiling.singleCoreN, BLOCK_CUBE) * BLOCK_CUBE;
    int alignSingleCoreK = Ceil(tiling.singleCoreK, K_ALIGN_64) * K_ALIGN_64; // even multiples of 32
    if constexpr (A_TYPE::pos == TPosition::TSCM) {
        pipe.InitBuffer(qidA1, 1, static_cast<uint32_t>(alignSingleCoreM * tiling.singleCoreK * inputTypeBit / 8));
    }
    if constexpr (A_TYPE::scalePosition == TPosition::TSCM) {
        pipe.InitBuffer(qidMxA1, 1, scaleFactorKa * alignSingleCoreM * alignSingleCoreK * sizeof(MX_T) / 32);
    }
    if constexpr (B_TYPE::pos == TPosition::TSCM) {
        pipe.InitBuffer(qidB1, 1, static_cast<uint32_t>(tiling.singleCoreK * tiling.singleCoreN * inputTypeBit / 8));
    }
    if constexpr (B_TYPE::scalePosition == TPosition::TSCM) {
        pipe.InitBuffer(qidMxB1, 1, scaleFactorKb * alignSingleCoreK * alignSingleCoreN * sizeof(MX_T) / 32);
    }
    if (tiling.isBias) {
        if constexpr (BIAS_TYPE::pos == TPosition::TSCM) {
            pipe.InitBuffer(qidBias, 1, alignSingleCoreN * sizeof(BiasT));
        }
    }

    mm.Init(&tiling, &pipe);

    if constexpr (A_TYPE::pos == TPosition::TSCM && B_TYPE::pos == TPosition::TSCM) {
        mm.SetOrgShape(tiling.singleCoreM, tiling.singleCoreN, tiling.Ka);
    } else if constexpr (A_TYPE::pos == TPosition::TSCM) {
        mm.SetOrgShape(tiling.singleCoreM, tiling.N, tiling.Ka);
    } else if constexpr (B_TYPE::pos == TPosition::TSCM) {
        mm.SetOrgShape(tiling.M, tiling.singleCoreN, tiling.Ka);
    }

    LocalTensor<A_T> bufferLeft;
    LocalTensor<MX_T> bufferMxLeft;
    LocalTensor<B_T> bufferRight;
    LocalTensor<MX_T> bufferMxRight;
    LocalTensor<C_T> bufferC;
    LocalTensor<BiasT> bufferBias;

    if constexpr (A_TYPE::pos == TPosition::VECOUT) {
        pipe.InitBuffer(leftMatrix, 1, tiling.M * tiling.Ka * sizeof(A_T));
        bufferLeft = leftMatrix.AllocTensor<A_T>();
        DataCopy(bufferLeft, gmA, tiling.M * tiling.Ka);
        pipe_barrier(PIPE_ALL);
        mm.SetTensorA(bufferLeft, isTransposeA);
    } else if constexpr (A_TYPE::pos == TPosition::TSCM) {
        bufferLeft = qidA1.AllocTensor<A_T>();
        if (!isTransposeA) {
            int blockLen = alignSingleCoreM * inputTypeC0Size * inputTypeBit / 8 / ONE_BLK_SIZE;
            DataCopy(bufferLeft, gmA,
                { static_cast<uint16_t>(tiling.singleCoreK / inputTypeC0Size), static_cast<uint16_t>(blockLen), 0, 0 });
        } else {
            int blockLen = tiling.Ka * inputTypeC0Size * inputTypeBit / 8 / ONE_BLK_SIZE;
            DataCopy(bufferLeft, gmA,
                { static_cast<uint16_t>(tiling.M / inputTypeC0Size), static_cast<uint16_t>(blockLen), 0, 0 });
        }
        pipe_barrier(PIPE_ALL);
        mm.SetTensorA(bufferLeft, isTransposeA);
    } else {
        mm.SetTensorA(gmA, isTransposeA);
    }
    if constexpr (A_TYPE::scalePosition == TPosition::VECOUT) {
        pipe.InitBuffer(leftMatrixScale, 1, tiling.M * tiling.Ka / 32);
        bufferMxLeft = leftMatrixScale.AllocTensor<MX_T>();
        DataCopy(bufferMxLeft, gmAMX, tiling.M * tiling.Ka / 32);
        pipe_barrier(PIPE_ALL);
        mm.SetTensorScaleA(bufferMxLeft, isTransposeAS);
    } else if constexpr (A_TYPE::scalePosition == TPosition::TSCM) {
        bufferMxLeft = qidMxA1.AllocTensor<MX_T>();
        int blockLen = tiling.singleCoreK / 32 * BLOCK_CUBE * sizeof(MX_T) / ONE_BLK_SIZE;
        uint64_t nburst = Ceil(alignSingleCoreM, BLOCK_CUBE);
        uint64_t dstStride = (scaleFactorKa - 1) * alignSingleCoreK / 32 * BLOCK_CUBE / 32;
        DataCopy(bufferMxLeft, gmAMX, {nburst, static_cast<uint64_t>(blockLen), 0, dstStride});
        pipe_barrier(PIPE_ALL);
        mm.SetTensorScaleA(bufferMxLeft, isTransposeAS);
    } else if constexpr (A_TYPE::scalePosition == TPosition::GM) {
        mm.SetTensorScaleA(gmAMX, isTransposeAS);
    }

    if constexpr (B_TYPE::pos == TPosition::VECOUT) {
        pipe.InitBuffer(rightMatrix, 1, tiling.Kb * tiling.N * sizeof(B_T));
        bufferRight = rightMatrix.AllocTensor<B_T>();
        DataCopy(bufferRight, gmB, tiling.Kb * tiling.N);
        pipe_barrier(PIPE_ALL);
        mm.SetTensorB(bufferRight, isTransposeB);
    } else if constexpr (B_TYPE::pos == TPosition::TSCM) {
        bufferRight = qidB1.AllocTensor<B_T>();
        if (!isTransposeB) {
            int blockLen = tiling.singleCoreK * inputTypeC0Size * inputTypeBit / 8 / ONE_BLK_SIZE;
            DataCopy(bufferRight, gmB,
                { static_cast<uint16_t>(tiling.Ka / inputTypeC0Size), static_cast<uint16_t>(blockLen), 0, 0 });
        } else {
            int blockLen = tiling.singleCoreN * inputTypeC0Size * inputTypeBit / 8 / ONE_BLK_SIZE;
            DataCopy(bufferRight, gmB,
                { static_cast<uint16_t>(tiling.singleCoreK / inputTypeC0Size), static_cast<uint16_t>(blockLen), 0, 0 });
        }
        pipe_barrier(PIPE_ALL);
        mm.SetTensorB(bufferRight, isTransposeB);
    } else {
        mm.SetTensorB(gmB, isTransposeB);
    }
    if constexpr (B_TYPE::scalePosition == TPosition::VECOUT) {
        pipe.InitBuffer(rightMatrixScale, 1, tiling.Kb * tiling.N / 32);
        bufferMxRight = rightMatrixScale.AllocTensor<MX_T>();
        DataCopy(bufferMxRight, gmBMX, tiling.Kb * tiling.N);
        pipe_barrier(PIPE_ALL);
        mm.SetTensorScaleB(bufferMxRight, isTransposeBS);
    } else if constexpr (B_TYPE::scalePosition == TPosition::TSCM) {
        bufferMxRight = qidMxB1.AllocTensor<MX_T>();
        int blockLen = tiling.singleCoreK / 32 * BLOCK_CUBE * sizeof(MX_T) / ONE_BLK_SIZE;
        uint64_t nburst = Ceil(alignSingleCoreM, BLOCK_CUBE);
        uint64_t dstStride = (scaleFactorKb - 1) * alignSingleCoreK / 32 * BLOCK_CUBE / 32;
        DataCopy(bufferMxRight, gmBMX, {nburst, static_cast<uint64_t>(blockLen), 0, dstStride});
        pipe_barrier(PIPE_ALL);
        mm.SetTensorScaleB(bufferMxRight, isTransposeBS);
    } else if constexpr (B_TYPE::scalePosition == TPosition::GM) {
        mm.SetTensorScaleB(gmBMX, isTransposeBS);
    }

    if (tiling.isBias) {
        if constexpr (BIAS_TYPE::pos == TPosition::VECOUT) {
            pipe.InitBuffer(biasQue, 1, tiling.N * sizeof(BiasT));
            bufferBias = biasQue.AllocTensor<BiasT>();
            DataCopy(bufferBias, gmBias, tiling.N);
            pipe_barrier(PIPE_ALL);
            mm.SetBias(bufferBias);
        } else if constexpr (BIAS_TYPE::pos == TPosition::TSCM) {
            bufferBias = qidBias.template AllocTensor<BiasT>();
            DataCopy(bufferBias, gmBias, tiling.N);
            qidBias.EnQue(bufferBias);
            bufferBias = qidBias.template DeQue<BiasT>();
            mm.SetBias(bufferBias);
        } else {
            mm.SetBias(gmBias);
        }
    }

    if constexpr (C_TYPE::pos == TPosition::VECIN) {
        pipe.InitBuffer(resultCMatrix, 1, tiling.M * tiling.N * sizeof(C_T));
        bufferC = resultCMatrix.AllocTensor<C_T>();
        int64_t mIter_ = Ceil(tiling.singleCoreM, tiling.baseM);
        int64_t nIter_ = Ceil(tiling.singleCoreN, tiling.baseN);
        int64_t mmIter_ = mIter_ * nIter_;
        mm.template Iterate<false>();
        for (int64_t i = 0; i < mmIter_; i++) {
            mm.template GetTensorC<false>(bufferC, false, false);
        }
        mm.End();
        pipe_barrier(PIPE_ALL);
        resultCMatrix.EnQue(bufferC);
        bufferC = resultCMatrix.DeQue<C_T>();
        DataCopy(gmC, bufferC, tiling.M * tiling.N);
        pipe_barrier(PIPE_ALL);
    } else {
        mm.IterateAll(gmC);
        mm.End();
    }

    pipe_barrier(PIPE_ALL);
    if constexpr (A_TYPE::pos == TPosition::VECOUT) {
        leftMatrix.FreeTensor(bufferLeft);
    }
    if constexpr (A_TYPE::scalePosition == TPosition::VECOUT) {
        leftMatrixScale.FreeTensor(bufferMxLeft);
    }

    if constexpr (A_TYPE::pos == TPosition::TSCM) {
        qidA1.FreeTensor(bufferLeft);
    }
    if constexpr (A_TYPE::scalePosition == TPosition::TSCM) {
        qidMxA1.FreeTensor(bufferMxLeft);
    }

    if constexpr (B_TYPE::pos == TPosition::VECOUT) {
        rightMatrix.FreeTensor(bufferRight);
    }
    if constexpr (B_TYPE::scalePosition == TPosition::VECOUT) {
        rightMatrixScale.FreeTensor(bufferMxRight);
    }

    if constexpr (B_TYPE::pos == TPosition::TSCM) {
        qidB1.FreeTensor(bufferRight);
    }
    if constexpr (B_TYPE::scalePosition == TPosition::TSCM) {
        qidMxB1.FreeTensor(bufferMxRight);
    }

    if (tiling.isBias) {
        if constexpr (BIAS_TYPE::pos == TPosition::VECOUT) {
            biasQue.FreeTensor(bufferBias);
        }
        if constexpr (BIAS_TYPE::pos == TPosition::TSCM) {
            qidBias.FreeTensor(bufferBias);
        }
    }

    if constexpr (C_TYPE::pos == TPosition::VECIN) {
        resultCMatrix.FreeTensor(bufferC);
    }
    SetAtomicNone();
}

class TEST_KERNEL_MATMUL : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

__aicore__ inline constexpr MatmulConfig GetStaticCfg(uint32_t singleM, uint32_t singleN, uint32_t singleK,
    uint32_t baseM, uint32_t baseN, uint32_t baseK, bool isBias, uint32_t configMode)
{
    MatmulShapeParams shapeParams = {singleM, singleN, singleK, baseM, baseN, baseK};
    auto mmCfg = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(shapeParams);
    if (configMode == 0) {
        mmCfg = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shapeParams);
    }
    mmCfg.enableSetBias = isBias;
    return mmCfg;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ constexpr MatmulApiStaticTiling GetCustomMatmulApiTiling(const MatmulConfig& mmCFG, int32_t mxTypePara)
{
    MatmulApiStaticTiling tiling;
    tiling = GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG);
    tiling.mxTypePara = mxTypePara;
    return tiling;
}


#define KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tilingParams, A_Pos, AS_Pos, B_Pos, BS_Pos, C_Pos, BIAS_Pos,\
    A_Format, AS_Format, B_Format, BS_Format, C_Format, BIAS_Format, A_DType, B_DType, C_DType, BIAS_DType,            \
    isTransposeA, isTransposeAS, isTransposeB, isTransposeBS, CFG_Mode)                             \
    namespace Kernel_Matmul_Case_##tilingParams##_##A_Pos##_##AS_Pos##_##B_Pos##_##BS_Pos##_##C_Pos##_##BIAS_Pos       \
    ##_##A_Format##_##AS_Format##_##B_Format##_##BS_Format##_##C_Format##_##BIAS_Format                                \
    ##_##A_DType##_##B_DType##_##C_DType##_##BIAS_DType                                                                \
    ##_##isTransposeA##_##isTransposeAS##_##isTransposeB##_##isTransposeBS##_##CFG_Mode{         \
    typedef MatmulTypeWithScale<TPosition::A_Pos, TPosition::AS_Pos, CubeFormat::A_Format, A_DType, isTransposeA,      \
                                TPosition::GM, CubeFormat::AS_Format, isTransposeAS, TPosition::GM> aType;             \
    typedef MatmulTypeWithScale<TPosition::B_Pos, TPosition::BS_Pos, CubeFormat::B_Format, B_DType, isTransposeB,      \
                                TPosition::GM, CubeFormat::BS_Format, isTransposeBS, TPosition::GM> bType;             \
    typedef MatmulType<TPosition::C_Pos, CubeFormat::C_Format, C_DType> cType;                                         \
    typedef MatmulType<TPosition::BIAS_Pos, CubeFormat::BIAS_Format, BIAS_DType> biasType;                             \
    constexpr static MatmulConfig mmCFG = CFG_Mode;                                                                    \
    TEST_F(TEST_KERNEL_MATMUL, Kernel_Matmul_Case_##tilingParams##_##A_Pos##_##AS_Pos##_##B_Pos##_##BS_Pos             \
    ##_##C_Pos##_##BIAS_Pos##_##A_Format##_##AS_Format##_##B_Format##_##BS_Format##_##C_Format                         \
    ##_##BIAS_Format##_##A_DType##_##B_DType##_##C_DType##_##BIAS_DType##_##isTransposeA##_##isTransposeAS             \
    ##_##isTransposeB##_##isTransposeBS##_##CFG_Mode)                                            \
    {                                                                                                                  \
        const int32_t left_data_size = tilingParams.M_ * tilingParams.K_;                                              \
        const int32_t left_scale_data_size = tilingParams.M_ * ((tilingParams.K_ + 63) / 64 * 2);                          \
        const int32_t right_data_size = tilingParams.K_ * tilingParams.N_;                                             \
        const int32_t right_scale_data_size = ((tilingParams.K_ + 63) / 64 * 2) * tilingParams.N_;                         \
        const int32_t bias_data_size = tilingParams.N_;                                                                \
        const int32_t output_data_size = tilingParams.M_ * tilingParams.N_;                                            \
        uint8_t left_global[left_data_size * sizeof(A_DType)] = {0};                                                   \
        uint8_t left_scale_global[left_scale_data_size * sizeof(fp8_e8m0_t)] = {0};                                 \
        uint8_t right_global[right_data_size * sizeof(B_DType)] = {0};                                                 \
        uint8_t right_scale_global[right_scale_data_size * sizeof(fp8_e8m0_t)] = {0};                               \
        uint8_t bias_global[bias_data_size * sizeof(BIAS_DType)] = {0};                                                \
        uint8_t output_global[output_data_size * sizeof(C_DType)] = {0};                                               \
        main_kernel_matmul<aType, bType, cType, biasType, mmCFG>(left_global, left_scale_global, right_global, right_scale_global,    \
            output_global, bias_global, tilingParams, isTransposeA, isTransposeAS, isTransposeB, isTransposeBS);   \
        for (int32_t i = 0; i < output_data_size * sizeof(C_DType); i++) {                                             \
            EXPECT_EQ(output_global[i], 0x00);                                                                         \
        }                                                                                                              \
    }                                                                                                                  \
}

#define KERNEL_MATMUL_TESTCASE_CONSTANT(TEST_KERNEL_MATMUL, tilingParams, A_Pos, AS_Pos, B_Pos, BS_Pos, C_Pos, BIAS_Pos,\
    A_Format, AS_Format, B_Format, BS_Format, C_Format, BIAS_Format, A_DType, B_DType, C_DType, BIAS_DType,            \
    isTransposeA, isTransposeAS, isTransposeB, isTransposeBS, CFG_Mode,                                                \
    singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, isBias, mxTypePara, configMode)                        \
    namespace Kernel_Matmul_Case_##tilingParams##_##A_Pos##_##AS_Pos##_##B_Pos##_##BS_Pos##_##C_Pos##_##BIAS_Pos       \
    ##_##A_Format##_##AS_Format##_##B_Format##_##BS_Format##_##C_Format##_##BIAS_Format                                \
    ##_##A_DType##_##B_DType##_##C_DType##_##BIAS_DType                                                                \
    ##_##isTransposeA##_##isTransposeAS##_##isTransposeB##_##isTransposeBS##_##CFG_Mode##_##enTiling{ \
    typedef MatmulTypeWithScale<TPosition::A_Pos, TPosition::AS_Pos, CubeFormat::A_Format, A_DType, isTransposeA,      \
                                TPosition::GM, CubeFormat::AS_Format, isTransposeAS, TPosition::GM> aType;             \
    typedef MatmulTypeWithScale<TPosition::B_Pos, TPosition::BS_Pos, CubeFormat::B_Format, B_DType, isTransposeB,      \
                                TPosition::GM, CubeFormat::BS_Format, isTransposeBS, TPosition::GM> bType;             \
    typedef MatmulType<TPosition::C_Pos, CubeFormat::C_Format, C_DType> cType;                                         \
    typedef MatmulType<TPosition::BIAS_Pos, CubeFormat::BIAS_Format, BIAS_DType> biasType;                             \
    constexpr static MatmulConfig staticCfg = GetStaticCfg(singleCoreM, singleCoreN,                                   \
        singleCoreK, baseM, baseN, baseK, isBias, configMode);                                                         \
    constexpr static MatmulApiStaticTiling mmTiling =                                                                  \
        GetCustomMatmulApiTiling<aType, bType, cType, biasType>(staticCfg, mxTypePara);                                \
    TEST_F(TEST_KERNEL_MATMUL, Kernel_Matmul_Case_##tilingParams##_##A_Pos##_##AS_Pos##_##B_Pos##_##BS_Pos             \
    ##_##C_Pos##_##BIAS_Pos##_##A_Format##_##AS_Format##_##B_Format##_##BS_Format##_##C_Format                         \
    ##_##BIAS_Format##_##A_DType##_##B_DType##_##C_DType##_##BIAS_DType##_##isTransposeA##_##isTransposeAS             \
    ##_##isTransposeB##_##isTransposeBS##_##CFG_Mode##_##enTiling)                               \
    {                                                                                                                  \
        const int32_t left_data_size = tilingParams.M_ * tilingParams.K_;                                              \
        const int32_t left_scale_data_size = tilingParams.M_ * ((tilingParams.K_ + 31) / 32);                          \
        const int32_t right_data_size = tilingParams.K_ * tilingParams.N_;                                             \
        const int32_t right_scale_data_size = ((tilingParams.K_ + 31) / 32) * tilingParams.N_;                         \
        const int32_t bias_data_size = tilingParams.N_;                                                                \
        const int32_t output_data_size = tilingParams.M_ * tilingParams.N_;                                            \
        uint8_t left_global[left_data_size * sizeof(A_DType)] = {0};                                                   \
        uint8_t left_scale_global[left_scale_data_size * sizeof(fp8_e8m0_t)] = {0};                                 \
        uint8_t right_global[right_data_size * sizeof(B_DType)] = {0};                                                 \
        uint8_t right_scale_global[right_scale_data_size * sizeof(fp8_e8m0_t)] = {0};                               \
        uint8_t bias_global[bias_data_size * sizeof(BIAS_DType)] = {0};                                                \
        uint8_t output_global[output_data_size * sizeof(C_DType)] = {0};                                               \
        main_kernel_matmul<aType, bType, cType, biasType, mmTiling>(left_global, left_scale_global, right_global, right_scale_global, \
        output_global, bias_global, tilingParams, isTransposeA, isTransposeAS, isTransposeB, isTransposeBS);   \
        for (int32_t i = 0; i < output_data_size * sizeof(C_DType); i++) {                                             \
            EXPECT_EQ(output_global[i], 0x00);                                                                         \
        }                                                                                                              \
    }                                                                                                                  \
}

#define KERNEL_MATMUL_TESTCASE_MXTYPE(TEST_KERNEL_MATMUL, tilingParams, A_Pos, AS_Pos, B_Pos, BS_Pos,                  \
    A_SrcPos, AS_SrcPos, B_SrcPos, BS_SrcPos, C_Pos, BIAS_Pos,                                                         \
    A_Format, AS_Format, B_Format, BS_Format, C_Format, BIAS_Format, A_DType, B_DType, C_DType, BIAS_DType,            \
    isTransposeA, isTransposeAS, isTransposeB, isTransposeBS, CFG_Mode)                                                \
namespace Kernel_Matmul_Case_##tilingParams##_##A_Pos##_##AS_Pos##_##B_Pos##_##BS_Pos                                  \
    ##_##A_SrcPos##_##AS_SrcPos##_##B_SrcPos##_##BS_SrcPos##_##C_Pos##_##BIAS_Pos                                      \
    ##_##A_Format##_##AS_Format##_##B_Format##_##BS_Format##_##C_Format##_##BIAS_Format                                \
    ##_##A_DType##_##B_DType##_##C_DType##_##BIAS_DType                                                                \
    ##_##isTransposeA##_##isTransposeAS##_##isTransposeB##_##isTransposeBS##_##CFG_Mode{                               \
    typedef MatmulTypeWithScale<TPosition::A_Pos, TPosition::AS_Pos, CubeFormat::A_Format, A_DType, isTransposeA,      \
                                TPosition::A_SrcPos, CubeFormat::AS_Format, isTransposeAS, TPosition::AS_SrcPos> aType;\
    typedef MatmulTypeWithScale<TPosition::B_Pos, TPosition::BS_Pos, CubeFormat::B_Format, B_DType, isTransposeB,      \
                                TPosition::B_SrcPos, CubeFormat::BS_Format, isTransposeBS, TPosition::BS_SrcPos> bType;\
    typedef MatmulType<TPosition::C_Pos, CubeFormat::C_Format, C_DType> cType;                                         \
    typedef MatmulType<TPosition::BIAS_Pos, CubeFormat::BIAS_Format, BIAS_DType> biasType;                             \
    constexpr static MatmulConfig mmCFG = CFG_Mode;                                                                    \
    TEST_F(TEST_KERNEL_MATMUL, Kernel_Matmul_Case_##tilingParams##_##A_Pos##_##AS_Pos##_##B_Pos##_##BS_Pos             \
    ##_##C_Pos##_##BIAS_Pos##_##A_Format##_##AS_Format##_##B_Format##_##BS_Format##_##C_Format                         \
    ##_##BIAS_Format##_##A_DType##_##B_DType##_##C_DType##_##BIAS_DType##_##isTransposeA##_##isTransposeAS             \
    ##_##isTransposeB##_##isTransposeBS##_##CFG_Mode)                                                                  \
    {                                                                                                                  \
        MatmulImpl<aType, bType, cType, biasType, mmCFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>,                \
            AscendC::Impl::Detail::MatmulWithScalePolicy> mm;                                                          \
        TPipe pipe;                                                                                                    \
        TCubeTiling tiling;                                                                                            \
        tilingParams.GetTiling(tiling);                                                                                \
        mm.Init(&tiling, &pipe);                                                                                       \
    }                                                                                                                  \
}

// coreNum, M, N, K, singleCoreM, singleCoreN, singleCoreK, baseM, baseN, baseK, depthA1, depthB1, stepM, stepN, stepKa, stepKb, isBias, iterateOrder, mxTypePara
TilingParamsMx tiling_params_mx_case1 = {1, 128, 128, 128, 128, 128, 128, 128, 128, 128, 1, 1, 1, 1, 1, 1, 1, 0, 0};
TilingParamsMx tiling_params_mx_case2 = {1, 465, 251, 192, 465, 251, 192, 256, 256, 192, 2, 1, 2, 1, 1, 1, 0, 1, 16843009};
TilingParamsMx tiling_params_mx_case3 = {1, 32, 128, 128, 32, 128, 128, 32, 128, 128, 1, 1, 1, 1, 1, 1, 1, 0, 0};
TilingParamsMx tiling_params_mx_case4 = {8, 60, 510, 256, 64, 64, 256, 64, 64, 256, 1, 1, 1, 1, 1, 1, 0, 1, 0};
TilingParamsMx tiling_params_mx_case5 = {8, 64, 510, 250, 64, 64, 256, 64, 64, 256, 1, 1, 1, 1, 1, 1, 1, 0, 16843009};
TilingParamsMx tiling_params_mx_case6 = {1, 339, 288, 768, 339, 288, 768, 176, 256, 64, 1, 1, 1, 1, 1, 1, 0, 0, 16845836};
TilingParamsMx tiling_params_mx_case7 = {1, 448, 256, 64, 448, 256, 64, 128, 128, 128, 1, 1, 1, 1, 1, 1, 0, 0, 16843009};
TilingParamsMx tiling_params_mx_case8 = {1, 150, 64, 576, 150, 64, 576, 80, 64, 384, 6, 3, 2, 1, 3, 3, 1, 0, 0};
TilingParamsMx tiling_params_mx_case9 = {1, 304, 1, 320, 304, 1, 320, 256, 32, 256, 4, 2, 2, 1, 2, 2, 1, 0, 0};
TilingParamsMx tiling_params_mx_case10 = {1, 1, 684, 314, 1, 684, 314, 32, 256, 128, 3, 9, 1, 3, 3, 3, 1, 0, 16843009};
TilingParamsMx tiling_params_mx_case11 = {1, 428, 161, 132, 428, 161, 132, 144, 192, 128, 6, 2, 3, 1, 2, 2, 1, 0, 16843009};
TilingParamsMx tiling_params_mx_case12 = {1, 187, 32, 31, 187, 32, 31, 48, 32, 64, 4, 1, 4, 1, 1, 1, 0, 0, 16843009};
TilingParamsMx tiling_params_mx_case13 = {1, 208, 64, 96, 208, 64, 96, 96, 208, 64, 2, 2, 1, 1, 2, 2, 1, 0, 16843009};
TilingParamsMx tiling_params_mx_case14 = {1, 16, 96, 32, 16, 96, 32, 16, 64, 64, 1, 2, 1, 2, 1, 1, 0, 0, 16843009};
TilingParamsMx tiling_params_mx_case15 = {1, 107, 322, 800, 107, 322, 800, 112, 256, 128, 7, 2, 1, 1, 7, 1, 1, 0, 16844545};
TilingParamsMx tiling_params_mx_case16 = {1, 224, 449, 404, 224, 449, 404, 128, 256, 256, 4, 4, 2, 2, 2, 2, 0, 0, 16843009};
TilingParamsMx tiling_params_mx_case17 = {1, 408, 28, 272, 408, 28, 272, 208, 32, 256, 4, 2, 2, 1, 2, 2, 0, 0, 16843009};
TilingParamsMx tiling_params_mx_case18 = {1, 122, 205, 8, 122, 205, 8, 128, 112, 64, 1, 2, 1, 2, 1, 1, 0, 0, 16843009};
TilingParamsMx tiling_params_mx_case19 = {1, 800, 176, 128, 800, 176, 128, 160, 192, 128, 2, 1, 2, 1, 1, 1, 0, 0, 16843009};
TilingParamsMx tiling_params_mx_case20 = {1, 624, 800, 128, 624, 800, 128, 128, 256, 128, 5, 4, 5, 4, 1, 1, 0, 0, 16843009};

// dynamics
// TEST_KERNEL_MATMUL, tilingParams, A_Pos, AS_Pos, B_Pos, BS_Pos, C_Pos, BIAS_Pos, A_Format, AS_Format, B_Format, BS_Format, C_Format, BIAS_Format,
// A_DType, B_DType, C_DType, BIAS_DType, isTransposeA, isTransposeAS, isTransposeB, isTransposeBS, CFG_Mode
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case1, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, 0, 0, 0, 0, CFG_NORM);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case3, TSCM, GM, TSCM, GM, GM, TSCM, NZ, ND, NZ, ND, ND, ND,
    fp8_e5m2_t, fp8_e4m3fn_t, float, half, 1, 1, 0, 0, CFG_NORM);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case5, GM, GM, TSCM, TSCM, VECIN, GM, ND, ND, NZ, NZ, NZ, ND,
    fp4x2_e1m2_t, fp4x2_e2m1_t, float, bfloat16_t, 1, 0, 1, 1, CFG_MDL);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case7, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, 0, 0, 0, 0, CFG_NORM);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case8, TSCM, TSCM, TSCM, TSCM, GM, TSCM, NZ, NZ, NZ, NZ, NZ, NZ,
    fp8_e5m2_t, fp8_e5m2_t, float, float, 1, 0, 1, 0, CFG_NORM);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case9, GM, TSCM, TSCM, GM, GM, GM, ND, NZ, NZ, ND, ND, ND,
    fp4x2_e1m2_t, fp4x2_e1m2_t, float, half, 0, 0, 1, 0, CFG_NORM);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case10, TSCM, GM, GM, TSCM, GM, GM, NZ, ND, ND, NZ, ND, ND,
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, bfloat16_t, 1, 0, 0, 0, CFG_MDL);
// for Ceil(K,32) is odd
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case11, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, bfloat16_t, 1, 0, 0, 0, CFG_MDL);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case12, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, bfloat16_t, 1, 0, 0, 0, CFG_NORM);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case18, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp8_e5m2_t, fp8_e4m3fn_t, float, float, 1, 0, 1, 0, CFG_MDL);
// FP8 ND input
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case15, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp8_e4m3fn_t, fp8_e5m2_t, float, float, 0, 1, 0, 1, CFG_MDL);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case15, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp8_e4m3fn_t, fp8_e5m2_t, float, float, 0, 1, 0, 1, CFG_NORM);
// FP8 NZ input
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case13, GM, GM, GM, GM, GM, GM, NZ, NZ, NZ, NZ, ND, ND,
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, bfloat16_t, 0, 0, 1, 0, CFG_MDL);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case14, GM, GM, GM, GM, GM, GM, NZ, NZ, NZ, NZ, ND, ND,
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, bfloat16_t, 1, 0, 0, 0, CFG_NORM);
// FP4 ND input
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case16, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp4x2_e1m2_t, fp4x2_e1m2_t, bfloat16_t, float, 0, 0, 1, 1, CFG_MDL);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case17, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp4x2_e1m2_t, fp4x2_e2m1_t, float, float, 1, 1, 0, 0, CFG_MDL);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case16, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp4x2_e1m2_t, fp4x2_e1m2_t, bfloat16_t, float, 0, 0, 1, 1, CFG_NORM);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case17, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp4x2_e1m2_t, fp4x2_e2m1_t, float, float, 1, 1, 0, 0, CFG_NORM);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case19, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp4x2_e1m2_t, fp4x2_e2m1_t, half, float, 1, 0, 1, 1, CFG_MDL);

KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case1, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, NZ, ND,
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, 0, 0, 0, 0, CFG_NORM);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case3, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, NZ, ND,
    fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 0, 0, 0, 0, CFG_NORM);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case3, GM, GM, GM, GM, GM, GM, NZ, NZ, NZ, ND, NZ, ND,
    fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 0, 0, 0, 0, CFG_NORM);

// constant
// TEST_KERNEL_MATMUL, tilingParams, A_Pos, AS_Pos, B_Pos, BS_Pos, C_Pos, BIAS_Pos, A_Format, AS_Format, B_Format, BS_Format, C_Format, BIAS_Format,
// A_DType, B_DType, C_DType, BIAS_DType, isTransposeA, isTransposeAS, isTransposeB, isTransposeBS, CFG_Mode,
// singleM, singleN, singleK, baseM, baseN, baseK, isBias, mxTypePara, configMode
KERNEL_MATMUL_TESTCASE_CONSTANT(TEST_KERNEL_MATMUL, tiling_params_mx_case2, GM, GM, GM, GM, GM, GM, NZ, NZ, NZ, NZ, NZ, NZ,
    fp4x2_e1m2_t, fp4x2_e1m2_t, float, half, 1, 0, 1, 1, CONFIG_MDL,
    465, 251, 192, 256, 256, 192, 0, 16843009, 1);
KERNEL_MATMUL_TESTCASE_CONSTANT(TEST_KERNEL_MATMUL, tiling_params_mx_case4, GM, TSCM, GM, TSCM, VECIN, GM, ND, NZ, ND, NZ, ND, ND,
    fp4x2_e2m1_t, fp4x2_e1m2_t, float, float, 0, 0, 1, 1, CONFIG_NORM,
    64, 64, 256, 64, 64, 256, 0, 0, 0);
KERNEL_MATMUL_TESTCASE_CONSTANT(TEST_KERNEL_MATMUL, tiling_params_mx_case6, TSCM, TSCM, GM, GM, GM, GM, NZ, NZ, ND, ND, NZ, ND,
    fp8_e4m3fn_t, fp8_e5m2_t, float, float, 1, 1, 1, 0, CONFIG_MDL,
    339, 288, 768, 176, 256, 64, 0, 16845836, 1);

// mxtype
// TEST_KERNEL_MATMUL, tilingParams, A_Pos, AS_Pos, B_Pos, BS_Pos, A_SrcPos, AS_SrcPos, B_SrcPos, BS_SrcPos,
// C_Pos, BIAS_Pos, A_Format, AS_Format, B_Format, BS_Format, C_Format, BIAS_Format,
// A_DType, B_DType, C_DType, BIAS_DType, isTransposeA, isTransposeAS, isTransposeB, isTransposeBS, CFG_Mode
KERNEL_MATMUL_TESTCASE_MXTYPE(TEST_KERNEL_MATMUL, tiling_params_mx_case14, GM, GM, GM, GM, GM, GM, GM, GM,
    GM, GM, ND, ND, ND, ND, ND, ND, fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, 0, 0, 0, 0, CFG_NORM);
KERNEL_MATMUL_TESTCASE_MXTYPE(TEST_KERNEL_MATMUL, tiling_params_mx_case14, VECOUT, VECOUT, VECOUT, VECOUT, GM, GM, GM, GM,
    GM, GM, ND, ND, ND, ND, ND, ND, fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, 0, 0, 0, 0, CFG_MDL);
KERNEL_MATMUL_TESTCASE_MXTYPE(TEST_KERNEL_MATMUL, tiling_params_mx_case14, TSCM, TSCM, TSCM, TSCM, GM, GM, GM, GM,
    GM, GM, ND, ND, ND, ND, ND, ND, fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, 0, 0, 0, 0, CFG_NORM);
KERNEL_MATMUL_TESTCASE_MXTYPE(TEST_KERNEL_MATMUL, tiling_params_mx_case14, TSCM, TSCM, TSCM, TSCM, VECOUT, VECOUT, VECOUT, VECOUT,
    GM, GM, ND, ND, ND, ND, ND, ND, fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, 0, 0, 0, 0, CFG_MDL);


__aicore__ constexpr MatmulConfig GetOrderNConfig()
{
    MatmulConfig cfg = GetMDLConfig();
    cfg.iterateOrder = IterateOrder::ORDER_N;
    cfg.scheduleType = ScheduleType::OUTER_PRODUCT;
    return cfg;
}

__aicore__ constexpr MatmulConfig GetOrderMConfig()
{
    MatmulConfig cfg = GetMDLConfig();
    cfg.iterateOrder = IterateOrder::ORDER_M;
    cfg.scheduleType = ScheduleType::OUTER_PRODUCT;
    return cfg;
}

constexpr static MatmulConfig orderNConfig = GetOrderNConfig();
constexpr static MatmulConfig orderMConfig = GetOrderMConfig();

KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case20, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, 1, 0, 0, 0, orderNConfig);

KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case20, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, 1, 0, 0, 0, orderMConfig);

KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case20, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 1, 0, 0, 0, orderNConfig);

KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case20, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 1, 0, 0, 0, orderMConfig);

TilingParamsMx tiling_params_mx_case21 = {1, 802, 2426, 44, 802, 2462, 44, 128, 256, 64, 7, 10, 7, 5, 1, 1, 0, 1, 16843009};
TilingParamsMx tiling_params_mx_case22 = {1, 32, 1904, 256, 32, 1904, 256, 32, 256, 128, 4, 8, 1, 2, 2, 2, 0, 0, 16843009};
constexpr static MatmulConfig preloadM = GetMDLConfig(false, false, 1);
constexpr static MatmulConfig preloadN = GetMDLConfig(false, false, 2);

KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case21, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, 1, 0, 0, 0, preloadM);

KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case21, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 0, 1, 1, 1, preloadM);

KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case22, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, 1, 0, 0, 0, preloadN);

KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case22, GM, GM, GM, GM, GM, GM, ND, ND, ND, ND, ND, ND,
    fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 1, 0, 0, 0, preloadN);

// Gemv
TilingParamsMx tiling_params_mx_case23 = {1, 1, 48, 6272, 1, 48, 6272, 16, 32, 1024, 1, 1, 1, 1, 1, 1, 0, 0, 16843009};

KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case23, GM, GM, TSCM, GM, GM, GM, VECTOR, VECTOR, NZ, ND, NZ, ND,
    fp8_e4m3fn_t, fp8_e5m2_t, float, float, 0, 0, 0, 0, CFG_MDL);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case23, GM, GM, GM, TSCM, GM, GM, VECTOR, VECTOR, NZ, ND, NZ, ND,
    fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 0, 0, 0, 0, CFG_NORM);

// scaleKa = 2, scaleKb = 2, scaleM = 3, scaleN = 2
TilingParamsMx tiling_params_mx_case24 = {1, 608, 320, 192, 608, 320, 192, 256, 256, 128, 2, 2, 1, 1, 1, 1, 0, 0, 33751554};
// scaleKa = 1, scaleKb = 1, scaleM = 3, scaleN = 2
TilingParamsMx tiling_params_mx_case25 = {1, 576, 448, 192, 576, 448, 192, 256, 256, 192, 1, 1, 1, 1, 1, 1, 0, 0, 33751297};
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case24, TSCM, GM, TSCM, GM, GM, GM, NZ, ND, NZ, NZ, ND, ND,
    fp8_e4m3fn_t, fp8_e5m2_t, float, float, 1, 0, 0, 1, CFG_MDL);
KERNEL_MATMUL_TESTCASE_DYNAMICS(TEST_KERNEL_MATMUL, tiling_params_mx_case25, TSCM, GM, TSCM, GM, GM, GM, NZ, NZ, NZ, ND, ND, ND,
    fp4x2_e1m2_t, fp4x2_e1m2_t, bfloat16_t, float, 1, 0, 0, 0, CFG_MDL);