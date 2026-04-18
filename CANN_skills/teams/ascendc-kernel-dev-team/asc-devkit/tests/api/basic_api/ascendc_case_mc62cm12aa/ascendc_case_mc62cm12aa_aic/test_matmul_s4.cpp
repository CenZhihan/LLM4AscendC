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
using namespace AscendC;

enum class CubeFormat {
    ND = 0,
    NZ,
    ZN,
    ZZ,
    NN,
    ND_ALIGN,
    SCALAR,
    VECTOR,
};

template <TPosition POSITION, CubeFormat FORMAT, typename TYPE> struct InputInfo {
    constexpr static TPosition pos = POSITION;
    constexpr static CubeFormat format = FORMAT;
    using T = TYPE;
};
constexpr int32_t NZ_MASK_VAlUE = 2;
int32_t constexpr GetNdNzMask(CubeFormat dstFormat, CubeFormat srcFormat)
{
    if ((srcFormat == CubeFormat::ND) && (dstFormat == CubeFormat::NZ)) {
        return 1;
    } else if ((srcFormat == CubeFormat::NZ) && (dstFormat == CubeFormat::ND)) {
        return NZ_MASK_VAlUE;
    }
    return 0;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class L0C_TYPE, class BIAS_TYPE> class E2eCase {
    using SrcT = typename A_TYPE::T;
    using Src1T = typename B_TYPE::T;
    using DstT = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using L0cT = typename L0C_TYPE::T;

public:
    __aicore__ inline E2eCase() {}
    __aicore__ inline void Init(TPipe *tpipe, int32_t m, int32_t n, int32_t k, int32_t enableBias, int32_t transA, int32_t path)
    {
        pipe = tpipe;
        mLength = m;
        nLength = n;
        kLength = k;
        enBias = enableBias;
        transA_ = transA;
        path_ = path;
        pipe->InitBuffer(qidA1_, 1, m * k * sizeof(SrcT));
        pipe->InitBuffer(qidB1_, 1, n * k * sizeof(Src1T));

        pipe->InitBuffer(qidA2_, 1, m * k * sizeof(SrcT));
        pipe->InitBuffer(qidB2_, 1, n * k * sizeof(Src1T));
        if (enBias == 1) {
            pipe->InitBuffer(qidBias_, 1, n * sizeof(BiasT));
        }
        pipe->InitBuffer(qidCO1_, 1, m * n * sizeof(L0cT));
    }
    __aicore__ inline void SetTensorA(const GlobalTensor<SrcT> &gm)
    {
        aGlobal_ = gm;
    }
    __aicore__ inline void SetTensorB(const GlobalTensor<Src1T> &gm)
    {
        bGlobal_ = gm;
    }
    __aicore__ inline void SetBias(const GlobalTensor<BiasT> &biasGlobal)
    {
        biasGlobal_ = biasGlobal;
    }
    __aicore__ inline uint16_t CeilDiv(uint16_t num1, uint16_t num2)
    {
        ASSERT(num2 > 0);
        return (num1 + num2 - 1) / num2;
    }

    __aicore__ inline void CopyGmToA1()
    {
        LocalTensor<SrcT> leftMatrix = qidA1_.template AllocTensor<SrcT>();
        DataCopy(leftMatrix, aGlobal_, mLength * kLength);
        qidA1_.EnQue(leftMatrix);
    }

    __aicore__ inline void CopyGmToB1()
    {
        LocalTensor<Src1T> rightMatrix = qidB1_.template AllocTensor<Src1T>();
        DataCopy(rightMatrix, bGlobal_, kLength * nLength);
        qidB1_.EnQue(rightMatrix);
    }

    __aicore__ inline void Compute()
    {
        MmadParams mmadParams;
        mmadParams.m = mLength;
        mmadParams.n = nLength;
        mmadParams.k = kLength;
        mmadParams.isBias = enBias;
        auto co1Local = qidCO1_.AllocTensor<L0cT>();
        auto a2 = qidA2_.DeQue<SrcT>();
        auto b2 = qidB2_.DeQue<Src1T>();
        Mmad(co1Local, a2, b2, mmadParams);
        qidA2_.FreeTensor(a2);
        qidB2_.FreeTensor(b2);
        qidCO1_.EnQue(co1Local);
    }

    __aicore__ inline void Load2DA1ToL0A()
    {
        auto leftMatrix = qidA1_.template DeQue<SrcT>();
        LocalTensor<SrcT> a2 = qidA2_.AllocTensor<SrcT>();
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = DivCeil(mLength, 16);
        loadDataParams.kStep = DivCeil(kLength * sizeof(SrcT), 32);

        if (IsSameType<SrcT, int4b_t>::value) {
            loadDataParams.kStep = DivCeil(loadDataParams.kStep, 2);
        }

        loadDataParams.srcStride = DivCeil(mLength, 16);
        loadDataParams.dstStride = DivCeil(mLength, 16);

        loadDataParams.sid = 0;
        loadDataParams.ifTranspose = transA_;
        LoadData(a2, leftMatrix, loadDataParams);

        qidA2_.EnQue(a2);
        qidA1_.FreeTensor(leftMatrix);
    }

    __aicore__ inline void Load2DA1ToL0B()
    {
        auto rightMatrix = qidB1_.template DeQue<Src1T>();
        LocalTensor<Src1T> b2 = qidB2_.AllocTensor<Src1T>();
        uint16_t kAlign = CeilDiv(kLength, 16);
        uint16_t nAlign = CeilDiv(nLength * sizeof(Src1T), 32);

        if (IsSameType<Src1T, int4b_t>::value) {
            nAlign = DivCeil(nAlign, 2);
        }

        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = kAlign;
        loadDataParams.kStep = nAlign;
        loadDataParams.srcStride = kAlign;
        loadDataParams.dstStride = nAlign;

        if (IsSameType<Src1T, float>::value) {
            loadDataParams.dstStride = nAlign / 2;
        }

        if (IsSameType<Src1T, int8_t>::value) {
            loadDataParams.dstStride = nAlign * 2;
        }
        loadDataParams.sid = 0;
        loadDataParams.ifTranspose = true;
        LoadData(b2, rightMatrix, loadDataParams);

        qidB2_.EnQue(b2);
        qidB1_.FreeTensor(rightMatrix);
    }

    __aicore__ inline void Load2DA1ToL0BTranspose()
    {
        auto rightMatrix = qidB1_.template DeQue<Src1T>();
        LocalTensor<Src1T> b2 = qidB2_.AllocTensor<Src1T>();

        if constexpr (IsSameType<Src1T, half>::value || IsSameType<Src1T, bfloat16_t>::value) {
            uint16_t kStep = CeilDiv(kLength, 16);
            uint16_t nStep = CeilDiv(nLength, 16);
            for (uint16_t i = 0; i < kStep; i++) {
                LoadData2dTransposeParamsV2 loadDataParams;
                loadDataParams.startIndex = i;
                loadDataParams.repeatTimes = nStep;
                loadDataParams.srcStride = kStep;
                loadDataParams.dstGap = 0;
                LoadDataWithTranspose(b2[256 * i * nStep], rightMatrix, loadDataParams);
            }
        }

        if constexpr (IsSameType<Src1T, int8_t>::value) {
            uint16_t fracNum = 2;
            uint16_t kStep = CeilDiv(kLength, 16);
            uint16_t nStep = CeilDiv(nLength, 32);
            for (uint16_t i = 0; i < nStep; i ++) {
                LoadData2dTransposeParamsV2 loadDataParams;
                loadDataParams.startIndex = i * kStep;
                loadDataParams.repeatTimes = kStep / 2;               // original is n
                loadDataParams.srcStride = 2;
                loadDataParams.dstGap = nStep*2 - 1;
                LoadDataWithTranspose(b2[1024*i], rightMatrix, loadDataParams);
            }
        }

        if constexpr(IsSameType<Src1T, float>::value) {
            uint16_t kStep = CeilDiv(kLength, 16);
            uint16_t nStep = CeilDiv(nLength, 8);

            for (uint16_t i = 0; i < kStep; i ++) {
                LoadData2dTransposeParamsV2 loadDataParams;
                loadDataParams.startIndex = i;
                loadDataParams.repeatTimes = nStep / 2;               // original is n
                loadDataParams.srcStride = kStep * 2;
                loadDataParams.srcFracGap = kStep - 1;
                loadDataParams.dstGap = 0;
                loadDataParams.dstFracGap = nStep / 2 - 1;
                LoadDataWithTranspose(b2[128 * nStep*i], rightMatrix, loadDataParams);
            }

        }

        qidB2_.EnQue(b2);
        qidB1_.FreeTensor(rightMatrix);
    }

    __aicore__ inline void L0CToGm(const GlobalTensor<DstT> &gm)
    {
        auto co1Local = qidCO1_.DeQue<L0cT>();

        uint16_t srcStride = DivCeil(mLength, BLOCK_CUBE) * BLOCK_CUBE;
        uint16_t dstStride = mLength * sizeof(DstT) / 2;
        DataCopyCO12DstParams intriParams(nLength, mLength, dstStride, srcStride, QuantMode_t::NoQuant, 0, 0, 0);
        DataCopy(gm, co1Local, intriParams);
        qidCO1_.FreeTensor(co1Local);
    }

    __aicore__ inline void IterateAll(const GlobalTensor<DstT> &gm)
    {
        CopyGmToA1();
        pipe_barrier(PIPE_ALL);
        CopyGmToB1();
        pipe_barrier(PIPE_ALL);
        Load2DA1ToL0A();
        pipe_barrier(PIPE_ALL);
        Load2DA1ToL0B();
        Compute();
        L0CToGm(gm);
    }

private:
    TPipe *pipe = nullptr;
    int32_t mLength = 0;
    int32_t nLength = 0;
    int32_t kLength = 0;
    int32_t enBias = 0;
    int32_t transA_ = 0;
    int32_t path_ = 0;

    TQue<TPosition::A1, 1> qidA1_;
    TQue<TPosition::B1, 1> qidB1_;
    TQue<TPosition::C1, 1> qidBias_;
    TQue<TPosition::A2, 1> qidA2_;
    TQue<TPosition::B2, 1> qidB2_;
    TQue<TPosition::CO1, 1> qidCO1_;

    LocalTensor<SrcT> leftMatrix_;
    LocalTensor<Src1T> rightMatrix_;
    LocalTensor<BiasT> inputBias_;
    GlobalTensor<SrcT> aGlobal_;
    GlobalTensor<Src1T> bGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
};

template <class A_TYPE, class B_TYPE, class L0CType, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void E2eKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, int32_t m, int32_t n, int32_t k,
    int32_t usedCoreNum, int hasBias, int transA, int path)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using L0C_T = typename L0CType::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    if (block_idx >= usedCoreNum) {
        return;
    }
    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;
    GlobalTensor<BiasT> biasGlobal;
    GlobalTensor<A_T> tmpGlobal;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM), m * k);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), k * n);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM), m * n);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), n);
    tmpGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(cGM), m * n);

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    int offsetBias = 0;

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];
    auto gmBias = biasGlobal[offsetBias];
    auto gmTmp = tmpGlobal[0];

    set_atomic_none();
    TPipe que;
    E2eCase<A_TYPE, B_TYPE, L0CType, C_TYPE, BIAS_TYPE> ins;
    ins.Init(&que, m, n, k, hasBias, transA, path);
    ins.SetTensorA(gmA);
    ins.SetTensorB(gmB);
    if (hasBias == 1) {
        ins.SetBias(gmBias);
    }

    ins.IterateAll(gmC);
    set_atomic_none();
    return;
}

class TEST_KERNEL_MATMUL : public testing::Test {
protected:
    void SetUp()
    {
        g_coreType = AIC_TYPE;
    }
    void TearDown()
    {
        g_coreType = MIX_TYPE;
    }
};

#define KERNEL_E2E(TEST_KERNEL_MATMUL, coreNum, M, N, K, A_Pos, B_Pos, C_Pos, BIAS_Pos, A_Format, B_Format, C_Format, BIAS_Format, A_DType, B_DType, C_DType, BIAS_DType, HAS_BIAS, TRANS_A, PATH) \
    TEST_F(TEST_KERNEL_MATMUL,                                                                                                                                                                                \
        kernel_e2e_##coreNum##_##M##_##N##_##K##_##A_Pos##_##B_Pos##_##C_Pos##_##BIAS_Pos##_##A_Format##_##B_Format##_##C_Format##_##BIAS_Format##_##A_DType##_##B_DType##_##C_DType##_##BIAS_DType##_##HAS_BIAS##_##TRANS_A##_##PATH)                                                         \
    {                                                                                                                  \
        typedef InputInfo<TPosition::A_Pos, CubeFormat::A_Format, A_DType> aType;                             \
        typedef InputInfo<TPosition::B_Pos, CubeFormat::B_Format, B_DType> bType;                             \
        typedef InputInfo<TPosition::CO1, CubeFormat::C_Format, C_DType> l0cType;                             \
        typedef InputInfo<TPosition::C_Pos, CubeFormat::C_Format, C_DType> cType;                             \
        typedef InputInfo<TPosition::BIAS_Pos, CubeFormat::BIAS_Format, BIAS_DType> biasType;                 \
        const int32_t left_data_size = M * K;                                                             \
        const int32_t right_data_size = K * N;                                                            \
        const int32_t bias_data_size = N;                                                                 \
        const int32_t output_data_size = M * N;                                                           \
        uint8_t* aGM = new uint8_t[left_data_size * sizeof(A_DType) * 1000]{0};                                              \
        uint8_t* bGM = new uint8_t[right_data_size * sizeof(B_DType) * 1000]{0};                                             \
        uint8_t* biasGM = new uint8_t[bias_data_size * sizeof(BIAS_DType) * 1000]{0};                                        \
        uint8_t* cGM = new uint8_t[output_data_size * sizeof(C_DType) * 1000]{0};                                            \
        E2eKernel<aType, bType, l0cType, cType, biasType>(aGM, bGM, biasGM, cGM, M, N, K, coreNum, HAS_BIAS, TRANS_A, PATH); \
        for (int32_t i = 0; i < output_data_size * sizeof(C_DType); i++) {                                                  \
            EXPECT_EQ(cGM[i], 0x00);                                                                                        \
        }                                                                                                                   \
        delete[] aGM;\
        delete[] bGM;\
        delete[] biasGM;\
        delete[] cGM;\
    }

KERNEL_E2E(TEST_KERNEL_MATMUL, 1, 16, 32, 32, GM, GM, GM, GM, NZ, NZ, NZ, ND, int8_t, int8_t, int32_t, int32_t, 0, 0, 0)
KERNEL_E2E(TEST_KERNEL_MATMUL, 1, 32, 32, 32, GM, GM, GM, GM, NZ, NZ, NZ, ND, int16_t, int16_t, int32_t, int32_t, 0, 0, 0)
KERNEL_E2E(TEST_KERNEL_MATMUL, 1, 32, 32, 32, GM, GM, GM, GM, NZ, NZ, NZ, ND, int16_t, int8_t, int32_t, int32_t, 0, 0, 0)
KERNEL_E2E(TEST_KERNEL_MATMUL, 1, 32, 64, 64, GM, GM, GM, GM, NZ, NZ, NZ, ND, int8_t, int8_t, int32_t, int32_t, 0, 0, 0)
KERNEL_E2E(TEST_KERNEL_MATMUL, 1, 64, 64, 64, GM, GM, GM, GM, NZ, NZ, NZ, ND, int4b_t, int4b_t, int32_t, int32_t, 0, 0, 0)
KERNEL_E2E(TEST_KERNEL_MATMUL, 1, 32, 64, 64, GM, GM, GM, GM, NZ, NZ, NZ, ND, int8_t, int4b_t, int32_t, int32_t, 0, 0, 0)

KERNEL_E2E(TEST_KERNEL_MATMUL, 1, 16, 32, 32, GM, GM, GM, GM, NZ, NZ, NZ, ND, int8_t, int8_t, int32_t, int32_t, 0, 1, 0)
KERNEL_E2E(TEST_KERNEL_MATMUL, 1, 32, 32, 32, GM, GM, GM, GM, NZ, NZ, NZ, ND, int16_t, int16_t, int32_t, int32_t, 0, 1, 0)
KERNEL_E2E(TEST_KERNEL_MATMUL, 1, 32, 32, 32, GM, GM, GM, GM, NZ, NZ, NZ, ND, int16_t, int8_t, int32_t, int32_t, 0, 1, 0)
KERNEL_E2E(TEST_KERNEL_MATMUL, 1, 32, 64, 64, GM, GM, GM, GM, NZ, NZ, NZ, ND, int8_t, int8_t, int32_t, int32_t, 0, 1, 0)
KERNEL_E2E(TEST_KERNEL_MATMUL, 1, 64, 64, 64, GM, GM, GM, GM, NZ, NZ, NZ, ND, int4b_t, int4b_t, int32_t, int32_t, 0, 1, 0)
KERNEL_E2E(TEST_KERNEL_MATMUL, 1, 32, 64, 64, GM, GM, GM, GM, NZ, NZ, NZ, ND, int8_t, int4b_t, int32_t, int32_t, 0, 1, 0)