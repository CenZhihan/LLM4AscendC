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

template <TPosition POSITION, CubeFormat FORMAT, typename Type> struct InputInfo {
    constexpr static TPosition pos = POSITION;
    constexpr static CubeFormat format = FORMAT;
    using T = Type;
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

template <class AType, class BType, class CType, class L0CType, class BiasType, bool hasBias1> class E2eCase {
    using SrcT = typename AType::T;
    using Src1T = typename BType::T;
    using DstT = typename CType::T;
    using BiasT = typename BiasType::T;
    using L0cT = typename L0CType::T;

public:
    __aicore__ inline E2eCase() {}
    __aicore__ inline void Init(TPipe *tpipe, int32_t m, int32_t n, int32_t k, int32_t enableBias)
    {
        pipe = tpipe;
        mLength = m;
        nLength = n;
        kLength = k;
        enBias = enableBias;

        pipe->InitBuffer(qidA1_, 1, m * k * sizeof(SrcT)); // sizeof(fp4) = 1
        pipe->InitBuffer(qidB1_, 1, n * k * sizeof(Src1T));
        pipe->InitBuffer(qidA1MX_, 1, m * k * sizeof(fp8_e8m0_t));
        pipe->InitBuffer(qidB1MX_, 1, n * k * sizeof(fp8_e8m0_t));

        pipe->InitBuffer(qidA2, 1, m * k * sizeof(SrcT));
        pipe->InitBuffer(qidB2, 1, n * k * sizeof(Src1T));
        if (enBias == 1) {
            pipe->InitBuffer(inQueueC1, 1, n * sizeof(BiasT));
            pipe->InitBuffer(outQueueC2, 1, n * sizeof(L0cT));
        }
        pipe->InitBuffer(qidCO1, 1, m * n * sizeof(L0cT));
    }
    __aicore__ inline void SetTensorA(const GlobalTensor<SrcT> &gm, const GlobalTensor<fp8_e8m0_t> &amgm)
    {
        aGlobal1 = gm;
        aMxGlobal = amgm;
    }
    __aicore__ inline void SetTensorB(const GlobalTensor<Src1T> &gm, const GlobalTensor<fp8_e8m0_t> &bmgm)
    {
        bGlobal1 = gm;
        bMxGlobal = bmgm;

    }
    __aicore__ inline void SetBias(const GlobalTensor<BiasT> &biasGlobal)
    {
        biasGlobal1 = biasGlobal;
    }
    __aicore__ inline uint16_t CeilDiv(uint16_t num1, uint16_t num2)
    {
        ASSERT(num2 > 0);
        return (num1 + num2 - 1) / num2;
    }

    __aicore__ inline void CopyGmToA1Nd2Nz()
    {
        LocalTensor<SrcT> leftMatrix = qidA1_.template AllocTensor<SrcT>();

        // fp4 输入按照B8配置参数，要求内轴偶数
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = mLength;
        nd2nzParams.dValue = kLength / 2;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = kLength / 2;
        nd2nzParams.dstNzC0Stride = mLength;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;

        DataCopy(leftMatrix, aGlobal1, nd2nzParams);
        qidA1_.EnQue(leftMatrix);
    }

    __aicore__ inline void CopyScalarGmToA1Dn2Nz()
    {
        LocalTensor<fp8_e8m0_t> leftMxMatrix = qidA1MX_.template AllocTensor<fp8_e8m0_t>();

        // [m, k/32], k在内轴 DN2NZ.b16
        Dn2NzParams dn2nzParams;
        dn2nzParams.dnNum = 1;
        dn2nzParams.dValue = mLength;
        dn2nzParams.nValue = kLength / 32 / 2;
        dn2nzParams.srcDnMatrixStride = 0; // loop4
        dn2nzParams.srcDValue = kLength / 32 / 2;
        dn2nzParams.dstNzC0Stride = kLength / 32 / 2;
        dn2nzParams.dstNzNStride = 1;
        dn2nzParams.dstNzMatrixStride = 0;

        GlobalTensor<half> aMxGlobalB16;
        aMxGlobalB16.SetGlobalBuffer(((__gm__ half*)(aMxGlobal.GetPhyAddr())), mLength * kLength / 32 / 2);
        auto workLocal = leftMxMatrix.ReinterpretCast<half>();
        DataCopy(workLocal, aMxGlobalB16, dn2nzParams); // [m, k/32] fp8_e8m0

        qidA1MX_.EnQue(leftMxMatrix);
    }
    __aicore__ inline void CopyGmToB1Nd2Nz()
    {
        LocalTensor<Src1T> rightMatrix = qidB1_.template AllocTensor<Src1T>();
        LocalTensor<fp8_e8m0_t> rightMxMatrix = qidB1MX_.template AllocTensor<fp8_e8m0_t>();

        Nd2NzParams nd2nzParams; // 按照b8 配置参数
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = kLength;
        nd2nzParams.dValue = nLength / 2;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = nLength / 2;
        nd2nzParams.dstNzC0Stride = kLength;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;

        DataCopy(rightMatrix, bGlobal1, nd2nzParams);
        DataCopy(rightMxMatrix, bMxGlobal, kLength * nLength / 32); // [k/32, n]， k在外轴

        qidB1_.EnQue(rightMatrix);
        qidB1MX_.EnQue(rightMxMatrix);
    }

    __aicore__ inline void CopyGmToA1()
    {
        LocalTensor<SrcT> leftMatrix = qidA1_.template AllocTensor<SrcT>();
        LocalTensor<fp8_e8m0_t> leftMxMatrix = qidA1MX_.template AllocTensor<fp8_e8m0_t>();

        DataCopy(leftMatrix, aGlobal1, mLength * kLength); // 直接填fp4实际元素个数，代码内部处理
        DataCopy(leftMxMatrix, aMxGlobal, mLength * kLength / 32); // [m, k/32] fp8_e8m0_t
        qidA1_.EnQue(leftMatrix);
        qidA1MX_.EnQue(leftMxMatrix);
    }

    __aicore__ inline void CopyGmToB1()
    {
        LocalTensor<Src1T> rightMatrix = qidB1_.template AllocTensor<Src1T>();
        LocalTensor<fp8_e8m0_t> rightMxMatrix = qidB1MX_.template AllocTensor<fp8_e8m0_t>();
        DataCopy(rightMatrix, bGlobal1, kLength * nLength);
        DataCopy(rightMxMatrix, bMxGlobal, kLength * nLength / 32); // [k/32, n] fp8_e8m0_t
        qidB1_.EnQue(rightMatrix);
        qidB1MX_.EnQue(rightMxMatrix);
    }

    __aicore__ inline void Compute()
    {
        MmadParams mmadParams;
        mmadParams.m = mLength;
        mmadParams.n = nLength;
        mmadParams.k = kLength;
        mmadParams.isBias = enBias;

        auto co1Local = qidCO1.AllocTensor<L0cT>();
        auto a2 = qidA2.DeQue<SrcT>();
        auto b2 = qidB2.DeQue<Src1T>();
        pipe_barrier(PIPE_ALL);

        Mmad(co1Local, a2, b2, mmadParams);

        qidA2.FreeTensor(a2);
        qidB2.FreeTensor(b2);
        qidCO1.EnQue(co1Local);
    }

    __aicore__ inline void ComputeBt()
    {
        MmadParams mmadParams;
        mmadParams.m = mLength;
        mmadParams.n = nLength;
        mmadParams.k = kLength;
        mmadParams.isBias = enBias;
        mmadParams.cmatrixInitVal = false;

        auto co1Local = qidCO1.AllocTensor<L0cT>();
        auto a2 = qidA2.DeQue<SrcT>();
        auto b2 = qidB2.DeQue<Src1T>();
        auto bias2Local = outQueueC2.DeQue<L0cT>();
        pipe_barrier(PIPE_ALL);
        // mx, with bias, no uint64_t bias
        Mmad(co1Local, a2, b2, mmadParams);
        Mmad(co1Local, a2, b2, bias2Local, mmadParams);

        qidA2.FreeTensor(a2);
        qidB2.FreeTensor(b2);
        qidCO1.EnQue(co1Local);
        outQueueC2.FreeTensor(bias2Local);
    }

    __aicore__ inline void CopyL0CToGm(const GlobalTensor<DstT> &gm)
    {
        auto co1Local = qidCO1.DeQue<L0cT>();
        if constexpr (CType::format == CubeFormat::ND) {
            FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams(nLength, mLength, mLength, nLength);
            fixpipeParams.params = { 1, 0, 0 };
            if (IsSameType<DstT, half>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322F16;
            } else if (IsSameType<DstT, bfloat16_t>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322BF16;
            }
            Fixpipe<DstT, L0cT, CFG_ROW_MAJOR>(gm, co1Local, fixpipeParams);
        } else if constexpr (CType::format == CubeFormat::NZ) {
            FixpipeParamsArch3510<CO2Layout::NZ> fixpipeParams(nLength, mLength, mLength, nLength * BLOCK_CUBE);
            if (IsSameType<DstT, half>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322F16;
            } else if (IsSameType<DstT, bfloat16_t>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322BF16;
            }
            Fixpipe<DstT, L0cT, CFG_NZ>(gm, co1Local, fixpipeParams);
        }
        qidCO1.FreeTensor(co1Local);
    }

    __aicore__ inline void CopyInBias()
    {
        LocalTensor<BiasT> bias1Local = inQueueC1.AllocTensor<BiasT>();
        // bt:gm->l1
        DataCopy(bias1Local, biasGlobal1, nLength);
        inQueueC1.EnQue(bias1Local);
    }

    __aicore__ inline void SplitBias()
    {
        LocalTensor<BiasT> bias1Local = inQueueC1.DeQue<BiasT>();
        LocalTensor<L0cT> bias2Local = outQueueC2.AllocTensor<L0cT>();
        // bt:l1->bt
        DataCopy(bias2Local, bias1Local, { 1, static_cast<uint16_t>(nLength * sizeof(L0cT) / 32), 0, 0 });

        outQueueC2.EnQue<L0cT>(bias2Local);
        inQueueC1.FreeTensor(bias1Local);
    }

    __aicore__ inline void Load2DA1ToL0A(bool transpose)
    {
        auto leftMatrix = qidA1_.template DeQue<SrcT>();
        LocalTensor<SrcT> a2 = qidA2.AllocTensor<SrcT>();
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = DivCeil(mLength, 16);
        loadDataParams.kStep = DivCeil(kLength * sizeof(SrcT), 32);
        loadDataParams.srcStride = DivCeil(mLength, 16);
        loadDataParams.dstStride = DivCeil(mLength, 16);

        loadDataParams.sid = 0;
        loadDataParams.ifTranspose = transpose;

        if constexpr (SupportType<SrcT, fp4x2_e2m1_t, fp4x2_e1m2_t>()) {
            auto leftMxMatrix = qidA1MX_.template DeQue<fp8_e8m0_t>(); // mx 先按照b8定义 本质是float8_e8m0

            loadDataParams.kStep = DivCeil(kLength, 64); // sizeof(fp4x2_e2m1_t) is 1

            LoadData2DMxParams loadDataMxParams;
            loadDataMxParams.xStartPosition = 0;
            loadDataMxParams.yStartPosition = 0;
            loadDataMxParams.xStep = DivCeil(kLength, 64); // xstep
            loadDataMxParams.yStep = DivCeil(mLength, 16); // ystep
            loadDataMxParams.srcStride = DivCeil(mLength, 16);
            loadDataMxParams.dstStride = DivCeil(mLength, 16);
            LoadData(a2, leftMatrix, leftMxMatrix, loadDataParams, loadDataMxParams);
        } else {
            LoadData(a2, leftMatrix, loadDataParams);
        }

        qidA2.EnQue(a2);
        qidA1_.FreeTensor(leftMatrix);
    }

    __aicore__ inline void Load2DA1ToL0B(bool transpose)
    {
        auto rightMatrix = qidB1_.template DeQue<Src1T>();
        LocalTensor<Src1T> b2 = qidB2.AllocTensor<Src1T>();
        uint16_t nAlign = CeilDiv(nLength * sizeof(Src1T), 32);
        uint16_t kAlign = CeilDiv(kLength, 16);
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
        loadDataParams.ifTranspose = transpose;

        if constexpr (SupportType<SrcT, fp4x2_e2m1_t, fp4x2_e1m2_t>()) {
            auto rightMxMatrix = qidB1MX_.template DeQue<fp8_e8m0_t>();

            loadDataParams.mStep = DivCeil(kLength, 16);
            loadDataParams.kStep = DivCeil(nLength, 64); // sizeof(fp4x2_e2m1_t) is 1
            loadDataParams.srcStride = DivCeil(kLength, 16);
            // loadDataParams.dstStride = loadDataParams.kStep * 4;// unit is 512B
            loadDataParams.dstStride = DivCeil(nLength, 16);

            LoadData2DMxParams loadDataMxParams;
            loadDataMxParams.xStartPosition = 0;
            loadDataMxParams.yStartPosition = 0;
            loadDataMxParams.xStep = DivCeil(kLength, 64); // xstep 横方向
            loadDataMxParams.yStep = DivCeil(nLength, 16);
            loadDataMxParams.srcStride = DivCeil(nLength, 16);
            loadDataMxParams.dstStride = DivCeil(nLength, 16);

            LoadData(b2, rightMatrix, rightMxMatrix, loadDataParams, loadDataMxParams);
        } else {
            LoadData(b2, rightMatrix, loadDataParams);
        }

        qidB2.EnQue(b2);
        qidB1_.FreeTensor(rightMatrix);
    }

    __aicore__ inline void IterateAll(const GlobalTensor<DstT> &gm, bool transpose)
    {
        /* 1.非transpose场景(K在内轴) FP4 ND | Data: ND2NZ + Load2d;
                                            | Exp: DN2NZ.B16 + Load_L1_to_L0A/B_MX
           2. transpose场景(K在外轴)   FP4 ND | Data: ND2NZ + Load2d.b4.transpose;
                                             | Exp: ND2NZ.B16 + Load_L1_to_L0A/B_MX
        */
        if constexpr (AType::format == CubeFormat::ND) {
            CopyGmToA1Nd2Nz(); // matrix A: ND2NZ
            CopyScalarGmToA1Dn2Nz(); // scale matrix A : DN2NZ.b16
            CopyGmToB1Nd2Nz(); // matrix B: ND2NZ + scale matrix B ND2NZ
            Load2DA1ToL0A(transpose);
            Load2DA1ToL0B(transpose);
        } else if constexpr (AType::format == CubeFormat::NZ) {
            // gm ->l1(nz->nz), l1 ->loa/lob(with transpose)
            CopyGmToA1();
            CopyGmToB1();
            Load2DA1ToL0A(transpose);
            Load2DA1ToL0B(transpose);
        }

        if constexpr(hasBias1) {
            CopyInBias();
            SplitBias();
            ComputeBt();
        } else {
            Compute();
        }

        if constexpr (CType::format == CubeFormat::ND) {
            CopyL0CToGm(gm);
        }
    }

private:
    TPipe *pipe = nullptr;
    int32_t mLength = 0;
    int32_t nLength = 0;
    int32_t kLength = 0;
    int32_t enBias = 0;

    TQue<TPosition::A1, 1, GetNdNzMask(CubeFormat::NZ, AType::format)> qidA1_;
    TQue<TPosition::B1, 1, GetNdNzMask(CubeFormat::NZ, BType::format)> qidB1_;
    TQue<TPosition::A1, 1, GetNdNzMask(CubeFormat::NZ, AType::format)> qidA1MX_;
    TQue<TPosition::B1, 1, GetNdNzMask(CubeFormat::NZ, BType::format)> qidB1MX_;

    TQue<TPosition::A2, 1> qidA2;
    TQue<TPosition::B2, 1> qidB2;
    TQue<TPosition::CO1, 1> qidCO1;
    TQue<TPosition::C1, 1> inQueueC1;
    TQue<TPosition::C2, 1> outQueueC2;

    LocalTensor<SrcT> leftMatrix1;
    LocalTensor<Src1T> rightMatrix1;
    LocalTensor<BiasT> inputBias;
    GlobalTensor<SrcT> aGlobal1;
    GlobalTensor<Src1T> bGlobal1;
    GlobalTensor<fp8_e8m0_t> aMxGlobal;
    GlobalTensor<fp8_e8m0_t> bMxGlobal;
    GlobalTensor<BiasT> biasGlobal1;
};

template <class AType, class BType, class L0CType, class CType, class BiasType, bool hasBias1>
__aicore__ inline void E2eKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR amxGM, GM_ADDR bmxGM,
    int32_t m, int32_t n, int32_t k, int32_t usedCoreNum, int hasBias, bool transpose)
{
    // cube core cases, ignore vector core
    if (g_coreType == AIV) {
        return;
    }

    AscendCUtils::SetOverflow(1);
    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using L0C_T = typename L0CType::T;
    using C_T = typename CType::T;
    using BiasT = typename BiasType::T;

    if (block_idx >= usedCoreNum) {
        return;
    }
    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;
    GlobalTensor<BiasT> biasGlobal;
    GlobalTensor<fp8_e8m0_t> amxGlobal;
    GlobalTensor<fp8_e8m0_t> bmxGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM), m * k);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), k * n);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM), m * n);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), n);
    amxGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e8m0_t *>(amxGM), n);
    bmxGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e8m0_t *>(bmxGM), n);

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    int offsetBias = 0;

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];
    auto gmBias = biasGlobal[offsetBias];

    set_atomic_none();
    TPipe que;
    E2eCase<AType, BType, L0CType, CType, BiasType, hasBias1> ins;
    ins.Init(&que, m, n, k, hasBias);
    ins.SetTensorA(gmA, amxGlobal);
    ins.SetTensorB(gmB, bmxGlobal);
    if (hasBias == 1) {
        ins.SetBias(gmBias);
    }

    ins.IterateAll(gmC, transpose);
    set_atomic_none();
    return;
}

class TEST_KERNEL_MATMUL_FP4 : public testing::Test {
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

#define KERNEL_MATMUL_FP4(testKernelMatmulFp4, coreNum, m1, n1, k1, aPos, bPos, cPos, biasPos, aFormat, bFormat, cFormat, biasFormat,                                                                     \
    aDType, bDType, cDType, biasDType, hasBias1, transpose)                                                                                                                                                              \
    TEST_F(testKernelMatmulFp4,                                                                                                                                                                                \
        kernel_e2e_##coreNum##_##m1##_##n1##_##k1##_##aPos##_##bPos##_##cPos##_##biasPos##_##aFormat##_##bFormat##_##cFormat##_##biasFormat##_##aDType##_##bDType##_##cDType##_##biasDType##_##hasBias1##_##transpose) \
    {                                                                                                                                                                                                             \
        typedef InputInfo<TPosition::aPos, CubeFormat::aFormat, aDType> aType;                                                                                                                        \
        typedef InputInfo<TPosition::bPos, CubeFormat::bFormat, bDType> bType;                                                                                                                        \
        typedef InputInfo<TPosition::CO1, CubeFormat::cFormat, cDType> l0cType;                                                                                                                        \
        typedef InputInfo<TPosition::cPos, CubeFormat::cFormat, cDType> cType;                                                                                                                        \
        typedef InputInfo<TPosition::biasPos, CubeFormat::biasFormat, biasDType> biasType;                                                                                                            \
        const int32_t leftDataSize = m1 * k1;                                                             \
        const int32_t rightDataSize = k1 * n1;                                                            \
        const int32_t biasDataSize = n1;                                                                 \
        const int32_t outputDataSize = m1 * n1;                                                           \
        uint8_t aGM[leftDataSize * sizeof(aDType)] = {0};                                              \
        uint8_t amxGM[leftDataSize / 32 * sizeof(fp8_e8m0_t)] = {0};                                 \
        uint8_t bGM[rightDataSize * sizeof(bDType)] = {0};                                             \
        uint8_t bmxGM[rightDataSize / 32 * sizeof(fp8_e8m0_t)] = {0};                                \
        uint8_t biasGM[biasDataSize * sizeof(biasDType)] = {0};                                        \
        uint8_t cGM[outputDataSize * sizeof(cDType)] = {0};                                            \
        E2eKernel<aType, bType, l0cType, cType, biasType, hasBias1>(aGM, bGM, cGM, biasGM, amxGM, bmxGM, m1, n1, k1, coreNum, hasBias1, transpose); \
        for (int32_t i = 0; i < outputDataSize * sizeof(cDType); i++) {                                                  \
            EXPECT_EQ(cGM[i], 0x00);                                                                                        \
        }                                                                                                                   \
    }

KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 128, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e2m1_t, fp4x2_e2m1_t, float, float, 0, true)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 64, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e2m1_t, fp4x2_e2m1_t, float, float, 0, true)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 128, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 0, true)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 64, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 0, true)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 64, GM, GM, GM, GM, NZ, NZ, ND, ND, fp4x2_e2m1_t, fp4x2_e1m2_t, float, float, 0, true)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 0, true)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 32, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 0, true)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 128, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e2m1_t, fp4x2_e2m1_t, float, float, 0, false)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 64, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e2m1_t, fp4x2_e2m1_t, float, float, 0, false)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 128, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 0, false)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 64, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 0, false)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 64, GM, GM, GM, GM, NZ, NZ, ND, ND, fp4x2_e2m1_t, fp4x2_e1m2_t, float, float, 0, false)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 0, false)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 32, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 0, false)

KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 128, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e2m1_t, fp4x2_e2m1_t, float, float, 1, true)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 64, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e2m1_t, fp4x2_e2m1_t, float, float, 1, true)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 128, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 1, true)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 64, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 1, true)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 64, GM, GM, GM, GM, NZ, NZ, ND, ND, fp4x2_e2m1_t, fp4x2_e1m2_t, float, float, 1, true)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 1, true)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 32, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 1, true)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 128, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e2m1_t, fp4x2_e2m1_t, float, float, 1, false)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 64, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e2m1_t, fp4x2_e2m1_t, float, float, 1, false)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 128, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 1, false)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 64, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 1, false)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 64, GM, GM, GM, GM, NZ, NZ, ND, ND, fp4x2_e2m1_t, fp4x2_e1m2_t, float, float, 1, false)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 1, false)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 32, 128, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e1m2_t, float, float, 1, false)
KERNEL_MATMUL_FP4(TEST_KERNEL_MATMUL_FP4, 1, 32, 64, 64, GM, GM, GM, GM, ND, ND, ND, ND, fp4x2_e1m2_t, fp4x2_e2m1_t, float, float, 1, true)



template <class AType, class BType, class CType, class L0CType, class BiasType, bool hasBias1> class E2eCaseNonMx {
    using SrcT = typename AType::T;
    using Src1T = typename BType::T;
    using DstT = typename CType::T;
    using BiasT = typename BiasType::T;
    using L0cT = typename L0CType::T;

public:
    __aicore__ inline E2eCaseNonMx() {}
    __aicore__ inline void Init(TPipe *tpipe, int32_t m, int32_t n, int32_t k, int32_t enableBias)
    {
        pipe = tpipe;
        mLength = m;
        nLength = n;
        kLength = k;
        enBias = enableBias;

        pipe->InitBuffer(qidA1_, 1, m * k * sizeof(SrcT));
        pipe->InitBuffer(qidB1_, 1, n * k * sizeof(Src1T));

        pipe->InitBuffer(qidA2, 1, m * k * sizeof(SrcT));
        pipe->InitBuffer(qidB2, 1, n * k * sizeof(Src1T));

        if (enBias == 1) {
            pipe->InitBuffer(inQueueC1, 1, n * sizeof(BiasT));
            pipe->InitBuffer(outQueueC2, 1, n * sizeof(L0cT));
        }


        pipe->InitBuffer(qidCO1, 1, m * n * sizeof(L0cT));
    }
    __aicore__ inline void SetTensorA(const GlobalTensor<SrcT> &gm)
    {
        aGlobal1 = gm;
    }
    __aicore__ inline void SetTensorB(const GlobalTensor<Src1T> &gm)
    {
        bGlobal1 = gm;
    }
    __aicore__ inline void SetBias(const GlobalTensor<BiasT> &biasGlobal)
    {
        biasGlobal1 = biasGlobal;
    }
    __aicore__ inline uint16_t CeilDiv(uint16_t num1, uint16_t num2)
    {
        ASSERT(num2 > 0);
        return (num1 + num2 - 1) / num2;
    }

    __aicore__ inline void CopyGmToA1Nd2Nz()
    {
        LocalTensor<SrcT> leftMatrix = qidA1_.template AllocTensor<SrcT>();
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = mLength;
        nd2nzParams.dValue = kLength;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = kLength;
        nd2nzParams.dstNzC0Stride = mLength;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        DataCopy(leftMatrix, aGlobal1, nd2nzParams);
        qidA1_.EnQue(leftMatrix);
    }

    __aicore__ inline void CopyGmToB1Nd2Nz()
    {
        LocalTensor<Src1T> rightMatrix = qidB1_.template AllocTensor<Src1T>();
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = kLength;
        nd2nzParams.dValue = nLength;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = nLength;
        nd2nzParams.dstNzC0Stride = kLength;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        DataCopy(rightMatrix, bGlobal1, nd2nzParams);
        qidB1_.EnQue(rightMatrix);
    }

    __aicore__ inline void CopyGmToA1()
    {
        LocalTensor<SrcT> leftMatrix = qidA1_.template AllocTensor<SrcT>();
        DataCopy(leftMatrix, aGlobal1, mLength * kLength);
        qidA1_.EnQue(leftMatrix);
    }

    __aicore__ inline void CopyGmToB1()
    {
        LocalTensor<Src1T> rightMatrix = qidB1_.template AllocTensor<Src1T>();
        DataCopy(rightMatrix, bGlobal1, kLength * nLength);
        qidB1_.EnQue(rightMatrix);
    }
    __aicore__ inline void Load3DA1ToL0A()
    {
        auto leftMatrix = qidA1_.template DeQue<SrcT>();
        LocalTensor<SrcT> a2 = qidA2.AllocTensor<SrcT>();
        LoadData3DParamsV2<SrcT> loadData3dParams;
        loadData3dParams.l1W = 1;
        loadData3dParams.l1H = mLength;
        loadData3dParams.channelSize = kLength;
        loadData3dParams.kExtension = kLength;
        loadData3dParams.mExtension = mLength;
        loadData3dParams.kStartPt = 0;
        loadData3dParams.mStartPt = 0;
        loadData3dParams.strideW = 1;
        loadData3dParams.strideH = 1;
        loadData3dParams.filterW = 1;
        loadData3dParams.filterH = 1;
        loadData3dParams.dilationFilterW = 1;
        loadData3dParams.dilationFilterH = 1;
        loadData3dParams.enTranspose = false;
        loadData3dParams.enSmallK = false;
        loadData3dParams.padValue = 0;
        loadData3dParams.filterSizeW = 0;
        loadData3dParams.filterSizeH = 0;
        loadData3dParams.fMatrixCtrl = false;
        uint16_t dstStride = DivCeil(mLength, 16);
        SetLoadDataRepeatWithStride({0, 1, 0, dstStride});
        LoadDataWithStride(a2, leftMatrix, loadData3dParams);
        qidA2.EnQue(a2);
        qidA1_.FreeTensor(leftMatrix);
    }

    __aicore__ inline void Load3DB1ToL0B()
    {
        auto rightMatrix = qidB1_.template DeQue<Src1T>();
        LocalTensor<Src1T> b2 = qidB2.AllocTensor<Src1T>();
        LoadData3DParamsV2<Src1T> loadData3dParams;
        loadData3dParams.l1W = 1;
        loadData3dParams.l1H = kLength;
        loadData3dParams.channelSize = nLength;
        loadData3dParams.kExtension = nLength;
        loadData3dParams.mExtension = kLength;
        loadData3dParams.kStartPt = 0;
        loadData3dParams.mStartPt = 0;
        loadData3dParams.strideW = 1;
        loadData3dParams.strideH = 1;
        loadData3dParams.filterW = 1;
        loadData3dParams.filterH = 1;
        loadData3dParams.dilationFilterW = 1;
        loadData3dParams.dilationFilterH = 1;
        loadData3dParams.enTranspose = true;
        loadData3dParams.enSmallK = false;
        loadData3dParams.padValue = 0;
        loadData3dParams.filterSizeW = 0;
        loadData3dParams.filterSizeH = 0;
        loadData3dParams.fMatrixCtrl = false;
        uint8_t padList[PAD_SIZE] = {0, 0, 0, 0};
        SetFmatrix(1, kLength, padList, FmatrixMode::FMATRIX_LEFT);
        SetLoadDataPaddingValue(0); // bit cast
        uint16_t dstStride = DivCeil(nLength, 16);
        SetLoadDataRepeatWithStride({0, 1, 0, dstStride});
        static constexpr IsResetLoad3dConfig LOAD3D_CONFIG = {false, false};
        LoadDataWithStride<Src1T, LOAD3D_CONFIG>(b2, rightMatrix, loadData3dParams);
        qidB2.EnQue(b2);
        qidB1_.FreeTensor(rightMatrix);
    }

    __aicore__ inline void Compute()
    {
        MmadParams mmadParams;
        mmadParams.m = mLength;
        mmadParams.n = nLength;
        mmadParams.k = kLength;
        mmadParams.isBias = enBias;

        auto co1Local = qidCO1.AllocTensor<L0cT>();
        auto a2 = qidA2.DeQue<SrcT>();
        auto b2 = qidB2.DeQue<Src1T>();
        pipe_barrier(PIPE_ALL);
        Mmad(co1Local, a2, b2, mmadParams);

        qidA2.FreeTensor(a2);
        qidB2.FreeTensor(b2);
        qidCO1.EnQue(co1Local);
    }

    __aicore__ inline void CopyL0CToGm(const GlobalTensor<DstT> &gm)
    {
        auto co1Local = qidCO1.DeQue<L0cT>();
        if constexpr (CType::format == CubeFormat::ND) {
            FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams(nLength, mLength, mLength, nLength);
            fixpipeParams.params = { 1, 0, 0 };
            if (IsSameType<DstT, half>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322F16;
            } else if (IsSameType<DstT, bfloat16_t>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322BF16;
            }
            Fixpipe<DstT, L0cT, CFG_ROW_MAJOR>(gm, co1Local, fixpipeParams);
        } else if constexpr (CType::format == CubeFormat::NZ) {
            FixpipeParamsArch3510<CO2Layout::NZ> fixpipeParams(nLength, mLength, mLength, nLength * BLOCK_CUBE);
            if (IsSameType<DstT, half>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322F16;
            } else if (IsSameType<DstT, bfloat16_t>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322BF16;
            }
            Fixpipe<DstT, L0cT, CFG_NZ>(gm, co1Local, fixpipeParams);
        }
        qidCO1.FreeTensor(co1Local);
    }

    __aicore__ inline void CopyInBias()
    {
        LocalTensor<BiasT> bias1Local = inQueueC1.AllocTensor<BiasT>();
        // bt:gm->l1
        DataCopy(bias1Local, biasGlobal1, nLength);
        inQueueC1.EnQue(bias1Local);
    }

    __aicore__ inline void SplitBias()
    {
        LocalTensor<BiasT> bias1Local = inQueueC1.DeQue<BiasT>();
        LocalTensor<L0cT> bias2Local = outQueueC2.AllocTensor<L0cT>();
        // bt:l1->bt
        DataCopy(bias2Local, bias1Local, { 1, static_cast<uint16_t>(nLength * sizeof(L0cT) / 32), 0, 0 });

        outQueueC2.EnQue<L0cT>(bias2Local);
        inQueueC1.FreeTensor(bias1Local);
    }

    __aicore__ inline void ComputeBt()
    {
        MmadParams mmadParams;
        mmadParams.m = mLength;
        mmadParams.n = nLength;
        mmadParams.k = kLength;
        mmadParams.isBias = enBias;

        auto co1Local = qidCO1.AllocTensor<L0cT>();
        auto a2 = qidA2.DeQue<SrcT>();
        auto b2 = qidB2.DeQue<Src1T>();
        auto bias2Local = outQueueC2.DeQue<L0cT>();
        pipe_barrier(PIPE_ALL);

        Mmad(co1Local, a2, b2, bias2Local, mmadParams);

        qidA2.FreeTensor(a2);
        qidB2.FreeTensor(b2);
        qidCO1.EnQue(co1Local);
        outQueueC2.FreeTensor(bias2Local);
    }

    __aicore__ inline void Load2DA1ToL0A()
    {
        auto leftMatrix = qidA1_.template DeQue<SrcT>();
        LocalTensor<SrcT> a2 = qidA2.AllocTensor<SrcT>();
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = DivCeil(mLength, 16);
        loadDataParams.kStep = DivCeil(kLength * sizeof(SrcT), 32);
        loadDataParams.srcStride = DivCeil(mLength, 16);
        loadDataParams.dstStride = DivCeil(mLength, 16);

        loadDataParams.sid = 0;
        loadDataParams.ifTranspose = false;
        LoadData(a2, leftMatrix, loadDataParams);
        qidA2.EnQue(a2);
        qidA1_.FreeTensor(leftMatrix);
    }

    __aicore__ inline void Load2DA1ToL0B()
    {
        auto rightMatrix = qidB1_.template DeQue<Src1T>();
        LocalTensor<Src1T> b2 = qidB2.AllocTensor<Src1T>();
        uint16_t nAlign = CeilDiv(nLength * sizeof(Src1T), 32);
        uint16_t kAlign = CeilDiv(kLength, 16);
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
        if (SupportType<Src1T, int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t>()) {
            loadDataParams.dstStride = nAlign * 2;
        }
        loadDataParams.sid = 0;
        loadDataParams.ifTranspose = true;
        LoadData(b2, rightMatrix, loadDataParams);

        qidB2.EnQue(b2);
        qidB1_.FreeTensor(rightMatrix);
    }

    __aicore__ inline void CopyL0CToL1ToGm(const GlobalTensor<DstT> &gm)
    {
        LocalTensor<DstT> resMatrix = qidA1_.template AllocTensor<DstT>();
        auto co1Local = qidCO1.DeQue<L0cT>();
        if constexpr (CType::format == CubeFormat::ND) {
            FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams(nLength,
                mLength, mLength, nLength);
            fixpipeParams.params = { 1, 0, 0 };
            if (IsSameType<DstT, half>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322F16;
            } else if (IsSameType<DstT, bfloat16_t>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322BF16;
            }
            Fixpipe<DstT, L0cT, CFG_ROW_MAJOR>(resMatrix, co1Local, fixpipeParams);
        } else if constexpr (CType::format == CubeFormat::NZ) {
            FixpipeParamsArch3510<CO2Layout::NZ> fixpipeParams(nLength,
                mLength, mLength, nLength * BLOCK_CUBE);
            if (IsSameType<DstT, half>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322F16;
            } else if (IsSameType<DstT, bfloat16_t>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322BF16;
            }
            Fixpipe<DstT, L0cT, CFG_NZ>(resMatrix, co1Local, fixpipeParams);
        }
        qidCO1.FreeTensor(co1Local);
        pipe_barrier(PIPE_ALL);
        DataCopyParams dataCopyInfo;
        dataCopyInfo.blockCount = 1;
        dataCopyInfo.blockLen = nLength * mLength * sizeof(DstT) / ONE_BLK_SIZE;
        DataCopy(gm, resMatrix, dataCopyInfo);
        qidA1_.FreeTensor(resMatrix);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void IterateAll(const GlobalTensor<DstT> &gm)
    {
        if (AType::format == CubeFormat::ND) {
            CopyGmToA1Nd2Nz();
            CopyGmToB1Nd2Nz();
            Load2DA1ToL0A();
            Load2DA1ToL0B();
        } else if (AType::format == CubeFormat::NZ) {
            // gm ->l1(nz->nz), l1 ->loa/lob(with transpose)
            CopyGmToA1();
            CopyGmToB1();
            Load2DA1ToL0A();
            Load2DA1ToL0B();
        }

        if constexpr(hasBias1) {
            CopyInBias();
            SplitBias();
            ComputeBt();
            if constexpr(hasBias1) {
                if (CType::format == CubeFormat::ND) {
                    CopyL0CToGm(gm);
                } else if (CType::format == CubeFormat::NZ) {
                    CopyL0CToGm(gm); // in 1982 l1->gm has removed
                }
            }
            return;
        }

        Compute();
        if (CType::format == CubeFormat::ND) {
            CopyL0CToGm(gm);
        } else if (CType::format == CubeFormat::NZ) {
            CopyL0CToGm(gm); // in 1982 l1->gm has removed
        }
    }

private:
    TPipe *pipe = nullptr;
    int32_t mLength = 0;
    int32_t nLength = 0;
    int32_t kLength = 0;
    int32_t enBias = 0;

    TQue<TPosition::A1, 1, GetNdNzMask(CubeFormat::NZ, AType::format)> qidA1_;
    TQue<TPosition::B1, 1, GetNdNzMask(CubeFormat::NZ, BType::format)> qidB1_;
    TQue<TPosition::A2, 1> qidA2;
    TQue<TPosition::B2, 1> qidB2;
    TQue<TPosition::CO1, 1> qidCO1;
    TQue<TPosition::C1, 1> inQueueC1;
    TQue<TPosition::C2, 1> outQueueC2;

    LocalTensor<SrcT> leftMatrix1;
    LocalTensor<Src1T> rightMatrix1;
    LocalTensor<BiasT> inputBias;
    GlobalTensor<SrcT> aGlobal1;
    GlobalTensor<Src1T> bGlobal1;
    GlobalTensor<BiasT> biasGlobal1;
};

template <class AType, class BType, class L0CType, class CType, class BiasType, bool hasBias1>
__aicore__ inline void E2eKernelNonMx(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, int32_t m, int32_t n, int32_t k,
    int32_t usedCoreNum, int hasBias)
{
    // cube core cases, ignore vector core
    if (g_coreType == AIV) {
        return;
    }

    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using L0C_T = typename L0CType::T;
    using C_T = typename CType::T;
    using BiasT = typename BiasType::T;

    if (block_idx >= usedCoreNum) {
        return;
    }
    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;
    GlobalTensor<BiasT> biasGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM), m * k);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), k * n);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM), m * n);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), n);

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    int offsetBias = 0;

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];
    auto gmBias = biasGlobal[offsetBias];

    set_atomic_none();
    TPipe que;
    E2eCaseNonMx<AType, BType, L0CType, CType, BiasType, hasBias1> ins;
    ins.Init(&que, m, n, k, hasBias);
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

#define KERNEL_MATMUL_NONMX(testKernelMatmul, coreNum, m1, n1, k1, aPos, bPos, cPos, biasPos, aFormat, bFormat, cFormat, biasFormat,                                                                     \
    aDType, bDType, cDType, biasDType, hasBias1, transpose)                                                                                                                                                              \
    TEST_F(testKernelMatmul,                                                                                                                                                                                \
        kernel_e2e_nonmx_##coreNum##_##m1##_##n1##_##k1##_##aPos##_##bPos##_##cPos##_##biasPos##_##aFormat##_##bFormat##_##cFormat##_##biasFormat##_##aDType##_##bDType##_##cDType##_##biasDType##_##hasBias1##_##transpose) \
    {                                                                                                                                                                                                             \
        typedef InputInfo<TPosition::aPos, CubeFormat::aFormat, aDType> aType;                                                                                                                        \
        typedef InputInfo<TPosition::bPos, CubeFormat::bFormat, bDType> bType;                                                                                                                        \
        typedef InputInfo<TPosition::CO1, CubeFormat::cFormat, cDType> l0cType;                                                                                                                        \
        typedef InputInfo<TPosition::cPos, CubeFormat::cFormat, cDType> cType;                                                                                                                        \
        typedef InputInfo<TPosition::biasPos, CubeFormat::biasFormat, biasDType> biasType;                                                                                                            \
        const int32_t leftDataSize = m1 * k1;                                                             \
        const int32_t rightDataSize = k1 * n1;                                                            \
        const int32_t biasDataSize = n1;                                                                 \
        const int32_t outputDataSize = m1 * n1;                                                           \
        uint8_t aGM[leftDataSize * sizeof(aDType)] = {0};                                              \
        uint8_t bGM[rightDataSize * sizeof(bDType)] = {0};                                             \
        uint8_t biasGM[biasDataSize * sizeof(biasDType)] = {0};                                        \
        uint8_t cGM[outputDataSize * sizeof(cDType)] = {0};                                            \
        E2eKernelNonMx<aType, bType, l0cType, cType, biasType, hasBias1>(aGM, bGM, cGM, biasGM, m1, n1, k1, coreNum, hasBias1); \
        for (int32_t i = 0; i < outputDataSize * sizeof(cDType); i++) {                                                  \
            EXPECT_EQ(cGM[i], 0x00);                                                                                        \
        }                                                                                                                   \
    }

KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 16, 16, 16, GM, GM, GM, GM, ND, ND, ND, ND, bfloat16_t, bfloat16_t, float, float, 1, false)
KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 16, 16, 16, GM, GM, GM, GM, NZ, NZ, NZ, ND, bfloat16_t, bfloat16_t, float, float, 1, false)
KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 16, 16, 16, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 1, false)
KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 16, 16, 16, GM, GM, GM, GM, NZ, NZ, NZ, ND, half, half, float, float, 1, false)
KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 16, 16, 16, GM, GM, GM, GM, ND, ND, ND, ND, float, float, float, float, 1, false)
KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 16, 32, 32, GM, GM, GM, GM, ND, ND, ND, ND, int8_t, int8_t, int32_t, int32_t, 1, false)
KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 128, 64, 128, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 1, false)
KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 16, 16, 32, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 1, false)
KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 128, 64, 128, GM, GM, GM, GM, ND, ND, ND, ND, float, float, float, float, 1, false)
KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 128, 128, 128, GM, GM, GM, GM, ND, ND, ND, ND, int8_t, int8_t, int32_t, int32_t, 1, false)
KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 32, 64, 64, GM, GM, GM, GM, ND, ND, ND, ND, int8_t, int8_t, int32_t, int32_t, 1, false)
KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 16, 32, 32, GM, GM, GM, GM, ND, ND, ND, ND, fp8_e5m2_t, fp8_e5m2_t, float, float, 1, false)
KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 16, 32, 32, GM, GM, GM, GM, ND, ND, ND, ND, fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, 1, false)
KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 16, 32, 32, GM, GM, GM, GM, ND, ND, ND, ND, fp8_e5m2_t, fp8_e4m3fn_t, float, float, 1, false)
KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 16, 32, 32, GM, GM, GM, GM, ND, ND, ND, ND, fp8_e4m3fn_t, fp8_e5m2_t, float, float, 1, false)
KERNEL_MATMUL_NONMX(TEST_KERNEL_MATMUL, 1, 16, 32, 32, GM, GM, GM, GM, ND, ND, ND, ND, hifloat8_t, hifloat8_t, float, float, 1, false)