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
#include "mockcpp/mockcpp.hpp"

//using namespace std;
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

template <class AType, class BType, class CType, class BiasType> class E2eCase {
    using SrcT = typename AType::T;
    using Src1T = typename BType::T;
    using DstT = typename CType::T;
    using BiasT = typename BiasType::T;

public:
    __aicore__ inline E2eCase() {}
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
            pipe->InitBuffer(qidBias, 1, n * sizeof(BiasT));
        }
        pipe->InitBuffer(qidCO1, 1, m * n * sizeof(float));
        pipe->InitBuffer(tmpQueue, 1, m * k * sizeof(SrcT));
    }
    __aicore__ inline void SetTensorA(const GlobalTensor<SrcT> &gm)
    {
        aGlobal1 = gm;
    }
    __aicore__ inline void SetTensorB(const GlobalTensor<SrcT> &gm)
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
        Load3DBitModeParam load3DBitModeParam(loadData3dParams);

        LoadData<TPosition::A2, TPosition::A1>(a2, leftMatrix, load3DBitModeParam);
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
        SetFMatrixBitModeParams setFMatrixBitModeParams(loadData3dParams);
        SetFmatrix(setFMatrixBitModeParams, FmatrixMode::FMATRIX_LEFT);
        SetLoadDataPaddingValue(0); // bit cast
        uint16_t dstStride = DivCeil(nLength, 16);
        static constexpr IsResetLoad3dConfig LOAD3D_CONFIG = {false, false};
        SetLoadDataRepeat({0, 1, 0});
        LoadData<Src1T, LOAD3D_CONFIG>(b2, rightMatrix, loadData3dParams);
        SetLoadDataRepeatWithStride({0, 1, 0, dstStride});
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

        auto co1Local = qidCO1.AllocTensor<float>();
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
        auto co1Local = qidCO1.DeQue<float>();
        if constexpr (CType::format == CubeFormat::ND) {
            FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams(nLength, mLength, mLength, nLength);
            fixpipeParams.params = { 1, 0, 0 };
            if (IsSameType<DstT, half>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322F16;
            } else if (IsSameType<DstT, bfloat16_t>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322BF16;
            }
            Fixpipe(gm, co1Local, fixpipeParams);
        } else if constexpr (CType::format == CubeFormat::NZ) {
            FixpipeParamsArch3510<CO2Layout::NZ> fixpipeParams(nLength, mLength, mLength, mLength * BLOCK_CUBE);
            if (IsSameType<DstT, half>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322F16;
            } else if (IsSameType<DstT, bfloat16_t>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322BF16;
            }
            Fixpipe(gm, co1Local, fixpipeParams);
        }
        qidCO1.FreeTensor(co1Local);
    }

    __aicore__ inline void Load2DA1ToL0A()
    {
        auto leftMatrix = qidA1_.template DeQue<SrcT>();
        LocalTensor<SrcT> a2 = qidA2.AllocTensor<SrcT>();
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = CeilDiv(mLength, 16);
        loadDataParams.kStep = CeilDiv(kLength * sizeof(SrcT), 32);
        loadDataParams.srcStride = CeilDiv(mLength, 16);
        loadDataParams.dstStride = CeilDiv(mLength, 16);
        loadDataParams.sid = 0;
        loadDataParams.ifTranspose = false;
        Load2DBitModeParam loadDataBitModeParam(loadDataParams);

        LoadData<TPosition::A2, TPosition::A1>(a2, leftMatrix, loadDataBitModeParam);
        // failed case
        TBuf<TPosition::C2> tbufFailed;
        pipe->InitBuffer(tbufFailed, mLength * kLength * sizeof(SrcT));
        LocalTensor<SrcT> fmLocal = tbufFailed.Get<SrcT>();
        MOCKER(raise).stubs().will(returnValue(int(0)));
        LoadData<SrcT>(fmLocal, leftMatrix, loadDataParams);
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
        loadDataParams.sid = 0;
        loadDataParams.ifTranspose = true;
        Load2DBitModeParam loadDataBitModeParam(loadDataParams);
        LoadData<TPosition::B2, TPosition::B1>(b2, rightMatrix, loadDataBitModeParam);

        if (IsSameType<Src1T, half>::value) {
            uint16_t kStep = CeilDiv(kLength, 16);
            uint16_t nStep = CeilDiv(nLength, 16);
            for (uint16_t i = 0; i < kStep; i++) {
                LoadData2dTransposeParamsV2 loadDataTransParams;
                loadDataTransParams.startIndex = i;
                loadDataTransParams.repeatTimes = nStep;
                loadDataTransParams.srcStride = kStep;
                loadDataTransParams.dstGap = 0;
                LoadDataWithTranspose(b2[256 * i * nStep], rightMatrix, loadDataTransParams);
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

        qidB2.EnQue(b2);
        qidB1_.FreeTensor(rightMatrix);
    }

    __aicore__ inline void Load2dGMToL1Left()
    {
        auto leftMatrix = qidA1_.template AllocTensor<SrcT>();
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.srcStride = CeilDiv(mLength, 16);
        loadDataParams.dstStride = CeilDiv(mLength, 16);
        loadDataParams.mStep = CeilDiv(mLength, 16);
        loadDataParams.kStep = CeilDiv(kLength * sizeof(SrcT), 32);

        LoadData(leftMatrix, aGlobal1, loadDataParams);
        qidA1_.EnQue(leftMatrix);
    }

    __aicore__ inline void Load2dGMToL0A()
    {
        auto leftMatrix = qidA2.template AllocTensor<SrcT>();
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.srcStride = 0;
        loadDataParams.dstStride = 0;
        loadDataParams.mStep = 0;
        loadDataParams.kStep = 0;

        LoadData(leftMatrix, aGlobal1, loadDataParams);
        qidA2.EnQue(leftMatrix);
        leftMatrix = qidA2.template DeQue<SrcT>();
        qidA2.FreeTensor(leftMatrix);
    }

    __aicore__ inline void Load2dGMToL0B()
    {
        auto leftMatrix = qidB2.template AllocTensor<SrcT>();
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.srcStride = 0;
        loadDataParams.dstStride = 0;
        loadDataParams.mStep = 0;
        loadDataParams.kStep = 0;

        LoadData(leftMatrix, bGlobal1, loadDataParams);
        qidB2.EnQue(leftMatrix);
        leftMatrix = qidB2.template DeQue<SrcT>();
        qidB2.FreeTensor(leftMatrix);
    }

    __aicore__ inline void Load2dGMToL1Right()
    {
        auto rightMatrix = qidB1_.template AllocTensor<Src1T>();
        auto tmp = tmpQueue.template AllocTensor<Src1T>();
        LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.srcStride = CeilDiv(kLength, 16);
        loadDataParams.dstStride = CeilDiv(kLength, 16);
        loadDataParams.mStep = CeilDiv(kLength, 16);
        loadDataParams.kStep = CeilDiv(nLength * sizeof(Src1T), 32);

        LoadData(rightMatrix, bGlobal1, loadDataParams);
        qidB1_.EnQue(rightMatrix);
    }

    __aicore__ inline void IterateAll(const GlobalTensor<DstT> &gm)
    {
            CopyGmToA1Nd2Nz();
            CopyGmToB1Nd2Nz();
            Load2DA1ToL0A();
            Load2DA1ToL0B();
            Load2dGMToL1Left();
            Load2dGMToL1Right();
    }

private:
    TPipe *pipe = nullptr;
    int32_t mLength = 0;
    int32_t nLength = 0;
    int32_t kLength = 0;
    int32_t enBias = 0;

    TQue<TPosition::A1, 1, GetNdNzMask(CubeFormat::NZ, AType::format)> qidA1_;
    TQue<TPosition::B1, 1, GetNdNzMask(CubeFormat::NZ, BType::format)> qidB1_;
    TQue<TPosition::C1, 1> qidBias;
    TQue<TPosition::A2, 1> qidA2;
    TQue<TPosition::B2, 1> qidB2;
    TQue<TPosition::CO1, 1> qidCO1;

    TQue<TPosition::VECIN, 1> tmpQueue;
    LocalTensor<SrcT> leftMatrix1;
    LocalTensor<Src1T> rightMatrix1;
    LocalTensor<BiasT> inputBias;
    GlobalTensor<SrcT> aGlobal1;
    GlobalTensor<Src1T> bGlobal1;
    GlobalTensor<BiasT> biasGlobal1;
};

template <class AType, class BType, class CType, class BiasType>
__aicore__ inline void E2eKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, int32_t m, int32_t n, int32_t k,
    int hasBias)
{
    // cube core cases, ignore vector core
    if (g_coreType == AIV) {
        return;
    }

    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using C_T = typename CType::T;
    using BiasT = typename BiasType::T;

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
    E2eCase<AType, BType, CType, BiasType> ins;
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

template <typename T1, typename T2>
__global__ __aicore__ void kernelE2E(__gm__ uint8_t *aGm, __gm__ uint8_t *bGm, __gm__ uint8_t *cGm,
    __gm__ uint8_t *biasGm, __gm__ uint16_t m, __gm__ uint16_t n, __gm__ uint16_t k)
{
    typedef InputInfo<TPosition::GM, CubeFormat::ND, T1> aType;
    typedef InputInfo<TPosition::GM, CubeFormat::ND, T1> bType;
    typedef InputInfo<TPosition::GM, CubeFormat::ND, T2> cType;
    typedef InputInfo<TPosition::GM, CubeFormat::ND, T2> biasType;
    E2eKernel<aType, bType, cType, biasType>(aGm, bGm, cGm, biasGm, static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k), 0);
}

struct E2eParams {
    uint16_t m;
    uint16_t n;
    uint16_t k;
    void (*cal_func)(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint16_t, uint16_t, uint16_t);
};

class E2eBitModeTestsuite : public testing::Test {
protected:
    void SetUp()
    {
        g_coreType = AIC_TYPE;
    }
    void TearDown()
    {
        g_coreType = MIX_TYPE;
        GlobalMockObject::verify();
    }
};

namespace AscendC {
// T, U for gm->l1 nd2nz
template <typename T> class KernelDataCopyGm2L1Nd2Nz {
public:
    __aicore__ inline KernelDataCopyGm2L1Nd2Nz() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, Nd2NzParams& intriParamsIn)
    {
        intriParams = intriParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrcA1, 1,
            (((intriParams.dValue * sizeof(T) - 1) / 32 + 1) * intriParams.dstNzC0Stride * 32));
        pipe.InitBuffer(inQueueSrcVecOut, 1,
            (((intriParams.dValue * sizeof(T) - 1) / 32 + 1) * intriParams.dstNzC0Stride * 32));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<T> srcLocal = inQueueSrcA1.AllocTensor<T>();
        DataCopy(srcLocal, srcGlobal, intriParams);
        inQueueSrcA1.EnQue(srcLocal);
    }
    __aicore__ inline void Compute() {}
    __aicore__ inline void CopyOut() {}

private:
    TPipe pipe;
    TQue<TPosition::A1, 1> inQueueSrcA1;
    TQue<TPosition::VECOUT, 1> inQueueSrcVecOut;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    Nd2NzParams intriParams;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void MainDataCopyGm2L1Nd2Nz(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
    Nd2NzParams& intriParams)
{
    KernelDataCopyGm2L1Nd2Nz<T> op;
    op.Init(dstGm, srcGm, intriParams);
    op.Process();
}

struct DataCopyGm2L1Nd2NzTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, Nd2NzParams&);
    Nd2NzParams intriParams;
};

class DataCopyGm2L1Nd2NzTestsuite : public testing::Test,
    public testing::WithParamInterface<DataCopyGm2L1Nd2NzTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DATACOPYGM2L1ND2NZ, DataCopyGm2L1Nd2NzTestsuite,
    ::testing::Values(DataCopyGm2L1Nd2NzTestParams { 2, MainDataCopyGm2L1Nd2Nz<half>, { 2, 2, 13, 48, 16, 11, 2, 48 } },
    DataCopyGm2L1Nd2NzTestParams { 4, MainDataCopyGm2L1Nd2Nz<float>, { 2, 2, 13, 48, 16, 11, 2, 40 } }));

namespace AscendC {
template <typename Src0T, bool UseFill = false> class KernelCreatMartix {
public:
    __aicore__ inline void Init()
    {
        c0 = 32 / sizeof(Src0T);
        c1Fmap = channelSizeFmap / c0;
        c1Weight = channelSizeWeight / c0;
        hoFmap = hFmap - dilationH * (khFmap - 1);
        woFmap = wFmap - dilationW * (kwFmap - 1);
        howoFmap = hoFmap * woFmap;
        howoRoundFmap = ((howoFmap + 16 - 1) / 16) * 16;

        hoWeight = hWeight - dilationH * (khWeight - 1);
        woWeight = wWeight - dilationW * (kwWeight - 1);
        howoWeight = hoWeight * woWeight;
        howoRoundWeight = ((howoWeight + 16 - 1) / 16) * 16;
        featureMapA1Size = c1Fmap * hFmap * wFmap * c0;
        weightA1Size = c1Weight * hWeight * wWeight * c0;

        pipe.InitBuffer(inQueueFmA1, 1, featureMapA1Size * sizeof(Src0T));
        pipe.InitBuffer(inQueueWeightB1, 1, weightA1Size * sizeof(Src0T));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
    }

private:
    __aicore__ inline void CopyIn()
    {
        if constexpr (IsSameType<Src0T, int16_t>::value) {
            SetLoadDataPaddingValue(static_cast<int16_t>(0));
        }
        LocalTensor<Src0T> featureMapA1 = inQueueFmA1.AllocTensor<Src0T>();
        // set repeat
        set_l3d_rpt(0);
        // init l1
        if constexpr (UseFill) {
            Fill<Src0T>(featureMapA1,
                { 1, static_cast<uint16_t>(featureMapA1Size * sizeof(Src0T) / 32), 0, static_cast<Src0T>(1) });
        }else{
            InitConstValue<Src0T>(featureMapA1,
                { 1, static_cast<uint16_t>(featureMapA1Size * sizeof(Src0T) / 32), 0, static_cast<Src0T>(1) });
        }
        
        inQueueFmA1.FreeTensor(featureMapA1);
    }

private:
    TPipe pipe;
    TQue<TPosition::A1, 1> inQueueFmA1;
    TQue<TPosition::B1, 1> inQueueWeightB1;
    TQue<TPosition::CO1, 1> outQueueCO1;
    uint16_t channelSizeFmap = 32, channelSizeWeight = 16;
    uint16_t c0;
    uint16_t c1Fmap, c1Weight;
    uint16_t hFmap = 12, wFmap = 16, hWeight = 5, wWeight = 9;
    uint8_t khFmap = 1, kwFmap = 1, khWeight = 2, kwWeight = 2;
    uint8_t dilationH = 1, dilationW = 1;
    uint16_t hoFmap, woFmap, howoFmap, howoRoundFmap, hoWeight, woWeight, howoWeight, howoRoundWeight;
    uint16_t featureMapA1Size, weightA1Size, featureMapA2Size, weightB2Size;
};
} // namespace AscendC

template <typename Src0T>
__global__ __aicore__ void creat_martix_AscendC()
{
    KernelCreatMartix<Src0T, false> op;
    op.Init();
    op.Process();
}

template <typename Src0T>
__global__ __aicore__ void FillAscendC()
{
    AscendC::KernelCreatMartix<Src0T, true> op;
    op.Init();
    op.Process();
}

struct CreatmartixTestParams {
    uint16_t m;
    uint16_t n;
    uint16_t k;
    void (*cal_func)();
    uint8_t sizeofSrc0;
};

class CreatmartixTestsuite : public testing::Test, public testing::WithParamInterface<CreatmartixTestParams> {
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

INSTANTIATE_TEST_CASE_P(SetCreatmartixTest, CreatmartixTestsuite,
    ::testing::Values(CreatmartixTestParams { 192, 32, 64, creat_martix_AscendC<half>, 2 },
    CreatmartixTestParams { 192, 32, 64, creat_martix_AscendC<int16_t>, 2 },
    CreatmartixTestParams { 192, 32, 64, creat_martix_AscendC<uint16_t>, 2 },
    CreatmartixTestParams { 192, 32, 64, creat_martix_AscendC<bfloat16_t>, 2 },
    CreatmartixTestParams { 192, 32, 64, creat_martix_AscendC<int32_t>, 4 },
    CreatmartixTestParams { 192, 32, 64, creat_martix_AscendC<uint32_t>, 4 },
    CreatmartixTestParams { 192, 32, 64, creat_martix_AscendC<float>, 4 }
    ));

INSTANTIATE_TEST_CASE_P(SetFillTest, CreatmartixTestsuite,
    ::testing::Values(CreatmartixTestParams { 192, 32, 64, FillAscendC<half>, 2 },
    CreatmartixTestParams { 192, 32, 64, FillAscendC<int16_t>, 2 },
    CreatmartixTestParams { 192, 32, 64, FillAscendC<uint16_t>, 2 },
    CreatmartixTestParams { 192, 32, 64, FillAscendC<bfloat16_t>, 2 },
    CreatmartixTestParams { 192, 32, 64, FillAscendC<int32_t>, 4 },
    CreatmartixTestParams { 192, 32, 64, FillAscendC<uint32_t>, 4 },
    CreatmartixTestParams { 192, 32, 64, FillAscendC<float>, 4 }
    ));
