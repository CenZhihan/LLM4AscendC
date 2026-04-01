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
#include "kernel_utils.h"
#include "mockcpp/mockcpp.hpp"

using namespace AscendC;
namespace AscendC {
__aicore__ inline uint32_t GetScalarBitcodeMm(float scalarValue)
{
    union ScalarBitcode {
    __aicore__ ScalarBitcode() {}
        float input;
        uint32_t output;
    } data;

    data.input = scalarValue;

    return data.output;
}
template <typename DstT, typename Src0T, typename Src1T, typename Src2T> class KernelLoadData {
public:
    __aicore__ inline KernelLoadData(__gm__ uint16_t inputN)
    {
        c0 = 32 / sizeof(Src0T);
        cout = inputN;
        coutBlocks = (cout + 16 - 1) / 16;

        c1Fmap = channelSizeFmap / c0;
        hoFmap = hFmap - dilationH * (khFmap - 1);
        woFmap = wFmap - dilationW * (kwFmap - 1);
        howoFmap = hoFmap * woFmap;
        howoRoundFmap = ((howoFmap + 16 - 1) / 16) * 16;
        featureMapA1Size = c1Fmap * hFmap * wFmap * c0;
        featureMapA2Size = howoRoundFmap * (c1Fmap * khFmap * kwFmap * c0);
        dstCO1Size = coutBlocks * howoRoundFmap * 16;

        c1Weight = channelSizeWeight / c0;
        hoWeight = hWeight - dilationH * (khWeight - 1);
        woWeight = wWeight - dilationW * (kwWeight - 1);
        howoWeight = hoWeight * woWeight;
        howoRoundWeight = ((howoWeight + 16 - 1) / 16) * 16;
        weightA1Size = c1Weight * hWeight * wWeight * c0;
        weightB2Size = howoRoundWeight * (c1Weight * khWeight * kwWeight * c0);
    }
    __aicore__ inline void Init(__gm__ uint8_t* fmGm, __gm__ uint8_t* weightGm, __gm__ uint8_t* dstGm)
    {
        fmGlobal.SetGlobalBuffer((__gm__ Src0T*)fmGm);
        weightGlobal.SetGlobalBuffer((__gm__ Src1T*)weightGm);
        dstGlobal.SetGlobalBuffer((__gm__ DstT*)dstGm);

        pipe.InitBuffer(inQueueFmA1, 1, featureMapA1Size * sizeof(Src0T));
        pipe.InitBuffer(inQueueFmA2, 1, featureMapA2Size * sizeof(Src0T));
        pipe.InitBuffer(inQueueWeightB1, 1, weightA1Size * sizeof(Src1T));
        pipe.InitBuffer(inQueueWeightB2, 1, weightB2Size * sizeof(Src1T));
        pipe.InitBuffer(inQueueFb, 1, weightB2Size * sizeof(Src1T));
        pipe.InitBuffer(outQueueCO1, 1, dstCO1Size * sizeof(Src2T));
    }
    __aicore__ inline void Process(__gm__ uint16_t inputM, __gm__ uint16_t inputN, __gm__ uint16_t inputK, __gm__ bool doLoadData3dv2ProIn)
    {
        m = inputM;
        n = inputN;
        k = inputK;
        doLoadData3dv2Pro = doLoadData3dv2ProIn;
        CopyIn();
        Split();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<Src0T> featureMapA1 = inQueueFmA1.AllocTensor<Src0T>();
        LocalTensor<Src1T> weightB1 = inQueueWeightB1.AllocTensor<Src1T>();
        // gm->L1
        DataCopy(featureMapA1, fmGlobal, { 1, static_cast<uint16_t>(featureMapA1Size * sizeof(Src0T) / 32), 0, 0 });
        // gm->L1
        DataCopy(weightB1, weightGlobal, { 1, static_cast<uint16_t>(weightA1Size * sizeof(Src1T) / 32), 0, 0 });
        inQueueFmA1.EnQue(featureMapA1);
        inQueueWeightB1.EnQue(weightB1);
    }
    __aicore__ inline void Split()
    {
        LocalTensor<Src0T> featureMapA2 = inQueueFmA2.AllocTensor<Src0T>();
        LocalTensor<Src1T> weightB2 = inQueueWeightB2.AllocTensor<Src1T>();
        LocalTensor<Src1T> Fbuffer = inQueueFb.AllocTensor<Src1T>();
        LocalTensor<Src0T> featureMapA1 = inQueueFmA1.DeQue<Src0T>();
        LocalTensor<Src1T> weightB1 = inQueueWeightB1.DeQue<Src1T>();
        uint8_t padList[PAD_SIZE] = {0, 0, 0, 0};
        // L1->L0A
        // loaddata3dv2
        uint64_t repeat = 0;
        if (sizeof(Src0T) == sizeof(half)) {
            repeat = ConstCeil(k, BLOCK_CUBE * 16)  << 16 ;
            repeat |= 1;
            create_ca_matrix((__ca__ Src0T*)featureMapA2.GetPhyAddr(), repeat, static_cast<half>(1));
        } else if (sizeof(Src0T) == sizeof(float)) {
            repeat = ConstCeil(k, BLOCK_CUBE * 8)  << 16 ;
            repeat |= 1;
            create_ca_matrix((__ca__ Src0T*)featureMapA2.GetPhyAddr(), repeat, GetScalarBitcodeMm(static_cast<float>(1)));
        }

        if (doLoadData3dv2Pro) {
            SetLoadDataBoundary(static_cast<uint64_t>(0));
            SetLoadDataPaddingValue(static_cast<uint64_t>(0));
            LoadData3DParamsV2Pro loadData3DV2;
            loadData3DV2.channelSize = channelSizeFmap;
            loadData3DV2.extConfig = (static_cast<uint64_t>(0) << LOAD_M_START_POSITION) |
                (static_cast<uint64_t>(0) << LOAD_K_START_POSITION) | (static_cast<uint64_t>(192) << LOAD_M_EXTENSION) | static_cast<uint64_t>(32);
            loadData3DV2.filterConfig = (static_cast<uint64_t>(1) << LOAD_DILATION_FILTER_H) |
                (static_cast<uint64_t>(1) << LOAD_DILATION_FILTER_W) | (static_cast<uint64_t>(1) << LOAD_FILTER_H) |
                (static_cast<uint64_t>(1) << LOAD_FILTER_W) | (static_cast<uint64_t>(1) << LOAD_STRIDE_H) | static_cast<uint64_t>(1);
            LoadData<Src0T>(featureMapA2, featureMapA1, loadData3DV2);

            // failed case
            TBuf<TPosition::C2> tbufFailed;
            pipe.InitBuffer(tbufFailed, featureMapA1Size * sizeof(Src0T));
            LocalTensor<Src0T> fmLocal = tbufFailed.Get<Src0T>();
            MOCKER(raise).stubs().will(returnValue(int(0)));
            AscendC::LoadData<Src0T>(fmLocal, featureMapA1, loadData3DV2);
        } else {
            LoadData(featureMapA2, featureMapA1,
                { padList, hFmap, wFmap, channelSizeFmap, 32, 192, 0, 0, 1, 1, 1, 1, 1, 1, false, false, 0 });
            LoadData<bfloat16_t, IS_RESER_LOAD3D_DEFAULT_CONFIG>(
                featureMapA2.template ReinterpretCast<bfloat16_t>(),
                featureMapA1.template ReinterpretCast<bfloat16_t>(),
                { padList, hFmap, wFmap, channelSizeFmap, 32, 192, 0, 0, 1, 1, 1, 1, 1, 1, false, false, 0 });
            LoadData<bfloat16_t>(
                featureMapA2.template ReinterpretCast<bfloat16_t>(),
                featureMapA1.template ReinterpretCast<bfloat16_t>(),
                { padList, hFmap, wFmap, channelSizeFmap, 32, 192, 0, 0, 1, 1, 1, 1, 1, 1, false, false, 0 });
            LoadData<IS_RESER_LOAD3D_DEFAULT_CONFIG>(
                featureMapA2.template ReinterpretCast<bfloat16_t>(),
                featureMapA1.template ReinterpretCast<bfloat16_t>(),
                { padList, hFmap, wFmap, channelSizeFmap, 32, 192, 0, 0, 1, 1, 1, 1, 1, 1, false, false, 0 });
        }
        // //L1->L0B
        // loaddata3dv2
        if (doLoadData3dv2Pro) {
            SetLoadDataBoundary(static_cast<uint64_t>(0));
            SetLoadDataPaddingValue(static_cast<uint64_t>(0));
            LoadData3DParamsV2Pro loadData3DToBV2;
            loadData3DToBV2.channelSize = channelSizeWeight;
            loadData3DToBV2.extConfig = (static_cast<uint64_t>(0) << LOAD_M_START_POSITION) |
                (static_cast<uint64_t>(0) << LOAD_K_START_POSITION) | (static_cast<uint64_t>(32) << LOAD_M_EXTENSION) | static_cast<uint64_t>(64);
            loadData3DToBV2.filterConfig = (static_cast<uint64_t>(1) << LOAD_DILATION_FILTER_H) |
                (static_cast<uint64_t>(1) << LOAD_DILATION_FILTER_W) | (static_cast<uint64_t>(2) << LOAD_FILTER_H) |
                (static_cast<uint64_t>(2) << LOAD_FILTER_W) | (static_cast<uint64_t>(1) << LOAD_STRIDE_H) | static_cast<uint64_t>(1);
            LoadData<Src1T>(weightB2, weightB1, loadData3DToBV2);
        } else {
            LoadData(weightB2, weightB1,
                { padList, hWeight, wWeight, channelSizeWeight, 64, 32, 0, 0, 1, 1, 2, 2, 1, 1, false, false, 0 });
            LoadData<bfloat16_t, IS_RESER_LOAD3D_DEFAULT_CONFIG>(
                weightB2.template ReinterpretCast<bfloat16_t>(),
                weightB1.template ReinterpretCast<bfloat16_t>(),
                { padList, hWeight, wWeight, channelSizeWeight, 64, 32, 0, 0, 1, 1, 2, 2, 1, 1, false, false, 0 });
            LoadData<IS_RESER_LOAD3D_DEFAULT_CONFIG>(
                weightB2.template ReinterpretCast<bfloat16_t>(),
                weightB1.template ReinterpretCast<bfloat16_t>(),
                { padList, hWeight, wWeight, channelSizeWeight, 64, 32, 0, 0, 1, 1, 2, 2, 1, 1, false, false, 0 });
            LoadData<bfloat16_t>(
                weightB2.template ReinterpretCast<bfloat16_t>(),
                weightB1.template ReinterpretCast<bfloat16_t>(),
                { padList, hWeight, wWeight, channelSizeWeight, 64, 32, 0, 0, 1, 1, 2, 2, 1, 1, false, false, 0 });
        }
        // datacopy B1->FB
        DataCopyParams dataCopyToFbParams = {1, weightB2Size * sizeof(Src1T) / 128, 0, 0};
        DataCopy(Fbuffer, weightB1, dataCopyToFbParams);

        inQueueFmA2.EnQue<Src0T>(featureMapA2);
        inQueueWeightB2.EnQue<Src1T>(weightB2);
        inQueueFmA1.FreeTensor(featureMapA1);
        inQueueWeightB1.FreeTensor(weightB1);
        inQueueFb.FreeTensor(Fbuffer);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<Src0T> featureMapA2 = inQueueFmA2.DeQue<Src0T>();
        LocalTensor<Src1T> weightB2 = inQueueWeightB2.DeQue<Src1T>();
        LocalTensor<Src2T> dstCO1 = outQueueCO1.AllocTensor<Src2T>();
        // L0A*L0B->L0C
        // mad
        Mmad(dstCO1, featureMapA2, weightB2, { m, n, k, 0, false, true });
        outQueueCO1.EnQue<Src2T>(dstCO1);
        inQueueFmA2.FreeTensor(featureMapA2);
        inQueueWeightB2.FreeTensor(weightB2);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<Src2T> dstCO1 = outQueueCO1.DeQue<Src2T>();
        // L0C->gm
        uint16_t cburstNum = n / BLOCK_CUBE;
        uint16_t burstLen = m * BLOCK_CUBE * sizeof(Src2T) / ONE_BLK_SIZE;
        FixpipeParams<Src2T> fixpipeParams(cburstNum, burstLen, 0, 0);
        fixpipeParams.quantParams = { QuantMode_t::F322F16 };
        fixpipeParams.reluEn = true;
        fixpipeParams.nz2ndParams = { true, 1, 0, 0, n };

        Fixpipe(dstGlobal, dstCO1, fixpipeParams);
        outQueueCO1.FreeTensor(dstCO1);
    }

private:
    TPipe pipe;
    TQue<TPosition::A1, 1> inQueueFmA1;
    TQue<TPosition::A2, 1> inQueueFmA2;
    TQue<TPosition::B1, 1> inQueueWeightB1;
    TQue<TPosition::B2, 1> inQueueWeightB2;
    TQue<TPosition::C2PIPE2GM, 1> inQueueFb;
    TQue<TPosition::CO1, 1> outQueueCO1;

    GlobalTensor<Src0T> fmGlobal;
    GlobalTensor<Src1T> weightGlobal;
    GlobalTensor<DstT> dstGlobal;

    uint16_t channelSizeFmap = 36, channelSizeWeight = 20;
    uint16_t c0, cout, coutBlocks;
    uint16_t c1Fmap, c1Weight;
    uint16_t hFmap = 12, wFmap = 16, hWeight = 5, wWeight = 9;
    uint8_t khFmap = 1, kwFmap = 1, khWeight = 2, kwWeight = 2;
    uint8_t dilationH = 1, dilationW = 1;
    uint16_t hoFmap, woFmap, howoFmap, howoRoundFmap, hoWeight, woWeight, howoWeight, howoRoundWeight;
    uint16_t featureMapA1Size, weightA1Size, featureMapA2Size, weightB2Size, dstCO1Size;
    uint16_t m, k, n;
    bool doLoadData3dv2Pro;
};
} // namespace AscendC

template <typename DstT, typename Src0T, typename Src1T, typename Src2T>
__global__ __aicore__ void LoadDataAscendC(__gm__ uint8_t* fmGm, __gm__ uint8_t* weightGm, __gm__ uint8_t* dstGm,
    __gm__ uint16_t m, __gm__ uint16_t n, __gm__ uint16_t k, __gm__ bool doLoadData3dv2Pro)
{
    AscendC::KernelLoadData<DstT, Src0T, Src1T, Src2T> op(n);
    op.Init(fmGm, weightGm, dstGm);
    op.Process(m, n, k, doLoadData3dv2Pro);
}

struct LoadDataTestParams {
    uint16_t m;
    uint16_t n;
    uint16_t k;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint16_t, uint16_t, uint16_t, bool);
    uint8_t sizeofDst;
    uint8_t sizeofSrc0;
    uint8_t sizeofSrc1;
    uint8_t sizeofSrc2;
    bool doLoadData3dv2Pro;
};

class LoadDataTestsuite : public testing::Test, public testing::WithParamInterface<LoadDataTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(SetLoadDataTest1, LoadDataTestsuite,
    ::testing::Values(LoadDataTestParams { 192, 32, 64, LoadDataAscendC<half, half, half, float>, 2, 2, 2, 4, false },
    LoadDataTestParams { 192, 32, 64, LoadDataAscendC<half, half, half, float>, 2, 2, 2, 4, true }));

TEST_P(LoadDataTestsuite, LoadDataTestCase)
{
    auto param = GetParam();
    uint8_t fmGm[param.m * param.k * param.sizeofSrc0];
    uint8_t weightGm[param.k * param.n * param.sizeofSrc1];
    uint8_t dstGm[param.m * param.n * param.sizeofDst];
    param.cal_func(fmGm, weightGm, dstGm, param.m, param.n, param.k, param.doLoadData3dv2Pro);
}

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

template <class AType, class BType, class CType>
class E2eMain {
    using SrcT = typename AType::T;
    using DstT = typename CType::T;
public:
    __aicore__ inline E2eMain() {}
    __aicore__ inline void Init(TPipe* tpipe, int32_t m, int32_t n, int32_t k)
    {
        pipe = tpipe;
        mLength = m;
        nLength = n;
        kLength = k;

        pipe->InitBuffer(qidA1_, 1, m * k * sizeof(SrcT));
        pipe->InitBuffer(qidB1_, 1, n * k * sizeof(SrcT));

        pipe->InitBuffer(qidA2, 1, m * k * sizeof(SrcT));
        pipe->InitBuffer(qidB2, 1, n * k * sizeof(SrcT));

        pipe->InitBuffer(qidCO1, 1, m * n * sizeof(int32_t));
    }
    __aicore__ inline void SetTensorA(const GlobalTensor<SrcT>& gm)
    {
        aGlobal1 = gm;
    }
    __aicore__ inline void SetTensorB(const GlobalTensor<SrcT>& gm)
    {
        bGlobal1 = gm; 
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
        LocalTensor<SrcT> rightMatrix = qidB1_.template AllocTensor<SrcT>();
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

    __aicore__ inline void Load2DA1ToL0A()
    {
        auto leftMatrix = qidA1_.template DeQue<SrcT>();
        LocalTensor<SrcT> a2 = qidA2.AllocTensor<SrcT>();
        LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = ConstCeil(kLength, 32);
        loadDataParams.dstGap = 0;
        loadDataParams.srcStride = mLength / 16;
        loadDataParams.ifTranspose = false;
        int dstOffset = kLength / 32 * 512;
        int srcOffset = 512;
        for (int i = 0; i < mLength / 16; ++i) {
            LoadData(a2[i * dstOffset], leftMatrix[i * srcOffset], loadDataParams);
        }
        qidA2.EnQue(a2);
        qidA1_.FreeTensor(leftMatrix);
    }

    __aicore__ inline void Load2DA1ToL0B()
    {
        auto rightMatrix = qidB1_.template DeQue<SrcT>();
        LocalTensor<SrcT> b2 = qidB2.AllocTensor<SrcT>();
        uint32_t dstOffset = nLength / 32 * 1024;
        uint32_t srcOffset = 1024;
        for (int i = 0; i < kLength / 32; ++i) {
            load_cbuf_to_cb_transpose((__cb__ int8_t*)(b2[i * dstOffset].GetPhyAddr()),
                (__cbuf__ int8_t*)(rightMatrix[i * srcOffset].GetPhyAddr()), 0, nLength / 32, kLength / 32, 1, inc, 0);
        }
        qidB2.EnQue(b2);
        qidB1_.FreeTensor(rightMatrix);
    }

    __aicore__ inline void Compute()
    {
        MmadParams mmadParams;
        mmadParams.m = mLength;
        mmadParams.n = nLength;
        mmadParams.k = kLength;
        mmadParams.cmatrixInitVal = !enBias;

        auto co1Local = qidCO1.AllocTensor<int32_t>();
        auto a2 = qidA2.DeQue<SrcT>();
        auto b2 = qidB2.DeQue<SrcT>();
        pipe_barrier(PIPE_ALL);
        Mmad(co1Local, a2, b2, mmadParams);
        qidA2.FreeTensor(a2);
        qidB2.FreeTensor(b2);
        qidCO1.EnQue(co1Local);
    }

     __aicore__ inline void CopyL0CToGm(const GlobalTensor<DstT>& gm)
    {
        auto co1Local = qidCO1.DeQue<int32_t>();
        FixpipeParams<int32_t> fixpipeParams(nLength / BLOCK_CUBE,
            static_cast<uint16_t>(mLength * BLOCK_CUBE * sizeof(float) / ONE_BLK_SIZE),
            0, static_cast<uint16_t>(nLength));
        fixpipeParams.nz2ndParams = { true, 1, 0, 0, static_cast<uint16_t>(nLength) };
        if (IsSameType<DstT, half>::value) {
            fixpipeParams.quantParams = { QuantMode_t::F322F16 };
        } else if (IsSameType<DstT, bfloat16_t>::value) {
            fixpipeParams.quantParams = { QuantMode_t::F322BF16 };
        }
        Fixpipe(gm, co1Local, fixpipeParams);

        qidCO1.FreeTensor(co1Local);
    }

    __aicore__ inline void IterateAll(const GlobalTensor<DstT>& gm)
    {
        CopyGmToA1Nd2Nz();
        CopyGmToB1Nd2Nz();
        Load2DA1ToL0A();
        Load2DA1ToL0B();

        Compute();
        CopyL0CToGm(gm);
    }

private:
    TPipe* pipe = nullptr;
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

    LocalTensor<SrcT> leftMatrix;
    LocalTensor<SrcT> rightMatrix;
    GlobalTensor<SrcT> aGlobal1;
    GlobalTensor<SrcT> bGlobal1;
};

template <class AType, class BType, class CType>
__aicore__ inline void E2eKernel(__gm__ uint8_t* aGM, __gm__ uint8_t* bGM, __gm__ uint8_t* cGM,
    int32_t m, int32_t n, int32_t k)
{
    if (g_coreType == AIV) {
        return;
    }

    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using C_T = typename CType::T;

    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(aGM), m * k);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(bGM), k * n);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T*>(cGM), m * n);

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];

    set_atomic_none();
    TPipe que;
    E2eMain<AType, BType, CType> ins;
    ins.Init(&que, m, n, k);
    ins.SetTensorA(gmA);
    ins.SetTensorB(gmB);

    ins.IterateAll(gmC);
    set_atomic_none();
    return;
}

__global__ __aicore__ void kernelE2e(__gm__ uint8_t* aGm, __gm__ uint8_t* bGm, __gm__ uint8_t* cGm,
    __gm__ uint16_t m, __gm__ uint16_t n, __gm__ uint16_t k)
{
    typedef InputInfo<AscendC::TPosition::GM, CubeFormat::ND, int8_t> aType;
    typedef InputInfo<AscendC::TPosition::GM, CubeFormat::ND, int8_t> bType;
    typedef InputInfo<AscendC::TPosition::GM, CubeFormat::ND, int32_t> cType;
    E2eKernel<aType, bType, cType>(aGm, bGm, cGm, static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
}

struct E2eParams {
    uint16_t m;
    uint16_t n;
    uint16_t k;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint16_t, uint16_t, uint16_t);
};

class E2eTestsuite : public testing::Test, public testing::WithParamInterface<E2eParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(E2eCase, E2eTestsuite,
    ::testing::Values(E2eParams { 16, 32, 32, kernelE2e}));

TEST_P(E2eTestsuite, E2eCase1)
{
    auto param = GetParam();
    uint8_t aGm[param.m * param.k];
    uint8_t bGm[param.n * param.k];
    uint8_t cGm[param.m * param.n * 4] = {0};
    param.cal_func(aGm, bGm, cGm, param.m, param.n, param.k);
    for (int32_t i = 0; i <param.m * param.n * 4; i++) {
        EXPECT_EQ(cGm[i], 0x00);
    }
}

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
        featureMapA2Size = howoRoundFmap * (c1Fmap * khFmap * kwFmap * c0);
        weightA1Size = c1Weight * hWeight * wWeight * c0;
        weightB2Size = howoRoundWeight * (c1Weight * khWeight * kwWeight * c0);

        pipe.InitBuffer(inQueueFmA1, 1, featureMapA1Size * sizeof(Src0T));
        pipe.InitBuffer(inQueueFmA2, 1, featureMapA2Size * sizeof(Src0T));
        pipe.InitBuffer(inQueueWeightB1, 1, weightA1Size * sizeof(Src0T));
        pipe.InitBuffer(inQueueWeightB2, 1, weightB2Size * sizeof(Src0T));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
    }

private:
    __aicore__ inline void CopyIn()
    {
        if constexpr (AscendC::IsSameType<Src0T, int16_t>::value) {
            SetLoadDataPaddingValue(static_cast<int16_t>(0));
        }
        AscendC::LocalTensor<Src0T> featureMapA1 = inQueueFmA1.AllocTensor<Src0T>();
        AscendC::LocalTensor<Src0T> featureMapA2 = inQueueFmA2.AllocTensor<Src0T>();
        AscendC::LocalTensor<Src0T> weightB2 = inQueueWeightB2.AllocTensor<Src0T>();
        // set repeat
        set_l3d_rpt(0);
        if constexpr (UseFill) {
            // init l1
            Fill<Src0T>(featureMapA1,
                { 1, static_cast<uint16_t>(featureMapA1Size * sizeof(Src0T) / 32), 0, static_cast<Src0T>(1) });
            // init l0a
            Fill<Src0T>(featureMapA2,
                { 1, static_cast<uint16_t>(1), 0, static_cast<Src0T>(1) });
            // init l0b
            Fill<Src0T>(weightB2,
                { 1, static_cast<uint16_t>(1), 0, static_cast<Src0T>(1) });
        }else{
            // init l1
            InitConstValue<Src0T>(featureMapA1,
                { 1, static_cast<uint16_t>(featureMapA1Size * sizeof(Src0T) / 32), 0, static_cast<Src0T>(1) });
            // init l0a
            InitConstValue<Src0T>(featureMapA2,
                { 1, static_cast<uint16_t>(1), 0, static_cast<Src0T>(1) });
            // init l0b
            InitConstValue<Src0T>(weightB2,
                { 1, static_cast<uint16_t>(1), 0, static_cast<Src0T>(1) });
        }

        inQueueFmA1.FreeTensor(featureMapA1);
        inQueueFmA2.FreeTensor(featureMapA2);
        inQueueWeightB2.FreeTensor(weightB2);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::A1, 1> inQueueFmA1;
    AscendC::TQue<TPosition::A2, 1> inQueueFmA2;
    AscendC::TQue<TPosition::B1, 1> inQueueWeightB1;
    AscendC::TQue<TPosition::B2, 1> inQueueWeightB2;
    AscendC::TQue<TPosition::CO1, 1> outQueueCO1;
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
__global__ __aicore__ void CreatMartixAscendC()
{
    AscendC::KernelCreatMartix<Src0T, false> op;
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
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(SetCreatmartixTest, CreatmartixTestsuite,
    ::testing::Values(CreatmartixTestParams { 192, 32, 64, CreatMartixAscendC<half>, 2 },
    CreatmartixTestParams { 192, 32, 64, CreatMartixAscendC<int16_t>, 2 },
    CreatmartixTestParams { 192, 32, 64, CreatMartixAscendC<uint16_t>, 2 },
    CreatmartixTestParams { 192, 32, 64, CreatMartixAscendC<bfloat16_t>, 2 },
    CreatmartixTestParams { 192, 32, 64, CreatMartixAscendC<int32_t>, 4 },
    CreatmartixTestParams { 192, 32, 64, CreatMartixAscendC<uint32_t>, 4 },
    CreatmartixTestParams { 192, 32, 64, CreatMartixAscendC<float>, 4 }
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

TEST_P(CreatmartixTestsuite, CreatmartixTest)
{
    auto param = GetParam();
    param.cal_func();
}
