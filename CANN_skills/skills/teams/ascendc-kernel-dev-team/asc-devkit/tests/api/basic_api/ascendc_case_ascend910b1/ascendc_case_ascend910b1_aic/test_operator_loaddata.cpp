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

using  namespace AscendC;

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
        pipe.InitBuffer(outQueueCO1, 1, dstCO1Size * sizeof(Src2T));
    }
    __aicore__ inline void Process(__gm__ uint16_t inputM, __gm__ uint16_t inputN, __gm__ uint16_t inputK)
    {
        m = inputM;
        n = inputN;
        k = inputK;
        CopyIn();
        Split();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<TensorTrait<Src0T>> featureMapA1 = inQueueFmA1.AllocTensor<TensorTrait<Src0T>>();
        AscendC::LocalTensor<TensorTrait<Src1T>> weightB1 = inQueueWeightB1.AllocTensor<TensorTrait<Src1T>>();
        // gm->L1
        AscendC::DataCopy(featureMapA1, fmGlobal, { 1, static_cast<uint16_t>(featureMapA1Size * sizeof(Src0T) / 32), 0, 0 });
        // gm->L1
        AscendC::DataCopy(weightB1, weightGlobal, { 1, static_cast<uint16_t>(weightA1Size * sizeof(Src1T) / 32), 0, 0 });
        inQueueFmA1.EnQue(featureMapA1);
        inQueueWeightB1.EnQue(weightB1);
    }
    __aicore__ inline void Split()
    {
        AscendC::LocalTensor<TensorTrait<Src0T>> featureMapA2 = inQueueFmA2.AllocTensor<TensorTrait<Src0T>>();
        AscendC::LocalTensor<TensorTrait<Src1T>> weightB2 = inQueueWeightB2.AllocTensor<TensorTrait<Src1T>>();
        AscendC::LocalTensor<TensorTrait<Src0T>> featureMapA1 = inQueueFmA1.DeQue<TensorTrait<Src0T>>();
        AscendC::LocalTensor<TensorTrait<Src1T>> weightB1 = inQueueWeightB1.DeQue<TensorTrait<Src1T>>();
        uint8_t padList[PAD_SIZE] = {0, 0, 0, 0};
        SetLoadDataBoundary(static_cast<uint32_t>(0));
        SetLoadDataPaddingValue(static_cast<DstT>(0));
        // L1->L0A
        LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.srcStride = 1;
        loadDataParams.addrMode = 0;

        loadDataParams.repeatTimes = 1;
        loadDataParams.dstGap = 16;
        loadDataParams.dstFracGap = 0;
        LoadDataWithTranspose<TensorTrait<int8_t>>(featureMapA2.template ReinterpretCast<TensorTrait<int8_t>>(),
            featureMapA1.template ReinterpretCast<TensorTrait<int8_t>>(), loadDataParams);
        // loaddata3dv2
        uint64_t repeat = 0;
        if (sizeof(Src0T) == sizeof(half)) {
            repeat = AscendC::ConstCeil(k, AscendC::BLOCK_CUBE * 16)  << 16 ;
            repeat |= 1;
            create_ca_matrix((__ca__ Src0T*)featureMapA2.GetPhyAddr(), repeat, static_cast<half>(1));
        } else if (sizeof(Src0T) == sizeof(float)) {
            repeat = AscendC::ConstCeil(k, AscendC::BLOCK_CUBE * 8)  << 16 ;
            repeat |= 1;
            create_ca_matrix((__ca__ Src0T*)featureMapA2.GetPhyAddr(), repeat, GetScalarBitcodeMm(static_cast<float>(1)));
        }
        LoadData(featureMapA2, featureMapA1,
            { padList, hFmap, wFmap, channelSizeFmap, 32, 192, 0, 0, 1, 1, 1, 1, 1, 1, false, false, 0 });
        LoadData<TensorTrait<bfloat16_t>, IS_RESER_LOAD3D_DEFAULT_CONFIG>(
            featureMapA2.template ReinterpretCast<TensorTrait<bfloat16_t>>(),
            featureMapA1.template ReinterpretCast<TensorTrait<bfloat16_t>>(),
            { padList, hFmap, wFmap, channelSizeFmap, 32, 192, 0, 0, 1, 1, 1, 1, 1, 1, false, false, 0 });
        LoadData<TensorTrait<bfloat16_t>>(
            featureMapA2.template ReinterpretCast<TensorTrait<bfloat16_t>>(),
            featureMapA1.template ReinterpretCast<TensorTrait<bfloat16_t>>(),
            { padList, hFmap, wFmap, channelSizeFmap, 32, 192, 0, 0, 1, 1, 1, 1, 1, 1, false, false, 0 });
        LoadData<TensorTrait<bfloat16_t>, IS_RESER_LOAD3D_DEFAULT_CONFIG>(
            featureMapA2.template ReinterpretCast<TensorTrait<bfloat16_t>>(),
            featureMapA1.template ReinterpretCast<TensorTrait<bfloat16_t>>(),
            { padList, hFmap, wFmap, channelSizeFmap, 32, 192, 0, 0, 1, 1, 1, 1, 1, 1, false, false, 0 });
        // //L1->L0B
        // loaddata3dv2
        LoadData(weightB2, weightB1,
            { padList, hWeight, wWeight, channelSizeWeight, 64, 32, 0, 0, 1, 1, 2, 2, 1, 1, false, false, 0 });
        LoadData<TensorTrait<bfloat16_t>, IS_RESER_LOAD3D_DEFAULT_CONFIG>(
            weightB2.template ReinterpretCast<TensorTrait<bfloat16_t>>(), weightB1.template ReinterpretCast<TensorTrait<bfloat16_t>>(),
            { padList, hWeight, wWeight, channelSizeWeight, 64, 32, 0, 0, 1, 1, 2, 2, 1, 1, false, false, 0 });
        LoadData<TensorTrait<bfloat16_t>>(
            weightB2.template ReinterpretCast<TensorTrait<bfloat16_t>>(), weightB1.template ReinterpretCast<TensorTrait<bfloat16_t>>(),
            { padList, hWeight, wWeight, channelSizeWeight, 64, 32, 0, 0, 1, 1, 2, 2, 1, 1, false, false, 0 });
        LoadData<TensorTrait<bfloat16_t>, IS_RESER_LOAD3D_DEFAULT_CONFIG>(
            weightB2.template ReinterpretCast<TensorTrait<bfloat16_t>>(), weightB1.template ReinterpretCast<TensorTrait<bfloat16_t>>(),
            { padList, hWeight, wWeight, channelSizeWeight, 64, 32, 0, 0, 1, 1, 2, 2, 1, 1, false, false, 0 });
        // loaddata3dv2 pro
        AscendC::LoadData3DParamsV2Pro loadData3DV2;
        loadData3DV2.channelSize = 32;
        loadData3DV2.extConfig = (static_cast<uint64_t>(0) << AscendC::LOAD_M_START_POSITION) |
                                 (static_cast<uint64_t>(0) << AscendC::LOAD_K_START_POSITION) |
                                 (static_cast<uint64_t>(16) << AscendC::LOAD_M_EXTENSION) | static_cast<uint64_t>(128);
        loadData3DV2.filterConfig = (static_cast<uint64_t>(2) << AscendC::LOAD_DILATION_FILTER_H) |
                                    (static_cast<uint64_t>(2) << AscendC::LOAD_DILATION_FILTER_W) |
                                    (static_cast<uint64_t>(1) << AscendC::LOAD_FILTER_H) | (static_cast<uint64_t>(1) << AscendC::LOAD_FILTER_W) |
                                    (static_cast<uint64_t>(1) << AscendC::LOAD_STRIDE_H) | static_cast<uint64_t>(1);
        AscendC::LoadData<AscendC::TensorTrait<bfloat16_t>>(weightB2.template ReinterpretCast<TensorTrait<bfloat16_t>>(),
            weightB1.template ReinterpretCast<TensorTrait<bfloat16_t>>(), loadData3DV2);
        inQueueFmA2.EnQue(featureMapA2);
        inQueueWeightB2.EnQue(weightB2);
        inQueueFmA1.FreeTensor(featureMapA1);
        inQueueWeightB1.FreeTensor(weightB1);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<TensorTrait<Src0T>> featureMapA2 = inQueueFmA2.DeQue<TensorTrait<Src0T>>();
        AscendC::LocalTensor<TensorTrait<Src1T>> weightB2 = inQueueWeightB2.DeQue<TensorTrait<Src1T>>();
        AscendC::LocalTensor<TensorTrait<Src2T>> dstCO1 = outQueueCO1.AllocTensor<TensorTrait<Src2T>>();
        // L0A*L0B->L0C
        // mad
        Mmad(dstCO1, featureMapA2, weightB2, { m, n, k, 0, false, true });
        outQueueCO1.EnQue(dstCO1);
        inQueueFmA2.FreeTensor(featureMapA2);
        inQueueWeightB2.FreeTensor(weightB2);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<TensorTrait<Src2T>> dstCO1 = outQueueCO1.DeQue<TensorTrait<Src2T>>();
        // L0C->gm
        uint16_t cburstNum = n / AscendC::BLOCK_CUBE;
        uint16_t burstLen = m * AscendC::BLOCK_CUBE * sizeof(Src2T) / AscendC::ONE_BLK_SIZE;
        FixpipeParams<Src2T> fixpipeParams(cburstNum, burstLen, 0, 0);
        fixpipeParams.quantParams = { QuantMode_t::F322F16 };
        fixpipeParams.reluEn = true;
        fixpipeParams.nz2ndParams = { true, 1, 0, 0, n };

        AscendC::Fixpipe(dstGlobal, dstCO1, fixpipeParams);
        outQueueCO1.FreeTensor(dstCO1);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::A1, 1> inQueueFmA1;
    AscendC::TQue<TPosition::A2, 1> inQueueFmA2;
    AscendC::TQue<TPosition::B1, 1> inQueueWeightB1;
    AscendC::TQue<TPosition::B2, 1> inQueueWeightB2;
    AscendC::TQue<TPosition::CO1, 1> outQueueCO1;

    AscendC::GlobalTensor<TensorTrait<Src0T>> fmGlobal;
    AscendC::GlobalTensor<TensorTrait<Src1T>> weightGlobal;
    AscendC::GlobalTensor<TensorTrait<DstT>> dstGlobal;

    uint16_t channelSizeFmap = 36, channelSizeWeight = 20;
    uint16_t c0, cout, coutBlocks;
    uint16_t c1Fmap, c1Weight;
    uint16_t hFmap = 12, wFmap = 16, hWeight = 5, wWeight = 9;
    uint8_t khFmap = 1, kwFmap = 1, khWeight = 2, kwWeight = 2;
    uint8_t dilationH = 1, dilationW = 1;
    uint16_t hoFmap, woFmap, howoFmap, howoRoundFmap, hoWeight, woWeight, howoWeight, howoRoundWeight;
    uint16_t featureMapA1Size, weightA1Size, featureMapA2Size, weightB2Size, dstCO1Size;
    uint16_t m, k, n;
};
} // namespace AscendC

template <typename DstT, typename Src0T, typename Src1T, typename Src2T>
__global__ __aicore__ void LoadDataAscendC(__gm__ uint8_t* fmGm, __gm__ uint8_t* weightGm, __gm__ uint8_t* dstGm,
    __gm__ uint16_t m, __gm__ uint16_t n, __gm__ uint16_t k)
{
    AscendC::KernelLoadData<DstT, Src0T, Src1T, Src2T> op(n);
    op.Init(fmGm, weightGm, dstGm);
    op.Process(m, n, k);
}

struct LoadDataTestParams {
    uint16_t m;
    uint16_t n;
    uint16_t k;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint16_t, uint16_t, uint16_t);
    uint8_t sizeofDst;
    uint8_t sizeofSrc0;
    uint8_t sizeofSrc1;
    uint8_t sizeofSrc2;
};

class LoadDataTestsuite : public testing::Test, public testing::WithParamInterface<LoadDataTestParams> {
protected:
    void SetUp()
    {
        g_coreType = AscendC::AIC_TYPE;
    }
    void TearDown()
    {
        AscendC::CheckSyncState();
        g_coreType = AscendC::MIX_TYPE;
    }
};

INSTANTIATE_TEST_CASE_P(SetLoadDataTest1, LoadDataTestsuite,
    ::testing::Values(LoadDataTestParams { 192, 32, 64, LoadDataAscendC<half, half, half, float>, 2, 2, 2, 4 },
    LoadDataTestParams { 192, 32, 64, LoadDataAscendC<half, float, float, float>, 2, 4, 4, 4 }));

TEST_P(LoadDataTestsuite, LoadDataTestCase)
{
    auto param = GetParam();
    uint8_t fmGm[param.m * param.k * param.sizeofSrc0];
    uint8_t weightGm[param.k * param.n * param.sizeofSrc1];
    uint8_t dstGm[param.m * param.n * param.sizeofDst];
    param.cal_func(fmGm, weightGm, dstGm, param.m, param.n, param.k);
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

template <AscendC::TPosition POSITION, CubeFormat FORMAT, typename Type> struct InputInfo {
    constexpr static AscendC::TPosition pos = POSITION;
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
    __aicore__ inline void Init(AscendC::TPipe* tpipe, int32_t m, int32_t n, int32_t k)
    {
        pipe = tpipe;
        mLength = m;
        nLength = n;
        kLength = k;

        pipe->InitBuffer(qidA1, 1, m * k * sizeof(SrcT));
        pipe->InitBuffer(qidB1, 1, n * k * sizeof(SrcT));

        pipe->InitBuffer(qidA2, 1, m * k * sizeof(SrcT));
        pipe->InitBuffer(qidB2, 1, n * k * sizeof(SrcT));

        pipe->InitBuffer(qidCO1, 1, m * n * sizeof(int32_t));
    }
    __aicore__ inline void SetTensorA(const AscendC::GlobalTensor<TensorTrait<SrcT>>& gm)
    {
        aGlobal = gm;
    }
    __aicore__ inline void SetTensorB(const AscendC::GlobalTensor<TensorTrait<SrcT>>& gm)
    {
        bGlobal = gm; 
    }

    __aicore__ inline void CopyGmToA1Nd2Nz()
    {
        AscendC::LocalTensor<TensorTrait<SrcT>> leftMatrix = qidA1.template AllocTensor<TensorTrait<SrcT>>();
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = mLength;
        nd2nzParams.dValue = kLength;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = kLength;
        nd2nzParams.dstNzC0Stride = mLength;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(leftMatrix, aGlobal, nd2nzParams);
        qidA1.EnQue(leftMatrix);
    }

    __aicore__ inline void CopyGmToB1Nd2Nz()
    {
        AscendC::LocalTensor<TensorTrait<SrcT>> rightMatrix = qidB1.template AllocTensor<TensorTrait<SrcT>>();
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = kLength;
        nd2nzParams.dValue = nLength;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = nLength;
        nd2nzParams.dstNzC0Stride = kLength;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        AscendC::DataCopy(rightMatrix, bGlobal, nd2nzParams);
        qidB1.EnQue(rightMatrix);
    }

    __aicore__ inline void Load2DA1ToL0A()
    {
        auto leftMatrix = qidA1.template DeQue<TensorTrait<SrcT>>();
        AscendC::LocalTensor<TensorTrait<SrcT>> a2 = qidA2.AllocTensor<TensorTrait<SrcT>>();
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = AscendC::ConstCeil(kLength, 32);
        loadDataParams.dstGap = 0;
        loadDataParams.srcStride = mLength / 16;
        loadDataParams.ifTranspose = false;
        int dstOffset = kLength / 32 * 512;
        int srcOffset = 512;
        for (int i = 0; i < mLength / 16; ++i) {
            AscendC::LoadData(a2[i * dstOffset], leftMatrix[i * srcOffset], loadDataParams);
        }
        qidA2.EnQue(a2);
        qidA1.FreeTensor(leftMatrix);
    }

    __aicore__ inline void Load2DA1ToL0B()
    {
        auto rightMatrix = qidB1.template DeQue<TensorTrait<SrcT>>();
        AscendC::LocalTensor<TensorTrait<SrcT>> b2 = qidB2.AllocTensor<TensorTrait<SrcT>>();
        uint32_t dstOffset = nLength / 32 * 1024;
        uint32_t srcOffset = 1024;
        for (int i = 0; i < kLength / 32; ++i) {
            load_cbuf_to_cb_transpose((__cb__ int8_t*)(b2[i * dstOffset].GetPhyAddr()),
                (__cbuf__ int8_t*)(rightMatrix[i * srcOffset].GetPhyAddr()), 0, nLength / 32, kLength / 32, 1, inc, 0);
        }
        qidB2.EnQue(b2);
        qidB1.FreeTensor(rightMatrix);
    }

    __aicore__ inline void Compute()
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = mLength;
        mmadParams.n = nLength;
        mmadParams.k = kLength;
        mmadParams.cmatrixInitVal = !enBias;

        auto co1Local = qidCO1.AllocTensor<TensorTrait<int32_t>>();
        auto a2 = qidA2.DeQue<TensorTrait<SrcT>>();
        auto b2 = qidB2.DeQue<TensorTrait<SrcT>>();
        pipe_barrier(PIPE_ALL);
        AscendC::Mmad(co1Local, a2, b2, mmadParams);
        qidA2.FreeTensor(a2);
        qidB2.FreeTensor(b2);
        qidCO1.EnQue(co1Local);
    }

     __aicore__ inline void CopyL0CToGm(const AscendC::GlobalTensor<TensorTrait<DstT>>& gm)
    {
        auto co1Local = qidCO1.DeQue<TensorTrait<int32_t>>();
        AscendC::FixpipeParams<int32_t> fixpipeParams(nLength / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mLength * AscendC::BLOCK_CUBE * sizeof(float) / AscendC::ONE_BLK_SIZE),
            0, static_cast<uint16_t>(nLength));
        fixpipeParams.nz2ndParams = { true, 1, 0, 0, static_cast<uint16_t>(nLength) };
        if (AscendC::IsSameType<DstT, half>::value) {
            fixpipeParams.quantParams = { QuantMode_t::F322F16 };
        } else if (AscendC::IsSameType<DstT, bfloat16_t>::value) {
            fixpipeParams.quantParams = { QuantMode_t::F322BF16 };
        }
        AscendC::Fixpipe(gm, co1Local, fixpipeParams);

        qidCO1.FreeTensor(co1Local);
    }


    __aicore__ inline void IterateAll(const AscendC::GlobalTensor<TensorTrait<DstT>>& gm)
    {
        CopyGmToA1Nd2Nz();
        CopyGmToB1Nd2Nz();
        Load2DA1ToL0A();
        Load2DA1ToL0B();

        Compute();
        CopyL0CToGm(gm);
    }

private:
    AscendC::TPipe* pipe = nullptr;
    int32_t mLength = 0;
    int32_t nLength = 0;
    int32_t kLength = 0;
    int32_t enBias = 0;

    AscendC::TQue<AscendC::TPosition::A1, 1, GetNdNzMask(CubeFormat::NZ, AType::format)> qidA1;
    AscendC::TQue<AscendC::TPosition::B1, 1, GetNdNzMask(CubeFormat::NZ, BType::format)> qidB1;
    AscendC::TQue<AscendC::TPosition::C1, 1> qidBias;
    AscendC::TQue<AscendC::TPosition::A2, 1> qidA2;
    AscendC::TQue<AscendC::TPosition::B2, 1> qidB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> qidCO1;

    AscendC::LocalTensor<TensorTrait<SrcT>> leftMatrix;
    AscendC::LocalTensor<TensorTrait<SrcT>> rightMatrix;
    AscendC::GlobalTensor<TensorTrait<SrcT>> aGlobal;
    AscendC::GlobalTensor<TensorTrait<SrcT>> bGlobal;
};

template <class AType, class BType, class CType>
__aicore__ inline void E2eKernel(__gm__ uint8_t* aGM, __gm__ uint8_t* bGM, __gm__ uint8_t* cGM,
    int32_t m, int32_t n, int32_t k)
{
    if (g_coreType == AscendC::AIV) {
        return;
    }

    using AT = typename AType::T;
    using BT = typename BType::T;
    using CT = typename CType::T;

    AscendC::GlobalTensor<TensorTrait<AT>> aGlobal;
    AscendC::GlobalTensor<TensorTrait<BT>> bGlobal;
    AscendC::GlobalTensor<TensorTrait<CT>> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ AT*>(aGM), m * k);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BT*>(bGM), k * n);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ CT*>(cGM), m * n);

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];

    set_atomic_none();
    AscendC::TPipe que;
    E2eMain<AType, BType, CType> ins;
    ins.Init(&que, m, n, k);
    ins.SetTensorA(gmA);
    ins.SetTensorB(gmB);

    ins.IterateAll(gmC);
    set_atomic_none();
    return;
}

__global__ __aicore__ void KernelE2e(__gm__ uint8_t* aGm, __gm__ uint8_t* bGm, __gm__ uint8_t* cGm,
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
    void SetUp()
    {
        g_coreType = AscendC::AIC_TYPE;
    }
    void TearDown()
    {
        CheckSyncState();
        g_coreType = AscendC::MIX_TYPE;
    }
};

INSTANTIATE_TEST_CASE_P(E2eCase, E2eTestsuite,
    ::testing::Values(E2eParams { 16, 32, 32, KernelE2e}));

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
        AscendC::LocalTensor<TensorTrait<Src0T>> featureMapA1 = inQueueFmA1.AllocTensor<TensorTrait<Src0T>>();
        AscendC::LocalTensor<TensorTrait<Src0T>> featureMapA2 = inQueueFmA2.AllocTensor<TensorTrait<Src0T>>();
        AscendC::LocalTensor<TensorTrait<Src0T>> weightB2 = inQueueWeightB2.AllocTensor<TensorTrait<Src0T>>();
        // set repeat
        set_l3d_rpt(0);
        if constexpr (UseFill) {
            // init l1
            Fill<TensorTrait<Src0T>>(featureMapA1,
                { 1, static_cast<uint16_t>(featureMapA1Size * sizeof(Src0T) / 32), 0, static_cast<Src0T>(1) });
            // init l0a
            Fill<TensorTrait<Src0T>>(featureMapA2,
                { 1, static_cast<uint16_t>(1), 0, static_cast<Src0T>(1) });
            // init l0b
            Fill<TensorTrait<Src0T>>(weightB2,
                { 1, static_cast<uint16_t>(1), 0, static_cast<Src0T>(1) });
        }else{
            // init l1
            InitConstValue<TensorTrait<Src0T>>(featureMapA1,
                { 1, static_cast<uint16_t>(featureMapA1Size * sizeof(Src0T) / 32), 0, static_cast<Src0T>(1) });
            // init l0a
            InitConstValue<TensorTrait<Src0T>>(featureMapA2,
                { 1, static_cast<uint16_t>(1), 0, static_cast<Src0T>(1) });
            // init l0b
            InitConstValue<TensorTrait<Src0T>>(weightB2,
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
    void SetUp()
    {
        g_coreType = AscendC::AIC_TYPE;
    }
    void TearDown()
    {
        CheckSyncState();
        g_coreType = AscendC::MIX_TYPE;
    }
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