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

using namespace std;

namespace AscendC {
#define LOCAL_TENSOR_REGISTER(tensorName, type, quePos, initAddr, dataSize) \
    AscendC::LocalTensor<type> tensorName;                                     \
    AscendC::TBuffAddr tbuf_##tensorName;                                      \
    tbuf_##tensorName.logicPos = static_cast<uint8_t>(AscendC::TPosition::quePos);       \
    tensorName.SetAddr(tbuf_##tensorName);                                    \
    tensorName.InitBuffer(initAddr, dataSize);

#define ALIGN_ADDR(addr) (((addr) + 31) / 32 * 32)

enum TestInstr {
    Load2dv2,
    Load3dv2,
    LoadWithTranspose,
    Load2dv2Gm2L1,
    Load2dv2Gm2L0,
};

/* **************************************************************************************************
 * AscendC::LoadData                                             *
 * ************************************************************************************************* */
// out load data -> l1
// l1 load data -> l0a & l0b
// l0a l0b mmad -> l0c
// l0c data copy -> out
template <typename DstT, typename Src0T, typename Src1T, uint8_t instrType>
void MainLoadData(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm,
                  __gm__ uint8_t* __restrict__ src1Gm, __gm__ uint16_t m, __gm__ uint16_t n, __gm__ uint16_t k,
                  __gm__ uint16_t channelSize)
{
    AscendC::TPipe tpipe;
    // mmad c = a * b
    uint16_t aSize = m * k;
    uint16_t bSize = k * n;
    uint16_t cSize = m * n;
    AscendC::GlobalTensor<Src0T> input0Global;
    AscendC::GlobalTensor<Src1T> input1Global;
    AscendC::GlobalTensor<DstT> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ Src0T*>(src0Gm), aSize);
    input1Global.SetGlobalBuffer(reinterpret_cast<__gm__ Src1T*>(src1Gm), bSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(dstGm), cSize);

    LOCAL_TENSOR_REGISTER(src0_l1, Src0T, A1, 0, aSize)
    LOCAL_TENSOR_REGISTER(src1_l1, Src1T, B1, ALIGN_ADDR(aSize * sizeof(Src0T)), bSize)
    LOCAL_TENSOR_REGISTER(src0_l0a, Src0T, A2, 0, aSize)
    LOCAL_TENSOR_REGISTER(src1_l0b, Src1T, B2, 0, bSize)
    LOCAL_TENSOR_REGISTER(dst_l0c, DstT, CO1, 0, cSize)

    // load2dv2
    if constexpr (instrType == 0) {
        AscendC::DataCopy(src0_l1, input0Global, m * k);
        AscendC::DataCopy(src1_l1, input1Global, k * n);
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = DivCeil(m, 16);
        loadDataParams.kStep = DivCeil(k * sizeof(Src0T), 32);
        if (AscendC::IsSameType<Src0T, int4b_t>::value) {
            loadDataParams.kStep = DivCeil(loadDataParams.kStep, 2);
        }
        loadDataParams.srcStride = DivCeil(m, 16);
        loadDataParams.dstStride = DivCeil(m, 16);
        loadDataParams.sid = 0;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(src0_l0a, src0_l1, loadDataParams);

        uint16_t kAlign = DivCeil(k, 16);
        uint16_t nAlign = DivCeil(n * sizeof(Src1T), 32);

        if (AscendC::IsSameType<Src1T, int4b_t>::value) {
            nAlign = DivCeil(nAlign, 2);
        }

        AscendC::LoadData2DParamsV2 loadDataParams2;
        loadDataParams2.mStartPosition = 0;
        loadDataParams2.kStartPosition = 0;
        loadDataParams2.mStep = kAlign;
        loadDataParams2.kStep = nAlign;
        loadDataParams2.srcStride = kAlign;
        loadDataParams2.dstStride = nAlign;

        if (AscendC::IsSameType<Src1T, int8_t>::value) {
            loadDataParams.dstStride = nAlign * 2;
        }
        loadDataParams.sid = 0;
        loadDataParams.ifTranspose = true;
        AscendC::LoadData(src1_l0b, src1_l1, loadDataParams2);
        // load3dv2
    } else if constexpr (instrType == 1) {
        AscendC::DataCopy(src0_l1, input0Global, m * k);
        AscendC::DataCopy(src1_l1, input1Global, k * n);
        AscendC::LoadDataRepeatParam repeatParams;
        repeatParams.repeatTime = 1;
        repeatParams.dstStride = m / 16;
        AscendC::SetLoadDataRepeat(repeatParams);
        LoadData3DParamsV2<Src0T> loadData3dParams;
        loadData3dParams.l1W = 1;
        loadData3dParams.l1H = m;
        loadData3dParams.channelSize = k;
        loadData3dParams.kExtension = k;
        loadData3dParams.mExtension = m;
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
        AscendC::LoadData(src0_l0a, src0_l1, loadData3dParams);

        // failed case
        TBuf<TPosition::C2> tbufFailed;
        tpipe.InitBuffer(tbufFailed, aSize * sizeof(Src0T));
        LocalTensor<Src0T> fmLocal = tbufFailed.Get<Src0T>();
        MOCKER(raise).stubs().will(returnValue(int(0)));
        AscendC::LoadData<Src0T>(fmLocal, src0_l1, loadData3dParams);

        AscendC::LoadDataRepeatParam repeatParams1;
        repeatParams1.repeatTime = 1;
        repeatParams1.dstStride = n / 16;
        AscendC::SetLoadDataRepeat(repeatParams1);
        LoadData3DParamsV2<Src1T> loadData3dParams1;
        loadData3dParams1.l1W = 1;
        loadData3dParams1.l1H = k;
        loadData3dParams1.channelSize = n;
        loadData3dParams1.kExtension = n;
        loadData3dParams1.mExtension = k;
        loadData3dParams1.kStartPt = 0;
        loadData3dParams1.mStartPt = 0;
        loadData3dParams1.strideW = 1;
        loadData3dParams1.strideH = 1;
        loadData3dParams1.filterW = 1;
        loadData3dParams1.filterH = 1;
        loadData3dParams1.dilationFilterW = 1;
        loadData3dParams1.dilationFilterH = 1;
        loadData3dParams1.enTranspose = true;
        loadData3dParams1.enSmallK = false;
        loadData3dParams1.padValue = 0;
        loadData3dParams1.filterSizeW = 0;
        loadData3dParams1.filterSizeH = 0;
        loadData3dParams1.fMatrixCtrl = false;
        AscendC::LoadData(src1_l0b, src1_l1, loadData3dParams1);  
    } else if constexpr (instrType == 2) {
        AscendC::DataCopy(src0_l1, input0Global, m * k);
        AscendC::DataCopy(src1_l1, input1Global, k * n);
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.mStep = DivCeil(m, 16);
        loadDataParams.kStep = DivCeil(k * sizeof(Src0T), 32);
        if (AscendC::IsSameType<Src0T, int4b_t>::value) {
            loadDataParams.kStep = DivCeil(loadDataParams.kStep, 2);
        }
        loadDataParams.srcStride = DivCeil(m, 16);
        loadDataParams.dstStride = DivCeil(m, 16);
        loadDataParams.sid = 0;
        loadDataParams.ifTranspose = false;
        AscendC::LoadData(src0_l0a, src0_l1, loadDataParams);

        uint16_t kStep = DivCeil(k, 16);
        uint16_t nStep = DivCeil(n * sizeof(Src1T), 32);

        for (uint16_t i = 0; i < nStep; i++) {
            LoadData2dTransposeParamsV2 loadDataParams;
            loadDataParams.startIndex = i * kStep;
            loadDataParams.repeatTimes = kStep;
            loadDataParams.srcStride = 1;
            loadDataParams.dstGap = nStep - 1;
            LoadDataWithTranspose(src1_l0b[256 * i], src1_l1, loadDataParams);
        }
    } else if constexpr (instrType == 3) {
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.srcStride = DivCeil(m, 16);
        loadDataParams.dstStride = DivCeil(m, 16);
        loadDataParams.mStep = DivCeil(m, 16);
        loadDataParams.kStep = DivCeil(k * sizeof(Src0T), 32);
        if (AscendC::IsSameType<Src1T, int4b_t>::value) {
            loadDataParams.kStep = DivCeil(loadDataParams.kStep, 2);
        }
        AscendC::LoadData(src0_l1, input0Global, loadDataParams);
        AscendC::DataCopy(src1_l1, input1Global, k * n);
        uint16_t kAlign = DivCeil(k, 16);
        uint16_t nAlign = DivCeil(n * sizeof(Src1T), 32);
        AscendC::LoadData2DParamsV2 loadDataParams2;
        loadDataParams2.mStartPosition = 0;
        loadDataParams2.kStartPosition = 0;
        loadDataParams2.mStep = kAlign;
        loadDataParams2.kStep = nAlign;
        loadDataParams2.srcStride = kAlign;
        loadDataParams2.dstStride = nAlign;

        if (AscendC::IsSameType<Src1T, int8_t>::value) {
            loadDataParams.dstStride = nAlign * 2;
        }
        loadDataParams.sid = 0;
        loadDataParams.ifTranspose = true;
        AscendC::LoadData(src1_l0b, src1_l1, loadDataParams2);
    } else if constexpr (instrType == 4) {
        AscendC::LoadData2DParamsV2 loadDataParams;
        loadDataParams.mStartPosition = 0;
        loadDataParams.kStartPosition = 0;
        loadDataParams.srcStride = DivCeil(m, 16);
        loadDataParams.dstStride = DivCeil(m, 16);
        loadDataParams.mStep = DivCeil(m, 16);
        loadDataParams.kStep = DivCeil(k * sizeof(Src0T), 32);
        if (AscendC::IsSameType<Src0T, int4b_t>::value) {
            loadDataParams.kStep = DivCeil(loadDataParams.kStep, 2);
        }
        AscendC::LoadData(src0_l0a, input0Global, loadDataParams);

        AscendC::LoadData2DParamsV2 loadDataParams2;
        loadDataParams2.mStartPosition = 0;
        loadDataParams2.kStartPosition = 0;
        loadDataParams2.srcStride = DivCeil(k, 16);
        loadDataParams2.dstStride = DivCeil(k, 16);
        loadDataParams2.mStep = DivCeil(k, 16);
        loadDataParams2.kStep = DivCeil(n * sizeof(Src1T), 32);
        if (AscendC::IsSameType<Src1T, int4b_t>::value) {
            loadDataParams2.kStep = DivCeil(loadDataParams2.kStep, 2);
        }
        AscendC::LoadData(src1_l0b, input1Global, loadDataParams2);
    }

    MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.isBias = false;
    Mmad(dst_l0c, src0_l0a, src1_l0b, mmadParams);

    FixpipeParamsV220 fixpipeParams;
    fixpipeParams.nSize = n;
    fixpipeParams.mSize = m;
    fixpipeParams.dstStride = m * BLOCK_CUBE * sizeof(DstT) / ONE_BLK_SIZE;
    fixpipeParams.srcStride = DivCeil(m, BLOCK_CUBE) * BLOCK_CUBE;
    Fixpipe<DstT, DstT, CFG_NZ>(outputGlobal, dst_l0c, fixpipeParams);
}

struct LoadDataTestParams {
    uint16_t m;
    uint16_t n;
    uint16_t k;
    uint8_t channelSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint16_t, uint16_t, uint16_t, uint16_t);
    uint8_t sizeofDst;
    uint8_t sizeofSrc0;
    uint8_t sizeofSrc1;
};

class LoadData610LiteTestsuite
    : public testing::Test
    , public testing::WithParamInterface<LoadDataTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(
    TEST_LOAD_DATA, LoadData610LiteTestsuite,
    ::testing::Values(
        LoadDataTestParams{ 16, 128, 32, 32, MainLoadData<float, half, half, uint8_t(TestInstr::Load2dv2)>, 4, 2, 2 },
        LoadDataTestParams{ 16, 32, 32, 32, MainLoadData<int32_t, int8_t, int8_t, uint8_t(TestInstr::Load2dv2)>, 4, 1,
                            1 },
        LoadDataTestParams{ 32, 64, 64, 32, MainLoadData<int32_t, int4b_t, int4b_t, uint8_t(TestInstr::Load2dv2)>, 4, 2,
                            2 },
        LoadDataTestParams{ 16, 128, 32, 32, MainLoadData<float, half, half, uint8_t(TestInstr::Load3dv2)>, 4, 2, 2 },
        LoadDataTestParams{ 16, 128, 32, 32, MainLoadData<float, half, half, uint8_t(TestInstr::LoadWithTranspose)>, 4,
                            2, 2 },
        LoadDataTestParams{ 32, 64, 64, 32, MainLoadData<int32_t, int4b_t, int4b_t, TestInstr::Load2dv2Gm2L1>, 4, 2,
                            2 },
        LoadDataTestParams{ 16, 128, 32, 32, MainLoadData<float, half, half, TestInstr::Load2dv2Gm2L1>, 4, 2, 2 },
        LoadDataTestParams{ 16, 128, 32, 32, MainLoadData<float, half, half, uint8_t(TestInstr::Load2dv2Gm2L0)>, 4, 2,
                            2 }));

TEST_P(LoadData610LiteTestsuite, LoadDataTestCase)
{
    auto param = GetParam();
    uint8_t dstGm[param.m * param.n * param.sizeofDst];
    uint8_t src0Gm[param.m * param.k * param.sizeofSrc0];
    uint8_t src1Gm[param.k * param.n * param.sizeofSrc1];
    param.cal_func(dstGm, src0Gm, src1Gm, param.m, param.n, param.k, param.channelSize);
    for (int32_t i = 0; i < param.m * param.n * param.sizeofDst; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

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
        pipe.InitBuffer(outQueueCO1, 1, featureMapA1Size * sizeof(Src0T));
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
        LocalTensor<Src0T> featureMapA1 = inQueueFmA1.AllocTensor<Src0T>();
        LocalTensor<Src0T> featureMapA2 = inQueueFmA2.AllocTensor<Src0T>();
        LocalTensor<Src0T> weightB2 = inQueueWeightB2.AllocTensor<Src0T>();
        LocalTensor<Src0T> featureMapCO1 = outQueueCO1.AllocTensor<Src0T>();

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

            // init l0b
            Fill<Src0T>(weightB2,
                { 1, static_cast<uint16_t>(1), 0, static_cast<Src0T>(1) });
            
            // init l0c error case
            MOCKER(raise).stubs().will(returnValue(int(0)));
            Fill<Src0T>(featureMapCO1,
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

            // init l0b
            InitConstValue<Src0T>(weightB2,
                { 1, static_cast<uint16_t>(1), 0, static_cast<Src0T>(1) });
            
            // init l0c error case
            MOCKER(raise).stubs().will(returnValue(int(0)));
            InitConstValue<Src0T>(featureMapCO1,
                { 1, static_cast<uint16_t>(1), 0, static_cast<Src0T>(1) });
        }
        inQueueFmA1.FreeTensor(featureMapA1);
        inQueueFmA2.FreeTensor(featureMapA2);
        inQueueWeightB2.FreeTensor(weightB2);
    }

private:
    TPipe pipe;
    TQue<TPosition::A1, 1> inQueueFmA1;
    TQue<TPosition::A2, 1> inQueueFmA2;
    TQue<TPosition::B1, 1> inQueueWeightB1;
    TQue<TPosition::B2, 1> inQueueWeightB2;
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

template <typename Src0T>
__global__ __aicore__ void creat_martix_AscendC()
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
        g_coreType = AscendC::MIX_TYPE;
    }
};

INSTANTIATE_TEST_CASE_P(SetCreatmartixTest, CreatmartixTestsuite,
    ::testing::Values(CreatmartixTestParams { 192, 32, 64, creat_martix_AscendC<half>, 2 },
    CreatmartixTestParams { 192, 32, 64, creat_martix_AscendC<int16_t>, 2 },
    CreatmartixTestParams { 192, 32, 64, creat_martix_AscendC<uint16_t>, 2 }
    ));

INSTANTIATE_TEST_CASE_P(SetFillTest, CreatmartixTestsuite,
    ::testing::Values(CreatmartixTestParams { 192, 32, 64, FillAscendC<half>, 2 },
    CreatmartixTestParams { 192, 32, 64, FillAscendC<int16_t>, 2 },
    CreatmartixTestParams { 192, 32, 64, FillAscendC<uint16_t>, 2 }
    ));

TEST_P(CreatmartixTestsuite, CreatmartixTest)
{
    auto param = GetParam();
    param.cal_func();
}
}  // namespace AscendC
