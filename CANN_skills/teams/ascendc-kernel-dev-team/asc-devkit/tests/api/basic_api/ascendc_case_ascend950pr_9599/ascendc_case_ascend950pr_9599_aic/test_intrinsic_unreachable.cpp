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
#include "mockcpp/mockcpp.hpp"
#include "kernel_operator.h"
// #include "model/model_factory.h"
#include "test_utils.h"

using namespace std;
using namespace AscendC;

template <typename T>
class KernelUnreachable {
public:
    __aicore__ inline KernelUnreachable()
    {}

    __aicore__ inline void RunCopyGm2CBufMultiNd2Nz(GM_ADDR dstGm, GM_ADDR srcGm)
    {
        uint8_t sid = 0;
        uint64_t loop1SrcStride = 32;
        uint8_t l2CacheCtl = 0;
        uint16_t nValue = 32;
        uint32_t dValue = 32;
        uint64_t loop4SrcStride = 0;
        bool smallc0En = false;
        copy_gm_to_cbuf_multi_nd2nz((__gm__ T *)dstGm, (__gm__ T *)srcGm, sid, loop1SrcStride, l2CacheCtl,
            nValue,  dValue, loop4SrcStride, smallc0En);
    }

    __aicore__ inline void RunCopyGm2UbufAlignV2(GM_ADDR dstGm, GM_ADDR srcGm)
    {
        uint8_t sid = 0;
        uint32_t burstNum = 1;
        uint32_t burstLen = 8;
        uint8_t leftPaddingCount = 0;
        uint8_t rightPaddingCount = 0;
        bool constantPaddingCtl = false;
        uint8_t l2CacheCtl = 0;
        uint64_t burstSrcStride = 8;
        uint32_t burstDstStride = 8;
        copy_gm_to_ubuf_align_v2((__gm__ T *)dstGm, (__gm__ T *)srcGm, sid, burstNum,
            burstLen, leftPaddingCount, rightPaddingCount, constantPaddingCtl,
            l2CacheCtl, burstSrcStride, burstDstStride);

    }

    __aicore__ inline void RunCopyGm2CbufAlignV2(GM_ADDR dstGm, GM_ADDR srcGm)
    {
        uint8_t sid = 0;
        uint32_t burstNum = 1;
        uint32_t burstLen = 8;
        uint8_t leftPaddingCount = 0;
        uint8_t rightPaddingCount = 0;
        bool constantPaddingCtl = false;
        uint8_t l2CacheCtl = 0;
        uint64_t burstSrcStride = 8;
        uint32_t burstDstStride = 8;
        copy_gm_to_cbuf_align_v2((__gm__ T *)dstGm, (__gm__ T *)srcGm, sid, burstNum,
            burstLen, leftPaddingCount, rightPaddingCount, constantPaddingCtl,
            l2CacheCtl, burstSrcStride, burstDstStride);
    }

    __aicore__ inline void RunImg2ColV2Cbuf2Cb(GM_ADDR dstGm, GM_ADDR srcGm)
    {
        uint16_t kExtension = 1;
        uint16_t mExtension = 1;
        uint16_t kStartPt = 0;
        uint16_t mStartPt = 0;
        uint8_t strideW = 1;
        uint8_t strideH = 1;
        uint8_t wk = 1;
        uint8_t hk = 1;
        uint8_t dilationFilterW = 1;
        uint8_t dilationFilterH = 1;
        bool filterSizeW = false;
        bool filterSizeH = false;
        bool transpose = false;
        bool fMatrixCtrl = false;
        uint16_t sizeChannel = 1;
        // AscendC::ModelFactory::Instance().SetSprFMatrix(1000000);
        // img2colv2_cbuf_to_cb((__gm__ T *)dstGm, (__gm__ T *)srcGm, kExtension, mExtension, kStartPt, mStartPt,
        //     strideW, strideH, wk, hk, dilationFilterW, dilationFilterH, filterSizeW, filterSizeH, transpose,
        //     fMatrixCtrl, sizeChannel);
    }

    __aicore__ inline void Run(GM_ADDR dstGm, GM_ADDR srcGm)
    {
        RunCopyGm2CBufMultiNd2Nz(dstGm, srcGm);
        RunCopyGm2UbufAlignV2(dstGm, srcGm);
        RunCopyGm2CbufAlignV2(dstGm, srcGm);
        RunImg2ColV2Cbuf2Cb(dstGm, srcGm);
    }
};

void RunKernelOperatorMmImplCoverage()
{
    constexpr uint16_t mLength = 16;
    constexpr uint16_t nLength = 16;
    constexpr uint16_t kLength = 16;
    constexpr int32_t aSize = mLength * kLength;
    constexpr int32_t bSize = kLength * nLength;
    constexpr int32_t cSize = mLength * nLength;
    uint8_t gmBufferA[aSize * sizeof(half)] = {0};
    uint8_t gmBufferB[bSize * sizeof(half)] = {0};

    LOCAL_TENSOR_REGISTER(srcA1, half, A1, 0, aSize)
    LOCAL_TENSOR_REGISTER(srcB1, half, B1, 0, bSize)
    LOCAL_TENSOR_REGISTER(dstA2, half, A2, 0, aSize)
    LOCAL_TENSOR_REGISTER(dstB2, half, B2, 0, bSize)
    LOCAL_TENSOR_REGISTER(dstUb, half, VECIN, 0, aSize)
    LOCAL_TENSOR_REGISTER(dstL0c, float, CO1, 0, cSize)

    LoadData2DParamsV2 loadData2dA = {0, 0, 1, 1, 1, 1, false, 0};
    Load2DBitModeParam loadData2dABitMode(loadData2dA);
    LoadData2DL12L0ACal((__ca__ half *)dstA2.GetPhyAddr(), (__cbuf__ half *)srcA1.GetPhyAddr(), loadData2dABitMode);
    loadData2dA.ifTranspose = true;
    Load2DBitModeParam loadData2dATransBitMode(loadData2dA);
    LoadData2DL12L0ACal((__ca__ half *)dstA2.GetPhyAddr(), (__cbuf__ half *)srcA1.GetPhyAddr(),
        loadData2dATransBitMode);

    LoadData2DParamsV2 loadData2dB = {0, 0, 1, 1, 1, 1, false, 0};
    Load2DBitModeParam loadData2dBBitMode(loadData2dB);
    LoadData2DL12L0BCal((__cb__ half *)dstB2.GetPhyAddr(), (__cbuf__ half *)srcB1.GetPhyAddr(), loadData2dBBitMode);
    loadData2dB.ifTranspose = true;
    Load2DBitModeParam loadData2dBTransBitMode(loadData2dB);
    LoadData2DL12L0BCal((__cb__ half *)dstB2.GetPhyAddr(), (__cbuf__ half *)srcB1.GetPhyAddr(),
        loadData2dBTransBitMode);

    MmadParams mmadParams;
    mmadParams.m = mLength;
    mmadParams.k = kLength;
    mmadParams.n = nLength;
    mmadParams.cmatrixSource = false;
    mmadParams.cmatrixInitVal = false;
    mmadParams.disableGemv = false;
    MmadBitModeParams mmadBitModeParams(mmadParams);
    MmadCal((__cc__ float *)dstL0c.GetPhyAddr(), (__ca__ half *)dstA2.GetPhyAddr(), (__cb__ half *)dstB2.GetPhyAddr(),
        mmadBitModeParams);

    LoadData3DParamsV2<half> loadData3dParams;
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
    loadData3dParams.filterSizeW = 0;
    loadData3dParams.filterSizeH = 0;
    loadData3dParams.enTranspose = true;
    loadData3dParams.fMatrixCtrl = false;
    SetFmatrix(SetFMatrixBitModeParams(loadData3dParams), FmatrixMode::FMATRIX_LEFT);
    SetLoadDataPaddingValue(0);
    SetLoadDataRepeat({0, 1, 0});
    LoadData3DV2L12L0BCal((__cb__ half *)dstB2.GetPhyAddr(), (__cbuf__ half *)srcB1.GetPhyAddr(), loadData3dParams);

    MOCKER(raise).stubs().will(returnValue(0));
    LoadData2DGM2L0ACal((__ca__ half *)dstA2.GetPhyAddr(), (__gm__ half *)gmBufferA, loadData2dA);
    LoadData2DGM2L0BCal((__cb__ half *)dstB2.GetPhyAddr(), (__gm__ half *)gmBufferB, loadData2dB);
    InitConstValueParams<half> initParams = {1, 1, 0, static_cast<half>(1)};
    InitL0ANzMatrixCal((__ca__ half *)dstA2.GetPhyAddr(), initParams);
    InitL0BNzMatrixCal((__cb__ half *)dstB2.GetPhyAddr(), initParams);
    LoadData3DV2L12UBCal((__ubuf__ half *)dstUb.GetPhyAddr(), (__cbuf__ half *)srcA1.GetPhyAddr(), loadData3dParams);
    SetSysWorkSpacePtr(GetSysWorkSpacePtr());
}


class TEST_INTRINSIC_UNREACHABLE : public testing::Test {
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

#define REGIST_INTRINSIC_UNREACHABLE(type)                                                    \
    TEST_F(TEST_INTRINSIC_UNREACHABLE, DATACOPY_NZ2NZ_Case_Bias_##type##_##high##_##width)      \
    {                                                                                           \
        KernelUnreachable<type> op;                                                             \
        uint8_t srcGm[10000] = {0};                                                               \
        uint8_t dstGm[10000] = {0};                                                               \
        op.Run(dstGm, srcGm);                                                                  \
    }


REGIST_INTRINSIC_UNREACHABLE(fp8_e5m2_t);
REGIST_INTRINSIC_UNREACHABLE(fp8_e4m3fn_t);

TEST_F(TEST_INTRINSIC_UNREACHABLE, KERNEL_OPERATOR_MM_IMPL_BRANCH_COVERAGE)
{
    RunKernelOperatorMmImplCoverage();
}
