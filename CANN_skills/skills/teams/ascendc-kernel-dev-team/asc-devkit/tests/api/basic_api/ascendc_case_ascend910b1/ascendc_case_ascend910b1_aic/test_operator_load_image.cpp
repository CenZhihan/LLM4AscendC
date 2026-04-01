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

using namespace std;

class TEST_LOADIMAGE : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(1);
    }
    void TearDown()
    {
        AscendC::CheckSyncState();
        AscendC::SetGCoreType(0);
    }
};

namespace AscendC {
template <typename SrcT, typename FMapT, typename WeightT, typename DstCO1T, typename DstT> class KernelLoadImage {
public:
    __aicore__ inline KernelLoadImage(uint16_t horizSizeIn, uint16_t vertSizeIn, uint16_t horizStartPosIn,
        uint16_t vertStartPosIn, uint16_t srcHorizSizeIn, uint16_t srcVertSizeIn, uint8_t topPadSizeIn,
        uint8_t botPadSizeIn, uint16_t leftPadSizeIn, uint16_t rightPadSizeIn, AscendC::AippInputFormat inputFormatIn,
        uint32_t areaPadModeIn, uint32_t cPadModeIn, FMapT cPaddingValueIn, bool isSwapIn, bool isSingleLineCopyIn,
        bool isCscEnableIn, int16_t cscMatrixValueIn, uint8_t biasInValueIn, uint8_t biasOutValueIn,
        uint32_t dstHorizSizeIn, uint32_t dstVertSizeIn)
    {
        // loadImage params
        horizSize = horizSizeIn;
        vertSize = vertSizeIn;
        horizStartPos = horizStartPosIn;
        vertStartPos = vertStartPosIn;
        srcHorizSize = srcHorizSizeIn;
        topPadSize = topPadSizeIn;
        botPadSize = botPadSizeIn;
        leftPadSize = leftPadSizeIn;
        rightPadSize = rightPadSizeIn;
        // aipp config
        inputFormat = inputFormatIn;
        areaPadMode = areaPadModeIn;
        cPadMode = cPadModeIn;
        cPaddingValue = cPaddingValueIn;
        isSwap = isSwapIn;
        isSingleLineCopy = isSingleLineCopyIn;
        isCscEnable = isCscEnableIn;
        cscMatrixValue = cscMatrixValueIn;
        biasInValue = biasInValueIn;
        biasOutValue = biasOutValueIn;
        // src shape
        srcVertSize = srcVertSizeIn;

        //dst shape
        dstHorizSize = dstHorizSizeIn;
        dstVertSize = dstVertSizeIn;

        if (isSingleLineCopy == true) {
            dstVertSize = 2;
        }

        gmSrc0Size = srcHorizSize * srcVertSize;

        if (cPadMode == 0) {
            cSize = 32 / sizeof(FMapT);
        } else {
            cSize = 4;
        }
        dstSize = dstHorizSize * dstVertSize * cSize;

        featureMapA1Size = dstSize;
        weightSize = dstHorizSize * dstHorizSize;
    }
    __aicore__ inline void Init(__gm__ uint8_t* fmGm, __gm__ uint8_t* dstGm)
    {
        fmGlobal.SetGlobalBuffer((__gm__ SrcT*)fmGm);
        dstGlobal.SetGlobalBuffer((__gm__ DstT*)dstGm);
        pipe.InitBuffer(inQueueA1, 1, featureMapA1Size * sizeof(FMapT));
        pipe.InitBuffer(outQueueUB, 1, featureMapA1Size * sizeof(FMapT));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<TensorTrait<FMapT>> featureMapA1 = inQueueA1.AllocTensor<TensorTrait<FMapT>>();

        uint64_t fm_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fmGlobal.GetPhyAddr()));

        // aipp config
        AippParams<FMapT> aippConfig;
        // 1、area pad
            // 1) padding mode
        aippConfig.paddingParams.paddingMode = areaPadMode;
            // 2) const padding value
        aippConfig.paddingParams.paddingValueCh0 = paddingValueCh0;
        aippConfig.paddingParams.paddingValueCh1 = paddingValueCh1;
        aippConfig.paddingParams.paddingValueCh2 = paddingValueCh2;
        aippConfig.paddingParams.paddingValueCh3 = paddingValueCh3;

        // 2、channel swap
        aippConfig.swapParams.isSwapRB = isSwap;
        aippConfig.swapParams.isSwapUV = isSwap;
        aippConfig.swapParams.isSwapAX = isSwap;

        // 3、singleline copy
        aippConfig.singleLineParams.isSingleLineCopy = isSingleLineCopy;

        // 4、csc mode
            // 1) if csc
        aippConfig.cscParams.isEnableCsc = isCscEnable;
            // 2) csc matrix
        aippConfig.cscParams.cscMatrixR0C0 = cscMatrixValue;
        aippConfig.cscParams.cscMatrixR0C1 = cscMatrixValue;
        aippConfig.cscParams.cscMatrixR0C2 = cscMatrixValue;
        aippConfig.cscParams.cscMatrixR1C0 = cscMatrixValue;
        aippConfig.cscParams.cscMatrixR1C1 = cscMatrixValue;
        aippConfig.cscParams.cscMatrixR1C2 = cscMatrixValue;
        aippConfig.cscParams.cscMatrixR2C0 = cscMatrixValue;
        aippConfig.cscParams.cscMatrixR2C1 = cscMatrixValue;
        aippConfig.cscParams.cscMatrixR2C2 = cscMatrixValue;
            // 3) YUV to RGB biasin
        aippConfig.cscParams.cscBiasIn0 = biasInValue;
        aippConfig.cscParams.cscBiasIn1 = biasInValue;
        aippConfig.cscParams.cscBiasIn2 = biasInValue;
            // 4) RGB to YUV biasout
        aippConfig.cscParams.cscBiasOut0 = biasOutValue;
        aippConfig.cscParams.cscBiasOut1 = biasOutValue;
        aippConfig.cscParams.cscBiasOut2 = biasOutValue;

        // 5、datatype conversion
        aippConfig.dtcParams.dtcMeanCh0 = meanValueCh0;
        aippConfig.dtcParams.dtcMeanCh1 = meanValueCh1;
        aippConfig.dtcParams.dtcMeanCh2 = meanValueCh2;
        aippConfig.dtcParams.dtcMinCh0 = minValueCh0;
        aippConfig.dtcParams.dtcMinCh1 = minValueCh1;
        aippConfig.dtcParams.dtcMinCh2 = minValueCh2;
        aippConfig.dtcParams.dtcVarCh0 = varValueCh0;
        aippConfig.dtcParams.dtcVarCh1 = varValueCh1;
        aippConfig.dtcParams.dtcVarCh2 = varValueCh2;

        // 6、channel pad
        aippConfig.cPaddingParams.cPaddingMode = cPadMode;
        aippConfig.cPaddingParams.cPaddingValue = cPaddingValue;

        if (inputFormat == AscendC::AippInputFormat::YUV420SP_U8) {
            SetAippFunctions(fmGlobal, fmGlobal[gmSrc0Size], inputFormat, aippConfig);
        } else {
            SetAippFunctions(fmGlobal, inputFormat, aippConfig);
        }
        LoadImageToLocal(featureMapA1, { horizSize, vertSize, horizStartPos, vertStartPos, srcHorizSize, topPadSize,
            botPadSize, leftPadSize, rightPadSize });

        inQueueA1.EnQue(featureMapA1);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<TensorTrait<FMapT>> featureMapA1 = inQueueA1.DeQue<TensorTrait<FMapT>>();

        event_t eventIdMTE2ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3);
        DataCopy(dstGlobal, featureMapA1, dstSize);

        inQueueA1.FreeTensor(featureMapA1);
    }
private:
    TPipe pipe;
    // feature map queue
    TQue<TPosition::A1, 1> inQueueA1;
    TQue<TPosition::VECOUT, 1> outQueueUB;


    GlobalTensor<TensorTrait<SrcT>> fmGlobal;
    GlobalTensor<TensorTrait<WeightT>> weGlobal;
    GlobalTensor<TensorTrait<DstT>> dstGlobal;

    uint16_t horizSize, vertSize, horizStartPos, vertStartPos, srcHorizSize, srcVertSize, leftPadSize, rightPadSize;
    uint32_t dstHorizSize, dstVertSize, cSize;
    uint8_t topPadSize, botPadSize;
    uint32_t gmSrc0Size = 0, gmSrc1Size = 0, featureMapA1Size, weightSize, dstSize;
    uint16_t m, k, n;
    uint8_t meanValueCh0 = 0, meanValueCh1 = 0, meanValueCh2 = 0, meanValueCh3 = 0;
    half minValueCh0 = 0.0, minValueCh1 = 0.0, minValueCh2 = 0.0, minValueCh3 = 0.0;
    half varValueCh0 = 0.0, varValueCh1 = 0.0, varValueCh2 = 0.0, varValueCh3 = 0.0;
    FMapT paddingValueCh0 = 2, paddingValueCh1 = 3, paddingValueCh2 = 5, paddingValueCh3 = 6;
    FMapT cPaddingValue = 1;
    AscendC::AippInputFormat inputFormat;
    uint32_t areaPadMode, cPadMode;
    bool isSwap, isSingleLineCopy, isCscEnable;
    int16_t cscMatrixValue = 0;
    uint8_t biasInValue = 0, biasOutValue = 0;
};
} // namespace AscendC

#define KERNEL_LOAD_IMAGE(srcType, fMapA1Type, weightType, co1Type, dstType, horizSizeIn, vertSizeIn,                        \
    horizStartPosIn, vertStartPosIn, srcHorizSizeIn, srcVertSizeIn, topPadSizeIn, botPadSizeIn, leftPadSizeIn, rightPadSizeIn,    \
    inputFormatIn, areaPadModeIn, cPadModeIn, cPaddingValueIn, isSwapIn, isSingleLineCopyIn, isCscEnableIn,                       \
    cscMatrixValueIn, biasInValueIn, biasOutValueIn, dstVertSizeIn, dstHorizSizeIn)                                               \
    TEST_F(TEST_LOADIMAGE, LOADIMAGE##_##inputFormatIn##_##Case)                                                                  \
    {                                                                                                                             \
        uint32_t cSizeIn = 32 / sizeof(fMapA1Type);                                                                              \
        if (cPadModeIn != 0) {                                                                                                    \
            cSizeIn = 4;                                                                                                          \
        }                                                                                                                         \
        uint32_t dstSizeIn = (dstHorizSizeIn + topPadSizeIn + botPadSizeIn) * (dstVertSizeIn + leftPadSizeIn + rightPadSizeIn)    \
            * cSizeIn;                                                                                                            \
        uint8_t fmGm[srcHorizSizeIn * srcVertSizeIn * sizeof(srcType)];                                                          \
        uint8_t dstGm[dstSizeIn * sizeof(dstType)];                                                                              \
        AscendC::KernelLoadImage<srcType, fMapA1Type, weightType, co1Type, dstType> op(horizSizeIn, vertSizeIn,              \
            horizStartPosIn, vertStartPosIn, srcHorizSizeIn, srcVertSizeIn, topPadSizeIn, botPadSizeIn, leftPadSizeIn,            \
            rightPadSizeIn, AscendC::AippInputFormat::inputFormatIn, areaPadModeIn, cPadModeIn, cPaddingValueIn, isSwapIn,                 \
            isSingleLineCopyIn, isCscEnableIn, cscMatrixValueIn, biasInValueIn, biasOutValueIn, dstHorizSizeIn,                   \
            dstVertSizeIn);                                                                                                       \
        op.Init(fmGm, dstGm);                                                                                                     \
        op.Process();                                                                                                             \
    }

// --test_YUV420SP_U8
KERNEL_LOAD_IMAGE(uint8_t, int8_t, int8_t, int32_t, int8_t, 32, 32, 
                    0, 0, 32, 32, 0, 0, 0, 0, 
                    YUV420SP_U8, 0, 0, 0, false, false, false, 
                    1, 1, 1, 32, 32);
// --test_XRGB8888_U8
KERNEL_LOAD_IMAGE(uint8_t, int8_t, int8_t, int32_t, int8_t, 32, 32, 
                    0, 0, 32, 32, 0, 0, 0, 0, 
                    XRGB8888_U8, 0, 0, 0, false, false, false, 
                    1, 1, 1, 32, 32);