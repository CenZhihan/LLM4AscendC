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
#include "test_utils.h"

using namespace std;

namespace AscendC {
struct FixpipeInputParamsV2 {
    __aicore__ FixpipeInputParamsV2() {}
    __aicore__ FixpipeInputParamsV2(const uint16_t c1In, const uint16_t hIn, const uint16_t wIn, const uint8_t khIn,
        const uint8_t kwIn, const uint16_t coutIn, const uint16_t c0In, const uint16_t dilationHIn,
        const uint16_t dilationWIn)
    {
        c1 = c1In;
        h = hIn;
        w = wIn;
        kh = khIn;
        kw = kwIn;
        cout = coutIn;
        c0 = c0In;
        dilationH = dilationHIn;
        dilationW = dilationWIn;

        coutBlocks = (cout + 16 - 1) / 16;
        ho = h - dilationH * (kh - 1);
        wo = w - dilationW * (kw - 1);
        howo = ho * wo;
        howoRound = ((howo + 16 - 1) / 16) * 16;

        featureMapSize = c1 * h * w * c0;     // [c1, h, w, c0]
        weightSize = c1 * kh * kw * cout * c0; // [c1, kh, kw, cout, c0]
        featureMapL0aSize = howoRound * (c1 * kh * kw * c0);
        weightL0bSize = (c1 * kh * kw * c0) * coutBlocks * 16;
        m = howo;
        k = c1 * kh * kw * c0;
        n = cout;
        deqSize = cout;                    // [cout]
        dstSize = coutBlocks * howo * 16; // [coutBlocks, howo, 16]
        dstL0cSize = coutBlocks * howoRound * 16;

        fmRepeat = featureMapL0aSize / (16 * c0);
        weRepeat = weightL0bSize / (16 * c0);
    }

    uint16_t c1;
    uint16_t h;
    uint16_t w;
    uint8_t kh;
    uint8_t kw;
    uint16_t cout;
    uint16_t c0;
    uint8_t dilationH;
    uint8_t dilationW;
    bool reluEnable;
    bool enNz2nd;

    uint16_t coutBlocks;
    uint16_t ho;
    uint16_t wo;
    uint16_t howo;
    uint16_t howoRound;

    uint32_t featureMapSize;
    uint32_t weightSize;
    uint32_t featureMapL0aSize;
    uint32_t weightL0bSize;
    uint16_t m;
    uint16_t k;
    uint16_t n;
    uint32_t deqSize;
    uint32_t dstSize;
    uint32_t dstL0cSize;

    uint8_t fmRepeat;
    uint8_t weRepeat;
};

/* **************************************************************************************************
 * Fixpipe                                             *
 * ************************************************************************************************* */
#define FIXPIPE_RELU_FUNC(deqMode, deqScalarValue, deqTensorGm, enRelu, fixpipeParams) fixpipeParams.reluEn = enRelu;

#define FIXPIPE_DEQ_CONV_RELU_FUNC(deqMode, deqScalarValue, deqTensorGm, enRelu, fixpipeParams)                        \
    fixpipeParams.quantPre = deqMode;                                                                                  \
    fixpipeParams.reluEn = enRelu;

#define FIXPIPE_DEQ_SCALAR_RELU_FUNC(deqMode, deqScalarValue, deqTensorGm, enRelu, fixpipeParams)                      \
    fixpipeParams.quantPre = deqMode;                                                                                  \
    fixpipeParams.deqScalar = deqScalarValue;                                                                          \
    fixpipeParams.reluEn = enRelu;

#define FIXPIPE_DEQ_TENSOR_RELU_FUNC(deqMode, deqScalarValue, deqTensorGm, enRelu, fixpipeParams)                      \
    fixpipeParams.quantPre = deqMode;                                                                                  \
    fixpipeParams.reluEn = enRelu;

#define KERNEL_FIXPIPE(                                                                                                \
    fmT, wT, l1OutT, dstT, fmTSize, wTSize, l1OutTSize, dstTSize, name, deqMode, FixpipeFunc)                          \
    extern "C" __global__ __aicore__ void kernel_fixpipe_##name(GM_ADDR fmData,                                        \
        GM_ADDR weData,                                                                                                \
        GM_ADDR deqTensor,                                                                                             \
        GM_ADDR outputData,                                                                                            \
        const FixpipeInputParamsV2 &inputParams)                                                                       \
    {                                                                                                                  \
        TPipe tpipe;                                                                                                   \
        const uint16_t c1 = inputParams.c1;                                                                            \
        const uint16_t h = inputParams.h;                                                                              \
        const uint16_t w = inputParams.w;                                                                              \
        const uint8_t kh = inputParams.kh;                                                                             \
        const uint8_t kw = inputParams.kw;                                                                             \
        const uint16_t cout = inputParams.cout;                                                                        \
        const uint16_t c0 = inputParams.c0;                                                                            \
        const uint8_t dilationH = inputParams.dilationH;                                                               \
        const uint8_t dilationW = inputParams.dilationW;                                                               \
        const bool reluEnable = inputParams.reluEnable;                                                                \
        const bool enNz2nd = inputParams.enNz2nd;                                                                      \
                                                                                                                       \
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);                                                                        \
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);                                                                       \
                                                                                                                       \
        const uint16_t coutBlocks = inputParams.coutBlocks;                                                            \
        const uint16_t ho = inputParams.ho;                                                                            \
        const uint16_t wo = inputParams.wo;                                                                            \
        const uint16_t howo = inputParams.howo;                                                                        \
        const uint16_t howoRound = inputParams.howoRound;                                                              \
                                                                                                                       \
        const uint32_t featureMapSize = inputParams.featureMapSize;                                                    \
        const uint32_t weightSize = inputParams.weightSize;                                                            \
        const uint32_t featureMapL0aSize = inputParams.featureMapL0aSize;                                              \
        const uint32_t weightL0bSize = inputParams.weightL0bSize;                                                      \
        const uint16_t m = inputParams.m;                                                                              \
        const uint16_t k = inputParams.k;                                                                              \
        const uint16_t n = inputParams.n;                                                                              \
        const uint32_t deqSize = inputParams.deqSize;                                                                  \
        const uint32_t dstSize = inputParams.dstSize;                                                                  \
        const uint32_t dstL0cSize = inputParams.dstL0cSize;                                                            \
                                                                                                                       \
        const uint8_t fmRepeat = inputParams.fmRepeat;                                                                 \
        const uint8_t weRepeat = inputParams.weRepeat;                                                                 \
                                                                                                                       \
        GlobalTensor<TensorTrait<fmT>> featureMapGm;                                                                   \
        GlobalTensor<TensorTrait<wT>> weightGm;                                                                        \
        GlobalTensor<TensorTrait<uint64_t>> deqTensorGm;                                                               \
        GlobalTensor<TensorTrait<dstT>> outputGm;                                                                      \
        featureMapGm.SetGlobalBuffer(reinterpret_cast<__gm__ fmT *>(fmData), featureMapSize);                          \
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ wT *>(weData), weightSize);                                   \
        deqTensorGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(deqTensor), deqSize);                          \
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ dstT *>(outputData), dstSize);                                \
                                                                                                                       \
        LOCAL_TENSOR_REGISTER(featureMapL1, TensorTrait<fmT>, A1, 0, featureMapSize)                                   \
        LOCAL_TENSOR_REGISTER(weightL1, TensorTrait<wT>, B1, featureMapSize *fmTSize, weightSize)                      \
                                                                                                                       \
        LOCAL_TENSOR_REGISTER(featureMapL0a, TensorTrait<fmT>, A2, 0, featureMapL0aSize)                               \
        LOCAL_TENSOR_REGISTER(weightL0b, TensorTrait<wT>, B2, 0, weightL0bSize)                                        \
        LOCAL_TENSOR_REGISTER(dstL0c, TensorTrait<l1OutT>, CO1, 0, dstL0cSize)                                         \
                                                                                                                       \
        DataCopy(featureMapL1, featureMapGm, {1, static_cast<uint16_t>(featureMapSize * fmTSize / 32), 0, 0});         \
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);                                                                     \
        DataCopy(weightL1, weightGm, {1, static_cast<uint16_t>(weightSize * wTSize / 32), 0, 0});                      \
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);                                                                     \
        uint8_t padList[PAD_SIZE] = {0, 0, 0, 0};                                                                      \
        LoadData<TensorTrait<fmT>>(featureMapL0a,                                                                      \
            featureMapL1,                                                                                              \
            {padList, h, w, 36, 128, 16, 0, 0, kw, kh, dilationW, dilationH, 2, 2, false, false, 0});                  \
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);                                                                    \
                                                                                                                       \
        LoadData(weightL0b, weightL1, {0, weRepeat, 1, 0, 0, false, 0});                                               \
                                                                                                                       \
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);                                                                        \
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);                                                                       \
                                                                                                                       \
        Mmad(dstL0c, featureMapL0a, weightL0b, {m, n, k, 0, false, true});                                             \
                                                                                                                       \
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);                                                                    \
                                                                                                                       \
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);                                                                         \
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);                                                                        \
                                                                                                                       \
        LOCAL_TENSOR_REGISTER(                                                                                         \
            cbufWorkspace, TensorTrait<uint64_t>, C1, featureMapSize *fmTSize + weightSize * wTSize, deqSize)          \
        if (enNz2nd) {                                                                                                 \
            FixpipeParamsV220 fixpipeParams;                                                                           \
            fixpipeParams = {n, m, m, n, reluEnable};                                                                  \
            fixpipeParams.ndNum = 1;                                                                                   \
            fixpipeParams.srcNdStride = 0;                                                                             \
            fixpipeParams.dstNdStride = 0;                                                                             \
            FixpipeFunc(deqMode, static_cast<float>(0.5), deqTensorGm, reluEnable, fixpipeParams);                     \
            if (fixpipeParams.quantPre == QuantMode_t::VDEQF16 ||                                                      \
                fixpipeParams.quantPre == QuantMode_t::VQF322B8_PRE || fixpipeParams.quantPre == QuantMode_t::VREQ8) { \
                Fixpipe<TensorTrait<dstT>, TensorTrait<l1OutT>, CFG_ROW_MAJOR>(outputGm, dstL0c, cbufWorkspace, fixpipeParams);              \
            } else {                                                                                                   \
                Fixpipe<TensorTrait<dstT>, TensorTrait<l1OutT>, CFG_ROW_MAJOR>(outputGm, dstL0c, fixpipeParams);       \
            }                                                                                                          \
        } else {                                                                                                       \
            FixpipeParamsV220 fixpipeParams;                                                                           \
            fixpipeParams = {n, m, m, static_cast<uint16_t>(m * BLOCK_CUBE * dstTSize / 32), reluEnable};              \
            FixpipeFunc(deqMode, static_cast<float>(0.5), deqTensorGm, reluEnable, fixpipeParams);                     \
            if (fixpipeParams.quantPre == QuantMode_t::VDEQF16 ||                                                      \
                fixpipeParams.quantPre == QuantMode_t::VQF322B8_PRE || fixpipeParams.quantPre == QuantMode_t::VREQ8) { \
                Fixpipe<TensorTrait<dstT>, TensorTrait<l1OutT>, CFG_NZ>(outputGm, dstL0c, cbufWorkspace, fixpipeParams);                     \
            } else {                                                                                                   \
                Fixpipe<TensorTrait<dstT>, TensorTrait<l1OutT>, CFG_NZ>(outputGm, dstL0c, fixpipeParams);              \
            }                                                                                                          \
        }                                                                                                              \
        PipeBarrier<PIPE_ALL>();                                                                                       \
    }

KERNEL_FIXPIPE(half, half, float, float, 2, 2, 4, 4, f322f32_relu_v2, QuantMode_t::NoQuant, FIXPIPE_RELU_FUNC)
KERNEL_FIXPIPE(int8_t, int8_t, int32_t, int32_t, 1, 1, 4, 4, s322s32_relu_v2, QuantMode_t::NoQuant, FIXPIPE_RELU_FUNC)
KERNEL_FIXPIPE(half, half, float, half, 2, 2, 4, 2, f322f16_relu_v2, QuantMode_t::F322F16, FIXPIPE_DEQ_CONV_RELU_FUNC)
KERNEL_FIXPIPE(int8_t, int8_t, int32_t, half, 1, 1, 4, 2, s322f16_scalar_relu_v2, QuantMode_t::DEQF16,
    FIXPIPE_DEQ_SCALAR_RELU_FUNC)
KERNEL_FIXPIPE(int8_t, int8_t, int32_t, half, 1, 1, 4, 2, s322f16_tensor_relu_v2, QuantMode_t::VDEQF16,
    FIXPIPE_DEQ_TENSOR_RELU_FUNC)
KERNEL_FIXPIPE(half, half, float, int8_t, 2, 2, 4, 1, f322s8_scalar_relu_v2, QuantMode_t::QF322B8_PRE,
    FIXPIPE_DEQ_SCALAR_RELU_FUNC)
KERNEL_FIXPIPE(half, half, float, uint8_t, 2, 2, 4, 1, f322u8_tensor_relu_v2, QuantMode_t::VQF322B8_PRE,
    FIXPIPE_DEQ_TENSOR_RELU_FUNC)
KERNEL_FIXPIPE(int8_t, int8_t, int32_t, int8_t, 1, 1, 4, 1, s322s8_scalar_relu_v2, QuantMode_t::REQ8,
    FIXPIPE_DEQ_SCALAR_RELU_FUNC)
KERNEL_FIXPIPE(int8_t, int8_t, int32_t, uint8_t, 1, 1, 4, 1, s322u8_tensor_relu_v2, QuantMode_t::VREQ8,
    FIXPIPE_DEQ_TENSOR_RELU_FUNC)

struct FixpipeTestParamsV2 {
    FixpipeInputParamsV2 inputParams;
    uint8_t fmTSize;
    uint8_t wTSize;
    uint8_t l1OutTSize;
    uint8_t dstTSize;
    void (*calFunc)(uint8_t*, uint8_t*, uint8_t*, uint8_t*, const FixpipeInputParamsV2&);
    bool reluEnable;
    bool enNz2nd;
};

class FixpipeTestsuiteV2 : public testing::Test, public testing::WithParamInterface<FixpipeTestParamsV2> {
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

INSTANTIATE_TEST_CASE_P(TEST_FIXPIPEV2, FixpipeTestsuiteV2,
    ::testing::Values(
    FixpipeTestParamsV2 { { 2, 4, 4, 2, 2, 128, 16, 2, 2 }, 2, 2, 4, 4, kernel_fixpipe_f322f32_relu_v2, true, true },
    FixpipeTestParamsV2 { { 2, 4, 4, 2, 2, 16, 16, 2, 2 }, 2, 2, 4, 4, kernel_fixpipe_f322f32_relu_v2, false, false },
    FixpipeTestParamsV2 { { 1, 4, 4, 2, 2, 128, 32, 1, 1 }, 1, 1, 4, 4, kernel_fixpipe_s322s32_relu_v2, true, true },
    FixpipeTestParamsV2 { { 1, 4, 4, 2, 2, 32, 32, 1, 1 }, 1, 1, 4, 4, kernel_fixpipe_s322s32_relu_v2, false, false },
    FixpipeTestParamsV2 { { 2, 4, 4, 2, 2, 128, 16, 2, 2 }, 2, 2, 4, 2, kernel_fixpipe_f322f16_relu_v2, true, true },
    FixpipeTestParamsV2 { { 2, 4, 4, 2, 2, 16, 16, 2, 2 }, 2, 2, 4, 2, kernel_fixpipe_f322f16_relu_v2, true, true },
    FixpipeTestParamsV2 { { 2, 4, 4, 2, 2, 16, 16, 2, 2 }, 2, 2, 4, 2, kernel_fixpipe_f322f16_relu_v2, false, false },
    FixpipeTestParamsV2 { { 1, 4, 4, 2, 2, 128, 32, 1, 1 }, 1, 1, 4, 2, kernel_fixpipe_s322f16_scalar_relu_v2, true, true },
    FixpipeTestParamsV2 { { 1, 4, 4, 2, 2, 32, 32, 1, 1 }, 1, 1, 4, 2, kernel_fixpipe_s322f16_scalar_relu_v2, false, false },
    FixpipeTestParamsV2 { { 1, 4, 4, 2, 2, 128, 32, 1, 1 }, 1, 1, 4, 2, kernel_fixpipe_s322f16_tensor_relu_v2, true, true },
    FixpipeTestParamsV2 { { 1, 4, 4, 2, 2, 128, 32, 1, 1 }, 1, 1, 4, 2, kernel_fixpipe_s322f16_tensor_relu_v2, true, false },
    FixpipeTestParamsV2 { { 1, 4, 4, 2, 2, 32, 32, 1, 1 }, 1, 1, 4, 2, kernel_fixpipe_s322f16_tensor_relu_v2, false, true },
    FixpipeTestParamsV2 { { 1, 4, 4, 2, 2, 32, 32, 1, 1 }, 1, 1, 4, 2, kernel_fixpipe_s322f16_tensor_relu_v2, false, false },
    FixpipeTestParamsV2 { { 1, 4, 4, 2, 2, 32, 32, 1, 1 }, 2, 2, 4, 1, kernel_fixpipe_f322s8_scalar_relu_v2, false, false },
    FixpipeTestParamsV2 { { 1, 4, 4, 2, 2, 128, 32, 1, 1 }, 2, 2, 4, 1, kernel_fixpipe_f322u8_tensor_relu_v2, true, true },
    FixpipeTestParamsV2 { { 1, 4, 4, 2, 2, 32, 32, 1, 1 }, 1, 1, 4, 1, kernel_fixpipe_s322s8_scalar_relu_v2, false, false },
    FixpipeTestParamsV2 { { 1, 4, 4, 2, 2, 128, 32, 1, 1 }, 1, 1, 4, 1, kernel_fixpipe_s322u8_tensor_relu_v2, true, true }
    ));

TEST_P(FixpipeTestsuiteV2, FixpipeTestCaseV2)
{
    auto param = GetParam();

    uint8_t fmData[param.inputParams.featureMapSize * param.fmTSize] = { 0 };
    uint8_t wtData[param.inputParams.weightSize * param.wTSize] = { 0 };
    uint8_t deqTensor[param.inputParams.deqSize * sizeof(uint64_t)] = { 0 };
    uint8_t outputData[param.inputParams.dstSize * param.dstTSize] = { 0 };

    param.inputParams.reluEnable = param.reluEnable;
    param.inputParams.enNz2nd = param.enNz2nd;
    param.calFunc(fmData, wtData, deqTensor, outputData, param.inputParams);
    for (int32_t i = 0; i < param.inputParams.dstSize * param.dstTSize; i++) {
        EXPECT_EQ(outputData[i], 0x00);
    }
}

TEST_P(FixpipeTestsuiteV2, FixpipeTestCaseCheckGmOverflowV2)
{
    auto param = GetParam();

    uint8_t fmData[param.inputParams.featureMapSize * param.fmTSize] = { 0 };
    uint8_t wtData[param.inputParams.weightSize * param.wTSize] = { 0 };
    uint8_t deqTensor[param.inputParams.deqSize * sizeof(uint64_t)] = { 0 };
    uint8_t outputData[param.inputParams.dstSize * param.dstTSize] = { 0 };
    constexpr size_t workspaceSize = AscendC::RESERVED_WORKSPACE;
    uint8_t* sysWorkSpacePtr = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    memset(sysWorkSpacePtr, 0, workspaceSize);
    if (sysWorkSpacePtr == nullptr) {
        printf("[error]g_sysWorkspaceReserved is null, g_sysWorkspaceReserved has been set or not\n");
    }
    g_sysWorkspaceReserved = sysWorkSpacePtr;
    uint8_t* workspace = GetSysWorkSpacePtr();
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024)) = 2;
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 8)) = 1;
    *((__gm__ uintptr_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 16)) = reinterpret_cast<uintptr_t>(fmData);
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 24)) = param.inputParams.featureMapSize * param.fmTSize;
    *((__gm__ uintptr_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 32)) = reinterpret_cast<uintptr_t>(wtData);
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 40)) = param.inputParams.weightSize * param.wTSize;
    *((__gm__ uintptr_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 48)) = reinterpret_cast<uintptr_t>(outputData);
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 56)) = param.inputParams.dstSize * param.dstTSize;

    param.inputParams.reluEnable = param.reluEnable;
    param.inputParams.enNz2nd = param.enNz2nd;
    param.calFunc(fmData, wtData, deqTensor, outputData, param.inputParams);
    AscendC::GmFree((void*)sysWorkSpacePtr);
    g_sysWorkspaceReserved = nullptr; 
    for (int32_t i = 0; i < param.inputParams.dstSize * param.dstTSize; i++) {
        EXPECT_EQ(outputData[i], 0x00);
    }
}
} // namespace AscendC
