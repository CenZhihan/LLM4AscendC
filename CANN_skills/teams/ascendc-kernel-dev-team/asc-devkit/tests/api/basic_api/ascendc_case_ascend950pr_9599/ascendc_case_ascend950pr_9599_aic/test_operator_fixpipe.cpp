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
#include "securec.h"

using namespace std;

namespace AscendC {

#define LOCAL_TENSOR_REGISTER(tensorName, type, quePos, initAddr, dataSize) \
    LocalTensor<type> tensorName;                                     \
    TBuffAddr tbuf_##tensorName;                                      \
    tbuf_##tensorName.logicPos = static_cast<uint8_t>(TPosition::quePos);       \
    tensorName.SetAddr(tbuf_##tensorName);                                    \
    tensorName.InitBuffer(initAddr, dataSize);

struct FixpipeInputParams {
    __aicore__ FixpipeInputParams() {}
    __aicore__ FixpipeInputParams(const uint16_t c1In, const uint16_t hIn, const uint16_t wIn, const uint8_t khIn,
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
    bool reluEn;
    CO2Layout format;

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
#define FIXPIPE_RELU_FUNC(deqMode, deqScalar1, enRelu, fixpipeParams) fixpipeParams.reluEn = enRelu;

#define FIXPIPE_DEQ_CONV_RELU_FUNC(deqMode, deqScalar1, enRelu, fixpipeParams) \
    {                                                                            \
        fixpipeParams.quantPre = deqMode;                                       \
        fixpipeParams.reluEn = enRelu;                                          \
    }

#define FIXPIPE_DEQ_SCALAR_RELU_FUNC(deqMode, deqScalar1, enRelu, fixpipeParams) \
    {                                                                              \
        fixpipeParams.quantPre = deqMode;                                         \
        fixpipeParams.deqScalar = deqScalar1;                                      \
        fixpipeParams.reluEn = enRelu;                                            \
    }  // namespace AscendC

#define KERNEL_FIXPIPE(                                                                                                \
    fmT, wT, l1outT, dstT, fmTSize, wTSize, l1outTSize, dstTSize, name, deqMode, fixpipeFunc)            \
    extern "C" __global__ __aicore__ void kernel_fixpipe_##name(GM_ADDR fmData,                                       \
        GM_ADDR weData,                                                                                               \
        GM_ADDR deqTensor,                                                                                            \
        GM_ADDR outputData,                                                                                           \
        const FixpipeInputParams &inputParams)                                                                         \
    {                                                                                                                  \
        TPipe tpipe;                                                                                                   \
        const uint16_t c1 = inputParams.c1;                                                                            \
        const uint16_t h = inputParams.h;                                                                              \
        const uint16_t w = inputParams.w;                                                                              \
        const uint8_t kh = inputParams.kh;                                                                             \
        const uint8_t kw = inputParams.kw;                                                                             \
        const uint16_t cout = inputParams.cout;                                                                        \
        const uint16_t c0 = inputParams.c0;                                                                            \
        const uint8_t dilationH = inputParams.dilationH;                                                             \
        const uint8_t dilationW = inputParams.dilationW;                                                             \
        const bool reluEn = inputParams.reluEn;                                                                      \
        const CO2Layout format = inputParams.format;                                                                   \
                                                                                                                       \
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);                                                                        \
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);                                                                       \
                                                                                                                       \
        const uint16_t coutBlocks = inputParams.coutBlocks;                                                          \
        const uint16_t ho = inputParams.ho;                                                                            \
        const uint16_t wo = inputParams.wo;                                                                            \
        const uint16_t howo = inputParams.howo;                                                                        \
        const uint16_t howoRound = inputParams.howoRound;                                                            \
                                                                                                                       \
        const uint32_t featureMapSize = inputParams.featureMapSize;                                                \
        const uint32_t weightSize = inputParams.weightSize;                                                          \
        const uint32_t featureMapL0aSize = inputParams.featureMapL0aSize;                                        \
        const uint32_t weightL0bSize = inputParams.weightL0bSize;                                                  \
        const uint16_t m = inputParams.m;                                                                              \
        const uint16_t k = inputParams.k;                                                                              \
        const uint16_t n = inputParams.n;                                                                              \
        const uint32_t deqSize = inputParams.deqSize;                                                                \
        const uint32_t dstSize = inputParams.dstSize;                                                                \
        const uint32_t dstL0cSize = inputParams.dstL0cSize;                                                        \
                                                                                                                       \
        const uint8_t fmRepeat = inputParams.fmRepeat;                                                               \
        const uint8_t weRepeat = inputParams.weRepeat;                                                               \
                                                                                                                       \
        GlobalTensor<TensorTrait<fmT>> featureMapGm;                                                                             \
        GlobalTensor<TensorTrait<wT>> weightGm;                                                                                   \
        GlobalTensor<TensorTrait<uint64_t>> deqTensorGm;                                                                          \
        GlobalTensor<TensorTrait<dstT>> outputGm;                                                                                 \
        featureMapGm.SetGlobalBuffer(reinterpret_cast<__gm__ fmT *>(fmData), featureMapSize);                    \
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ wT *>(weData), weightSize);                               \
        deqTensorGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(deqTensor), deqSize);                      \
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ dstT *>(outputData), dstSize);                            \
                                                                                                                       \
        LOCAL_TENSOR_REGISTER(feature_map_l1, TensorTrait<fmT>, A1, 0, featureMapSize)                                           \
        LOCAL_TENSOR_REGISTER(weight_l1, TensorTrait<wT>, B1, featureMapSize *fmTSize, weightSize)                            \
                                                                                                                       \
        LOCAL_TENSOR_REGISTER(feature_map_l0a, TensorTrait<fmT>, A2, 0, featureMapL0aSize)                                      \
        LOCAL_TENSOR_REGISTER(weight_l0b, TensorTrait<wT>, B2, 0, weightL0bSize)                                                 \
        LOCAL_TENSOR_REGISTER(dst_l0c, TensorTrait<l1outT>, CO1, 0, dstL0cSize)                                                  \
                                                                                                                       \
        DataCopy(feature_map_l1, featureMapGm, {1, static_cast<uint16_t>(featureMapSize * (fmTSize) / 32), 0, 0}); \
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);                                                                     \
        DataCopy(weight_l1, weightGm, {1, static_cast<uint16_t>(weightSize * (wTSize) / 32), 0, 0});               \
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);                                                                     \
        SetLoadDataRepeatWithStride({0, 1, 0, howoRound / 16});                                                                 \
        uint8_t padList[PAD_SIZE] = {0, 0, 0, 0};                                                                      \
        LoadDataWithStride<TensorTrait<fmT>>(feature_map_l0a,                                                                                \
            feature_map_l1,                                                                                            \
            {padList, h, w, 32, 128, 16, 0, 0, kw, kh, dilationW, dilationH, 2, 2, false, false, 0});                 \
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);                                                                    \
                                                                                                                       \
        LoadData2DParamsV2 loadDataParams = {0, 0, 0, 0, 0, 0, false, 0};                                              \
        LoadData(weight_l0b, weight_l1, loadDataParams);                                                               \
                                                                                                                       \
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);                                                                        \
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);                                                                       \
                                                                                                                       \
        Mmad(dst_l0c, feature_map_l0a, weight_l0b, {m, n, k, 0, false, true});                                         \
                                                                                                                       \
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);                                                                    \
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);                                                                    \
                                                                                                                       \
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);                                                                         \
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);                                                                        \
                                                                                                                       \
        LOCAL_TENSOR_REGISTER(                                                                                         \
            cbufWorkspace, uint64_t, C1, featureMapSize *fmTSize + weightSize * wTSize, deqSize)               \
        if (format == CO2Layout::ROW_MAJOR) {                                                                          \
            FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams = {n, m, static_cast<uint16_t>(AlignUp(m, BLOCK_CUBE)), n}; \
            fixpipeParams.params = {1, 0, 0};                                                                          \
            fixpipeFunc(deqMode, static_cast<float>(0.5), reluEn, fixpipeParams);                                                \
            if (fixpipeParams.quantPre == QuantMode_t::VDEQF16 ||                                                      \
                fixpipeParams.quantPre == QuantMode_t::VQF322B8_PRE ||                                                 \
                fixpipeParams.quantPre == QuantMode_t::VREQ8 || fixpipeParams.quantPre == QuantMode_t::VQF322FP8_PRE) { \
                Fixpipe<TensorTrait<dstT>, TensorTrait<l1outT>, CFG_ROW_MAJOR>(outputGm, dst_l0c, cbufWorkspace, fixpipeParams);              \
            } else {                                                                                                   \
                Fixpipe<TensorTrait<dstT>, TensorTrait<l1outT>, CFG_ROW_MAJOR>(outputGm, dst_l0c, fixpipeParams);                             \
            }                                                                                                          \
        } else if (format == CO2Layout::COLUMN_MAJOR) {                                                                \
            FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> fixpipeParams = {n, m, static_cast<uint16_t>(AlignUp(m, BLOCK_CUBE)), m}; \
            fixpipeParams.params = { 1, 0, 0, 1 };                                                                     \
            fixpipeFunc(deqMode, static_cast<float>(0.5), reluEn, fixpipeParams);                                                \
            if (fixpipeParams.quantPre == QuantMode_t::VDEQF16 ||                                                      \
                fixpipeParams.quantPre == QuantMode_t::VQF322B8_PRE ||                                                 \
                fixpipeParams.quantPre == QuantMode_t::VREQ8 || fixpipeParams.quantPre == QuantMode_t::VQF322FP8_PRE) { \
                Fixpipe<TensorTrait<dstT>, TensorTrait<l1outT>, CFG_COLUMN_MAJOR>(outputGm, dst_l0c, cbufWorkspace, fixpipeParams);           \
            } else {                                                                                                   \
                Fixpipe<TensorTrait<dstT>, TensorTrait<l1outT>, CFG_COLUMN_MAJOR>(outputGm, dst_l0c, fixpipeParams);                          \
            }                                                                                                          \
        } else if (format == CO2Layout::NZ) {                                                                          \
            FixpipeParamsArch3510<CO2Layout::NZ> fixpipeParams = {static_cast<uint16_t>(coutBlocks * BLOCK_CUBE), m, static_cast<uint16_t>(AlignUp(m, BLOCK_CUBE)), m * BLOCK_CUBE}; \
            fixpipeFunc(deqMode, static_cast<float>(0.5), reluEn, fixpipeParams);                                                \
            if (fixpipeParams.quantPre == QuantMode_t::VDEQF16 ||                                                      \
                fixpipeParams.quantPre == QuantMode_t::VQF322B8_PRE ||                                                 \
                fixpipeParams.quantPre == QuantMode_t::VREQ8 || fixpipeParams.quantPre == QuantMode_t::VQF322FP8_PRE) { \
                Fixpipe<TensorTrait<dstT>, TensorTrait<l1outT>, CFG_NZ>(outputGm, dst_l0c, cbufWorkspace, fixpipeParams);                     \
            } else {                                                                                                   \
                Fixpipe<TensorTrait<dstT>, TensorTrait<l1outT>, CFG_NZ>(outputGm, dst_l0c, fixpipeParams);                                    \
            }                                                                                                          \
        }                                                                                                              \
        pipe_barrier(PIPE_ALL);                                                                                        \
    }

KERNEL_FIXPIPE(half, half, float, float, 2, 2, 4, 4, f322f32_relu, QuantMode_t::NoQuant, FIXPIPE_RELU_FUNC)
KERNEL_FIXPIPE(int8_t, int8_t, int32_t, int32_t, 1, 1, 4, 4, s322s32_relu, QuantMode_t::NoQuant, FIXPIPE_RELU_FUNC)
KERNEL_FIXPIPE(half, half, float, half, 2, 2, 4, 2, f322f16_relu, QuantMode_t::F322F16, FIXPIPE_DEQ_CONV_RELU_FUNC)
KERNEL_FIXPIPE(int8_t, int8_t, int32_t, half, 1, 1, 4, 2, s322f16_scalar_relu, QuantMode_t::DEQF16,
    FIXPIPE_DEQ_SCALAR_RELU_FUNC)
KERNEL_FIXPIPE(int8_t, int8_t, int32_t, half, 1, 1, 4, 2, s322f16_tensor_relu, QuantMode_t::VDEQF16,
    FIXPIPE_DEQ_CONV_RELU_FUNC)
KERNEL_FIXPIPE(half, half, float, int8_t, 2, 2, 4, 1, f322s8_scalar_relu, QuantMode_t::QF322B8_PRE,
    FIXPIPE_DEQ_SCALAR_RELU_FUNC)
KERNEL_FIXPIPE(half, half, float, uint8_t, 2, 2, 4, 1, f322u8_tensor_relu, QuantMode_t::VQF322B8_PRE,
    FIXPIPE_DEQ_CONV_RELU_FUNC)
KERNEL_FIXPIPE(int8_t, int8_t, int32_t, int8_t, 1, 1, 4, 1, s322s8_scalar_relu, QuantMode_t::REQ8,
    FIXPIPE_DEQ_SCALAR_RELU_FUNC)
KERNEL_FIXPIPE(int8_t, int8_t, int32_t, uint8_t, 1, 1, 4, 1, s322u8_tensor_relu, QuantMode_t::VREQ8,
FIXPIPE_DEQ_CONV_RELU_FUNC)
KERNEL_FIXPIPE(half, half, float, bfloat16_t, 2, 2, 4, 2, f322bf16_tensor_relu, QuantMode_t::VQF322BF16_PRE,
    FIXPIPE_DEQ_CONV_RELU_FUNC)
KERNEL_FIXPIPE(int8_t, int8_t, int32_t, bfloat16_t, 1, 1, 4, 2, s322bf16_tensor_relu, QuantMode_t::VQS322BF16_PRE,
    FIXPIPE_DEQ_CONV_RELU_FUNC)
KERNEL_FIXPIPE(fp8_e4m3fn_t, fp8_e4m3fn_t, float, fp8_e4m3fn_t, 1, 1, 4, 1, f322fp8_tensor_relu, QuantMode_t::VQF322FP8_PRE,
    FIXPIPE_DEQ_CONV_RELU_FUNC)
KERNEL_FIXPIPE(fp8_e5m2_t, fp8_e5m2_t, float, fp8_e4m3fn_t, 1, 1, 4, 1, f322fp8_scalar_relu, QuantMode_t::QF322FP8_PRE,
    FIXPIPE_DEQ_SCALAR_RELU_FUNC)
KERNEL_FIXPIPE(hifloat8_t, hifloat8_t, float, hifloat8_t, 1, 1, 4, 1, f322hif8_scalar_relu, QuantMode_t::QF322HIF8_PRE,
    FIXPIPE_DEQ_SCALAR_RELU_FUNC)
KERNEL_FIXPIPE(half, half, float, float, 2, 2, 4, 4, f322f32_pre_scalar_relu, QuantMode_t::VQF322F32_PRE, FIXPIPE_DEQ_CONV_RELU_FUNC)

struct FixpipeTestParams {
    FixpipeInputParams inputParams;
    uint8_t fmTSize;
    uint8_t wTSize;
    uint8_t l1outTSize;
    uint8_t dstTSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint8_t*, const FixpipeInputParams&);
    bool reluEn;
    CO2Layout format;
};

class FixpipeTestsuite : public testing::Test, public testing::WithParamInterface<FixpipeTestParams> {
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

INSTANTIATE_TEST_CASE_P(TEST_FIXPIPE, FixpipeTestsuite,
    ::testing::Values(
    FixpipeTestParams { { 2, 4, 4, 2, 2, 128, 16, 2, 2 }, 2, 2, 4, 4, kernel_fixpipe_f322f32_relu, true, CO2Layout::ROW_MAJOR },
    FixpipeTestParams { { 2, 4, 4, 2, 2, 16, 16, 2, 2 }, 2, 2, 4, 4, kernel_fixpipe_f322f32_relu, false, CO2Layout::NZ },
    FixpipeTestParams { { 2, 4, 4, 2, 2, 16, 16, 2, 2 }, 2, 2, 4, 4, kernel_fixpipe_f322f32_relu, false, CO2Layout::COLUMN_MAJOR },
    FixpipeTestParams { { 1, 4, 4, 2, 2, 128, 32, 1, 1 }, 1, 1, 4, 4, kernel_fixpipe_s322s32_relu, true, CO2Layout::ROW_MAJOR },
    FixpipeTestParams { { 1, 4, 4, 2, 2, 32, 32, 1, 1 }, 1, 1, 4, 4, kernel_fixpipe_s322s32_relu, false, CO2Layout::NZ },
    FixpipeTestParams { { 2, 4, 4, 2, 2, 128, 16, 2, 2 }, 2, 2, 4, 2, kernel_fixpipe_f322f16_relu, true, CO2Layout::ROW_MAJOR },
    FixpipeTestParams { { 2, 4, 4, 2, 2, 16, 16, 2, 2 }, 2, 2, 4, 2, kernel_fixpipe_f322f16_relu, false, CO2Layout::NZ },
    FixpipeTestParams { { 1, 4, 4, 2, 2, 128, 32, 1, 1 }, 1, 1, 4, 2, kernel_fixpipe_s322f16_scalar_relu, true, CO2Layout::ROW_MAJOR },
    FixpipeTestParams { { 1, 4, 4, 2, 2, 32, 32, 1, 1 }, 1, 1, 4, 2, kernel_fixpipe_s322f16_scalar_relu, false, CO2Layout::NZ },
    FixpipeTestParams { { 1, 4, 4, 2, 2, 128, 32, 1, 1 }, 1, 1, 4, 2, kernel_fixpipe_s322f16_tensor_relu, true, CO2Layout::ROW_MAJOR },
    FixpipeTestParams { { 1, 4, 4, 2, 2, 128, 32, 1, 1 }, 1, 1, 4, 2, kernel_fixpipe_s322f16_tensor_relu, true, CO2Layout::NZ },
    FixpipeTestParams { { 1, 4, 4, 2, 2, 32, 32, 1, 1 }, 1, 1, 4, 2, kernel_fixpipe_s322f16_tensor_relu, false, CO2Layout::ROW_MAJOR },
    FixpipeTestParams { { 1, 4, 4, 2, 2, 32, 32, 1, 1 }, 1, 1, 4, 2, kernel_fixpipe_s322f16_tensor_relu, false, CO2Layout::NZ },
    FixpipeTestParams { { 1, 4, 4, 2, 2, 32, 32, 1, 1 }, 2, 2, 4, 1, kernel_fixpipe_f322s8_scalar_relu, false, CO2Layout::NZ },
    FixpipeTestParams { { 1, 4, 4, 2, 2, 128, 32, 1, 1 }, 2, 2, 4, 1, kernel_fixpipe_f322u8_tensor_relu, true, CO2Layout::ROW_MAJOR },
    FixpipeTestParams { { 1, 4, 4, 2, 2, 32, 32, 1, 1 }, 1, 1, 4, 1, kernel_fixpipe_s322s8_scalar_relu, false, CO2Layout::NZ },
    FixpipeTestParams { { 1, 4, 4, 2, 2, 128, 32, 1, 1 }, 1, 1, 4, 1, kernel_fixpipe_s322u8_tensor_relu, true, CO2Layout::ROW_MAJOR },
    FixpipeTestParams { { 1, 4, 4, 2, 2, 128, 32, 1, 1 }, 1, 1, 4, 1, kernel_fixpipe_f322fp8_tensor_relu, true, CO2Layout::ROW_MAJOR },
    FixpipeTestParams { { 1, 4, 4, 2, 2, 128, 32, 1, 1 }, 1, 1, 4, 1, kernel_fixpipe_f322fp8_scalar_relu, true, CO2Layout::ROW_MAJOR },
    FixpipeTestParams { { 1, 4, 4, 2, 2, 128, 32, 1, 1 }, 1, 1, 4, 1, kernel_fixpipe_f322hif8_scalar_relu, true, CO2Layout::ROW_MAJOR },
    FixpipeTestParams { { 2, 4, 4, 2, 2, 128, 16, 2, 2 }, 2, 2, 4, 4, kernel_fixpipe_f322f32_pre_scalar_relu, true, CO2Layout::ROW_MAJOR }
    ));

TEST_P(FixpipeTestsuite, FixpipeTestCase)
{
    auto param = GetParam();

    uint8_t fmData[param.inputParams.featureMapSize * param.fmTSize] = { 0 };
    uint8_t wtData[param.inputParams.weightSize * param.wTSize] = { 0 };
    uint8_t deqTensor[param.inputParams.deqSize * sizeof(uint64_t)] = { 0 };
    uint8_t outputData[param.inputParams.dstSize * param.dstTSize] = { 0 };

    param.inputParams.reluEn = param.reluEn;
    param.inputParams.format = param.format;
    param.cal_func(fmData, wtData, deqTensor, outputData, param.inputParams);
    for (int32_t i = 0; i < param.inputParams.dstSize * param.dstTSize; i++) {
        EXPECT_EQ(outputData[i], 0x00);
    }
}

TEST_P(FixpipeTestsuite, FixpipeTestCaseCheckGmOverflow)
{
    auto param = GetParam();

    uint8_t fmData[param.inputParams.featureMapSize * param.fmTSize] = { 0 };
    uint8_t wtData[param.inputParams.weightSize * param.wTSize] = { 0 };
    uint8_t deqTensor[param.inputParams.deqSize * sizeof(uint64_t)] = { 0 };
    uint8_t outputData[param.inputParams.dstSize * param.dstTSize] = { 0 };
    constexpr size_t workspaceSize = RESERVED_WORKSPACE;
    uint8_t* sysWorkSpacePtr = (uint8_t*)GmAlloc(workspaceSize);
    memset_s(sysWorkSpacePtr, workspaceSize, 0, workspaceSize);
    if (sysWorkSpacePtr == nullptr) {
        printf("[error]g_sysWorkspaceReserved is null, g_sysWorkspaceReserved has been set or not\n");
    }
    SetSysWorkSpacePtr(sysWorkSpacePtr);
    uint8_t* workspace = GetSysWorkSpacePtr();
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024)) = 2;
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 8)) = 1;
    *((__gm__ uintptr_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 16)) = reinterpret_cast<uintptr_t>(fmData);
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 24)) = param.inputParams.featureMapSize * param.fmTSize;
    *((__gm__ uintptr_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 32)) = reinterpret_cast<uintptr_t>(wtData);
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 40)) = param.inputParams.weightSize * param.wTSize;
    *((__gm__ uintptr_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 48)) = reinterpret_cast<uintptr_t>(outputData);
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 56)) = param.inputParams.dstSize * param.dstTSize;

    param.inputParams.reluEn = param.reluEn;
    param.inputParams.format = param.format;
    param.cal_func(fmData, wtData, deqTensor, outputData, param.inputParams);
    GmFree((void*)sysWorkSpacePtr);
    SetSysWorkSpacePtr(nullptr); 
    for (int32_t i = 0; i < param.inputParams.dstSize * param.dstTSize; i++) {
        EXPECT_EQ(outputData[i], 0x00);
    }
}

class TEST_FIXPIPE_SPR : public testing::Test {
protected:
    void SetUp()
    {
        SetGCoreType(1);
    }
    void TearDown()
    {
        SetGCoreType(0);
    }
};

TEST_F(TEST_FIXPIPE_SPR, FIXPIPE_SPR)
{
    LOCAL_TENSOR_REGISTER(quantPre, uint64_t, CO1, 0, 128)
    bool isUnitFlag = false;
    SetFixPipeConfig(quantPre, isUnitFlag);
    SetFixpipeNz2ndFlag(0, 0, 0);
    SetFixpipePreQuantFlag(0);
}
} // namespace AscendC
