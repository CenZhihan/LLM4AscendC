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

#define LOCAL_TENSOR_REGISTER(tensor_name, type, que_pos, init_addr, data_size) \
    LocalTensor<type> tensor_name;                                     \
    TBuffAddr tbuf_##tensor_name;                                      \
    tbuf_##tensor_name.logicPos = (uint8_t)(TPosition::que_pos);       \
    tensor_name.SetAddr(tbuf_##tensor_name);                                    \
    tensor_name.InitBuffer(init_addr, data_size);

struct FixpipeInputParams {
    __aicore__ FixpipeInputParams() {}
    __aicore__ FixpipeInputParams(const uint16_t C1_in, const uint16_t H_in, const uint16_t W_in, const uint8_t Kh_in,
        const uint8_t Kw_in, const uint16_t Cout_in, const uint16_t C0_in, const uint16_t dilation_h_in,
        const uint16_t dilation_w_in)
    {
        C1 = C1_in;
        H = H_in;
        W = W_in;
        Kh = Kh_in;
        Kw = Kw_in;
        Cout = Cout_in;
        C0 = C0_in;
        dilation_h = dilation_h_in;
        dilation_w = dilation_w_in;

        cout_blocks = (Cout + 16 - 1) / 16;
        ho = H - dilation_h * (Kh - 1);
        wo = W - dilation_w * (Kw - 1);
        howo = ho * wo;
        howo_round = ((howo + 16 - 1) / 16) * 16;

        feature_map_size = C1 * H * W * C0;     // [C1, H, W, C0]
        weight_size = C1 * Kh * Kw * Cout * C0; // [C1, Kh, Kw, Cout, C0]
        feature_map_l0a_size = howo_round * (C1 * Kh * Kw * C0);
        weight_l0b_size = (C1 * Kh * Kw * C0) * cout_blocks * 16;
        m = howo;
        k = C1 * Kh * Kw * C0;
        n = Cout;
        deq_size = Cout;                    // [Cout]
        dst_size = cout_blocks * howo * 16; // [cout_blocks, howo, 16]
        dst_l0c_size = cout_blocks * howo_round * 16;

        fm_repeat = feature_map_l0a_size / (16 * C0);
        we_repeat = weight_l0b_size / (16 * C0);
    }

    uint16_t C1;
    uint16_t H;
    uint16_t W;
    uint8_t Kh;
    uint8_t Kw;
    uint16_t Cout;
    uint16_t C0;
    uint8_t dilation_h;
    uint8_t dilation_w;
    bool relu_en;
    CO2Layout format;

    uint16_t cout_blocks;
    uint16_t ho;
    uint16_t wo;
    uint16_t howo;
    uint16_t howo_round;

    uint32_t feature_map_size;
    uint32_t weight_size;
    uint32_t feature_map_l0a_size;
    uint32_t weight_l0b_size;
    uint16_t m;
    uint16_t k;
    uint16_t n;
    uint32_t deq_size;
    uint32_t dst_size;
    uint32_t dst_l0c_size;

    uint8_t fm_repeat;
    uint8_t we_repeat;
};

/* **************************************************************************************************
 * Fixpipe                                             *
 * ************************************************************************************************* */
#define FIXPIPE_RELU_FUNC(deq_mode, deq_scalar, en_relu, fixpipeParams) fixpipeParams.reluEn = en_relu;

#define FIXPIPE_DEQ_CONV_RELU_FUNC(deq_mode, deq_scalar, en_relu, fixpipeParams) \
    {                                                                            \
        fixpipeParams.quantPre = deq_mode;                                       \
        fixpipeParams.reluEn = en_relu;                                          \
    }

#define FIXPIPE_DEQ_SCALAR_RELU_FUNC(deq_mode, deq_scalar, en_relu, fixpipeParams) \
    {                                                                              \
        fixpipeParams.quantPre = deq_mode;                                         \
        fixpipeParams.deqScalar = deq_scalar;                                      \
        fixpipeParams.reluEn = en_relu;                                            \
    }  // namespace AscendC

#define KERNEL_FIXPIPE(                                                                                                \
    fm_T, w_T, l1out_T, dst_T, fm_T_size, w_T_size, l1out_T_size, dst_T_size, name, deq_mode, fixpipe_func)            \
    extern "C" __global__ __aicore__ void kernel_fixpipe_##name(GM_ADDR fm_data,                                       \
        GM_ADDR we_data,                                                                                               \
        GM_ADDR deq_tensor,                                                                                            \
        GM_ADDR output_data,                                                                                           \
        const FixpipeInputParams &inputParams)                                                                         \
    {                                                                                                                  \
        TPipe tpipe;                                                                                                   \
        const uint16_t C1 = inputParams.C1;                                                                            \
        const uint16_t H = inputParams.H;                                                                              \
        const uint16_t W = inputParams.W;                                                                              \
        const uint8_t Kh = inputParams.Kh;                                                                             \
        const uint8_t Kw = inputParams.Kw;                                                                             \
        const uint16_t Cout = inputParams.Cout;                                                                        \
        const uint16_t C0 = inputParams.C0;                                                                            \
        const uint8_t dilation_h = inputParams.dilation_h;                                                             \
        const uint8_t dilation_w = inputParams.dilation_w;                                                             \
        const bool relu_en = inputParams.relu_en;                                                                      \
        const CO2Layout format = inputParams.format;                                                                   \
                                                                                                                       \
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);                                                                        \
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);                                                                       \
                                                                                                                       \
        const uint16_t cout_blocks = inputParams.cout_blocks;                                                          \
        const uint16_t ho = inputParams.ho;                                                                            \
        const uint16_t wo = inputParams.wo;                                                                            \
        const uint16_t howo = inputParams.howo;                                                                        \
        const uint16_t howo_round = inputParams.howo_round;                                                            \
                                                                                                                       \
        const uint32_t feature_map_size = inputParams.feature_map_size;                                                \
        const uint32_t weight_size = inputParams.weight_size;                                                          \
        const uint32_t feature_map_l0a_size = inputParams.feature_map_l0a_size;                                        \
        const uint32_t weight_l0b_size = inputParams.weight_l0b_size;                                                  \
        const uint16_t m = inputParams.m;                                                                              \
        const uint16_t k = inputParams.k;                                                                              \
        const uint16_t n = inputParams.n;                                                                              \
        const uint32_t deq_size = inputParams.deq_size;                                                                \
        const uint32_t dst_size = inputParams.dst_size;                                                                \
        const uint32_t dst_l0c_size = inputParams.dst_l0c_size;                                                        \
                                                                                                                       \
        const uint8_t fm_repeat = inputParams.fm_repeat;                                                               \
        const uint8_t we_repeat = inputParams.we_repeat;                                                               \
                                                                                                                       \
        GlobalTensor<fm_T> feature_map_gm;                                                                             \
        GlobalTensor<w_T> weight_gm;                                                                                   \
        GlobalTensor<uint64_t> deq_tensor_gm;                                                                          \
        GlobalTensor<dst_T> output_gm;                                                                                 \
        feature_map_gm.SetGlobalBuffer(reinterpret_cast<__gm__ fm_T *>(fm_data), feature_map_size);                    \
        weight_gm.SetGlobalBuffer(reinterpret_cast<__gm__ w_T *>(we_data), weight_size);                               \
        deq_tensor_gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(deq_tensor), deq_size);                      \
        output_gm.SetGlobalBuffer(reinterpret_cast<__gm__ dst_T *>(output_data), dst_size);                            \
                                                                                                                       \
        LOCAL_TENSOR_REGISTER(feature_map_l1, fm_T, A1, 0, feature_map_size)                                           \
        LOCAL_TENSOR_REGISTER(weight_l1, w_T, B1, feature_map_size *fm_T_size, weight_size)                            \
                                                                                                                       \
        LOCAL_TENSOR_REGISTER(feature_map_l0a, fm_T, A2, 0, feature_map_l0a_size)                                      \
        LOCAL_TENSOR_REGISTER(weight_l0b, w_T, B2, 0, weight_l0b_size)                                                 \
        LOCAL_TENSOR_REGISTER(dst_l0c, l1out_T, CO1, 0, dst_l0c_size)                                                  \
                                                                                                                       \
        DataCopy(feature_map_l1, feature_map_gm, {1, static_cast<uint16_t>(feature_map_size * (fm_T_size) / 32), 0, 0}); \
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);                                                                     \
        DataCopy(weight_l1, weight_gm, {1, static_cast<uint16_t>(weight_size * (w_T_size) / 32), 0, 0});               \
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);                                                                     \
        SetLoadDataRepeat({0, 1, 0, howo_round / 16});                                                                 \
        uint8_t padList[PAD_SIZE] = {0, 0, 0, 0};                                                                      \
        LoadData<fm_T>(feature_map_l0a,                                                                                \
            feature_map_l1,                                                                                            \
            {padList, H, W, 32, 128, 16, 0, 0, Kw, Kh, dilation_w, dilation_h, 2, 2, false, false, 0});                 \
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
            cbufWorkspace, uint64_t, C1, feature_map_size *fm_T_size + weight_size * w_T_size, deq_size)               \
        if (format == CO2Layout::ROW_MAJOR) {                                                                          \
            FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams = {n, m, static_cast<uint16_t>(AlignUp(m, BLOCK_CUBE)), n}; \
            fixpipeParams.params = {1, 0, 0};                                                                          \
            fixpipe_func(deq_mode, (float)0.5, relu_en, fixpipeParams);                                                \
            if (fixpipeParams.quantPre == QuantMode_t::VDEQF16 ||                                                      \
                fixpipeParams.quantPre == QuantMode_t::VQF322B8_PRE ||                                                 \
                fixpipeParams.quantPre == QuantMode_t::VREQ8 || fixpipeParams.quantPre == QuantMode_t::VQF322FP8_PRE) { \
                Fixpipe<dst_T, l1out_T, CFG_ROW_MAJOR>(output_gm, dst_l0c, cbufWorkspace, fixpipeParams);              \
            } else {                                                                                                   \
                Fixpipe<dst_T, l1out_T, CFG_ROW_MAJOR>(output_gm, dst_l0c, fixpipeParams);                             \
            }                                                                                                          \
        } else if (format == CO2Layout::COLUMN_MAJOR) {                                                                \
            FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> fixpipeParams = {n, m, static_cast<uint16_t>(AlignUp(m, BLOCK_CUBE)), m}; \
            fixpipeParams.params = { 1, 0, 0, 1 };                                                                     \
            fixpipe_func(deq_mode, (float)0.5, relu_en, fixpipeParams);                                                \
            if (fixpipeParams.quantPre == QuantMode_t::VDEQF16 ||                                                      \
                fixpipeParams.quantPre == QuantMode_t::VQF322B8_PRE ||                                                 \
                fixpipeParams.quantPre == QuantMode_t::VREQ8 || fixpipeParams.quantPre == QuantMode_t::VQF322FP8_PRE) { \
                Fixpipe<dst_T, l1out_T, CFG_COLUMN_MAJOR>(output_gm, dst_l0c, cbufWorkspace, fixpipeParams);           \
            } else {                                                                                                   \
                Fixpipe<dst_T, l1out_T, CFG_COLUMN_MAJOR>(output_gm, dst_l0c, fixpipeParams);                          \
            }                                                                                                          \
        } else if (format == CO2Layout::NZ) {                                                                          \
            FixpipeParamsArch3510<CO2Layout::NZ> fixpipeParams = {static_cast<uint16_t>(cout_blocks * BLOCK_CUBE), m, static_cast<uint16_t>(AlignUp(m, BLOCK_CUBE)), m * BLOCK_CUBE}; \
            fixpipe_func(deq_mode, (float)0.5, relu_en, fixpipeParams);                                                \
            if (fixpipeParams.quantPre == QuantMode_t::VDEQF16 ||                                                      \
                fixpipeParams.quantPre == QuantMode_t::VQF322B8_PRE ||                                                 \
                fixpipeParams.quantPre == QuantMode_t::VREQ8 || fixpipeParams.quantPre == QuantMode_t::VQF322FP8_PRE) { \
                Fixpipe<dst_T, l1out_T, CFG_NZ>(output_gm, dst_l0c, cbufWorkspace, fixpipeParams);                     \
            } else {                                                                                                   \
                Fixpipe<dst_T, l1out_T, CFG_NZ>(output_gm, dst_l0c, fixpipeParams);                                    \
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
    uint8_t fm_T_size;
    uint8_t w_T_size;
    uint8_t l1out_T_size;
    uint8_t dst_T_size;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint8_t*, const FixpipeInputParams&);
    bool relu_en;
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

    uint8_t fm_data[param.inputParams.feature_map_size * param.fm_T_size] = { 0 };
    uint8_t wt_data[param.inputParams.weight_size * param.w_T_size] = { 0 };
    uint8_t deq_tensor[param.inputParams.deq_size * sizeof(uint64_t)] = { 0 };
    uint8_t output_data[param.inputParams.dst_size * param.dst_T_size] = { 0 };

    param.inputParams.relu_en = param.relu_en;
    param.inputParams.format = param.format;
    param.cal_func(fm_data, wt_data, deq_tensor, output_data, param.inputParams);
    for (int32_t i = 0; i < param.inputParams.dst_size * param.dst_T_size; i++) {
        EXPECT_EQ(output_data[i], 0x00);
    }
}

TEST_P(FixpipeTestsuite, FixpipeTestCaseCheckGmOverflow)
{
    auto param = GetParam();

    uint8_t fm_data[param.inputParams.feature_map_size * param.fm_T_size] = { 0 };
    uint8_t wt_data[param.inputParams.weight_size * param.w_T_size] = { 0 };
    uint8_t deq_tensor[param.inputParams.deq_size * sizeof(uint64_t)] = { 0 };
    uint8_t output_data[param.inputParams.dst_size * param.dst_T_size] = { 0 };
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
    *((__gm__ uintptr_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 16)) = reinterpret_cast<uintptr_t>(fm_data);
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 24)) = param.inputParams.feature_map_size * param.fm_T_size;
    *((__gm__ uintptr_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 32)) = reinterpret_cast<uintptr_t>(wt_data);
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 40)) = param.inputParams.weight_size * param.w_T_size;
    *((__gm__ uintptr_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 48)) = reinterpret_cast<uintptr_t>(output_data);
    *((__gm__ uint64_t *)((__gm__ uint8_t *)workspace + 11 * 1024 * 1024 + 56)) = param.inputParams.dst_size * param.dst_T_size;

    param.inputParams.relu_en = param.relu_en;
    param.inputParams.format = param.format;
    param.cal_func(fm_data, wt_data, deq_tensor, output_data, param.inputParams);
    GmFree((void*)sysWorkSpacePtr);
    SetSysWorkSpacePtr(nullptr); 
    for (int32_t i = 0; i < param.inputParams.dst_size * param.dst_T_size; i++) {
        EXPECT_EQ(output_data[i], 0x00);
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
