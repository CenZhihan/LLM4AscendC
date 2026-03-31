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
// using namespace AscendC;

constexpr AscendC::FixpipeConfig CFG_NZ_UB = {AscendC::CO2Layout::NZ, true};
constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

template <typename DstT, typename SrcT>
constexpr QuantMode_t GetQuantMode() {
    if constexpr (std::is_same_v<SrcT, int32_t> && std::is_same_v<DstT, half>) {
        return QuantMode_t::DEQF16;
    } else if constexpr (std::is_same_v<SrcT, int32_t> && std::is_same_v<DstT, bfloat16_t>) {
        return QuantMode_t::QS322BF16_PRE;
    } else if constexpr (std::is_same_v<SrcT, int32_t> && std::is_same_v<DstT, int8_t>) {
        return QuantMode_t::REQ8;
    } else if constexpr (std::is_same_v<SrcT, int32_t> && std::is_same_v<DstT, uint8_t>) {
        return QuantMode_t::REQ8;
    } else if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, half>) {
        return QuantMode_t::F322F16;
    } else if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, bfloat16_t>) {
        return QuantMode_t::F322BF16;
    } else if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, int8_t>) {
        return QuantMode_t::QF322B8_PRE;
    } else if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, uint8_t>) {
        return QuantMode_t::QF322B8_PRE;
    } else if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, fp8_e4m3fn_t>) {
        return QuantMode_t::QF322FP8_PRE;
    } else if constexpr (std::is_same_v<SrcT, float> && std::is_same_v<DstT, hifloat8_t>) {
        return QuantMode_t::QF322HIF8_PRE;
    }
    return QuantMode_t::NoQuant;
}

template <typename dst_T, typename src0_T, typename src1_T, typename l1out_T, typename bias_T>
void main_cpu_cmp_mmad_bias_demo(__gm__ uint8_t* __restrict__ feature_gm, __gm__ uint8_t* __restrict__ weight_gm,
    __gm__ uint8_t* __restrict__ bias_gm, __gm__ uint8_t* __restrict__ result_gm, int32_t feature_data_size,
    int32_t weight_data_size, int32_t bias_data_size, int32_t output_data_size, bool isBias)
{
    AscendC::TPipe tpipe;
    AscendC::GlobalTensor<src0_T> feature_global;
    AscendC::GlobalTensor<src1_T> weight_global;
    AscendC::GlobalTensor<bias_T> bias_global;
    AscendC::GlobalTensor<dst_T> output_global;
    feature_global.SetGlobalBuffer(reinterpret_cast<__gm__ src0_T*>(feature_gm), feature_data_size);
    weight_global.SetGlobalBuffer(reinterpret_cast<__gm__ src1_T*>(weight_gm), weight_data_size);
    bias_global.SetGlobalBuffer(reinterpret_cast<__gm__ bias_T*>(bias_gm), bias_data_size);
    output_global.SetGlobalBuffer(reinterpret_cast<__gm__ dst_T*>(result_gm), output_data_size);

    AscendC::AscendCUtils::SetOverflow(1);

    // feature map / weight: gm -> l1 nd2nz
    LOCAL_TENSOR_REGISTER(feature_l1, src0_T, A1, 0, feature_data_size)
    LOCAL_TENSOR_REGISTER(weight_l1, src1_T, B1, 0, weight_data_size)
    LoadData(feature_l1, feature_global, {1, 0, 0, 0, 0, 0, 0, 0});
    LoadData(weight_l1, weight_global, {1, 0, 0, 0, 0, 0, 0, 0});


    // feature map: l1 ->l0a load2dv2
    LOCAL_TENSOR_REGISTER(feature_l0a, src0_T, A2, 0, feature_data_size)
    AscendC::LoadData2DParamsV2 loadDataParamsA = {0, 0, 0, 0, 0, 0, false, 0};
    LoadData(feature_l0a, feature_l1, loadDataParamsA);

    // weight: l1 -> l0b load2dv2 with transpose
    LOCAL_TENSOR_REGISTER(weight_l0b, src1_T, B2, 0, weight_data_size)
    AscendC::LoadData2DParamsV2 loadDataParamsB = {0, 0, 0, 0, 0, 0, true, 0};
    LoadData(weight_l0b, weight_l1, loadDataParamsB);

    // bias : gm -> l1
    LOCAL_TENSOR_REGISTER(bias_local, bias_T, C1, 0, bias_data_size)
    DataCopy(bias_local, bias_global, bias_data_size);

    // mmad c = a * b + bias
    LOCAL_TENSOR_REGISTER(l0c_out, l1out_T, CO1, 0, output_data_size)
    AscendC::MmadParams mmad_params;
    mmad_params.m = 112;
    mmad_params.k = 32;
    mmad_params.n = 128;
    mmad_params.cmatrixInitVal = !isBias;
    mmad_params.cmatrixSource = true; // bias in bt
    AscendC::Mmad<l1out_T, src0_T, src1_T>(l0c_out, feature_l0a, weight_l0b, mmad_params);

    // mov l0c to ub
    LOCAL_TENSOR_REGISTER(output_local, dst_T, VECIN, 0, output_data_size)
    uint16_t cburstNum = mmad_params.n / AscendC::BLOCK_CUBE;
    uint16_t burstLen = mmad_params.m * AscendC::BLOCK_CUBE * sizeof(l1out_T) / AscendC::ONE_BLK_SIZE;
    // in 950 l0c to ub nz
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> fixpipeParams(mmad_params.n, mmad_params.m, mmad_params.m, mmad_params.m * AscendC::BLOCK_CUBE);
    fixpipeParams.reluEn = true;
    if (isBias == true) {
        fixpipeParams.reluEn = false;
    }
    fixpipeParams.quantPre = GetQuantMode<dst_T, l1out_T>();
    AscendC::Fixpipe<dst_T, l1out_T, CFG_NZ_UB>(output_local, l0c_out, fixpipeParams);

    // mov l0c to l1
    LOCAL_TENSOR_REGISTER(output_local_l1, dst_T, A1, 0, output_data_size)
    AscendC::Fixpipe<dst_T, l1out_T, AscendC::CFG_NZ>(output_local_l1, l0c_out, fixpipeParams);

    // in 950 l0c to ub row_major
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams2(mmad_params.n, mmad_params.m, mmad_params.m, mmad_params.n);
    fixpipeParams2.dualDstCtl = 0b01;
    fixpipeParams2.quantPre = GetQuantMode<dst_T, l1out_T>();
    if (fixpipeParams2.quantPre == QuantMode_t::NoQuant) {
        AscendC::Fixpipe<dst_T, l1out_T, CFG_ROW_MAJOR_UB>(output_local, l0c_out, fixpipeParams2);
    }

    // mov ub to gm
    DataCopy(output_global, output_local, output_data_size);
    pipe_barrier(PIPE_ALL);
    SetSysWorkSpacePtr(GetSysWorkSpacePtr());
}

class TEST_MMAD_BIAS : public testing::Test {
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

#define VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, BiasOP, dst_Type, src0_Type, src1_Type, l1out_T, bias_T)             \
    TEST_F(TEST_MMAD_BIAS, MMAD_Case_Bias_##BiasOP##_##dst_Type##_##src0_Type##_##src1_Type##_##l1out_T##_##bias_T) \
    {                                                                                                               \
        const int32_t feature_data_size = 3584;                                                                     \
        const int32_t weight_data_size = 4096;                                                                      \
        const int32_t bias_data_size = 128;                                                                         \
        const int32_t output_data_size = 14336;                                                                     \
        uint8_t feature_global[feature_data_size * sizeof(src0_Type)] = {0};                                        \
        uint8_t weight_global[weight_data_size * sizeof(src1_Type)] = {0};                                          \
        uint8_t bias_global[bias_data_size * sizeof(bias_T)] = {0};                                                 \
        uint8_t output_global[output_data_size * sizeof(l1out_T)] = {0};                                            \
        main_cpu_cmp_mmad_bias_demo<dst_Type, src0_Type, src1_Type, l1out_T, bias_T>(feature_global, weight_global, \
            bias_global, output_global, feature_data_size, weight_data_size, bias_data_size, output_data_size,      \
            BiasOP);                                                                                                \
        for (int32_t i = 0; i < output_data_size * sizeof(dst_Type); i++) {                                         \
            EXPECT_EQ(output_global[i], 0x00);                                                                      \
        }                                                                                                           \
    }
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false, int32_t, int8_t, int8_t, int32_t, int32_t);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false, half, int8_t, int8_t, int32_t, int32_t);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false, bfloat16_t, int8_t, int8_t, int32_t, int32_t);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true, int8_t, int8_t, int8_t, int32_t, int32_t);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false, uint8_t, int8_t, int8_t, int32_t, int32_t);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true, int32_t, int16_t, int8_t, int32_t, int16_t);
