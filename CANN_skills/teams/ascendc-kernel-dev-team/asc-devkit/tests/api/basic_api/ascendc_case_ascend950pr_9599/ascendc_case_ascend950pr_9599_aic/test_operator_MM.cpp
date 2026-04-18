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

template <typename DstT, typename Src0T, typename Src1T, typename L1OutT, typename BiasT>
void MainCpuCmpMmadBiasDemo(__gm__ uint8_t* __restrict__ featureGm, __gm__ uint8_t* __restrict__ weightGm,
    __gm__ uint8_t* __restrict__ biasGm, __gm__ uint8_t* __restrict__ resultGm, int32_t featureDataSize,
    int32_t weightDataSize, int32_t biasDataSize, int32_t outputDataSize, bool isBias)
{
    AscendC::TPipe tpipe;
    AscendC::GlobalTensor<Src0T> featureGlobal;
    AscendC::GlobalTensor<Src1T> weightGlobal;
    AscendC::GlobalTensor<BiasT> biasGlobal;
    AscendC::GlobalTensor<DstT> outputGlobal;
    featureGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ Src0T*>(featureGm), featureDataSize);
    weightGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ Src1T*>(weightGm), weightDataSize);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(biasGm), biasDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(resultGm), outputDataSize);

    AscendC::AscendCUtils::SetOverflow(1);

    // feature map / weight: gm -> l1 nd2nz
    LOCAL_TENSOR_REGISTER(feature_l1, Src0T, A1, 0, featureDataSize)
    LOCAL_TENSOR_REGISTER(weight_l1, Src1T, B1, 0, weightDataSize)
    LoadData(feature_l1, featureGlobal, {1, 0, 0, 0, 0, 0, 0, 0});
    LoadData(weight_l1, weightGlobal, {1, 0, 0, 0, 0, 0, 0, 0});


    // feature map: l1 ->l0a load2dv2
    LOCAL_TENSOR_REGISTER(feature_l0a, Src0T, A2, 0, featureDataSize)
    AscendC::LoadData2DParamsV2 loadDataParamsA = {0, 0, 0, 0, 0, 0, true, 0};
    LoadData(feature_l0a, feature_l1, loadDataParamsA);

    // weight: l1 -> l0b load2dv2 with transpose
    LOCAL_TENSOR_REGISTER(weight_l0b, Src1T, B2, 0, weightDataSize)
    AscendC::LoadData2DParamsV2 loadDataParamsB = {0, 0, 0, 0, 0, 0, true, 0};
    LoadData(weight_l0b, weight_l1, loadDataParamsB);

    // bias : gm -> l1
    LOCAL_TENSOR_REGISTER(bias_local, BiasT, C1, 0, biasDataSize)
    DataCopy(bias_local, biasGlobal, biasDataSize);

    // mmad c = a * b + bias
    LOCAL_TENSOR_REGISTER(l0c_out, L1OutT, CO1, 0, outputDataSize)
    AscendC::MmadParams mmad_params;
    mmad_params.m = 112;
    mmad_params.k = 32;
    mmad_params.n = 128;
    mmad_params.cmatrixInitVal = !isBias;
    mmad_params.cmatrixSource = true; // bias in bt
    AscendC::Mmad<L1OutT, Src0T, Src1T>(l0c_out, feature_l0a, weight_l0b, mmad_params);

    // mov l0c to ub
    LOCAL_TENSOR_REGISTER(output_local, DstT, VECIN, 0, outputDataSize)
    uint16_t cburstNum = mmad_params.n / AscendC::BLOCK_CUBE;
    uint16_t burstLen = mmad_params.m * AscendC::BLOCK_CUBE * sizeof(L1OutT) / AscendC::ONE_BLK_SIZE;
    // in 950 l0c to ub nz
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> fixpipeParams(mmad_params.n, mmad_params.m, mmad_params.m, mmad_params.m * AscendC::BLOCK_CUBE);
    fixpipeParams.reluEn = true;
    fixpipeParams.quantPre = GetQuantMode<DstT, L1OutT>();
    AscendC::Fixpipe<DstT, L1OutT, CFG_NZ_UB>(output_local, l0c_out, fixpipeParams);

    // mov l0c to l1
    LOCAL_TENSOR_REGISTER(output_local_l1, DstT, A1, 0, outputDataSize)
    AscendC::Fixpipe<DstT, L1OutT, AscendC::CFG_NZ>(output_local_l1, l0c_out, fixpipeParams);

    // in 950 l0c to ub row_major
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams2(mmad_params.n, mmad_params.m, mmad_params.m, mmad_params.n);
    fixpipeParams2.dualDstCtl = 0b01;
    fixpipeParams2.quantPre = GetQuantMode<DstT, L1OutT>();
    if (fixpipeParams2.quantPre == QuantMode_t::NoQuant) {
        AscendC::Fixpipe<DstT, L1OutT, CFG_ROW_MAJOR_UB>(output_local, l0c_out, fixpipeParams2);
    }

    // mov ub to gm
    DataCopy(outputGlobal, output_local, outputDataSize);
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

#define VEC_MMAD_BIAS_TESTCASE(testMmadBias, biasOp, dstType, src0Type, src1Type, L1OutT, biasT1)             \
    TEST_F(testMmadBias, MMAD_Case_Bias_##biasOp##_##dstType##_##src0Type##_##src1Type##_##L1OutT##_##biasT1) \
    {                                                                                                               \
        const int32_t featureDataSize = 3584;                                                                     \
        const int32_t weightDataSize = 4096;                                                                      \
        const int32_t biasDataSize = 128;                                                                         \
        const int32_t outputDataSize = 14336;                                                                     \
        uint8_t featureGlobal[featureDataSize * sizeof(src0Type)] = {0};                                        \
        uint8_t weightGlobal[weightDataSize * sizeof(src1Type)] = {0};                                          \
        uint8_t biasGlobal[biasDataSize * sizeof(biasT1)] = {0};                                                 \
        uint8_t outputGlobal[outputDataSize * sizeof(L1OutT)] = {0};                                            \
        MainCpuCmpMmadBiasDemo<dstType, src0Type, src1Type, L1OutT, biasT1>(featureGlobal, weightGlobal, \
            biasGlobal, outputGlobal, featureDataSize, weightDataSize, biasDataSize, outputDataSize,      \
            biasOp);                                                                                                \
        for (int32_t i = 0; i < outputDataSize * sizeof(dstType); i++) {                                         \
            EXPECT_EQ(outputGlobal[i], 0x00);                                                                      \
        }                                                                                                           \
    }
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false, int32_t, int8_t, int8_t, int32_t, int32_t);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false, half, int8_t, int8_t, int32_t, int32_t);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false, bfloat16_t, int8_t, int8_t, int32_t, int32_t);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true, int8_t, int8_t, int8_t, int32_t, int32_t);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false, uint8_t, int8_t, int8_t, int32_t, int32_t);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true, float, half, half, float, half);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true, float, half, half, float, float);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true, float, float, float, float, half);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true, float, float, float, float, float);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true, float, bfloat16_t, bfloat16_t, float, float);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false, half, fp8_e5m2_t, fp8_e5m2_t, float, float);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false, bfloat16_t, fp8_e4m3fn_t, fp8_e4m3fn_t, float, float);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false, int8_t, fp8_e5m2_t, fp8_e4m3fn_t, float, float);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false, uint8_t, fp8_e4m3fn_t, fp8_e5m2_t, float, float);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false, fp8_e4m3fn_t, fp8_e4m3fn_t, fp8_e4m3fn_t, float, float);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false, hifloat8_t, hifloat8_t, hifloat8_t, float, float);
