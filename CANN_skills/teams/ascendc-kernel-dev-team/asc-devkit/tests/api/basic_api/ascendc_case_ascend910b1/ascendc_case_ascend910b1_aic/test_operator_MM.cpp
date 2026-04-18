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

template <typename DstT, typename Src0T, typename Src1T, typename L1outT, typename BiasT>
void MainCpuMmadBiasDemo(__gm__ uint8_t* __restrict__ featureGm, __gm__ uint8_t* __restrict__ weightGm,
                             __gm__ uint8_t* __restrict__ biasGm, __gm__ uint8_t* __restrict__ quantGm,
                             __gm__ uint8_t* __restrict__ resultGm, int32_t featureDataSize, int32_t weightDataSize,
                             int32_t biasDataSize, int32_t quantDataSize, int32_t outputDataSize, bool isBias,
                             bool doLoadData3dv2Pro, QuantMode_t quantMode)
{
    AscendC::TPipe tpipe;
    AscendC::GlobalTensor<AscendC::TensorTrait<Src0T>> featureGlobal;
    AscendC::GlobalTensor<AscendC::TensorTrait<Src1T>> weightGlobal;
    AscendC::GlobalTensor<AscendC::TensorTrait<BiasT>> biasGlobal;
    AscendC::GlobalTensor<AscendC::TensorTrait<uint64_t>> quantGlobal;
    AscendC::GlobalTensor<AscendC::TensorTrait<DstT>> outputGlobal;
    featureGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ Src0T*>(featureGm), featureDataSize);
    weightGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ Src1T*>(weightGm), weightDataSize);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(biasGm), biasDataSize);
    quantGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(quantGm), quantDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(resultGm), outputDataSize);
    AscendC::AscendCUtils::SetOverflow(1);
    // weight: gm -> l0b
    LOCAL_TENSOR_REGISTER(weightLocal, AscendC::TensorTrait<Src1T>, B2, 0, weightDataSize)
    // gm -> l0a
    LOCAL_TENSOR_REGISTER(weightLocalL0A, AscendC::TensorTrait<Src1T>, A2, 0, weightDataSize)

    // load2d: gm -> l0b
    AscendC::LoadData2DParams loadDataParam;
    loadDataParam.repeatTimes = 8;
    loadDataParam.srcStride = 1;
    AscendC::LoadData(weightLocal, weightGlobal, loadDataParam);

    AscendC::LoadData(weightLocalL0A, weightGlobal, loadDataParam);  // loadda2d: gm -> l0a

    // gm -> l1
    AscendC::LoadData2DParams loadDataParam1;
    loadDataParam1.repeatTimes = 8;
    loadDataParam1.srcStride = 1;
    loadDataParam1.addrMode = 1;
    LOCAL_TENSOR_REGISTER(weightLocalL1, AscendC::TensorTrait<Src1T>, A1, 0, weightDataSize)
    AscendC::LoadData(weightLocalL1, weightGlobal, loadDataParam1);  // loadda2d: gm -> l1
    loadDataParam1.addrMode = 0;
    AscendC::LoadData(weightLocalL1, weightGlobal, loadDataParam1);

    // feature map: gm -> l1 -> l0a
    LOCAL_TENSOR_REGISTER(L1Local, AscendC::TensorTrait<Src0T>, A1, 0, featureDataSize)
    DataCopy(L1Local, featureGlobal, featureDataSize);

    // load3dv2: l1 -> l0a
    LOCAL_TENSOR_REGISTER(featureLocal, AscendC::TensorTrait<Src0T>, A2, 0, featureDataSize)
    uint8_t padList[4] = { 0, 0, 0, 0 };
    if (doLoadData3dv2Pro) {
        AscendC::SetFmatrix(4, 4, padList, AscendC::FmatrixMode::FMATRIX_LEFT);
        AscendC::SetLoadDataBoundary(static_cast<uint64_t>(0));
        AscendC::SetLoadDataPaddingValue(static_cast<uint64_t>(0));
        AscendC::LoadData3DParamsV2Pro loadData3DV2;
        loadData3DV2.channelSize = 32;
        loadData3DV2.extConfig = (static_cast<uint64_t>(0) << AscendC::LOAD_M_START_POSITION) |
                                 (static_cast<uint64_t>(0) << AscendC::LOAD_K_START_POSITION) |
                                 (static_cast<uint64_t>(16) << AscendC::LOAD_M_EXTENSION) | static_cast<uint64_t>(128);
        loadData3DV2.filterConfig = (static_cast<uint64_t>(2) << AscendC::LOAD_DILATION_FILTER_H) |
                                    (static_cast<uint64_t>(2) << AscendC::LOAD_DILATION_FILTER_W) |
                                    (static_cast<uint64_t>(1) << AscendC::LOAD_FILTER_H) | (static_cast<uint64_t>(1) << AscendC::LOAD_FILTER_W) |
                                    (static_cast<uint64_t>(1) << AscendC::LOAD_STRIDE_H) | static_cast<uint64_t>(1);
        AscendC::LoadData<AscendC::TensorTrait<Src0T>>(featureLocal, L1Local, loadData3DV2);
    } else {
        if constexpr(AscendC::IsSameType<Src0T, half>::value) {
            AscendC::LoadData3DParamsV2<Src0T> ld3DV2 = { padList, 4, 4, 32, 128, 16, 0, 0, 1, 1, 1, 1, 2, 2, false, false, 0 };
            AscendC::LoadData<AscendC::TensorTrait<Src0T>>(featureLocal, L1Local, ld3DV2);
        }
        AscendC::LoadData3DParamsV2<Src0T> loadData3DV2 = { padList, 4, 4, 36, 128, 16,    0,     0, 1,
                                                            1,       1, 1, 2,  2,   false, false, 0 };
        AscendC::LoadData<AscendC::TensorTrait<Src0T>>(featureLocal, L1Local, loadData3DV2);
    }

    // bias : gm -> l1
    LOCAL_TENSOR_REGISTER(biasLocal, AscendC::TensorTrait<BiasT>, C1, 0, biasDataSize)
    AscendC::DataCopy(biasLocal, biasGlobal, biasDataSize);
    // bias : l1 ->bt
    LOCAL_TENSOR_REGISTER(biasC2, AscendC::TensorTrait<L1outT>, C2, 0, biasDataSize)
    AscendC::DataCopyParams dataCopyParams = { 1, biasDataSize * sizeof(BiasT) / 64, 0, 0 };
    AscendC::DataCopy<AscendC::TensorTrait<L1outT>, AscendC::TensorTrait<BiasT>>(biasC2, biasLocal, dataCopyParams);

    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID1);

    // quant : gm -> l1
    LOCAL_TENSOR_REGISTER(quantLocal, AscendC::TensorTrait<uint64_t>, A1, 0, quantDataSize)
    AscendC::DataCopy(quantLocal, quantGlobal, quantDataSize);

    AscendC::SetFlag<AscendC::HardEvent::MTE1_FIX>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_FIX>(EVENT_ID0);
    // quant : l1 ->fb
    LOCAL_TENSOR_REGISTER(quantFixBuffer, AscendC::TensorTrait<uint64_t>, C2PIPE2GM, 0, quantDataSize)
    AscendC::DataCopyParams dataCopyToFbParams = { 1, quantDataSize * sizeof(uint64_t) / 128, 0, 0 };
    AscendC::DataCopy(quantFixBuffer, quantLocal, dataCopyToFbParams);

    // mmad c = a * b + bias
    LOCAL_TENSOR_REGISTER(l0cOut, AscendC::TensorTrait<L1outT>, CO1, 0, outputDataSize)
    AscendC::MmadParams mmadParams;
    mmadParams.m = 112;
    mmadParams.k = 32;
    mmadParams.n = 128;
    mmadParams.cmatrixSource = true;  // bias in bt
    mmadParams.cmatrixInitVal = !isBias;

    AscendC::Mmad<AscendC::TensorTrait<L1outT>, AscendC::TensorTrait<Src0T>, AscendC::TensorTrait<Src1T>>(l0cOut, featureLocal, weightLocal, mmadParams);
    // test for add biasInput
    AscendC::Mmad<AscendC::TensorTrait<L1outT>, AscendC::TensorTrait<Src0T>, AscendC::TensorTrait<Src1T>>(l0cOut, featureLocal, weightLocal, biasC2, mmadParams);

    mmadParams.cmatrixSource = false;  // bias in L0c
    AscendC::Mmad<AscendC::TensorTrait<L1outT>, AscendC::TensorTrait<Src0T>, AscendC::TensorTrait<Src1T>>(l0cOut, featureLocal, weightLocal, l0cOut, mmadParams);

    // mov l0c to L1
    LOCAL_TENSOR_REGISTER(outputLocal, AscendC::TensorTrait<DstT>, A1, 0, outputDataSize)
    uint16_t cburstNum = mmadParams.n / AscendC::BLOCK_CUBE;
    uint16_t burstLen = mmadParams.m * AscendC::BLOCK_CUBE * sizeof(L1outT) / AscendC::ONE_BLK_SIZE;
    AscendC::FixpipeParams<L1outT> fixpipeParams(cburstNum, burstLen, 0, 0);
    if (quantMode == QuantMode_t::F322F16) {
        fixpipeParams.quantParams = { quantMode };
    } else {
        fixpipeParams.quantParams = { quantMode, static_cast<float>(0.5) };
    }
    fixpipeParams.reluEn = true;
    AscendC::Fixpipe(outputLocal, l0cOut, fixpipeParams);

    // mov L1 to gm
    AscendC::DataCopy(outputGlobal, outputLocal, outputDataSize);
    AscendC::PipeBarrier<PIPE_ALL>();
    g_sysWorkspaceReserved = GetSysWorkSpacePtr();
}

class TEST_MMAD_BIAS : public testing::Test {
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

#define VEC_MMAD_BIAS_TESTCASE(testMmadBias, biasOp, dstType, src0Type, src1Type, l1outT, biasT, doLoadData3dv2Pro, \
                               quantMode)                                                                             \
    TEST_F(testMmadBias,                                                                                            \
           MMAD_Case_Bias_##biasOp##_##dstType##_##src0Type##_##src1Type##_##l1outT##_##biasT##_##doLoadData3dv2Pro)  \
    {                                                                                                                 \
        const int32_t featureDataSize = 3584;                                                                         \
        const int32_t weightDataSize = 4096;                                                                          \
        const int32_t biasDataSize = 128;                                                                             \
        const int32_t quantDataSize = 128;                                                                            \
        const int32_t outputDataSize = 14336;                                                                         \
        uint8_t featureGlobal[featureDataSize * sizeof(src0Type)] = { 0 };                                            \
        uint8_t weightGlobal[weightDataSize * sizeof(src1Type)] = { 0 };                                              \
        uint8_t biasGlobal[biasDataSize * sizeof(biasT)] = { 0 };                                                     \
        uint8_t quantGlobal[quantDataSize * sizeof(uint64_t)] = { 0 };                                                \
        uint8_t outputGlobal[outputDataSize * sizeof(dstType)] = { 0 };                                               \
        MainCpuMmadBiasDemo<dstType, src0Type, src1Type, l1outT, biasT>(                                          \
            featureGlobal, weightGlobal, biasGlobal, quantGlobal, outputGlobal, featureDataSize, weightDataSize,      \
            biasDataSize, quantDataSize, outputDataSize, biasOp, doLoadData3dv2Pro, quantMode);                       \
        for (int32_t i = 0; i < outputDataSize * sizeof(dstType); i++) {                                              \
            EXPECT_EQ(outputGlobal[i], 0x00);                                                                         \
        }                                                                                                             \
    }
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false,   int8_t, int8_t,     int8_t,     int32_t,    int32_t,    false, QuantMode_t::REQ8);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false,   half,   half,       half,       float,      float,      false, QuantMode_t::F322F16);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false,   half,   float,      float,      float,      float,      false, QuantMode_t::F322F16);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true,    int8_t, int8_t,     int8_t,     int32_t,    int32_t,    false, QuantMode_t::REQ8);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true,    half,   half,       half,       float,      half,       false, QuantMode_t::F322F16);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true,    half,   half,       half,       float,      float,      false, QuantMode_t::F322F16);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true,    half,   float,      float,      float,      half,       false, QuantMode_t::F322F16);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true,    half,   float,      float,      float,       float,      false, QuantMode_t::F322F16);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false,   half,   bfloat16_t, bfloat16_t, float,      float,      false, QuantMode_t::F322F16);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true,    half,   bfloat16_t, bfloat16_t, float,      float,      false, QuantMode_t::F322F16);

// test load3dv2Pro
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false,   int8_t, int8_t,     int8_t,     int32_t,    int32_t,    true, QuantMode_t::REQ8);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false,   half,   half,       half,       float,      float,      true, QuantMode_t::F322F16);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false,   half,   float,      float,      float,      float,      true, QuantMode_t::F322F16);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true,    int8_t, int8_t,     int8_t,     int32_t,    int32_t,    true, QuantMode_t::REQ8);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true,    half,   half,       half,       float,      half,       true, QuantMode_t::F322F16);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true,    half,   half,       half,       float,      float,      true, QuantMode_t::F322F16);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true,    half,   float,      float,      float,      half,       true, QuantMode_t::F322F16);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true,    half,   float,      float,      float,      float,      true, QuantMode_t::F322F16);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, false,   half,   bfloat16_t, bfloat16_t, float,      float,      true, QuantMode_t::F322F16);
VEC_MMAD_BIAS_TESTCASE(TEST_MMAD_BIAS, true,    half,   bfloat16_t, bfloat16_t, float,      float,      true, QuantMode_t::F322F16);