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
using namespace AscendC;

template <typename T>
void MainNpuBroadcastDemo(__gm__ uint8_t* __restrict__ src0Gm, __gm__ uint8_t* __restrict__ dstGm,
    int32_t dataSize, int32_t outputDataSize)
{
    AscendCUtils::SetOverflow(0);
    TPipe tpipe;
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0Gm), dataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), outputDataSize);

    TBuf<TPosition::CO2> tbuf1;
    tpipe.InitBuffer(tbuf1, dataSize * sizeof(T));
    LocalTensor<T> inputLocal = tbuf1.Get<T>();

    TBuf<TPosition::CO1> tbuf2;
    tpipe.InitBuffer(tbuf2, outputDataSize * sizeof(T));
    LocalTensor<T> outputLocalC0 = tbuf2.Get<T>();

    TBuf<TPosition::CO2> tbuf3;
    tpipe.InitBuffer(tbuf3, outputDataSize * sizeof(T));
    LocalTensor<T> outputLocal = tbuf3.Get<T>();

    DataCopy(inputLocal, inputGlobal, dataSize);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    AscendCUtils::SetMask<uint8_t>(128);
    AscendC::BroadCastVecToMM(outputLocalC0, inputLocal, 1, 1, 0, 0);
    PipeBarrier<PIPE_V>();

    DataCopyParams dataParams;
    dataParams.blockCount = 1;
    dataParams.blockLen = 1;
    dataParams.dstStride = 0;
    dataParams.srcStride = 0;
    DataCopyEnhancedParams enhance_info;
    enhance_info.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(outputLocal, outputLocalC0, dataParams, enhance_info);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outputGlobal, outputLocal, outputDataSize);

    PipeBarrier<PIPE_ALL>();
}

class TEST_BROADCAST : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

#define VEC_BROADCAST_TESTCASE(testBroadcast, dstType)                           \
    TEST_F(testBroadcast, Broadcast_Case_##dstType)                              \
    {                                                                             \
        const int32_t srcDataSize = 16;                                           \
        const int32_t dstDataSize = 256;                                          \
        uint8_t srcGm[srcDataSize * sizeof(dstType)];                             \
        uint8_t dstGm[dstDataSize * sizeof(dstType)];                             \
        MainNpuBroadcastDemo<dstType>(srcGm, dstGm, srcDataSize, dstDataSize); \
        for (int32_t i = 0; i < dstDataSize * sizeof(dstType); i++) {             \
            EXPECT_EQ(dstGm[i], 0x00);                                            \
        }                                                                         \
    }

VEC_BROADCAST_TESTCASE(TEST_BROADCAST, int32_t);
VEC_BROADCAST_TESTCASE(TEST_BROADCAST, float);
VEC_BROADCAST_TESTCASE(TEST_BROADCAST, half);

template <typename Src0T, typename Src1T, typename DstT>
void MainCpuMmadDemo(__gm__ uint8_t* __restrict__ featureGm, __gm__ uint8_t* __restrict__ weightGm,
    __gm__ uint8_t* __restrict__ resultGm, int32_t featureDataSize, int32_t weightDataSize,
    int32_t outputDataSize, bool isBias)
{
    TPipe tpipe;
    GlobalTensor<Src0T> featureGlobal;
    GlobalTensor<Src1T> weightGlobal;
    GlobalTensor<DstT> outputGlobal;
    featureGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ Src0T*>(featureGm), featureDataSize);
    weightGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ Src1T*>(weightGm), weightDataSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(resultGm), outputDataSize);

    TBuf<TPosition::B2> tbuf;
    tpipe.InitBuffer(tbuf, weightDataSize * sizeof(Src1T));
    LocalTensor<Src1T> weightLocal = tbuf.Get<Src1T>();

    // load2dv1
    LoadData2DParams loadDataParam;
    loadDataParam.repeatTimes = 8;
    loadDataParam.srcStride = 1;
    LoadData(weightLocal, weightGlobal, loadDataParam);

    TBuf<TPosition::A1> tbuf1;
    tpipe.InitBuffer(tbuf1, featureDataSize * sizeof(Src0T));
    LocalTensor<Src0T> l1Local = tbuf1.Get<Src0T>();

    DataCopy(l1Local, featureGlobal, featureDataSize);

    TBuf<TPosition::A2> tbuf2;
    tpipe.InitBuffer(tbuf2, featureDataSize * sizeof(Src0T));
    LocalTensor<Src0T> featureLocal = tbuf2.Get<Src0T>();

    LoadData3DParamsV1<Src0T> loadData3DV1;
    loadData3DV1.l1H = 2;
    loadData3DV1.l1W = 56;
    loadData3DV1.strideW = 1;
    loadData3DV1.strideH = 1;
    loadData3DV1.filterW = 1;
    loadData3DV1.filterH = 1;
    loadData3DV1.dilationFilterW = 1;
    loadData3DV1.dilationFilterH = 1;
    loadData3DV1.jumpStride = 1;
    loadData3DV1.repeatMode = 1;
    loadData3DV1.repeatTime = 7;
    if (sizeof(Src0T) == sizeof(half)) {
        uint64_t repeat = ConstCeil(128, BLOCK_CUBE * 16)  << 16 ;
        repeat |= 1;
        create_ca_matrix(static_cast<__ca__ void*>(featureLocal.GetPhyAddr()), repeat, static_cast<half>(1));
        create_cbuf_matrix(static_cast<__cbuf__ void*>(l1Local.GetPhyAddr()), repeat, static_cast<half>(1));
    }
    SetLoadDataBoundary(static_cast<uint32_t>(0));
    LoadData<Src0T>(featureLocal, l1Local, loadData3DV1);

    TBuf<TPosition::C2> tbufFailed;
    tpipe.InitBuffer(tbufFailed, featureDataSize * sizeof(Src0T));
    LocalTensor<Src0T> fmLocal = tbufFailed.Get<Src0T>();
    MOCKER(raise).stubs().will(returnValue(int(0)));
    LoadData<Src0T>(fmLocal, l1Local, loadData3DV1);

    TBuf<TPosition::C2> tbuf3;
    tpipe.InitBuffer(tbuf3, outputDataSize * sizeof(DstT));
    LocalTensor<DstT> l0cOut = tbuf3.Get<DstT>();

    MmadParams mmadParams;
    mmadParams.m = 112;
    mmadParams.n = 32;
    mmadParams.k = 128;
    mmadParams.cmatrixSource = 0; // bias in L0C
    mmadParams.cmatrixInitVal = !isBias;
    Mmad<DstT, Src0T, Src1T>(l0cOut, featureLocal, weightLocal, mmadParams);


    TBuf<TPosition::CO2> tbuf4;
    tpipe.InitBuffer(tbuf4, outputDataSize * sizeof(DstT));
    LocalTensor<DstT> outLocal = tbuf4.Get<DstT>();

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = 56;
    DataCopyEnhancedParams dataCopyEnhance;
    dataCopyEnhance.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy<DstT>(outLocal, l0cOut, dataCopyParams, dataCopyEnhance);
    DataCopy(outputGlobal, outLocal, outputDataSize);
    PipeBarrier<PIPE_ALL>();
}

class TEST_MMAD : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

#define VEC_MMAD_TESTCASE(testMmad, biasOp, dstType, src0Type, src1Type)                                        \
    TEST_F(testMmad, MMAD_Case_Bias_##biasOp##_##dstType##_##src0Type##_##src1Type)                             \
    {                                                                                                            \
        const int32_t featureDataSize = 3584;                                                                    \
        const int32_t weightDataSize = 4096;                                                                     \
        const int32_t outputDataSize = 14336;                                                                    \
        uint8_t featureGlobal[featureDataSize * sizeof(src0Type)];                                               \
        uint8_t weightGlobal[weightDataSize * sizeof(src1Type)];                                                 \
        uint8_t outputGlobal[outputDataSize * sizeof(dstType)];                                                  \
        MainCpuMmadDemo<src0Type, src1Type, dstType>(                                                         \
            featureGlobal, weightGlobal, outputGlobal, featureDataSize, weightDataSize, outputDataSize, biasOp); \
        for (int32_t i = 0; i < outputDataSize * sizeof(dstType); i++) {                                         \
            EXPECT_EQ(outputGlobal[i], 0x00);                                                                    \
        }                                                                                                        \
    }

VEC_MMAD_TESTCASE(TEST_MMAD, true, int32_t, uint8_t, uint8_t);
VEC_MMAD_TESTCASE(TEST_MMAD, true, int32_t, int8_t, int8_t);
VEC_MMAD_TESTCASE(TEST_MMAD, true, int32_t, uint8_t, int8_t);
VEC_MMAD_TESTCASE(TEST_MMAD, true, half, half, half);
VEC_MMAD_TESTCASE(TEST_MMAD, true, float, half, half);

VEC_MMAD_TESTCASE(TEST_MMAD, false, int32_t, uint8_t, uint8_t);
VEC_MMAD_TESTCASE(TEST_MMAD, false, int32_t, int8_t, int8_t);
VEC_MMAD_TESTCASE(TEST_MMAD, false, int32_t, uint8_t, int8_t);
VEC_MMAD_TESTCASE(TEST_MMAD, false, half, half, half);
VEC_MMAD_TESTCASE(TEST_MMAD, false, float, half, half);
