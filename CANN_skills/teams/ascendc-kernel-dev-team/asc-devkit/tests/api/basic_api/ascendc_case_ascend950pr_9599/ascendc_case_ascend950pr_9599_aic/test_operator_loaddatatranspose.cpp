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

using namespace std;

namespace AscendC {
#define LOCAL_TENSOR_REGISTER(tensorName, type, quePos, initAddr, dataSize) \
    LocalTensor<type> tensorName;                                     \
    TBuffAddr tbuf_##tensorName;                                      \
    tbuf_##tensorName.logicPos = static_cast<uint8_t>(TPosition::quePos);       \
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
 * LoadDataWithTranspose                                             *
 * ************************************************************************************************* */
template <typename DstT, typename Src0T, typename Src1T>
void MainLoadData(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm,
                  __gm__ uint8_t* __restrict__ src1Gm, __gm__ uint16_t m, __gm__ uint16_t n, __gm__ uint16_t k,
                  __gm__ uint16_t channelSize)
{
    TPipe tpipe;
    // mmad c = a * b
    uint16_t aSize = m * k;
    uint16_t bSize = k * n;
    uint16_t cSize = m * n;
    GlobalTensor<Src0T> input0Global;
    GlobalTensor<Src1T> input1Global;
    GlobalTensor<DstT> outputGlobal;
    input0Global.SetGlobalBuffer(reinterpret_cast<__gm__ Src0T*>(src0Gm), aSize);
    input1Global.SetGlobalBuffer(reinterpret_cast<__gm__ Src1T*>(src1Gm), bSize);
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ DstT*>(dstGm), cSize);

    LOCAL_TENSOR_REGISTER(src0_l1, Src0T, A1, 0, aSize)
    LOCAL_TENSOR_REGISTER(src1_l1, Src1T, B1, ALIGN_ADDR(aSize * sizeof(Src0T)), bSize)
    LOCAL_TENSOR_REGISTER(src0_l0a, Src0T, A2, 0, aSize)
    LOCAL_TENSOR_REGISTER(src1_l0b, Src1T, B2, 0, bSize)
    LOCAL_TENSOR_REGISTER(dst_l0c, DstT, CO1, 0, cSize)

    DataCopy(src0_l1, input0Global, m * k);
    DataCopy(src1_l1, input1Global, k * n);
    LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = 0;
    loadDataParams.mStep = DivCeil(m, 16);
    loadDataParams.kStep = DivCeil(k * sizeof(Src0T), 32);
    if (IsSameType<Src0T, int4b_t>::value) {
        loadDataParams.kStep = DivCeil(loadDataParams.kStep, 2);
    }
    loadDataParams.srcStride = DivCeil(m, 16);
    loadDataParams.dstStride = DivCeil(m, 16);
    loadDataParams.sid = 0;
    loadDataParams.ifTranspose = false;
    LoadData(src0_l0a, src0_l1, loadDataParams);

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

    MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.isBias = false;
    Mmad(dst_l0c, src0_l0a, src1_l0b, mmadParams);
    uint16_t srcStride = DivCeil(m, BLOCK_CUBE) * BLOCK_CUBE;
    uint16_t dstStride = m * BLOCK_CUBE;
    DataCopyCO12DstParams intriParams(n, m, dstStride, srcStride, QuantMode_t::NoQuant, 0, 0, 0);
    DataCopy(outputGlobal, dst_l0c, intriParams);
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

class LoadDataTranspose910dTestsuite
    : public testing::Test
    , public testing::WithParamInterface<LoadDataTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(
    TEST_LOAD_DATA, LoadDataTranspose910dTestsuite,
    ::testing::Values(
        LoadDataTestParams{ 16, 128, 32, 32, MainLoadData<float, half, half>, 4, 2, 2 }));

TEST_P(LoadDataTranspose910dTestsuite, LoadDataTransposeTestCase)
{
    auto param = GetParam();
    uint8_t dstGm[param.m * param.n * param.sizeofDst];
    uint8_t src0Gm[param.m * param.k * param.sizeofSrc0];
    uint8_t src1Gm[param.k * param.n * param.sizeofSrc1];
    param.cal_func(dstGm, src0Gm, src1Gm, param.m, param.n, param.k, param.channelSize);
}
}  // namespace AscendC
