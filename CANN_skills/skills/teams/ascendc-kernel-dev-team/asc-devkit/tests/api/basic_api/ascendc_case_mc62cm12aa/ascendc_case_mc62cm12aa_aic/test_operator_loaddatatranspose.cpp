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
#define LOCAL_TENSOR_REGISTER(tensor_name, type, que_pos, init_addr, data_size) \
    LocalTensor<type> tensor_name;                                     \
    TBuffAddr tbuf_##tensor_name;                                      \
    tbuf_##tensor_name.logicPos = (uint8_t)TPosition::que_pos;       \
    tensor_name.SetAddr(tbuf_##tensor_name);                                    \
    tensor_name.InitBuffer(init_addr, data_size);

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
template <typename dst_T, typename src0_T, typename src1_T>
void MainLoadData(__gm__ uint8_t* __restrict__ dst_gm, __gm__ uint8_t* __restrict__ src0_gm,
                  __gm__ uint8_t* __restrict__ src1_gm, __gm__ uint16_t m, __gm__ uint16_t n, __gm__ uint16_t k,
                  __gm__ uint16_t channelSize)
{
    TPipe tpipe;
    // mmad c = a * b
    uint16_t a_size = m * k;
    uint16_t b_size = k * n;
    uint16_t c_size = m * n;
    GlobalTensor<src0_T> input0_global;
    GlobalTensor<src1_T> input1_global;
    GlobalTensor<dst_T> output_global;
    input0_global.SetGlobalBuffer(reinterpret_cast<__gm__ src0_T*>(src0_gm), a_size);
    input1_global.SetGlobalBuffer(reinterpret_cast<__gm__ src1_T*>(src1_gm), b_size);
    output_global.SetGlobalBuffer(reinterpret_cast<__gm__ dst_T*>(dst_gm), c_size);

    LOCAL_TENSOR_REGISTER(src0_l1, src0_T, A1, 0, a_size)
    LOCAL_TENSOR_REGISTER(src1_l1, src1_T, B1, ALIGN_ADDR(a_size * sizeof(src0_T)), b_size)
    LOCAL_TENSOR_REGISTER(src0_l0a, src0_T, A2, 0, a_size)
    LOCAL_TENSOR_REGISTER(src1_l0b, src1_T, B2, 0, b_size)
    LOCAL_TENSOR_REGISTER(dst_l0c, dst_T, CO1, 0, c_size)

    DataCopy(src0_l1, input0_global, m * k);
    DataCopy(src1_l1, input1_global, k * n);
    LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = 0;
    loadDataParams.mStep = DivCeil(m, 16);
    loadDataParams.kStep = DivCeil(k * sizeof(src0_T), 32);
    if (IsSameType<src0_T, int4b_t>::value) {
        loadDataParams.kStep = DivCeil(loadDataParams.kStep, 2);
    }
    loadDataParams.srcStride = DivCeil(m, 16);
    loadDataParams.dstStride = DivCeil(m, 16);
    loadDataParams.sid = 0;
    loadDataParams.ifTranspose = false;
    LoadData(src0_l0a, src0_l1, loadDataParams);

    LoadData2DParamsV2 loadDataParamsB;
    loadDataParamsB.mStartPosition = 0;
    loadDataParamsB.kStartPosition = 0;
    loadDataParamsB.mStep = DivCeil(m, 16);;
    loadDataParamsB.kStep = DivCeil(k * sizeof(src0_T), 32);
    if (IsSameType<src0_T, int4b_t>::value) {
        loadDataParams.kStep = DivCeil(loadDataParams.kStep, 2);
    }
    loadDataParamsB.srcStride = DivCeil(m, 16);
    loadDataParamsB.dstStride = DivCeil(m, 16);
    loadDataParamsB.sid = 0;
    loadDataParamsB.ifTranspose = false;
    LoadData(src1_l0b, src1_l1, loadDataParamsB);

    uint16_t kStep = DivCeil(k, 16);
    uint16_t nStep = DivCeil(n * sizeof(src1_T), 32);

    MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.isBias = false;
    Mmad(dst_l0c, src0_l0a, src1_l0b, mmadParams);
    uint16_t srcStride = DivCeil(m, BLOCK_CUBE) * BLOCK_CUBE;
    uint16_t dstStride = m * BLOCK_CUBE;
    DataCopyCO12DstParams intriParams(n, m, dstStride, srcStride, QuantMode_t::NoQuant, 0, 0, 0);
    DataCopy(output_global, dst_l0c, intriParams);
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
        LoadDataTestParams{ 16, 128, 32, 32, MainLoadData<int32_t, int16_t, int16_t>, 4, 2, 2 }));

TEST_P(LoadDataTranspose910dTestsuite, LoadDataTransposeTestCase)
{
    auto param = GetParam();
    uint8_t* dst_gm = new uint8_t[param.m * param.n * param.sizeofDst*10];
    uint8_t* src0_gm = new uint8_t[param.m * param.k * param.sizeofSrc0*10];
    uint8_t* src1_gm = new uint8_t[param.k * param.n * param.sizeofSrc1*10];
    param.cal_func(dst_gm, src0_gm, src1_gm, param.m, param.n, param.k, param.channelSize);
    delete [] dst_gm;
    delete [] src0_gm;
    delete [] src1_gm;
}
}  // namespace AscendC
