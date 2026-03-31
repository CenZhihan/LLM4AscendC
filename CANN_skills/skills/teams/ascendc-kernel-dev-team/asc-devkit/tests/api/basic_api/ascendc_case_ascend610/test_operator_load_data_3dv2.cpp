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
/* **************************************************************************************************
 * LoadData 3dv2                                             *
 * ************************************************************************************************* */
// out load data -> l1
// l1 load data -> l0a & l0b
// l0a l0b mmad -> l0c
// l0c data copy -> out
template <typename DstT, typename Src0T, typename Src1T>
void MainLoad3dv2(__gm__ uint8_t* __restrict__ dstGm, __gm__ uint8_t* __restrict__ src0Gm,
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
    LOCAL_TENSOR_REGISTER(dst_l0c, DstT, C2, 0, cSize)
    LOCAL_TENSOR_REGISTER(dst_ub, DstT, CO2, 0, cSize)

    LoadData(src0_l1, input0Global, { 0, aSize / BYTE_PER_FRACTAL, 0, 0, 0, false, 0 });
    LoadData(src0_l1, input0Global, { 0, aSize / BYTE_PER_FRACTAL, 0, 0, 0, false, 0 });
    LoadData(src1_l1, input1Global, { 0, bSize / BYTE_PER_FRACTAL, 0, 0, 0, false, 1 });

    uint8_t padList[PAD_SIZE] = {0, 0, 0, 0};
    LoadData(src0_l0a, src0_l1, { padList, 2, m / 2, channelSize, k, m, 0, 0, 1, 1, 1, 1, 1, 1, false, false, 0 });
    LoadData(src1_l0b, src1_l1, { padList, 2, n / 2, channelSize, k, n, 0, 0, 1, 1, 1, 1, 1, 1, false, false, 0 });

    // l1 -> ub, no purpose
    LOCAL_TENSOR_REGISTER(tmp_l1, Src0T, A1, ALIGN_ADDR(aSize * sizeof(Src0T) + bSize * sizeof(Src1T)), cSize)
    LOCAL_TENSOR_REGISTER(tmp_ub, Src0T, CO2, ALIGN_ADDR(cSize * sizeof(DstT)), cSize)
    LoadData(tmp_ub, tmp_l1,
        { padList, m, n / channelSize, channelSize, m, n, 0, 0, 1, 1, 1, 1, 1, 1, false, false, 0 });

    Mmad(dst_l0c, src0_l0a, src1_l0b, { m, n, k, 0, false, true });

    DataCopyEnhancedParams dataCopyEnhance;
    dataCopyEnhance.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(dst_ub, dst_l0c, { 1, cSize / BYTE_PER_FRACTAL, 0, 0 }, dataCopyEnhance);
    DataCopy(outputGlobal, dst_ub, cSize);
}

struct LoadData3dv2TestParams {
    uint16_t m;
    uint16_t n;
    uint16_t k;
    uint8_t channelSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint16_t, uint16_t, uint16_t, uint16_t);
    uint8_t sizeofDst;
    uint8_t sizeofSrc0;
    uint8_t sizeofSrc1;
};

class LoadData3dv2Testsuite : public testing::Test, public testing::WithParamInterface<LoadData3dv2TestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_LOAD_DATA_3DV2, LoadData3dv2Testsuite,
    ::testing::Values(
    LoadData3dv2TestParams { 128, 128, 32, 32, MainLoad3dv2<int32_t, int8_t, int8_t>, 4, 1, 1 },
    LoadData3dv2TestParams { 112, 128, 32, 36, MainLoad3dv2<half, half, half>, 2, 2, 2 }));

TEST_P(LoadData3dv2Testsuite, LoadData3dv2TestCase)
{
    auto param = GetParam();
    uint8_t dstGm[param.m * param.n * param.sizeofDst];
    uint8_t src0Gm[param.m * param.k * param.sizeofSrc0];
    uint8_t src1Gm[param.k * param.n * param.sizeofSrc1];

    param.cal_func(dstGm, src0Gm, src1Gm, param.m, param.n, param.k, param.channelSize);
    for (int32_t i = 0; i < param.m * param.n * param.sizeofDst; i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}
} // namespace AscendC
