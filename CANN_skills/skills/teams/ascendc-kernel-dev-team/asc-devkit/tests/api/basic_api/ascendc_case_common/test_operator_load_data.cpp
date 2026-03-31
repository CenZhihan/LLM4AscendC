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
 * LoadData 2dv1 v2                                             *
 * ************************************************************************************************* */
// out load data -> l1
// l1 load data -> l0a & l0b
// l0a l0b mmad -> l0c
// l0c data copy -> out
template <typename dst_T, typename src0_T, typename src1_T>
void MainLoad2dL1(__gm__ uint8_t* __restrict__ dst_gm, __gm__ uint8_t* __restrict__ src0_gm,
    __gm__ uint8_t* __restrict__ src1_gm, __gm__ uint16_t m, __gm__ uint16_t n, __gm__ uint16_t k, __gm__ bool ifTrans)
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
    LOCAL_TENSOR_REGISTER(dst_l0c, dst_T, C2, 0, c_size)
    LOCAL_TENSOR_REGISTER(dst_ub, dst_T, CO2, 0, c_size)

    LoadData(src0_l1, input0_global, { 0, a_size / BYTE_PER_FRACTAL, 0, 0, 0, false, 0 });
    LoadData(src1_l1, input1_global, { 0, b_size / BYTE_PER_FRACTAL, 0, 0, 0, false, 0 });
    LoadData(src0_l0a, src0_l1, { 0, a_size / BYTE_PER_FRACTAL, 0, 0, 0, ifTrans, 0 });
    LoadData(src1_l0b, src1_l1, { 0, b_size / BYTE_PER_FRACTAL, 0, 0, 0, ifTrans, 0 });

    Mmad(dst_l0c, src0_l0a, src1_l0b, { m, n, k, 0, false, true });

    DataCopyEnhancedParams data_copy_enhance;
    data_copy_enhance.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(dst_ub, dst_l0c, { 1, c_size / BYTE_PER_FRACTAL, 0, 0 }, data_copy_enhance);
    DataCopy(output_global, dst_ub, c_size);
}

// out load data -> l0a & l0b
// l0a l0b mmad -> l0c
// l0c data copy -> out
template <typename dst_T, typename src0_T, typename src1_T>
void MainLoad2dOut(__gm__ uint8_t* __restrict__ dst_gm, __gm__ uint8_t* __restrict__ src0_gm,
    __gm__ uint8_t* __restrict__ src1_gm, __gm__ uint16_t m, __gm__ uint16_t n, __gm__ uint16_t k, __gm__ bool ifTrans)
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

    LOCAL_TENSOR_REGISTER(src0_l0a, src0_T, A2, 0, a_size)
    LOCAL_TENSOR_REGISTER(src1_l0b, src1_T, B2, 0, b_size)
    LOCAL_TENSOR_REGISTER(dst_l0c, dst_T, C2, 0, c_size)
    LOCAL_TENSOR_REGISTER(dst_ub, dst_T, CO2, 0, c_size)
    LoadData(src0_l0a, input0_global, { 0, a_size / BYTE_PER_FRACTAL, 0, 0, 0, false, 0 });
    LoadData(src1_l0b, input1_global, { 0, b_size / BYTE_PER_FRACTAL, 0, 0, 0, false, 0 });

    Mmad(dst_l0c, src0_l0a, src1_l0b, { m, n, k, 0, false, true });

    DataCopyEnhancedParams data_copy_enhance;
    data_copy_enhance.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(dst_ub, dst_l0c, { 1, c_size / BYTE_PER_FRACTAL, 0, 0 }, data_copy_enhance);
    DataCopy(output_global, dst_ub, c_size);
}

struct LoadData2dTestParams {
    uint16_t m;
    uint16_t n;
    uint16_t k;
    bool ifTrans;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint16_t, uint16_t, uint16_t, bool);
    uint8_t sizeofDst;
    uint8_t sizeofSrc0;
    uint8_t sizeofSrc1;
};

class LoadData2dTestsuite : public testing::Test, public testing::WithParamInterface<LoadData2dTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

/* **************************************************************************************************
 * LoadData 3dv1                                             *
 * ************************************************************************************************* */
// out load data -> l1
// l1 load data -> l0a & l0b
// l0a l0b mmad -> l0c
// l0c data copy -> out
template <typename dst_T, typename src0_T, typename src1_T>
void MainLoad3dv1(__gm__ uint8_t* __restrict__ dst_gm, __gm__ uint8_t* __restrict__ src0_gm,
    __gm__ uint8_t* __restrict__ src1_gm, __gm__ uint16_t m, __gm__ uint16_t n, __gm__ uint16_t k, __gm__ uint8_t csize)
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
    LOCAL_TENSOR_REGISTER(dst_l0c, dst_T, C2, 0, c_size)
    LOCAL_TENSOR_REGISTER(dst_ub, dst_T, CO2, 0, c_size)

    LoadData(src0_l1, input0_global, { 0, a_size / BYTE_PER_FRACTAL, 0, 0, 0, false, 0 });
    LoadData(src1_l1, input1_global, { 0, b_size / BYTE_PER_FRACTAL, 0, 0, 0, false, 0 });

    uint8_t padList[PAD_SIZE] = {0, 0, 0, 0};
    LoadData(src0_l0a, src0_l1,
        { padList, 2, m / 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, a_size / BYTE_PER_FRACTAL, csize, 1 });
    LoadData(src1_l0b, src1_l1,
        { padList, 2, n / 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, b_size / BYTE_PER_FRACTAL, csize, 0 });

    // l1 -> ub, no purpose
    LOCAL_TENSOR_REGISTER(tmp_l1, src0_T, A1, ALIGN_ADDR(a_size * sizeof(src0_T) + b_size * sizeof(src1_T)), c_size)
    LOCAL_TENSOR_REGISTER(tmp_ub, src0_T, CO2, ALIGN_ADDR(c_size * sizeof(dst_T)), c_size)
    LoadData(tmp_ub, tmp_l1,
        { padList, 1, m, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, c_size / BYTE_PER_FRACTAL, csize, 0 });

    Mmad(dst_l0c, src0_l0a, src1_l0b, { m, n, k, 0, false, true });

    DataCopyEnhancedParams data_copy_enhance;
    data_copy_enhance.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(dst_ub, dst_l0c, { 1, c_size / BYTE_PER_FRACTAL, 0, 0 }, data_copy_enhance);
    DataCopy(output_global, dst_ub, c_size);
}

struct LoadData3dv1TestParams {
    uint16_t m;
    uint16_t n;
    uint16_t k;
    uint8_t csize;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint16_t, uint16_t, uint16_t, uint8_t);
    uint8_t sizeofDst;
    uint8_t sizeofSrc0;
    uint8_t sizeofSrc1;
};

class LoadData3dv1Testsuite : public testing::Test, public testing::WithParamInterface<LoadData3dv1TestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_LOAD_DATA_2D, LoadData2dTestsuite,
    ::testing::Values(
    LoadData2dTestParams { 112, 128, 32, false, MainLoad2dL1<int32_t, uint8_t, uint8_t>, 4, 1, 1 },
    LoadData2dTestParams { 112, 128, 32, false, MainLoad2dL1<int32_t, int8_t, int8_t>, 4, 1, 1 },
    LoadData2dTestParams { 112, 128, 32, false, MainLoad2dL1<int32_t, uint8_t, int8_t>, 4, 1, 1 },
    LoadData2dTestParams { 112, 128, 32, false, MainLoad2dL1<half, half, half>, 2, 2, 2 },
    LoadData2dTestParams { 112, 128, 32, false, MainLoad2dL1<float, half, half>, 4, 2, 2 },
    LoadData2dTestParams { 112, 128, 32, true, MainLoad2dL1<int32_t, uint8_t, uint8_t>, 4, 1, 1 },
    LoadData2dTestParams { 112, 128, 32, true, MainLoad2dL1<int32_t, int8_t, int8_t>, 4, 1, 1 },
    LoadData2dTestParams { 112, 128, 32, true, MainLoad2dL1<int32_t, uint8_t, int8_t>, 4, 1, 1 },
    LoadData2dTestParams { 112, 128, 32, true, MainLoad2dL1<half, half, half>, 2, 2, 2 },
    LoadData2dTestParams { 112, 128, 32, true, MainLoad2dL1<float, half, half>, 4, 2, 2 },
    LoadData2dTestParams { 112, 128, 32, false, MainLoad2dOut<int32_t, uint8_t, uint8_t>, 4, 1, 1 },
    LoadData2dTestParams { 112, 128, 32, false, MainLoad2dOut<int32_t, int8_t, int8_t>, 4, 1, 1 },
    LoadData2dTestParams { 112, 128, 32, false, MainLoad2dOut<int32_t, uint8_t, int8_t>, 4, 1, 1 },
    LoadData2dTestParams { 112, 128, 32, false, MainLoad2dOut<half, half, half>, 2, 2, 2 },
    LoadData2dTestParams { 112, 128, 32, false, MainLoad2dOut<float, half, half>, 4, 2, 2 }));

TEST_P(LoadData2dTestsuite, LoadData2dTestCase)
{
    auto param = GetParam();
    uint8_t dst_gm[param.m * param.n * param.sizeofDst];
    uint8_t src0_gm[param.m * param.k * param.sizeofSrc0];
    uint8_t src1_gm[param.k * param.n * param.sizeofSrc1];

    param.cal_func(dst_gm, src0_gm, src1_gm, param.m, param.n, param.k, param.ifTrans);
    for (int32_t i = 0; i < param.m * param.n * param.sizeofDst; i++) {
        EXPECT_EQ(dst_gm[i], 0x00);
    }
}

INSTANTIATE_TEST_CASE_P(TEST_LOAD_DATA_3DV1, LoadData3dv1Testsuite,
    ::testing::Values(
    LoadData3dv1TestParams { 112, 128, 32, 0, MainLoad3dv1<int32_t, int8_t, int8_t>, 4, 1, 1 },
    LoadData3dv1TestParams { 112, 128, 32, 0, MainLoad3dv1<half, half, half>, 2, 2, 2 },
    LoadData3dv1TestParams { 112, 128, 32, 1, MainLoad3dv1<int32_t, int8_t, int8_t>, 4, 1, 1 },
    LoadData3dv1TestParams { 112, 128, 32, 1, MainLoad3dv1<half, half, half>, 2, 2, 2 }));

TEST_P(LoadData3dv1Testsuite, LoadData3dv1TestCase)
{
    auto param = GetParam();
    uint8_t dst_gm[param.m * param.n * param.sizeofDst];
    uint8_t src0_gm[param.m * param.k * param.sizeofSrc0];
    uint8_t src1_gm[param.k * param.n * param.sizeofSrc1];

    param.cal_func(dst_gm, src0_gm, src1_gm, param.m, param.n, param.k, param.csize);
    for (int32_t i = 0; i < param.m * param.n * param.sizeofDst; i++) {
        EXPECT_EQ(dst_gm[i], 0x00);
    }
}
} // namespace AscendC
