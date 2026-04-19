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
using namespace AscendC;

template <typename T>
__global__ __aicore__ void MainDataCopySimple(__gm__ uint8_t* __restrict__ src_gm, __gm__ uint8_t* __restrict__ dst_gm,
    int32_t data_size, bool out2l1)
{
    TPipe tpipe;
    GlobalTensor<T> src_global;
    GlobalTensor<T> dst_global;
    src_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src_gm), data_size);
    dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst_gm), data_size);

    TBuf<TPosition::CO2> tbuf;
    tpipe.InitBuffer(tbuf, data_size * sizeof(T));
    LocalTensor<T> src_ub_0 = tbuf.Get<T>();

    TBuf<TPosition::CO2> tbuf1;
    tpipe.InitBuffer(tbuf1, data_size * sizeof(T));
    LocalTensor<T> src_ub_1 = tbuf1.Get<T>();

    TBuf<TPosition::A1> tbuf2;
    tpipe.InitBuffer(tbuf2, data_size * sizeof(T));
    LocalTensor<T> src_l1 = tbuf2.Get<T>();

    TBuf<TPosition::CO2> tbuf3;
    tpipe.InitBuffer(tbuf3, data_size * sizeof(T));
    LocalTensor<T> dst_ub = tbuf3.Get<T>();

    if (out2l1) {
        DataCopy(src_l1, src_global, data_size);
        pipe_barrier(PIPE_ALL);
        DataCopy(src_ub_0, src_l1, data_size);
        pipe_barrier(PIPE_ALL);
        DataCopy(dst_ub, src_ub_0, data_size);
        pipe_barrier(PIPE_ALL);
        DataCopy(dst_global, dst_ub, data_size);
    } else {
        DataCopy(src_ub_0, src_global, data_size);
        pipe_barrier(PIPE_ALL);
        DataCopy(src_l1, src_ub_0, data_size);
        pipe_barrier(PIPE_ALL);
        DataCopy(src_ub_1, src_l1, data_size);
        pipe_barrier(PIPE_ALL);
        DataCopy(dst_ub, src_ub_1, data_size);
        pipe_barrier(PIPE_ALL);
        DataCopy(dst_global, dst_ub, data_size);
    }

    pipe_barrier(PIPE_ALL);
}

struct DataCopyTestParams {
    int32_t data_size;
    int32_t data_bit_size;
    void (*cal_func)(uint8_t*, uint8_t*, int32_t, bool);
    bool out2l1;
};

class DataCopySimpleTestsuite : public testing::Test, public testing::WithParamInterface<DataCopyTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_DATA_COPY_SIMPLE, DataCopySimpleTestsuite,
    ::testing::Values(DataCopyTestParams { 512, 4, MainDataCopySimple<float>, true },
    DataCopyTestParams { 512, 2, MainDataCopySimple<half>, true },
    DataCopyTestParams { 512, 4, MainDataCopySimple<int32_t>, true },
    DataCopyTestParams { 512, 2, MainDataCopySimple<int16_t>, true },
    DataCopyTestParams { 512, 4, MainDataCopySimple<uint32_t>, true },
    DataCopyTestParams { 512, 2, MainDataCopySimple<uint16_t>, true },
    DataCopyTestParams { 512, 4, MainDataCopySimple<float>, false },
    DataCopyTestParams { 512, 2, MainDataCopySimple<half>, false },
    DataCopyTestParams { 512, 4, MainDataCopySimple<int32_t>, false },
    DataCopyTestParams { 512, 2, MainDataCopySimple<int16_t>, false },
    DataCopyTestParams { 512, 4, MainDataCopySimple<uint32_t>, false },
    DataCopyTestParams { 512, 2, MainDataCopySimple<uint16_t>, false }));

TEST_P(DataCopySimpleTestsuite, DataCopySimpleTestCase)
{
    auto param = GetParam();
    uint8_t src_gm[param.data_size * param.data_bit_size];
    uint8_t dst_gm[param.data_size * param.data_bit_size];

    param.cal_func(src_gm, dst_gm, param.data_size, param.out2l1);
    for (int32_t i = 0; i < param.data_size; i++) {
        EXPECT_EQ(dst_gm[i], 0x00);
    }
}

// test l0c -> ub deq8 deq16 vdeq8 vdeq16
// SRC_UB_T -> CC_T, CC_T -> DST_UB_T
template <typename SRC_UB_T, typename CC_T, typename DST_UB_T>
__global__ __aicore__ void MainDataCopyL0c2UbDeqModeDemo(__gm__ uint8_t* __restrict__ src_gm,
    __gm__ uint8_t* __restrict__ dst_gm, int32_t data_size, BlockMode blockMode, DeqScale deqScale, bool isRelu)
{
    TPipe tpipe;
    GlobalTensor<SRC_UB_T> src_global;
    GlobalTensor<DST_UB_T> dst_global;
    src_global.SetGlobalBuffer(reinterpret_cast<__gm__ SRC_UB_T*>(src_gm), data_size);
    dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ DST_UB_T*>(dst_gm), data_size);

    TBuf<TPosition::CO2> tbuf;
    tpipe.InitBuffer(tbuf, data_size * sizeof(SRC_UB_T));
    LocalTensor<SRC_UB_T> src_ub = tbuf.Get<SRC_UB_T>();

    TBuf<TPosition::CO1> tbuf1;
    tpipe.InitBuffer(tbuf1, data_size * sizeof(CC_T));
    LocalTensor<CC_T> src_cc = tbuf1.Get<CC_T>();

    TBuf<TPosition::CO2> tbuf2;
    tpipe.InitBuffer(tbuf2, data_size * sizeof(DST_UB_T) * 2);
    LocalTensor<DST_UB_T> dst_ub = tbuf2.Get<DST_UB_T>();

    DataCopy(src_ub, src_global, data_size);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    DataCopyEnhancedParams enhancedParams;
    enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    // ub to l0c
    DataCopy(src_cc, src_ub, { 1, 2, 0, 0 }, enhancedParams);
    pipe_barrier(PIPE_V);
    float a = 1;
    enhancedParams.deqScale = deqScale;
    enhancedParams.isRelu = isRelu;
    int32_t scalar = (int32_t)GetScalarBitcodeValue(a);
    if (deqScale == DeqScale::DEQ8) {
        enhancedParams.sidStoreMode = 2;
        enhancedParams.deqValue = scalar;
    } else if (deqScale == DeqScale::DEQ16) {
        enhancedParams.deqValue = scalar;
    } else if (deqScale == DeqScale::VDEQ8) {
        TBuf<TPosition::CO2> tbuf3;
        tpipe.InitBuffer(tbuf3, data_size * sizeof(uint64_t));
        LocalTensor<uint64_t> deqLocal = tbuf3.Get<uint64_t>();
        enhancedParams.deqTensorAddr = (uint64_t)deqLocal.GetPhyAddr();
        enhancedParams.sidStoreMode = 2;
    } else if (deqScale == DeqScale::VDEQ16) {
        TBuf<TPosition::CO2> tbuf3;
        tpipe.InitBuffer(tbuf3, data_size * sizeof(uint64_t));
        LocalTensor<uint64_t> deqLocal = tbuf3.Get<uint64_t>();
        enhancedParams.deqTensorAddr = (uint64_t)deqLocal.GetPhyAddr();
    }

    // l0c to ub
    DataCopy(dst_ub, src_cc, { 1, 2, 0, 0 }, enhancedParams);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(dst_global, dst_ub, data_size);

    pipe_barrier(PIPE_ALL);
}


// test l0c -> ub
// SRC_UB_T -> CC_T, CC_T -> DST_UB_T
template <typename SRC_UB_T, typename CC_T, typename DST_UB_T>
__global__ __aicore__ void MainDataCopyL0c2UbDemo(__gm__ uint8_t* __restrict__ src_gm,
    __gm__ uint8_t* __restrict__ dst_gm, int32_t data_size, BlockMode blockMode, DeqScale deqScale, bool isRelu)
{
    TPipe tpipe;
    GlobalTensor<SRC_UB_T> src_global;
    GlobalTensor<DST_UB_T> dst_global;
    src_global.SetGlobalBuffer(reinterpret_cast<__gm__ SRC_UB_T*>(src_gm), data_size);
    dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ DST_UB_T*>(dst_gm), data_size);

    TBuf<TPosition::CO2> tbuf;
    tpipe.InitBuffer(tbuf, data_size * sizeof(SRC_UB_T));
    LocalTensor<SRC_UB_T> src_ub = tbuf.Get<SRC_UB_T>();

    TBuf<TPosition::CO1> tbuf1;
    tpipe.InitBuffer(tbuf1, data_size * sizeof(CC_T));
    LocalTensor<CC_T> src_cc = tbuf1.Get<CC_T>();

    TBuf<TPosition::CO2> tbuf2;
    tpipe.InitBuffer(tbuf2, data_size * sizeof(DST_UB_T) * 2);
    LocalTensor<DST_UB_T> dst_ub = tbuf2.Get<DST_UB_T>();

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(src_ub, src_global, data_size);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // ub to l0c
    DataCopy(src_cc, src_ub, { 1, 2, 0, 0 },
        { BlockMode::BLOCK_MODE_VECTOR, DeqScale::DEQ_NONE, 0, 0, false, pad_t::PAD_NONE, 0 });

    pipe_barrier(PIPE_V);

    // l0c to ub
    DataCopy(dst_ub, src_cc, { 1, 2, 1, 1 }, { blockMode, deqScale, 0, 0, isRelu, pad_t::PAD_NONE, 0 });

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(dst_global, dst_ub, data_size);

    pipe_barrier(PIPE_ALL);
}

// test ub -> l0c
// SRC_UB_T -> CC_T, CC_T -> DST_UB_T
template <typename SRC_UB_T, typename CC_T, typename DST_UB_T>
__global__ __aicore__ void MainDataCopyUb2L0cDemo(__gm__ uint8_t* __restrict__ src_gm,
    __gm__ uint8_t* __restrict__ dst_gm, int32_t data_size, BlockMode blockMode, DeqScale deqScale, bool isRelu)
{
    TPipe tpipe;
    GlobalTensor<SRC_UB_T> src_global;
    GlobalTensor<DST_UB_T> dst_global;
    src_global.SetGlobalBuffer(reinterpret_cast<__gm__ SRC_UB_T*>(src_gm), data_size);
    dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ DST_UB_T*>(dst_gm), data_size);

    TBuf<TPosition::CO2> tbuf;
    tpipe.InitBuffer(tbuf, data_size * sizeof(SRC_UB_T) * 2);
    LocalTensor<SRC_UB_T> src_ub = tbuf.Get<SRC_UB_T>();

    TBuf<TPosition::CO1> tbuf1;
    tpipe.InitBuffer(tbuf1, data_size * sizeof(CC_T));
    LocalTensor<CC_T> src_cc = tbuf1.Get<CC_T>();

    TBuf<TPosition::CO2> tbuf2;
    tpipe.InitBuffer(tbuf2, data_size * sizeof(DST_UB_T) * 2);
    LocalTensor<DST_UB_T> dst_ub = tbuf2.Get<DST_UB_T>();

    AscendCUtils::SetMask<uint8_t>(256);

    DataCopy(src_ub, src_global, data_size);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // ub to l0c
    DataCopy(src_cc, src_ub, { 1, 2, 0, 0 }, { blockMode, deqScale, 0, 0, isRelu, pad_t::PAD_NONE, 0 });

    pipe_barrier(PIPE_V);

    // l0c to ub
    DataCopy(dst_ub, src_cc, { 1, 2, 1, 1 },
        { BlockMode::BLOCK_MODE_MATRIX, DeqScale::DEQ_NONE, 0, 0, false, pad_t::PAD_NONE, 0 });

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    DataCopy(dst_global, dst_ub, data_size);

    pipe_barrier(PIPE_ALL);
}

struct DataCopyEnhancedTestParams {
    void (*cal_func)(uint8_t*, uint8_t*, int32_t, BlockMode, DeqScale, bool);
    int32_t data_size;
    int32_t src_ub_bit_size;
    int32_t dst_ub_bit_size;
    BlockMode blockMode;
    DeqScale deqScale;
    bool isRelu;
};

class DataCopyEnhancedTestsuite : public testing::Test, public testing::WithParamInterface<DataCopyEnhancedTestParams> {
protected:
    void SetUp() {}

    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_DATA_COPY_ENHANCED, DataCopyEnhancedTestsuite,
    ::testing::Values(
    // l0c to ub, matrix
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<half, half, half>, 512, 2, 2, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::DEQ, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<half, half, half>, 512, 2, 2, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::DEQ_NONE, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<half, half, half>, 512, 2, 2, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<float, float, half>, 512, 4, 2, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::DEQ_NONE, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<float, float, half>, 512, 4, 2, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<float, float, float>, 512, 4, 4, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::DEQ_NONE, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<float, float, float>, 512, 4, 4, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDeqModeDemo<int32_t, int32_t, uint8_t>, 512, 4, 4, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::DEQ8, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDeqModeDemo<int32_t, int32_t, half>, 512, 4, 4, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::DEQ16, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDeqModeDemo<int32_t, int32_t, uint8_t>, 512, 4, 4, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::VDEQ8, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDeqModeDemo<int32_t, int32_t, half>, 512, 4, 4, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::VDEQ16, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDeqModeDemo<int32_t, int32_t, uint8_t>, 512, 4, 4, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::DEQ8, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDeqModeDemo<int32_t, int32_t, half>, 512, 4, 4, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::DEQ16, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDeqModeDemo<int32_t, int32_t, uint8_t>, 512, 4, 4, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::VDEQ8, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDeqModeDemo<int32_t, int32_t, half>, 512, 4, 4, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::VDEQ16, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<int32_t, int32_t, half>, 512, 4, 2,
    BlockMode::BLOCK_MODE_MATRIX, DeqScale::DEQ, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<int32_t, int32_t, half>, 512, 4, 2,
    BlockMode::BLOCK_MODE_MATRIX, DeqScale::DEQ16, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<int32_t, int32_t, int16_t>, 512, 4, 2,
    BlockMode::BLOCK_MODE_MATRIX, DeqScale::DEQ16, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<int32_t, int32_t, int16_t>, 512, 4, 2,
    BlockMode::BLOCK_MODE_MATRIX, DeqScale::DEQ_NONE, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<int32_t, int32_t, int16_t>, 512, 4, 2,
    BlockMode::BLOCK_MODE_MATRIX, DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<int32_t, int32_t, int32_t>, 512, 4, 4,
    BlockMode::BLOCK_MODE_MATRIX, DeqScale::DEQ_NONE, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<int32_t, int32_t, int32_t>, 512, 4, 4,
    BlockMode::BLOCK_MODE_MATRIX, DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<uint32_t, uint32_t, uint32_t>, 512, 4, 4,
    BlockMode::BLOCK_MODE_MATRIX, DeqScale::DEQ_NONE, false },

    // l0c to ub, vector
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<half, half, half>, 512, 2, 2, BlockMode::BLOCK_MODE_VECTOR,
    DeqScale::DEQ, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<half, half, half>, 512, 2, 2, BlockMode::BLOCK_MODE_VECTOR,
    DeqScale::DEQ_NONE, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<half, half, half>, 512, 2, 2, BlockMode::BLOCK_MODE_VECTOR,
    DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<float, float, half>, 512, 4, 2, BlockMode::BLOCK_MODE_VECTOR,
    DeqScale::DEQ_NONE, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<float, float, half>, 512, 4, 2, BlockMode::BLOCK_MODE_VECTOR,
    DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<float, float, float>, 512, 4, 4, BlockMode::BLOCK_MODE_VECTOR,
    DeqScale::DEQ_NONE, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<float, float, float>, 512, 4, 4, BlockMode::BLOCK_MODE_VECTOR,
    DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<int32_t, int32_t, half>, 512, 4, 2,
    BlockMode::BLOCK_MODE_VECTOR, DeqScale::DEQ, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<int32_t, int32_t, half>, 512, 4, 2,
    BlockMode::BLOCK_MODE_VECTOR, DeqScale::DEQ16, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<int32_t, int32_t, int16_t>, 512, 4, 2,
    BlockMode::BLOCK_MODE_VECTOR, DeqScale::DEQ16, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<int32_t, int32_t, int16_t>, 512, 4, 2,
    BlockMode::BLOCK_MODE_VECTOR, DeqScale::DEQ_NONE, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<int32_t, int32_t, int16_t>, 512, 4, 2,
    BlockMode::BLOCK_MODE_VECTOR, DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<int32_t, int32_t, int32_t>, 512, 4, 4,
    BlockMode::BLOCK_MODE_VECTOR, DeqScale::DEQ_NONE, true },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<int32_t, int32_t, int32_t>, 512, 4, 4,
    BlockMode::BLOCK_MODE_VECTOR, DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyL0c2UbDemo<uint32_t, uint32_t, uint32_t>, 512, 4, 4,
    BlockMode::BLOCK_MODE_VECTOR, DeqScale::DEQ_NONE, false },

    // ub -> l0c, matrix
    DataCopyEnhancedTestParams { MainDataCopyUb2L0cDemo<half, half, half>, 512, 2, 2, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyUb2L0cDemo<float, float, float>, 512, 4, 4, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyUb2L0cDemo<half, float, float>, 512, 2, 4, BlockMode::BLOCK_MODE_MATRIX,
    DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyUb2L0cDemo<int32_t, int32_t, int32_t>, 512, 4, 4,
    BlockMode::BLOCK_MODE_MATRIX, DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyUb2L0cDemo<uint32_t, uint32_t, uint32_t>, 512, 4, 4,
    BlockMode::BLOCK_MODE_MATRIX, DeqScale::DEQ_NONE, false },

    // ub -> l0c, vector
    DataCopyEnhancedTestParams { MainDataCopyUb2L0cDemo<half, half, half>, 512, 2, 2, BlockMode::BLOCK_MODE_VECTOR,
    DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyUb2L0cDemo<float, float, float>, 512, 4, 4, BlockMode::BLOCK_MODE_VECTOR,
    DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyUb2L0cDemo<half, float, float>, 512, 2, 4, BlockMode::BLOCK_MODE_VECTOR,
    DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyUb2L0cDemo<int32_t, int32_t, int32_t>, 512, 4, 4,
    BlockMode::BLOCK_MODE_VECTOR, DeqScale::DEQ_NONE, false },
    DataCopyEnhancedTestParams { MainDataCopyUb2L0cDemo<uint32_t, uint32_t, uint32_t>, 512, 4, 4,
    BlockMode::BLOCK_MODE_VECTOR, DeqScale::DEQ_NONE, false }));

TEST_P(DataCopyEnhancedTestsuite, DataCopyEnhancedTestCase)
{
    auto param = GetParam();
    uint8_t src_gm[param.data_size * param.src_ub_bit_size];
    uint8_t dst_gm[param.data_size * param.dst_ub_bit_size];

    param.cal_func(src_gm, dst_gm, param.data_size, param.blockMode, param.deqScale, param.isRelu);
    for (int32_t i = 0; i < param.data_size; i++) {
        EXPECT_EQ(dst_gm[i], 0x00);
    }
}

namespace AscendC {
// T, U for gm->l1 nd2nz
template <typename T> class KernelDataCopyGm2L1Nd2Nz {
public:
    __aicore__ inline KernelDataCopyGm2L1Nd2Nz() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, Nd2NzParams& intriParamsIn)
    {
        intriParams = intriParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrcA1, 1,
            (((intriParams.dValue * sizeof(T) - 1) / 32 + 1) * intriParams.dstNzC0Stride * 32));
        pipe.InitBuffer(inQueueSrcVecOut, 1,
            (((intriParams.dValue * sizeof(T) - 1) / 32 + 1) * intriParams.dstNzC0Stride * 32));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<T> srcLocal = inQueueSrcA1.AllocTensor<T>();

        DataCopy(srcLocal, srcGlobal, intriParams);
        Nd2NzParams param;
        DataCopy(srcLocal, srcGlobal, param);

        inQueueSrcA1.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        ;
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = inQueueSrcA1.DeQue<T>();
        LocalTensor<T> dstLocalVecOut = inQueueSrcVecOut.AllocTensor<T>();
        DataCopy(dstLocalVecOut, dstLocal,
            { 1, (((intriParams.dValue * sizeof(T) - 1) / 32 + 1) * intriParams.dstNzC0Stride * 32) / 32, 0, 0 });
        inQueueSrcVecOut.EnQue(dstLocalVecOut);
        inQueueSrcVecOut.DeQue<T>();
        DataCopy(dstGlobal, dstLocalVecOut,
            { 1, (((intriParams.dValue * sizeof(T) - 1) / 32 + 1) * intriParams.dstNzC0Stride * 32) / 32, 0, 0 });
        inQueueSrcVecOut.FreeTensor(dstLocalVecOut);
        inQueueSrcA1.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::A1, 1> inQueueSrcA1;
    TQue<TPosition::VECOUT, 1> inQueueSrcVecOut;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    Nd2NzParams intriParams;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void MainDataCopyGm2L1Nd2Nz(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
    Nd2NzParams& intriParams)
{
    AscendC::KernelDataCopyGm2L1Nd2Nz<T> op;
    op.Init(dstGm, srcGm, intriParams);
    op.Process();
}

struct DataCopyGm2L1Nd2NzTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, Nd2NzParams&);
    Nd2NzParams intriParams;
};

class DataCopyGm2L1Nd2NzTestsuite : public testing::Test,
    public testing::WithParamInterface<DataCopyGm2L1Nd2NzTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DATACOPYGM2L1ND2NZ, DataCopyGm2L1Nd2NzTestsuite,
    ::testing::Values(DataCopyGm2L1Nd2NzTestParams { 2, MainDataCopyGm2L1Nd2Nz<half>, { 2, 2, 13, 48, 16, 11, 2, 48 } },
    DataCopyGm2L1Nd2NzTestParams { 4, MainDataCopyGm2L1Nd2Nz<float>, { 2, 2, 13, 48, 16, 11, 2, 40 } }));

TEST_P(DataCopyGm2L1Nd2NzTestsuite, DataCopyGm2L1Nd2NzTestsuiteOpTestCase)
{
    auto param = GetParam();
    Nd2NzParams intriParams = param.intriParams;
    uint8_t srcGm[intriParams.ndNum * intriParams.srcNdMatrixStride * param.typeSize] = {0};
    uint8_t dstGm[((intriParams.dValue * param.typeSize -1 ) / 32 + 1) * intriParams.dstNzC0Stride * 32] = {0};

    param.cal_func(dstGm, srcGm, intriParams);
    for (int32_t i = 0; i < (sizeof(dstGm) / sizeof(dstGm[0])); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}

namespace AscendC {
// T, U for ub->out nz2nd
template <typename T> class KernelDataCopyUb2GmNz2Nd {
public:
    __aicore__ inline KernelDataCopyUb2GmNz2Nd() {}
    __aicore__ inline void Init(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm, Nz2NdParamsFull& intriParamsIn)
    {
        intriParams = intriParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T*)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dstGm);
        pipe.InitBuffer(inQueueSrcVecIn, 1, 65568);
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
        Nd2NzParams nd2nzPrams = {intriParams.ndNum, intriParams.nValue, intriParams.dValue,
            32, 32, 32, 32, 32};
        DataCopy(srcLocal, srcGlobal, nd2nzPrams);
        inQueueSrcVecIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        ;
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> dstLocal = inQueueSrcVecIn.DeQue<T>();

        set_atomic_none();
        DataCopy(dstGlobal, dstLocal, intriParams);
        set_atomic_f16();
        DataCopy(dstGlobal, dstLocal, intriParams);

        inQueueSrcVecIn.FreeTensor(dstLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueSrcVecIn;
    GlobalTensor<T> srcGlobal;
    GlobalTensor<T> dstGlobal;
    Nz2NdParamsFull intriParams;
};
} // namespace AscendC

template <typename T>
__global__ __aicore__ void MainDataCopyUb2GmNz2Nd(__gm__ uint8_t* dstGm, __gm__ uint8_t* srcGm,
    Nz2NdParamsFull& intriParams)
{
    AscendC::KernelDataCopyUb2GmNz2Nd<T> op;
    op.Init(dstGm, srcGm, intriParams);
    op.Process();
}

struct DataCopyUb2GmNz2NdTestParams {
    int32_t typeSize;
    void (*cal_func)(uint8_t*, uint8_t*, Nz2NdParamsFull&);
    Nz2NdParamsFull intriParams;
};

class DataCopyUb2GmNz2NdTestsuite : public testing::Test,
    public testing::WithParamInterface<DataCopyUb2GmNz2NdTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_OPEARATION_DATACOPYUb2GmNz2Nd, DataCopyUb2GmNz2NdTestsuite,
    ::testing::Values(DataCopyUb2GmNz2NdTestParams { 2, MainDataCopyUb2GmNz2Nd<half>, { 1, 32, 32, 1, 32, 32, 1 } },
    DataCopyUb2GmNz2NdTestParams { 2, MainDataCopyUb2GmNz2Nd<half>, { 1, 64, 32, 1, 64, 32, 1 } }));

TEST_P(DataCopyUb2GmNz2NdTestsuite, DataCopyUb2GmNz2NdTestsuiteOpTestCase)
{
    auto param = GetParam();
    Nz2NdParamsFull intriParams = param.intriParams;
    uint8_t srcGm[intriParams.nValue * intriParams.dValue * param.typeSize];
    uint8_t dstGm[intriParams.nValue * intriParams.dValue * param.typeSize];

    param.cal_func(dstGm, srcGm, intriParams);
    for (int32_t i = 0; i < (intriParams.nValue * intriParams.dValue); i++) {
        EXPECT_EQ(dstGm[i], 0x00);
    }
}