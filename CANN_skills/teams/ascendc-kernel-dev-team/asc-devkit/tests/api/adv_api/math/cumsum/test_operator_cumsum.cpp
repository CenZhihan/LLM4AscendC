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

enum TestMode {
    MODE_NORMAL,
    MODE_TMPBUFFER,
};

class TEST_CUMSUM : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(AscendC::AIV_TYPE);
    }
    void TearDown()
    {
        AscendC::SetGCoreType(AscendC::MIX_TYPE);
    }
};

template <typename T, bool isFirstAxis = false>
void main_vec_cumsum_demo(__gm__ uint8_t *__restrict__ dst_gm, __gm__ uint8_t *__restrict__ last_row_gm,
    __gm__ uint8_t *__restrict__ src_gm, const CumSumInfo &cumSumParams, TestMode test_mode)
{
    TPipe tpipe;
    GlobalTensor<T> input_global;
    GlobalTensor<T> output_global;
    GlobalTensor<T> last_raw_global;
    uint32_t srcSize = cumSumParams.outter * cumSumParams.inner;
    input_global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src_gm), srcSize);
    output_global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst_gm), srcSize);
    last_raw_global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(last_row_gm), cumSumParams.inner);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcSize * sizeof(T));
    LocalTensor<T> input_local = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, srcSize * sizeof(T));
    LocalTensor<T> output_local = tbuf1.Get<T>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, cumSumParams.inner * sizeof(T));
    LocalTensor<T> last_row_local = tbuf2.Get<T>();

    DataCopy(input_local, input_global, srcSize);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    static constexpr CumSumConfig cumSumConfig{!isFirstAxis, false, true};
    if (test_mode == MODE_NORMAL) {
        CumSum<T, cumSumConfig>(output_local, last_row_local, input_local, cumSumParams);
    } else if (test_mode == MODE_TMPBUFFER) {
        LocalTensor<uint8_t> sharedTmpBuffer;
        bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
        ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
        CumSum<T, cumSumConfig>(output_local, last_row_local, input_local, sharedTmpBuffer, cumSumParams);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(output_global, output_local, srcSize);
    DataCopy(last_raw_global, last_row_local, cumSumParams.inner);

    PipeBarrier<PIPE_ALL>();
}
#define VEC_CUMSUM_LASTDIM_TESTCASE(firstDim, lastDim, DATA_TYPE, TEST_MODE)                             \
    TEST_F(TEST_CUMSUM, CUMSUM##firstDim##lastDim##DATA_TYPE##TEST_MODE##Case)                           \
    {                                                                                                    \
        CumSumInfo cumSumParams{firstDim, lastDim};                                                      \
        uint32_t srcSize = firstDim * lastDim;                                                           \
        DATA_TYPE input_gm[srcSize];                                                                     \
        DATA_TYPE output_gm[srcSize];                                                                    \
        DATA_TYPE last_row_gm[lastDim];                                                                  \
        main_vec_cumsum_demo<DATA_TYPE>(                                                                 \
            (uint8_t *)output_gm, (uint8_t *)last_row_gm, (uint8_t *)input_gm, cumSumParams, TEST_MODE); \
                                                                                                         \
        for (uint32_t i = 0; i < srcSize; i++) {                                                         \
            EXPECT_EQ(output_gm[i], static_cast<DATA_TYPE>(0));                                          \
        }                                                                                                \
    }

VEC_CUMSUM_LASTDIM_TESTCASE(16, 16, half, MODE_NORMAL);
VEC_CUMSUM_LASTDIM_TESTCASE(16, 32, half, MODE_NORMAL);
VEC_CUMSUM_LASTDIM_TESTCASE(32, 16, half, MODE_NORMAL);

VEC_CUMSUM_LASTDIM_TESTCASE(16, 8, float, MODE_NORMAL);
VEC_CUMSUM_LASTDIM_TESTCASE(16, 16, float, MODE_NORMAL);
VEC_CUMSUM_LASTDIM_TESTCASE(32, 16, float, MODE_NORMAL);
VEC_CUMSUM_LASTDIM_TESTCASE(8, 8, float, MODE_NORMAL);

#define VEC_CUMSUM_FIRST_TESTCASE(firstDim, lastDim, DATA_TYPE, TEST_MODE)                               \
    TEST_F(TEST_CUMSUM, CUMSUM_FIRST_DIM##firstDim##lastDim##DATA_TYPE##TEST_MODE##Case)                 \
    {                                                                                                    \
        CumSumInfo cumSumParams{firstDim, lastDim};                                                      \
        uint32_t srcSize = firstDim * lastDim;                                                           \
        DATA_TYPE input_gm[srcSize];                                                                     \
        DATA_TYPE output_gm[srcSize];                                                                    \
        DATA_TYPE last_row_gm[lastDim];                                                                  \
        main_vec_cumsum_demo<DATA_TYPE, true>(                                                           \
            (uint8_t *)output_gm, (uint8_t *)last_row_gm, (uint8_t *)input_gm, cumSumParams, TEST_MODE); \
                                                                                                         \
        for (uint32_t i = 0; i < srcSize; i++) {                                                         \
            EXPECT_EQ(output_gm[i], static_cast<DATA_TYPE>(0));                                          \
        }                                                                                                \
    }

VEC_CUMSUM_FIRST_TESTCASE(16, 16, half, MODE_NORMAL);
VEC_CUMSUM_FIRST_TESTCASE(16, 32, half, MODE_NORMAL);

VEC_CUMSUM_FIRST_TESTCASE(16, 8, float, MODE_NORMAL);
VEC_CUMSUM_FIRST_TESTCASE(16, 16, float, MODE_NORMAL);