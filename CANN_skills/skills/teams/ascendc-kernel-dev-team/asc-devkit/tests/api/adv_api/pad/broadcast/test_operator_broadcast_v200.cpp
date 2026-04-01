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

class TEST_BROADCAST_200 : public testing::Test {
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

template <typename T, int32_t dim, int32_t axis>
void main_vec_broadcast_demo(__gm__ uint8_t *__restrict__ dst_gm, __gm__ uint8_t *__restrict__ src_gm,
    const uint32_t dstShape[dim], const uint32_t srcShape[dim], TestMode test_mode)
{
    TPipe tpipe;
    GlobalTensor<T> input_global;
    GlobalTensor<T> output_global;
    uint32_t srcSize = 1;
    uint32_t dstSize = 1;
    for (uint32_t i = 0; i < dim; i++) {
        srcSize *= srcShape[i];
        dstSize *= dstShape[i];
    }
    if (srcSize == 1) {
        srcSize = 32;
    }
    input_global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src_gm), srcSize);
    output_global.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst_gm), dstSize);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcSize * sizeof(T));
    LocalTensor<T> input_local = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, dstSize * sizeof(T));
    LocalTensor<T> output_local = tbuf1.Get<T>();

    DataCopy(input_local, input_global, srcSize);

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    if (test_mode == MODE_NORMAL) {
        Broadcast<T, dim, axis, false>(output_local, input_local, dstShape, srcShape);
    } else if (test_mode == MODE_TMPBUFFER) {
        LocalTensor<uint8_t> sharedTmpBuffer;
        bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
        ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
        Broadcast<T, dim, axis, false>(output_local, input_local, dstShape, srcShape, sharedTmpBuffer);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(output_global, output_local, dstSize);

    PipeBarrier<PIPE_ALL>();
}
#define VEC_BROADCAST_LASTDIM_TESTCASE(firstDim, lastDim, DATA_TYPE, TEST_MODE)        \
    TEST_F(TEST_BROADCAST_200, BROADCAST_LAST##_##firstDim##_##lastDim##_##DATA_TYPE##_##TEST_MODE##Case)   \
    {                                                                                  \
        uint32_t srcShape[] = {firstDim, 1};                                           \
        uint32_t dstShape[] = {firstDim, lastDim};                                     \
        uint32_t srcSize = firstDim;                                                   \
        uint32_t dstSize = firstDim * lastDim;                                         \
        DATA_TYPE input_gm[srcSize];                                                   \
        DATA_TYPE output_gm[firstDim][lastDim];                                        \
        main_vec_broadcast_demo<DATA_TYPE, 2, 1>(                                      \
            (uint8_t *)output_gm, (uint8_t *)input_gm, dstShape, srcShape, TEST_MODE); \
                                                                                       \
        for (uint32_t i = 0; i < firstDim; i++) {                                      \
            for (uint32_t j = 0; j < lastDim; j++) {                                   \
                EXPECT_EQ(output_gm[i][j], static_cast<DATA_TYPE>(0));                 \
            }                                                                          \
        }                                                                              \
    }

VEC_BROADCAST_LASTDIM_TESTCASE(32, 16, half, MODE_NORMAL);
VEC_BROADCAST_LASTDIM_TESTCASE(32, 15, half, MODE_NORMAL);
VEC_BROADCAST_LASTDIM_TESTCASE(32, 8, float, MODE_NORMAL);
VEC_BROADCAST_LASTDIM_TESTCASE(32, 7, float, MODE_NORMAL);
VEC_BROADCAST_LASTDIM_TESTCASE(2528, 3, float, MODE_NORMAL);
VEC_BROADCAST_LASTDIM_TESTCASE(32, 16, int8_t, MODE_NORMAL);
VEC_BROADCAST_LASTDIM_TESTCASE(32, 15, int8_t, MODE_NORMAL);
VEC_BROADCAST_LASTDIM_TESTCASE(32, 16, uint8_t, MODE_NORMAL);
VEC_BROADCAST_LASTDIM_TESTCASE(32, 15, uint8_t, MODE_NORMAL);

VEC_BROADCAST_LASTDIM_TESTCASE(32, 16, half, MODE_TMPBUFFER);
VEC_BROADCAST_LASTDIM_TESTCASE(32, 15, half, MODE_TMPBUFFER);
VEC_BROADCAST_LASTDIM_TESTCASE(32, 8, float, MODE_TMPBUFFER);
VEC_BROADCAST_LASTDIM_TESTCASE(32, 7, float, MODE_TMPBUFFER);
VEC_BROADCAST_LASTDIM_TESTCASE(2528, 3, float, MODE_TMPBUFFER);

#define VEC_BROADCAST_FIRSTDIM_TESTCASE(firstDim, lastDim, DATA_TYPE, TEST_MODE)        \
    TEST_F(TEST_BROADCAST_200, BROADCAST_FIRST##_##firstDim##_##lastDim##_##DATA_TYPE##_##TEST_MODE##Case)   \
    {                                                                                  \
        uint32_t srcShape[] = {1, lastDim};                                           \
        uint32_t dstShape[] = {firstDim, lastDim};                                     \
        uint32_t srcSize = lastDim;                                                   \
        uint32_t dstSize = firstDim * lastDim;                                         \
        DATA_TYPE input_gm[srcSize];                                                   \
        DATA_TYPE output_gm[firstDim][lastDim];                                        \
        main_vec_broadcast_demo<DATA_TYPE, 2, 0>(                                      \
            (uint8_t *)output_gm, (uint8_t *)input_gm, dstShape, srcShape, TEST_MODE); \
                                                                                       \
        for (uint32_t i = 0; i < firstDim; i++) {                                      \
            for (uint32_t j = 0; j < lastDim; j++) {                                   \
                EXPECT_EQ(output_gm[i][j], static_cast<DATA_TYPE>(0));                 \
            }                                                                          \
        }                                                                              \
    }

VEC_BROADCAST_FIRSTDIM_TESTCASE(32, 16, half, MODE_NORMAL);
VEC_BROADCAST_FIRSTDIM_TESTCASE(32, 8, float, MODE_NORMAL);
VEC_BROADCAST_FIRSTDIM_TESTCASE(32, 32, int8_t, MODE_NORMAL);
VEC_BROADCAST_FIRSTDIM_TESTCASE(32, 32, uint8_t, MODE_NORMAL);

VEC_BROADCAST_FIRSTDIM_TESTCASE(32, 16, half, MODE_TMPBUFFER);
VEC_BROADCAST_FIRSTDIM_TESTCASE(32, 8, float, MODE_TMPBUFFER);
VEC_BROADCAST_FIRSTDIM_TESTCASE(32, 32, int8_t, MODE_TMPBUFFER);
VEC_BROADCAST_FIRSTDIM_TESTCASE(32, 32, uint8_t, MODE_TMPBUFFER);

#define VEC_NO_BROADCAST_TESTCASE(firstDim, lastDim, DATA_TYPE, TEST_MODE)        \
    TEST_F(TEST_BROADCAST_200, NO_BROADCAST_##_##firstDim##_##lastDim##_##DATA_TYPE##_##TEST_MODE##Case)   \
    {                                                                                  \
        uint32_t srcShape[] = {firstDim, lastDim};                                           \
        uint32_t dstShape[] = {firstDim, lastDim};                                     \
        uint32_t srcSize = lastDim;                                                   \
        uint32_t dstSize = firstDim * lastDim;                                         \
        DATA_TYPE input_gm[srcSize];                                                   \
        DATA_TYPE output_gm[firstDim][lastDim];                                        \
        main_vec_broadcast_demo<DATA_TYPE, 2, 1>(                                      \
            (uint8_t *)output_gm, (uint8_t *)input_gm, dstShape, srcShape, TEST_MODE); \
                                                                                       \
        for (uint32_t i = 0; i < firstDim; i++) {                                      \
            for (uint32_t j = 0; j < lastDim; j++) {                                   \
                EXPECT_EQ(output_gm[i][j], static_cast<DATA_TYPE>(0));                 \
            }                                                                          \
        }                                                                              \
    }

VEC_NO_BROADCAST_TESTCASE(32, 1, half, MODE_NORMAL);
VEC_NO_BROADCAST_TESTCASE(32, 1, float, MODE_NORMAL);
VEC_NO_BROADCAST_TESTCASE(32, 1, int8_t, MODE_NORMAL);
VEC_NO_BROADCAST_TESTCASE(32, 1, uint8_t, MODE_NORMAL);

#define VEC_ONEDIM_BROADCAST_TESTCASE(dstDim, DATA_TYPE, TEST_MODE)                              \
    TEST_F(TEST_BROADCAST_200, ONEDIM_BROADCAST_dstDim##_##DATA_TYPE##_##TEST_MODE##Case)   \
    {                                                                                            \
        uint32_t srcShape[] = {1};                                                               \
        uint32_t dstShape[] = {dstDim};                                                          \
        uint32_t dstSize = dstDim;                                                               \
        DATA_TYPE input_gm[1];                                                                   \
        DATA_TYPE output_gm[dstDim];                                                             \
        main_vec_broadcast_demo<DATA_TYPE, 1, 0>(                                                \
            (uint8_t *)output_gm, (uint8_t *)input_gm, dstShape, srcShape, TEST_MODE);           \
                                                                                                 \
        for (uint32_t i = 0; i < dstDim; i++) {                                                  \
            EXPECT_EQ(output_gm[i], static_cast<DATA_TYPE>(0));                                  \
        }                                                                                        \
    }

VEC_ONEDIM_BROADCAST_TESTCASE(16, half, MODE_NORMAL);
VEC_ONEDIM_BROADCAST_TESTCASE(8, float, MODE_NORMAL);
VEC_ONEDIM_BROADCAST_TESTCASE(32, int8_t, MODE_NORMAL);
VEC_ONEDIM_BROADCAST_TESTCASE(32, uint8_t, MODE_NORMAL);