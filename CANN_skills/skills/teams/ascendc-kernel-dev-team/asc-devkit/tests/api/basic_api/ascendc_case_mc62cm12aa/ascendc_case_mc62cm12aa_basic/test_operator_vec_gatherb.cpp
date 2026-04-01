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
#define private public
#define protect public
#include "kernel_operator.h"
// #include "model/model_factory.h"
#include "common.h"
using namespace std;
using namespace AscendC;

struct GatherbTestParams {
    uint32_t srcLen;
    uint32_t offsetLen;
    uint32_t dataBitSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t);
};

class GatherbTestSuite : public testing::Test, public testing::WithParamInterface<GatherbTestParams> {
protected:
    void SetUp() {}
    void TearDown() {}
};

template <typename T>
void test_vgatherb_func(__gm__ uint8_t* __restrict__ dst_gm, __gm__ uint8_t* __restrict__ src0_gm,
    __gm__ uint8_t* __restrict__ offset_gm, __gm__ uint32_t srcLen, __gm__ uint32_t offsetLen)
{
    TPipe tpipe;
    GlobalTensor<T> input0_global;
    GlobalTensor<uint32_t> offset_global;
    GlobalTensor<T> output_global;
    uint32_t dstLen = srcLen;
    input0_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src0_gm), srcLen);
    offset_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(offset_gm), offsetLen);
    output_global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst_gm), dstLen);

    TBuf<TPosition::VECCALC> tbuf;
    tpipe.InitBuffer(tbuf, srcLen * sizeof(T));
    LocalTensor<T> input0_local = tbuf.Get<T>();

    TBuf<TPosition::VECCALC> tbuf1;
    tpipe.InitBuffer(tbuf1, offsetLen * sizeof(uint32_t));
    LocalTensor<uint32_t> offset_local = tbuf1.Get<uint32_t>();

    TBuf<TPosition::VECCALC> tbuf2;
    tpipe.InitBuffer(tbuf2, dstLen * sizeof(T));
    LocalTensor<T> output_local = tbuf2.Get<T>();

    for (int32_t i = 0; i < srcLen; ++i) {
        input0_local.SetValue(i, 1);
    }
    for (int32_t i = 0; i < offsetLen; ++i) {
        offset_local.SetValue(i, 0);
    }

    AscendCUtils::SetMask<uint8_t>(256);
    DataCopy(input0_local, input0_global, srcLen);
    DataCopy(offset_local, offset_global, offsetLen);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    GatherRepeatParams params { 1, 8 };
    uint8_t repeat = srcLen * sizeof(T) / 256;
    Gatherb(output_local, input0_local, offset_local, repeat, params);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    DataCopy(output_global, output_local, dstLen);
    pipe_barrier(PIPE_ALL);
}

INSTANTIATE_TEST_CASE_P(TEST_VGATHERB, GatherbTestSuite,
    ::testing::Values(
        GatherbTestParams { 128, 16, 1, test_vgatherb_func<uint8_t> },
        GatherbTestParams { 256, 32, 1, test_vgatherb_func<int8_t> },
        GatherbTestParams { 128, 8, 2, test_vgatherb_func<half> },
        GatherbTestParams { 128, 8, 2, test_vgatherb_func<bfloat16_t> },
        GatherbTestParams { 128, 8, 2, test_vgatherb_func<uint16_t> },
        GatherbTestParams { 256, 32, 4, test_vgatherb_func<int32_t> },
        GatherbTestParams { 128, 16, 4, test_vgatherb_func<uint32_t> },
        GatherbTestParams { 128, 16, 4, test_vgatherb_func<float> },
        GatherbTestParams { 64, 16, 8, test_vgatherb_func<int64_t> },
        GatherbTestParams { 64, 16, 8, test_vgatherb_func<uint64_t> }
    )
);

TEST_P(GatherbTestSuite, GatherbTestCase)
{
    auto param = GetParam();
    uint8_t dst_gm[param.srcLen * param.dataBitSize] = {0};
    uint8_t src_gm[param.srcLen * param.dataBitSize] = {0};
    uint8_t offset_gm[param.offsetLen * 4] = {0};
    param.cal_func(dst_gm, src_gm, offset_gm, param.srcLen, param.offsetLen);
    for (size_t i = 0; i < param.srcLen; ++i) {
        EXPECT_EQ(dst_gm[i], 0x00);
    }
    return;
}