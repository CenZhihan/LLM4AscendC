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
#define protected public
#include "kernel_operator.h"
// #include "model/model_factory.h"
#include "common.h"

using namespace std;
using namespace AscendC;

class TEST_GEMM : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TEST_GEMM SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "TEST_GEMM TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

extern "C" __global__ __aicore__ void main_gemm_test(__gm__ half* __restrict__ input0Gm,
    __gm__ half* __restrict__ input1Gm, __gm__ float* __restrict__ outputGm, int32_t m, int32_t k, int32_t n,
    int32_t c0Size, int32_t initValue, LoopMode mode)
{
    TPipe tpipe;
    int32_t roundm = DivCeil(m, 16) * 16;
    int32_t roundn = DivCeil(n, 16) * 16;
    int32_t roundk = DivCeil(k, c0Size) * c0Size;

    TBuf<TPosition::A1> tbuf;
    tpipe.InitBuffer(tbuf, roundm * roundk * sizeof(half));
    LocalTensor<half> input0 = tbuf.Get<half>();

    TBuf<TPosition::A1> tbuf1;
    tpipe.InitBuffer(tbuf1, roundk * roundn * sizeof(half));
    LocalTensor<half> input1 = tbuf1.Get<half>();

    TBuf<TPosition::C2> tbuf2;
    tpipe.InitBuffer(tbuf2, roundm * roundn * sizeof(float));
    LocalTensor<float> output = tbuf2.Get<float>();

    TBuf<TPosition::C1> tbuf3;
    tpipe.InitBuffer(tbuf3, roundm * roundn * sizeof(float));
    LocalTensor<float> ublocal = tbuf3.Get<float>();

    copy_gm_to_cbuf((__cbuf__ half*)input0.GetPhyAddr(), input0Gm, 0, 1, roundm * roundk * 2 / 32, 0, 0, PAD_NONE);
    copy_gm_to_cbuf((__cbuf__ half*)input1.GetPhyAddr(), input1Gm, 0, 1, roundk * roundn * 2 / 32, 0, 0, PAD_NONE);
    pipe_barrier(PIPE_ALL);
    GemmTiling tilling = GetGemmTiling<half>(m, k, n);
    tilling.loopMode = mode;
    Gemm(output, input0, input1, m, k, n, tilling, false, initValue);

    pipe_barrier(PIPE_ALL);
    copy_matrix_cc_to_ubuf((__ubuf__ float*)ublocal.GetPhyAddr(), (__cc__ float*)output.GetPhyAddr(), 0, 1,
        roundm * roundn * 4 / 1024, 0, 0, CRMODE_NONE);
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm(outputGm, (__ubuf__ float*)ublocal.GetPhyAddr(), 0, 1, roundm * roundn * 4 / 32, 0, 0);
    pipe_barrier(PIPE_ALL);
}

struct GemmTestParams {
    int32_t m;
    int32_t k;
    int32_t n;
    int32_t c0Size;
    int32_t initValue;
    LoopMode mode;
};

class GemmTestSuite : public testing::Test, public testing::WithParamInterface<GemmTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GemmTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "GemmTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_GEMM, GemmTestSuite,
    ::testing::Values(GemmTestParams { 32, 32, 32, 16, 0, LoopMode::MODE_NM },
    GemmTestParams { 64, 32, 64, 16, 0, LoopMode::MODE_NM },
    GemmTestParams { 256, 128, 32, 16, 1, LoopMode::MODE_NM },
    GemmTestParams { 2048, 128, 32, 16, 1, LoopMode::MODE_NM },

    GemmTestParams { 16, 16, 16, 16, 0, LoopMode::MODE_MN },
    GemmTestParams { 256, 128, 32, 16, 1, LoopMode::MODE_MN },
    GemmTestParams { 256, 129, 32, 16, 1, LoopMode::MODE_MN },
    GemmTestParams { 256, 129, 32, 16, 0, LoopMode::MODE_MN },
    GemmTestParams { 2048, 128, 32, 16, 1, LoopMode::MODE_MN }));

TEST_P(GemmTestSuite, GemmTestCase)
{
    auto param = GetParam();
    int32_t roundm = DivCeil(param.m, 16) * 16;
    int32_t roundn = DivCeil(param.n, 16) * 16;
    int32_t roundk = DivCeil(param.k, param.c0Size) * param.c0Size;
    half input0Gm[roundm * roundk] { 0x0000 };
    half input1Gm[roundn * roundk] { 0x0000 };
    float outputGm[roundm * roundn] { 0x00000000 };
    // main_gemm_test(input0Gm, input1Gm, outputGm, param.m, param.k, param.n, param.c0Size, param.initValue, param.mode);
    for (int32_t i = 0; i < param.m * param.n; i++) {
        EXPECT_EQ(outputGm[i], 0x00);
    }
}
