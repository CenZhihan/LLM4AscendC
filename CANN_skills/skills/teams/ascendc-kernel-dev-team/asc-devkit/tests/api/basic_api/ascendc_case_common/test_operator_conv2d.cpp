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

extern "C" __global__ __aicore__ void main_Conv2D_test(__gm__ half* __restrict__ input0Gm,
    __gm__ half* __restrict__ input1Gm, __gm__ float* __restrict__ outputGm, Conv2dParams& conv2dParams,
    Conv2dTilling& tilling)
{
    AscendC::TPipe tpipe;
    int32_t roundm = tilling.roundM;
    int32_t roundn = tilling.roundN;
    int32_t roundk = tilling.roundK;

    TBuf<TPosition::A1> tbuf;
    tpipe.InitBuffer(tbuf, roundm * roundk * sizeof(half));
    LocalTensor<half> input0 = tbuf.Get<half>();

    TBuf<TPosition::A1> tbuf1;
    tpipe.InitBuffer(tbuf1, roundk * roundn * sizeof(half));
    LocalTensor<half> input1 = tbuf1.Get<half>();


    TBuf<TPosition::CO1> tbuf2;
    tpipe.InitBuffer(tbuf2, roundm * roundn * sizeof(float));
    LocalTensor<float> output = tbuf2.Get<float>();

    TBuf<TPosition::C1> tbuf3;
    tpipe.InitBuffer(tbuf3, roundm * roundn * sizeof(float));
    LocalTensor<float> ublocal = tbuf3.Get<float>();

    copy_gm_to_cbuf((__cbuf__ half*)input0.GetPhyAddr(), input0Gm, 0, 1, roundm * roundk * 2 / 32, 0, 0, PAD_NONE);
    copy_gm_to_cbuf((__cbuf__ half*)input1.GetPhyAddr(), input1Gm, 0, 1, roundk * roundn * 2 / 32, 0, 0, PAD_NONE);
    pipe_barrier(PIPE_ALL);

    Conv2D(output, input0, input1, conv2dParams, tilling);

    pipe_barrier(PIPE_ALL);
    copy_matrix_cc_to_ubuf((__ubuf__ float*)ublocal.GetPhyAddr(), (__cc__ float*)output.GetPhyAddr(), 0, 1,
        roundm * roundn * 4 / 1024, 0, 0, CRMODE_NONE);
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm(outputGm, (__ubuf__ float*)ublocal.GetPhyAddr(), 0, 1, roundm * roundn * 4 / 32, 0, 0);
    pipe_barrier(PIPE_ALL);
}

extern "C" __global__ __aicore__ void main_Conv2D_doublebuffer_test(__gm__ half *__restrict__ input0Gm,
    __gm__ half *__restrict__ input1Gm, __gm__ float *__restrict__ outputGm, __gm__ float *__restrict__ bias,
    Conv2dParams &conv2dParams, Conv2dTilling &tilling)
{
    AscendC::TPipe tpipe;
    TQue<TPosition::A1, 1> inQueueSrcA1;
    tpipe.InitBuffer(inQueueSrcA1, 1, 32);
    int32_t roundm = tilling.roundM;
    int32_t roundn = tilling.roundN;
    int32_t roundk = tilling.roundK;

    int32_t src0Size = conv2dParams.cin * conv2dParams.imgShape[0] * conv2dParams.imgShape[1];
    int32_t src1Size = conv2dParams.cin * conv2dParams.cout * conv2dParams.kernelShape[0] * conv2dParams.kernelShape[1];

    GlobalTensor<half> src0_global;
    GlobalTensor<half> src1_global;
    GlobalTensor<float> dst_global;
    GlobalTensor<float> bias_global;

    src0_global.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(input0Gm), src0Size);
    src1_global.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(input1Gm), src1Size);
    dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputGm), roundm * roundn);
    bias_global.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias), conv2dParams.cout);

    TBuf<TPosition::A1> tbuf;
    tpipe.InitBuffer(tbuf, src0Size * sizeof(half));
    LocalTensor<half> input0 = tbuf.Get<half>();

    TBuf<TPosition::A1> tbuf1;
    tpipe.InitBuffer(tbuf1, src1Size * sizeof(half));
    LocalTensor<half> input1 = tbuf1.Get<half>();


    TBuf<TPosition::CO1> tbuf2;
    tpipe.InitBuffer(tbuf2, roundm * roundn * sizeof(float));
    LocalTensor<float> output = tbuf2.Get<float>();

    TBuf<TPosition::A1> tbuf3;
    tpipe.InitBuffer(tbuf3, conv2dParams.cout * sizeof(float));
    LocalTensor<float> biasfixlocal = tbuf3.Get<float>();

    TBuf<TPosition::CO2> tbuf4;
    tpipe.InitBuffer(tbuf4, conv2dParams.cout * sizeof(float));
    LocalTensor<float> biaslocal = tbuf4.Get<float>();

    DataCopy(biasfixlocal, bias_global, conv2dParams.cout);
    DataCopy(input0, src0_global, src0Size);
    DataCopy(input1, src1_global, src1Size);
    DataCopy(biaslocal, bias_global, conv2dParams.cout);

    Conv2D(output, biaslocal, input0, input1, conv2dParams, tilling);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510)
    FixpipeParams<float> fixpipeParams(static_cast<uint16_t>(conv2dParams.cout / BLOCK_CUBE),
        static_cast<uint16_t>(tilling.howo * BLOCK_CUBE * sizeof(float) / ONE_BLK_SIZE),
        0,
        0);
    Fixpipe(dst_global, output, fixpipeParams);
#endif
}

struct Conv2DTestParams {
    uint32_t imgShape[CONV2D_IMG_SIZE];
    uint32_t kernelShape[CONV2D_KERNEL_SIZE];
    uint32_t stride[CONV2D_STRIDE];
    uint32_t cin;
    uint32_t cout;
    uint32_t padList[CONV2D_PAD];
    uint32_t dilation[CONV2D_DILATION];
    uint32_t initY;
    bool partialSum;
    LoopMode mode;
};

class Conv2DTestSuite : public testing::Test, public testing::WithParamInterface<Conv2DTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Conv2DTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "Conv2DTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_Conv2D, Conv2DTestSuite,
    ::testing::Values(
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 1001)
    Conv2DTestParams { { 40, 40 }, { 5, 5 }, { 2, 2 }, 32, 16, { 0, 0, 0, 0 }, { 3, 3 }, 2, false, LoopMode::MODE_MN },
    Conv2DTestParams { { 16, 16 }, { 2, 2 }, { 1, 1 }, 32, 16, { 0, 0, 0, 0 }, { 1, 1 }, 0, false, LoopMode::MODE_NM }
#else
    Conv2DTestParams { { 4, 4 }, { 2, 2 }, { 1, 1 }, 32, 15, { 0, 0, 0, 0 }, { 2, 2 }, 0, false, LoopMode::MODE_NM },
    Conv2DTestParams { { 4, 1024 }, { 2, 2 }, { 1, 1 }, 32, 16, { 0, 0, 0, 0 }, { 2, 2 }, 1, false, LoopMode::MODE_NM },
    Conv2DTestParams { { 1024, 4 }, { 2, 2 }, { 1, 1 }, 32, 16, { 0, 0, 0, 0 }, { 2, 2 }, 1, true, LoopMode::MODE_NM },
    Conv2DTestParams { { 64, 32 }, { 2, 2 }, { 1, 1 }, 32, 16, { 0, 0, 0, 0 }, { 2, 2 }, 0, true, LoopMode::MODE_NM },
    Conv2DTestParams { { 32, 64 }, { 2, 2 }, { 1, 1 }, 32, 16, { 0, 0, 0, 0 }, { 2, 2 }, 0, false, LoopMode::MODE_MN },
    Conv2DTestParams { { 64, 64 }, { 3, 3 }, { 1, 1 }, 4, 16, { 0, 0, 0, 0 }, { 1, 1 }, 0, false, LoopMode::MODE_NM },
    Conv2DTestParams { { 128, 128 }, { 3, 3 }, { 1, 32 }, 4, 16, { 0, 0, 0, 0 }, { 1, 1 }, 1, false, LoopMode::MODE_NM },
    Conv2DTestParams { { 256, 256 }, { 3, 3 }, { 32, 1 }, 4, 16, { 0, 0, 0, 0 }, { 1, 1 }, 1, true, LoopMode::MODE_NM },
    Conv2DTestParams { { 512, 512 }, { 3, 3 }, { 32, 32 }, 4, 16, { 0, 0, 0, 0 }, { 1, 1 }, 0, true, LoopMode::MODE_NM }
#endif
    ));

TEST_P(Conv2DTestSuite, Conv2DTestCase)
{
    auto Params = GetParam();
    Conv2dParams conv2dParams(Params.imgShape, Params.kernelShape, Params.stride, Params.cin, Params.cout,
        Params.padList, Params.dilation, Params.initY, Params.partialSum);
    Conv2dTilling tilling = GetConv2dTiling<half>(conv2dParams);
    tilling.loopMode = Params.mode;

    int32_t roundm = tilling.roundM;
    int32_t roundn = tilling.roundN;
    int32_t roundk = tilling.roundK;

    half input0Gm[roundm * roundk]{0x0000};
    half input1Gm[roundn * roundk]{0x0000};
    float outputGm[roundm * roundn]{0x00000000};
    float bias[conv2dParams.cout]{0x00000000};
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 1001)
    main_Conv2D_doublebuffer_test(input0Gm, input1Gm, outputGm, bias, conv2dParams, tilling);
#else
    main_Conv2D_test(input0Gm, input1Gm, outputGm, conv2dParams, tilling);
#endif
    for (int32_t i = 0; i < tilling.mNum * tilling.nNum; i++) {
        EXPECT_EQ(outputGm[i], 0x00);
    }
}
