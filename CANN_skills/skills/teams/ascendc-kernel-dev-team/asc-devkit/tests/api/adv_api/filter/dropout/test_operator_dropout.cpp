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

using namespace std;
using namespace AscendC;

template <typename srcType>
void main_DropOut_test(__gm__ uint8_t* __restrict__ src_gm, __gm__ uint8_t* __restrict__ mask_gm,
    __gm__ uint8_t* __restrict__ dst_gm, __gm__ uint32_t firstAxis, __gm__ uint32_t srcLastAxis,
    __gm__ uint32_t maskLastAxis)
{
    uint32_t srcSize = firstAxis * srcLastAxis;
    uint32_t maskSize = firstAxis * maskLastAxis;

    GlobalTensor<srcType> src_global;
    GlobalTensor<uint8_t> mask_global;
    GlobalTensor<srcType> dst_global;

    src_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType*>(src_gm), srcSize);
    mask_global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mask_gm), maskSize);
    dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType*>(dst_gm), srcSize);

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECIN, 1> inQueueY;
    TQue<TPosition::VECOUT, 1> outQueue;

    pipe.InitBuffer(inQueueX, 1, srcSize * sizeof(srcType) + DEFAULT_BLOCK_SIZE);
    pipe.InitBuffer(inQueueY, 1, maskSize * sizeof(uint8_t));
    pipe.InitBuffer(outQueue, 1, srcSize * sizeof(srcType) + DEFAULT_BLOCK_SIZE);

    LocalTensor<srcType> srcLocal = inQueueX.AllocTensor<srcType>();
    LocalTensor<uint8_t> maskLocal = inQueueY.AllocTensor<uint8_t>();
    LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();

    DataCopy(srcLocal, src_global, srcSize);
    DataCopy(maskLocal, mask_global, maskSize);
    PipeBarrier<PIPE_ALL>();

    DropOutShapeInfo info;
    info.firstAxis = firstAxis;
    info.srcLastAxis = srcLastAxis;
    info.maskLastAxis = maskLastAxis;

    float probValue = 0.5;
    DropOut(dstLocal, srcLocal, maskLocal, probValue, info);
    PipeBarrier<PIPE_ALL>();

    DataCopy(dst_global, dstLocal, srcSize);
    inQueueX.FreeTensor(srcLocal);
    inQueueY.FreeTensor(maskLocal);
    outQueue.FreeTensor(dstLocal);
    PipeBarrier<PIPE_ALL>();
}

struct DropOutTestParams {
    uint32_t firstAxis;
    uint32_t srcLastAxis;
    uint32_t maskLastAxis;

    int32_t srcTypeSize;
    int32_t maskTypeSize;
    void (*cal_func)(uint8_t*, uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t);
};

class DropOutTestSuite : public testing::Test, public testing::WithParamInterface<DropOutTestParams> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DropOutTestSuite SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "DropOutTestSuite TearDownTestCase" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};

INSTANTIATE_TEST_CASE_P(TEST_PACKAGE_DropOut, DropOutTestSuite,
    ::testing::Values(
    DropOutTestParams { 4, 256, 256, sizeof(half), sizeof(uint8_t), main_DropOut_test<half> },
    DropOutTestParams { 4, 512, 512, sizeof(half), sizeof(uint8_t), main_DropOut_test<half> },
    DropOutTestParams { 4, 256, 256, sizeof(float), sizeof(uint8_t), main_DropOut_test<float> },
    DropOutTestParams { 4, 512, 512, sizeof(float), sizeof(uint8_t), main_DropOut_test<float> },

    DropOutTestParams { 4, 256, 32, sizeof(half), sizeof(uint8_t), main_DropOut_test<half> },
    DropOutTestParams { 4, 512, 64, sizeof(half), sizeof(uint8_t), main_DropOut_test<half> },
    DropOutTestParams { 4, 256, 32, sizeof(float), sizeof(uint8_t), main_DropOut_test<float> },
    DropOutTestParams { 4, 512, 64, sizeof(float), sizeof(uint8_t), main_DropOut_test<float> },

    DropOutTestParams { 4, 80, 96, sizeof(half), sizeof(uint8_t), main_DropOut_test<half> },
    DropOutTestParams { 4, 80, 96, sizeof(float), sizeof(uint8_t), main_DropOut_test<float> },

    DropOutTestParams { 4, 80, 32, sizeof(half), sizeof(uint8_t), main_DropOut_test<half> },
    DropOutTestParams { 4, 80, 32, sizeof(float), sizeof(uint8_t), main_DropOut_test<float> }
    ));

TEST_P(DropOutTestSuite, DropOutTestCase)
{
    auto param = GetParam();

    uint32_t srcSize = param.firstAxis * param.srcLastAxis * param.srcTypeSize;
    uint32_t maskSize = param.firstAxis * param.maskLastAxis * param.maskTypeSize;

    uint8_t inputGm[srcSize] { 0x00 };
    uint8_t maskGm[maskSize] { 0x00 };
    uint8_t outputGm[srcSize] { 0x00 };

    param.cal_func(inputGm, maskGm, outputGm, param.firstAxis, param.srcLastAxis, param.maskLastAxis);

    for (int32_t i = 0; i < srcSize; i++) {
        EXPECT_EQ(outputGm[i], 0x00);
    }
}