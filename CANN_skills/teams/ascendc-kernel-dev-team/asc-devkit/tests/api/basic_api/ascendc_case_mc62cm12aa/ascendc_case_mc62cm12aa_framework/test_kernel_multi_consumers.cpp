/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <gtest/gtest.h>
#include "kernel_tpipe_impl.h"
#include <vector>

namespace AscendC {
namespace {
constexpr int32_t DATA_SIZE = 1024;
} // namespace

class TestTPipe : public testing::Test {
protected:
    void SetUp() {
        AscendC::SetGCoreType(2);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
    }
};

TEST_F(TestTPipe, TestMultiConsumerCopyOutAndCompute)
{
    TPipe pipe;

    TQue<TPosition::VECIN, 1> que;
    pipe.InitBuffer(que, 1, DATA_SIZE * sizeof(float));

    LocalTensor<float> t = que.AllocTensor<float>();

    que.EnQue<TPosition::VECIN, TPosition::VECOUT>(t);
    t = que.DeQue<TPosition::VECIN, TPosition::VECOUT, float>();

    que.EnQue<TPosition::VECOUT, TPosition::VECCALC>(t);
    t = que.DeQue<TPosition::VECOUT, TPosition::VECCALC, float>();
    que.FreeTensor(t);
}

TEST_F(TestTPipe, TestMultiConsumerParallelCopyOutAndCompute)
{
    TPipe pipe;

    static constexpr TQueConfig consumerConfig = {.consumerSize = 2, .consumer = {TPosition::VECCALC, TPosition::VECOUT}};
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1, &consumerConfig> que;
    pipe.InitBuffer(que, 1, DATA_SIZE * sizeof(float));

    LocalTensor<float> t = que.AllocTensor<float>();

    que.EnQue<TPosition::VECIN, TPosition::VECOUT>(t);
    t = que.DeQue<TPosition::VECIN, TPosition::VECOUT, float>();

    que.EnQue<TPosition::VECIN, TPosition::VECCALC>(t);
    t = que.DeQue<TPosition::VECIN, TPosition::VECCALC, float>();
    que.FreeTensor(t);
}

TEST_F(TestTPipe, TestMultiConsumerCopyOutAndComputeAndCopyOut)
{
    TPipe pipe;

    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> que;
    pipe.InitBuffer(que, 1, DATA_SIZE * sizeof(float));

    auto t = que.AllocTensor<float>();

    que.EnQue<TPosition::VECIN, TPosition::VECOUT>(t);
    t = que.DeQue<TPosition::VECIN, TPosition::VECOUT, float>();

    que.EnQue<TPosition::VECOUT, TPosition::VECCALC>(t);
    t = que.DeQue<TPosition::VECOUT, TPosition::VECCALC, float>();

    que.EnQue<TPosition::VECCALC, TPosition::VECOUT>(t);
    t = que.DeQue<TPosition::VECCALC, TPosition::VECOUT, float>();

    que.FreeTensor(t);
}

TEST_F(TestTPipe, TestMultiConsumerCopyInAndOut)
{
    TPipe pipe;

    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> que;
    pipe.InitBuffer(que, 1, DATA_SIZE * sizeof(float));

    LocalTensor<float> t = que.AllocTensor<float>();

    que.EnQue(t);
    t = que.DeQue<float>();

    que.FreeTensor(t);
}

TEST_F(TestTPipe, TestMultiConsumerCopyInComputeAndOut)
{
    TPipe pipe;

    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> que;
    pipe.InitBuffer(que, 1, DATA_SIZE * sizeof(float));

    LocalTensor<float> t = que.AllocTensor<float>();

    que.EnQue<TPosition::VECIN, TPosition::VECCALC>(t);
    t = que.DeQue<TPosition::VECIN, TPosition::VECCALC, float>();

    que.EnQue<TPosition::VECCALC, TPosition::VECOUT>(t);
    t = que.DeQue<TPosition::VECCALC, TPosition::VECOUT, float>();

    que.FreeTensor(t);
}

TEST_F(TestTPipe, TestBufIdTtracker)
{
    BufIdTracker::GetInstance().AddBufIdEvent(0, PIPE_V, false, false);
    EXPECT_EQ(BufIdTracker::GetInstance().GetState(), true);
    BufIdTracker::GetInstance().AddBufIdEvent(0, PIPE_V, false, true);
    EXPECT_EQ(BufIdTracker::GetInstance().GetState(), true);
}
} // namespace AscendC